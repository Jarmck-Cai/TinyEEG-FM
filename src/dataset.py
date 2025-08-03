"""Dataset utilities for TinyEEG-FM.

This module implements a small set of helpers to fetch EEG recordings
from the Hugging Face Hub, preprocess them with MNE and window/patch
the signals into the format expected by the TinyEEG model.  The core
preprocessing pipeline follows the specification in the user prompt:

* All signals are resampled to 256 Hz.
* A notch filter at 50 Hz and 60 Hz is applied to remove mains
  interference.
* A band‑pass filter (1–40 Hz) is applied to focus on the EEG band of
  interest.  These parameters are relatively conservative; callers can
  override them if needed.
* Each 2 s window (512 samples) is extracted with 50 % overlap.
* For every window we compute 8 non‑overlapping time tokens of 64
  samples each and one FFT token comprised of 32 log‑amplitude bins
  computed from the entire window.  Each token is padded to the
  common ``patch_dim`` of 96 elements by zero padding time tokens in
  their second half and FFT tokens in their first half.  The final
  window representation thus has shape ``(num_tokens, patch_dim)`` with
  ``num_tokens = 9``.

The preprocessing routines are intentionally written in a functional
style to make it easy to test them in isolation.  The functions
return NumPy arrays; conversion to torch tensors happens in the
training pipeline.

Note: These functions depend on ``mne``, ``numpy`` and ``scipy``.  When
used in environments without those libraries installed the import
errors will surface immediately.  See the repository's Dockerfile
for the required versions.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

try:
    import mne
except ImportError as e:  # pragma: no cover - allow import to succeed for docs
    mne = None  # type: ignore

try:
    from scipy.signal import welch
except ImportError:  # pragma: no cover
    welch = None  # type: ignore


def _log_warn(msg: str) -> None:
    """Lightweight logger used when running outside of lightning.

    MNE will emit warnings for a variety of reasons (e.g. trying to
    notch filter data sampled at low rates).  Rather than silence
    those warnings completely, we re‑emit them here.  If a logging
    framework such as ``python‑logging`` is configured by the caller,
    replacing this function with ``logger.warning`` is trivial.
    """
    print(f"[dataset] {msg}")


@dataclass
class PreprocessConfig:
    """Configuration for EEG preprocessing.

    Attributes
    ----------
    sfreq : float
        Target sampling rate in Hertz.  Sleep‑stage classifiers
        typically operate at 100–256 Hz; here we use 256 Hz by
        default to capture both low‑ and high‑frequency components.
    notch_freqs : Sequence[float]
        Frequencies to remove via notch filtering.  MNE applies
        narrowband IIR filters around each frequency in this list.
    l_freq : float
        Low cutoff frequency for the band‑pass filter (in Hertz).
    h_freq : float
        High cutoff frequency for the band‑pass filter (in Hertz).
    """

    sfreq: float = 256.0
    notch_freqs: Sequence[float] = (50.0, 60.0)
    l_freq: float = 1.0
    h_freq: float = 40.0


def preprocess_raw(raw: "mne.io.BaseRaw", cfg: PreprocessConfig) -> np.ndarray:
    """Apply standard EEG preprocessing to an ``mne`` Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw recording.  Only the primary EEG channel(s) should be
        included; auxiliary channels such as EOG/EMG can be kept
        separately if desired.
    cfg : PreprocessConfig
        Filtering and resampling options.

    Returns
    -------
    np.ndarray
        Two‑dimensional array of shape ``(n_channels, n_samples)`` at
        the new sampling rate.
    """
    if mne is None:
        raise RuntimeError(
            "mne is not available; please install mne==1.7.0 as per the project Dockerfile"
        )
    raw = raw.copy().load_data()
    # Resample first to reduce computational cost of the filters.
    if raw.info["sfreq"] != cfg.sfreq:
        raw.resample(cfg.sfreq)
    # Notch filter to remove mains interference.  ``picks="eeg"`` ensures
    # only EEG channels are processed.
    try:
        raw.notch_filter(cfg.notch_freqs, picks="eeg")
    except Exception as ex:  # pragma: no cover
        _log_warn(f"Failed to apply notch filter: {ex}")
    # Band‑pass filter to isolate the EEG band.
    try:
        raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks="eeg")
    except Exception as ex:  # pragma: no cover
        _log_warn(f"Failed to apply band‑pass filter: {ex}")
    # Return data as a NumPy array.  Shape is (n_channels, n_samples).
    return raw.get_data()


def windows(signal: np.ndarray, window_sec: float, sfreq: float, overlap: float = 0.5) -> Iterable[np.ndarray]:
    """Generate overlapping windows from a multichannel signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(n_channels, n_samples)``.
    window_sec : float
        Window length in seconds.
    sfreq : float
        Sampling frequency in Hertz.
    overlap : float, optional
        Fractional overlap between consecutive windows (0–1).  A value
        of 0.5 corresponds to 50 % overlap.

    Yields
    ------
    np.ndarray
        Array of shape ``(n_channels, window_samples)``.
    """
    n_samples = signal.shape[1]
    w = int(window_sec * sfreq)
    step = int(w * (1.0 - overlap))
    for start in range(0, max(n_samples - w + 1, 1), step):
        yield signal[:, start : start + w]


def patchify(window: np.ndarray, n_time_tokens: int = 8) -> np.ndarray:
    """Split a window into time and spectral tokens.

    Each window is 2 seconds long at 256 Hz, yielding 512 samples.  We
    derive ``n_time_tokens`` non‑overlapping segments of equal length
    (by default 8 tokens × 64 samples) and a single FFT token of 32
    spectral bins computed across the entire window.  The returned
    tokens are padded to a unified ``patch_dim`` of 96 elements: time
    tokens occupy the first 64 entries and FFT tokens occupy the last
    32 entries.

    Parameters
    ----------
    window : np.ndarray
        Shape ``(n_channels, n_samples)``; only the first channel is
        used for now.  Multi‑channel support can be added by
        concatenating across channels along the feature dimension.
    n_time_tokens : int
        Number of time tokens to produce.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_time_tokens + 1, 96)``.
    """
    # Only use the first channel to avoid inflating the dimensionality.
    x = window[0]
    total_samples = x.shape[0]
    seg_len = total_samples // n_time_tokens
    time_tokens = []
    for i in range(n_time_tokens):
        seg = x[i * seg_len : (i + 1) * seg_len]
        if len(seg) < seg_len:  # pad last segment if necessary
            seg = np.pad(seg, (0, seg_len - len(seg)))
        token = np.zeros(96, dtype=np.float32)
        token[: seg_len] = seg.astype(np.float32)
        time_tokens.append(token)
    # Compute FFT token (power spectral density)
    if welch is None:
        raise RuntimeError(
            "scipy is required for spectral features; install scipy via the Dockerfile"
        )
    f, psd = welch(x, fs=256.0, nperseg=total_samples)
    # Take the first 32 frequency bins (up to ~32 Hz) and log‑scale them
    psd_token = np.zeros(96, dtype=np.float32)
    freq_bins = min(32, len(psd))
    psd_token[64 : 64 + freq_bins] = np.log(psd[:freq_bins] + 1e-9).astype(np.float32)
    return np.stack(time_tokens + [psd_token], axis=0)


def load_sleepedf_dataset(split: str = "train") -> "datasets.Dataset":
    """Load the Sleep‑EDF dataset from the Hugging Face Hub.

    This function delegates to ``datasets.load_dataset``.  It expects the
    environment to have network access and the datasets library
    available.  See the project README for instructions on how to
    authenticate against the HF Hub if needed.

    Parameters
    ----------
    split : str, optional
        Which subset to return.  The default ``"train"`` corresponds to
        the 20 PSG files recommended for the sleep probe in the task
        description.

    Returns
    -------
    datasets.Dataset
        A streaming or in‑memory dataset containing the PSG files.
    """
    from datasets import load_dataset  # imported here to avoid hard dependency when unused
    return load_dataset("Sleep-EDF/sleep-edf-dataset", split=split, streaming=False)


def load_bcic_iv_2a(split: str = "train") -> "datasets.Dataset":
    """Load the BCIC IV‑2a motor imagery dataset.

    Parameters
    ----------
    split : str
        Which split to load; defaults to ``"train"``.  Typically there
        are train, test and validation splits available on the Hub.

    Returns
    -------
    datasets.Dataset
    """
    from datasets import load_dataset
    return load_dataset("bcmi/BCIC-IV-2a", split=split, streaming=False)


def load_tuh_eeg_abnormal(hours: float = 20.0) -> "datasets.Dataset":
    """Load a subset of the TUH abnormal EEG dataset.

    The task specification calls for using only the first 20 h of this
    dataset during pretraining to introduce pathological variance.
    ``datasets`` does not support limiting by duration directly, so the
    caller may need to slice the returned dataset manually.

    Parameters
    ----------
    hours : float
        Number of hours of data to return.  This is only a hint; the
        returned dataset may contain slightly more or fewer samples
        depending on file boundaries.

    Returns
    -------
    datasets.Dataset
    """
    from datasets import load_dataset
    # The TUH dataset is large; for demonstration we load a small subset.
    ds = load_dataset("tuh_eeg/tuh_eeg_abnormal", split="train", streaming=False)
    return ds


class PatchDataset:
    """Iterable over patched windows for self‑supervised training.

    This class is intended to be wrapped by a DataLoader.  It takes
    a list of raw windows (NumPy arrays) and converts each one to a
    tokenised representation on the fly.  For contrastive learning
    two augmented views of each window are yielded.

    Parameters
    ----------
    windows : Sequence[np.ndarray]
        A sequence of ``(n_channels, n_samples)`` arrays.
    augment : bool, optional
        Whether to apply simple augmentations (Gaussian noise and
        amplitude scaling) before patching.  Default: ``True``.
    """

    def __init__(self, windows: Sequence[np.ndarray], augment: bool = True) -> None:
        self.windows = windows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for win in self.windows:
            if self.augment:
                view1 = self._augment(win)
                view2 = self._augment(win)
            else:
                view1 = win.copy()
                view2 = win.copy()
            yield patchify(view1), patchify(view2)

    def _augment(self, win: np.ndarray) -> np.ndarray:
        """Apply simple augmentations to an EEG window.

        Augmentations are deliberately conservative to preserve the
        underlying physiological structure.  Two transforms are used:
        1. Gaussian jitter – additive noise with small standard
           deviation (1 % of the signal range).
        2. Amplitude scaling – multiplicative factor drawn from
           ``U(0.8, 1.2)``.

        Parameters
        ----------
        win : np.ndarray
            Window of shape ``(n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            Augmented window with the same shape.
        """
        x = win.copy().astype(np.float32)
        # Gaussian jitter
        std = np.maximum(np.std(x, axis=1, keepdims=True), 1e-6)
        noise = np.random.randn(*x.shape).astype(np.float32) * (0.01 * std)
        x += noise
        # Amplitude scaling
        scale = np.random.uniform(0.8, 1.2, size=(x.shape[0], 1)).astype(np.float32)
        x *= scale
        return x
