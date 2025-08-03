"""Self‑supervised pretraining script for TinyEEG‑FM.

This script uses PyTorch Lightning to orchestrate the training of a
TinyEEGModel with a combination of masked EEG modelling and
contrastive (NT‑Xent) losses.  The configuration is supplied via a
YAML file (see ``configs/pretrain.yaml`` for an example) and
defines hyper‑parameters such as the learning rate, number of
epochs, batch size, masking ratio, temperature, and gradient
accumulation factor.

The training pipeline consists of the following stages:

1. Load raw recordings from the Hugging Face datasets as specified
   by the data configuration.  Preprocess each recording (resample,
   notch filter, band‑pass) and extract overlapping windows.
2. Construct a dataset of window pairs for contrastive learning.  Two
   augmented views of each window are generated on the fly using
   amplitude scaling and Gaussian noise.
3. Sample batches of patch tokens, apply random masking, and feed
   them through the TinyEEG model.  Compute reconstruction loss on
   the masked patches and NT‑Xent loss on the CLS representations.
4. Optimise with AdamW and a cosine LR scheduler.  Optionally
   integrate LoRA adapters for memory‑efficient training.

Note that this script does not save checkpoints or push models to
the hub automatically; those operations are handled by the main
``train_ssl.py`` entry point when run from the repository root.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from .dataset import PatchDataset, PreprocessConfig, preprocess_raw, windows, load_sleepedf_dataset, patchify
from .model import TinyEEGConfig, TinyEEGModel, add_lora


class SSLDataset(IterableDataset):
    """Lightning iterable dataset that yields two augmented views of EEG windows."""

    def __init__(self, window_list: List[np.ndarray]) -> None:
        self.dataset = PatchDataset(window_list, augment=True)

    def __iter__(self):
        for x1, x2 in self.dataset:
            yield torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)


class SSLModule(pl.LightningModule):
    """Lightning module encapsulating the TinyEEG self‑supervised objective."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = TinyEEGConfig()
        self.model = TinyEEGModel(model_cfg)
        # Optionally wrap with LoRA
        if cfg.get("lora", False):
            self.model = add_lora(
                self.model,
                r=cfg.get("lora_r", 8),
                alpha=cfg.get("lora_alpha", 16.0),
                dropout=cfg.get("lora_dropout", 0.05),
                target_modules=cfg.get("lora_targets", None),
            )
        self.mask_ratio = cfg.get("mask_ratio", 0.4)
        self.temperature = cfg.get("temperature", 0.07)
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 1e-2)
        self.num_epochs = cfg.get("epochs", 50)
        self.save_hyperparameters()

    def _apply_mask(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Randomly mask a fraction of patches.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, seq_len, patch_dim)``.

        Returns
        -------
        (masked_x, mask)
            ``masked_x`` has the same shape as ``x`` with masked patches
            set to zero.  ``mask`` is a boolean tensor marking the
            positions that were masked (True for masked).
        """
        b, l, d = x.shape
        mask = torch.rand(b, l, device=x.device) < self.mask_ratio
        x_masked = x.clone()
        # Zero out masked tokens
        x_masked[mask] = 0.0
        return x_masked, mask

    @staticmethod
    def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
        """Compute the NT‑Xent (InfoNCE) loss on a pair of batches.

        Parameters
        ----------
        z1, z2 : Tensor
            Two batches of embeddings of shape ``(batch, dim)``.
        temperature : float
            Temperature parameter controlling the sharpness of the
            similarity distribution.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, d)
        # Compute similarity matrix
        sim = torch.matmul(z, z.t()) / temperature  # (2B, 2B)
        # Mask out self‑similarities
        diag = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(diag, -float("inf"))
        # For each sample, the positive example is at index i+batch_size or i-batch_size
        pos_idx = torch.arange(batch_size, device=z.device)
        pos_sim_1 = sim[pos_idx, pos_idx + batch_size]
        pos_sim_2 = sim[pos_idx + batch_size, pos_idx]
        # Denominator: logsumexp over all similarities
        denom = torch.logsumexp(sim, dim=1)
        loss_1 = -pos_sim_1 + denom[:batch_size]
        loss_2 = -pos_sim_2 + denom[batch_size:]
        return 0.5 * (loss_1.mean() + loss_2.mean())

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x1, x2 = batch  # each is (B, L, D)
        # Apply masking independently to each view
        x1_masked, mask1 = self._apply_mask(x1)
        x2_masked, mask2 = self._apply_mask(x2)
        # Forward pass
        cls1, recon1 = self.model(x1_masked)
        cls2, recon2 = self.model(x2_masked)
        # Reconstruction loss (MSE on masked positions)
        # Expand mask to match feature dimension for broadcasting
        mask1_exp = mask1.unsqueeze(-1).expand_as(recon1)
        mask2_exp = mask2.unsqueeze(-1).expand_as(recon2)
        mse1 = F.mse_loss(recon1[mask1_exp], x1[mask1_exp]) if mask1_exp.any() else torch.tensor(0.0, device=self.device)
        mse2 = F.mse_loss(recon2[mask2_exp], x2[mask2_exp]) if mask2_exp.any() else torch.tensor(0.0, device=self.device)
        recon_loss = 0.5 * (mse1 + mse2)
        # Contrastive loss on CLS embeddings
        contrastive_loss = self.nt_xent_loss(cls1, cls2, self.temperature)
        total_loss = recon_loss + contrastive_loss
        # Logging
        self.log_dict(
            {
                "train/recon_loss": recon_loss.detach(),
                "train/contrastive_loss": contrastive_loss.detach(),
                "train/total_loss": total_loss.detach(),
            },
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": None,
            },
        }


def build_window_dataset(cfg: Dict[str, Any]) -> List[np.ndarray]:
    """Preload EEG windows from multiple datasets.

    This helper orchestrates loading the Sleep‑EDF, BCIC and TUH
    datasets, preprocessing them, and extracting overlapping windows.
    The returned list contains all windows concatenated across
    recordings.  Downsampling to a small subset can be achieved via
    the ``max_windows`` configuration option.
    """
    preprocess_cfg = PreprocessConfig()
    windows_list: List[np.ndarray] = []
    # Sleep‑EDF for base signals
    if cfg.get("use_sleepedf", True):
        ds = load_sleepedf_dataset()
        for rec in ds:
            path = rec.get("path") or rec.get("file", None)
            if path is None:
                continue
            import mne  # local import to avoid unconditional dependency
            raw = mne.io.read_raw_edf(path, preload=False)
            sig = preprocess_raw(raw, preprocess_cfg)
            for win in windows(sig, window_sec=2.0, sfreq=preprocess_cfg.sfreq, overlap=0.5):
                windows_list.append(win)
    # Additional datasets could be added here in a similar fashion
    max_windows = cfg.get("max_windows", None)
    if max_windows is not None:
        windows_list = windows_list[: max_windows]
    return windows_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Self‑supervised pretraining for TinyEEG‑FM")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to YAML config file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--max_windows", type=int, default=None, help="Limit total windows for debugging")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.max_windows is not None:
        cfg["max_windows"] = args.max_windows
    # Preload windows
    print("[train_ssl] Loading and preprocessing windows…")
    window_list = build_window_dataset(cfg)
    dataset = SSLDataset(window_list)
    batch_size = cfg.get("batch_size", 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
    # Initialise module
    model = SSLModule(cfg)
    # Mixed precision
    precision = cfg.get("precision", "bf16")
    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 50),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        precision=f"{precision}-mixed" if precision == "bf16" else precision,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    # Train
    trainer.fit(model, dataloader)
    # Save checkpoint
    ckpt_path = os.path.join("checkpoints", "tinyeeg_epoch{0}.ckpt".format(cfg.get("epochs", 50)))
    rank_zero_only(lambda: trainer.save_checkpoint(ckpt_path))()
    print(f"[train_ssl] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()