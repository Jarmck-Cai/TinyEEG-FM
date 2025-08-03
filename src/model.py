"""Model definitions for TinyEEG‑FM.

This module defines a small transformer encoder model for EEG patch
tokens along with its configuration class.  It follows the
``transformers`` library conventions so that checkpoints can be
seamlessly uploaded to and downloaded from the Hugging Face Hub.

In addition to the core model, we provide helper functions to
instantiate LoRA adapters via the ``peft`` library.  When loaded
with LoRA, the base model weights remain frozen and only the
adapter weights are trained, greatly reducing the memory footprint
and enabling efficient fine‑tuning.

Example
-------
>>> from tiny_eeg_fm.src.model import TinyEEGConfig, TinyEEGModel, add_lora
>>> cfg = TinyEEGConfig()
>>> base_model = TinyEEGModel(cfg)
>>> lora_model = add_lora(base_model)
>>> print(lora_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


class TinyEEGConfig(PretrainedConfig):
    """Configuration for the TinyEEG model.

    Attributes
    ----------
    model_type : str
        Identifier used by the transformers library.
    d_model : int
        Embedding size of the transformer.
    num_hidden_layers : int
        Number of transformer encoder layers.
    num_attention_heads : int
        Number of self‑attention heads.
    patch_dim : int
        Dimensionality of each patch token (default 96: 64 time + 32 FFT).
    """

    model_type = "tiny_eeg"

    def __init__(
        self,
        d_model: int = 256,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 4,
        patch_dim: int = 96,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_dim = patch_dim


class TinyEEGModel(PreTrainedModel):
    """A lightweight transformer encoder for EEG tokens.

    The model processes a sequence of patch tokens and produces a
    pooled CLS embedding along with reconstructed patch vectors for
    masked modelling.  Prototypes (centroids) can be learned to
    facilitate clustering; they are exposed as a parameter.
    """

    config_class = TinyEEGConfig

    def __init__(self, config: TinyEEGConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config.patch_dim, config.d_model)
        # CLS token: learns an embedding per model rather than being static
        self.cls = nn.Parameter(torch.randn(1, 1, config.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=4 * config.d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_hidden_layers)
        # Reconstruction head: maps back to patch dimension
        self.head_recon = nn.Linear(config.d_model, config.patch_dim)
        # Learned prototypes (e.g. for clustering or vector quantisation)
        self.proto = nn.Parameter(torch.randn(32, config.d_model))
        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following a normal distribution."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Run the transformer encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(batch, seq_len, patch_dim)``.
        mask : Tensor, optional
            Boolean mask of shape ``(batch, seq_len)`` indicating which
            tokens are masked (True for masked).  The encoder itself
            ignores this mask; it is used externally to compute the
            reconstruction loss on masked tokens.

        Returns
        -------
        Tuple[Tensor, Tensor]
            * ``cls`` – pooled representation of shape ``(batch, d_model)``.
            * ``recon`` – reconstructed patches of shape ``(batch, seq_len, patch_dim)``.
        """
        # Project tokens to embedding space
        z = self.proj(x)
        # Prepend CLS token to every sequence
        cls_tokens = self.cls.expand(z.size(0), -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)  # (B, 1+L, d_model)
        # Run encoder
        z = self.encoder(z)
        cls = z[:, 0]  # (B, d_model)
        # Reconstruct only the non‑CLS part
        recon = self.head_recon(z[:, 1:])  # (B, L, patch_dim)
        return cls, recon


def add_lora(
    model: TinyEEGModel,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> TinyEEGModel:
    """Wrap a TinyEEGModel with LoRA adapters.

    Parameters
    ----------
    model : TinyEEGModel
        The base model to be adapted.
    r : int
        Rank of the LoRA matrices.
    alpha : float
        Scaling factor for the LoRA layers.
    dropout : float
        Dropout applied within LoRA layers.
    target_modules : Sequence[str], optional
        List of module names to which LoRA should be applied.  By
        default all linear layers in the projection, encoder and head
        are adapted.

    Returns
    -------
    TinyEEGModel
        A model with LoRA parameters registered.  If the ``peft``
        library is unavailable, returns the original model.
    """
    if LoraConfig is None or get_peft_model is None:
        # PEFT is optional; return model unchanged if not installed
        print(
            "[model] peft is not available – returning the original model without LoRA adapters"
        )
        return model
    if target_modules is None:
        target_modules = ["proj", "encoder", "head_recon"]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    peft_model = get_peft_model(model, config)
    # Freeze the base parameters (optional); LoRA handles trainable params
    for name, param in peft_model.named_parameters():
        if not any([name.startswith(t) for t in target_modules]):
            param.requires_grad = False
    return peft_model
