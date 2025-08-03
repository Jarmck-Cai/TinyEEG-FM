"""Finetune TinyEEG on BCIC IV‑2a motor imagery.

This script takes a pretrained TinyEEG checkpoint, freezes the backbone
and attaches a small multi‑layer perceptron (MLP) classifier with
three fully connected layers.  The model is trained end‑to‑end on
window‑level CLS embeddings extracted from the BCIC IV‑2a dataset.
Training is constrained to a maximum wall‑time of 30 minutes by
limiting the number of epochs and using early stopping.

Usage
-----
>>> python -m tiny_eeg_fm.src.finetune_mi --checkpoint path/to.ckpt

Notes
-----
The BCIC dataset contains trials labelled with left/right/foot/tongue
motor imagery.  To construct inputs we segment each trial into
2‑second windows (50 % overlap), compute CLS embeddings with the
frozen TinyEEG backbone and average them across the trial.  The
resulting vectors are fed to the MLP classifier.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import evaluate

from .dataset import (
    PreprocessConfig,
    preprocess_raw,
    windows,
    load_bcic_iv_2a,
    patchify,
)
from .model import TinyEEGModel, TinyEEGConfig


class BCICDataset(Dataset):
    """Dataset returning averaged CLS embeddings and labels for BCIC trials."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = embeddings
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class ClassifierModule(pl.LightningModule):
    """Freeze backbone and train a small MLP classifier."""

    def __init__(self, backbone: TinyEEGModel, num_classes: int = 4, lr: float = 1e-3) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        d_model = backbone.config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = evaluate.load("accuracy")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        return opt


@torch.inference_mode()
def compute_trial_embeddings(
    model: TinyEEGModel, trials: List[np.ndarray], device: str = "cpu"
) -> np.ndarray:
    """Compute averaged CLS embeddings for a list of trials."""
    model.eval()
    all_embeds: List[np.ndarray] = []
    for trial in trials:
        # trial: (n_channels, n_samples)
        embeds: List[np.ndarray] = []
        for win in windows(trial, window_sec=2.0, sfreq=256.0, overlap=0.5):
            tokens = patchify(win)
            x = torch.tensor(tokens, dtype=torch.float32, device=device).unsqueeze(0)
            cls, _ = model(x, mask=None)
            embeds.append(cls.squeeze(0).cpu().numpy())
        if embeds:
            all_embeds.append(np.mean(np.stack(embeds, axis=0), axis=0))
        else:
            # If no windows extracted, append zeros
            all_embeds.append(np.zeros(model.config.d_model, dtype=np.float32))
    return np.stack(all_embeds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune TinyEEG on BCIC motor imagery")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained TinyEEG checkpoint (.ckpt)")
    parser.add_argument("--test_size", type=float, default=0.3, help="Validation split fraction")
    parser.add_argument("--max_trials", type=int, default=None, help="Limit number of trials for debugging")
    args = parser.parse_args()
    # Load backbone
    cfg = TinyEEGConfig()
    backbone = TinyEEGModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    backbone.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    # Load BCIC dataset
    ds = load_bcic_iv_2a()
    trials: List[np.ndarray] = []
    labels: List[int] = []
    preprocess_cfg = PreprocessConfig()
    for rec in ds:
        path = rec.get("path") or rec.get("file", None)
        label = rec.get("label") or rec.get("y", None)
        if path is None or label is None:
            continue
        import mne
        raw = mne.io.read_raw_gdf(path, preload=False)
        sig = preprocess_raw(raw, preprocess_cfg)
        trials.append(sig)
        labels.append(int(label))
        if args.max_trials is not None and len(trials) >= args.max_trials:
            break
    print(f"[finetune_mi] Loaded {len(trials)} trials")
    # Compute embeddings
    emb = compute_trial_embeddings(backbone, trials)
    labels_arr = np.array(labels, dtype=np.int64)
    # Train/val split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        emb, labels_arr, test_size=args.test_size, random_state=42, stratify=labels_arr
    )
    train_ds = BCICDataset(X_train, y_train)
    val_ds = BCICDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    # Lightning classifier
    clf = ClassifierModule(backbone, num_classes=len(set(labels_arr)), lr=1e-3)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="cpu",
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(clf, train_loader, val_loader)
    # Final validation accuracy
    val_acc = trainer.callback_metrics.get("val/acc").item() if trainer.callback_metrics else None
    print(f"Validation accuracy: {val_acc}")


if __name__ == "__main__":
    main()