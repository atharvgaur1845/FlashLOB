"""Evaluation metrics: classification accuracy / weighted + per-class F1, and
realized-volatility MAE."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.evaluation.protocol import N_CLASSES


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == targets).float().mean().item())


def f1_weighted(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1).cpu().numpy()
    y = targets.cpu().numpy()
    return float(f1_score(y, pred, average="weighted", zero_division=0))


def f1_per_class(logits: torch.Tensor, targets: torch.Tensor) -> list[float]:
    pred = logits.argmax(dim=-1).cpu().numpy()
    y = targets.cpu().numpy()
    return [float(x) for x in
            f1_score(y, pred, labels=list(range(N_CLASSES)),
                     average=None, zero_division=0)]


def volatility_mae(pred_vol: torch.Tensor, target_vol: torch.Tensor) -> float:
    return float((pred_vol.flatten() - target_vol.flatten()).abs().mean().item())


def compute_all(logits_dir, pred_vol, y_dir, y_vol) -> dict:
    """Aggregate metric dict from concatenated predictions/targets."""
    per_class = f1_per_class(logits_dir, y_dir)
    return {
        "accuracy": accuracy(logits_dir, y_dir),
        "f1_weighted": f1_weighted(logits_dir, y_dir),
        "f1_down": per_class[0],
        "f1_stable": per_class[1],
        "f1_up": per_class[2],
        "volatility_mae": volatility_mae(pred_vol, y_vol),
    }
