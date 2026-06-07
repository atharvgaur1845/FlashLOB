"""Prediction heads: 3-class direction (classification) + scalar volatility
(regression). Both read the representation of the sequence's last event."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.direction = nn.Linear(d_model, n_classes)
        self.volatility = nn.Linear(d_model, 1)

    def forward(self, h_last: torch.Tensor):
        """h_last: [B, d_model] -> (logits_direction [B,3], logits_volatility [B,1])."""
        h = self.norm(h_last)
        return self.direction(h), self.volatility(h)
