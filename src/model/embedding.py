"""Continuous event embedding. No discretization — the 8 continuous features go
straight through a small MLP into d_model."""

from __future__ import annotations

import torch
import torch.nn as nn


class EventEmbedding(nn.Module):
    """Project an 8-dim event token to d_model via Linear-LN-GELU-Linear."""

    def __init__(self, in_dim: int = 8, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, 8] -> [B, L, d_model]."""
        return self.net(x)
