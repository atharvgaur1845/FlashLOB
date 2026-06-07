"""Multi-head attention with selectable backends.

Backends:
    'flash2'   -> F.scaled_dot_product_attention (FlashAttention-2 path). Fast,
                  memory-efficient, but returns no attention weights.
    'standard' -> explicit softmax. Slower / O(L^2) memory, but exposes attention
                  weights for the Phase -1/0 clustering analysis.
    'flashlob' -> sparse local + hot-global + Hawkes-bias attention. Uses the
                  Triton kernel when available (src.kernel.flash_lob), else a
                  pure-PyTorch masked fallback so the model always runs.

All backends apply TimeAwareRoPE to Q/K first and respect causality.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.positional import TimeAwareRoPE


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        backend: str = "flash2",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.backend = backend
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = TimeAwareRoPE(self.d_head)

    def _project(self, x: torch.Tensor):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                 # each [B, L, H, d_head]
        # -> [B, H, L, d_head]
        return (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        attn_mask: torch.Tensor | None = None,   # bool [B,1,L,L] True=keep, or None
        attn_bias: torch.Tensor | None = None,   # float [B,1,L,L] additive, or None
        need_weights: bool = False,
    ):
        """Returns (out [B,L,D], weights or None)."""
        B, L, _ = x.shape
        q, k, v = self._project(x)
        q, k = self.rope(q, k, times)

        if self.backend == "flash2" and not need_weights and attn_bias is None and attn_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            weights = None
        else:
            out, weights = self._manual_attention(
                q, k, v, attn_mask, attn_bias, need_weights
            )

        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out(out), weights

    def _manual_attention(self, q, k, v, attn_mask, attn_bias, need_weights):
        """Explicit softmax attention with causal + optional sparse mask + bias.

        Builds a fused additive float mask so a single softmax handles causality,
        sparsity (attn_mask) and the Hawkes prior (attn_bias).
        """
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,L,L]

        # Causal mask (lower-triangular keep).
        causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        keep = causal.view(1, 1, L, L)
        if attn_mask is not None:
            keep = keep & attn_mask  # both must allow

        if attn_bias is not None:
            scores = scores + attn_bias.to(scores.dtype)

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~keep, neg_inf)

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)  # fully-masked rows -> 0
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)
        out = torch.matmul(weights, v)
        return out, (weights if need_weights else None)
