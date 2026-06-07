"""LOBTransformer — causal transformer encoder over LOB event sequences.

Pre-norm encoder blocks (attention + FFN). Attention backend is selectable:
'flash2' (Phase 0 baseline / dense transformer), 'standard' (returns weights for
analysis), or 'flashlob' (sparse local + hot-global + Hawkes bias).

For the sparse backend the model accepts an ``attn_spec`` describing the sliding
window / global selection and an optional ``hawkes_bias`` module; masks and bias
are built per-forward and passed to every layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.embedding import EventEmbedding
from src.model.head import PredictionHeads
from src.model import masks as M


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, backend):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, backend)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, times, attn_mask, attn_bias, need_weights):
        a, w = self.attn(self.norm1(x), times, attn_mask, attn_bias, need_weights)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w


class LOBTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        context_length: int | None = None,
        attention_backend: str = "flash2",
        n_classes: int = 3,
        attn_spec: dict | None = None,
        hawkes_bias: nn.Module | None = None,
        hawkes_params: dict | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.backend = attention_backend
        # attn_spec example: {'window':512, 'global':'volume_p90'|'hawkes_topk',
        #                     'n_hot':64, 'volume_pct':90, 'type_local':True}
        self.attn_spec = attn_spec or {}
        self.hawkes_bias = hawkes_bias
        self.hawkes_params = hawkes_params

        self.embed = EventEmbedding(8, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, attention_backend)
            for _ in range(n_layers)
        ])
        self.heads = PredictionHeads(d_model, n_classes)

    # ------------------------------------------------------------------ masks
    def _build_mask(self, meta: dict, L: int, device, B: int):
        """Construct the sparse keep-mask from attn_spec (None => dense)."""
        spec = self.attn_spec
        if not spec or self.backend != "flashlob":
            return None
        window = spec.get("window", 512)
        keep = M.sliding_window_mask(L, window, device, batch=B)

        glob = spec.get("global")
        if glob == "volume_p90":
            keep = keep | M.volume_global_mask(meta["log_sizes"].to(device),
                                               spec.get("volume_pct", 90))
        elif glob == "hawkes_topk":
            hot_idx = meta.get("hot_idx")
            if hot_idx is None and self.hawkes_params is not None:
                from src.hawkes.bias import build_hot_index
                hot_idx = build_hot_index(
                    meta["log_sizes"].to(device), meta["event_types"].to(device),
                    self.hawkes_params, meta["times"].to(device),
                    n_hot=spec.get("n_hot", 64))
            if hot_idx is not None:
                keep = keep | M.hot_index_global_mask(hot_idx.to(device), L)

        if spec.get("type_local"):
            keep = keep | M.type_local_mask(meta["event_types"].to(device), window)
        return keep

    def _build_bias(self, meta: dict, device):
        if self.hawkes_bias is None:
            return None
        return self.hawkes_bias({
            k: v.to(device) for k, v in meta.items()
            if k in ("event_types", "log_sizes", "times")
        })

    # ---------------------------------------------------------------- forward
    def forward(self, events, times, meta=None, return_attn=False):
        """events [B,L,8], times [B,L], meta dict -> (logits_dir, logits_vol, attn?)."""
        B, L, _ = events.shape
        device = events.device
        meta = meta or {}

        attn_mask = self._build_mask(meta, L, device, B)
        attn_bias = self._build_bias(meta, device)

        x = self.embed(events)
        attns = []
        for layer in self.layers:
            x, w = layer(x, times, attn_mask, attn_bias, return_attn)
            if return_attn:
                attns.append(w)

        h_last = x[:, -1, :]                       # predict from last event
        logits_dir, logits_vol = self.heads(h_last)
        if return_attn:
            return logits_dir, logits_vol, attns
        return logits_dir, logits_vol

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
