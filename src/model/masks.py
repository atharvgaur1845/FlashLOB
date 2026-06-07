"""Boolean attention-mask builders for the sparse-attention variants.

All masks are [B, 1, L, L] boolean with True = "query may attend to key", and are
combined with causality inside the attention module (so they need not be causal
themselves, but we make them causal anyway for clarity). These implement the
Phase 1 ablation variants using plain PyTorch (no kernel):

    sliding window      -> local_mask
    + volume threshold  -> local_mask | volume_global_mask
    + hawkes top-k      -> local_mask | hawkes_global_mask
"""

from __future__ import annotations

import torch


def causal_keep(L: int, device) -> torch.Tensor:
    return torch.ones(L, L, dtype=torch.bool, device=device).tril().view(1, 1, L, L)


def sliding_window_mask(L: int, window: int, device, batch: int = 1) -> torch.Tensor:
    """Keep keys within [q-window, q] (causal sliding window). [B,1,L,L]."""
    idx = torch.arange(L, device=device)
    diff = idx.view(L, 1) - idx.view(1, L)            # q - k
    keep = (diff >= 0) & (diff < window)
    return keep.view(1, 1, L, L).expand(batch, 1, L, L).clone()


def volume_global_mask(log_sizes: torch.Tensor, percentile: float = 90.0) -> torch.Tensor:
    """Global columns for keys above the per-sequence ``percentile`` of log_size.

    log_sizes: [B, L] -> keep [B, 1, L, L] where any query may attend to those
    high-volume key columns (subject to causality applied later).
    """
    B, L = log_sizes.shape
    thr = torch.quantile(log_sizes, percentile / 100.0, dim=1, keepdim=True)  # [B,1]
    is_hot = log_sizes >= thr                                                 # [B,L]
    return is_hot.view(B, 1, 1, L).expand(B, 1, L, L).clone()


def hot_index_global_mask(hot_idx: torch.Tensor, L: int) -> torch.Tensor:
    """Global columns for an explicit set of hot key indices.

    hot_idx: [B, n_hot] long -> keep [B,1,L,L] columns at those indices.
    """
    B, n_hot = hot_idx.shape
    cols = torch.zeros(B, L, dtype=torch.bool, device=hot_idx.device)
    cols.scatter_(1, hot_idx.clamp(0, L - 1), True)
    return cols.view(B, 1, 1, L).expand(B, 1, L, L).clone()


def type_local_mask(event_types: torch.Tensor, window: int) -> torch.Tensor:
    """Keep recent keys (within window) that share the query's event type.

    event_types: [B, L] -> [B,1,L,L].
    """
    B, L = event_types.shape
    same = event_types.view(B, 1, L, 1) == event_types.view(B, 1, 1, L)  # [B,1,L,L]
    idx = torch.arange(L, device=event_types.device)
    diff = idx.view(L, 1) - idx.view(1, L)
    recent = ((diff >= 0) & (diff < window)).view(1, 1, L, L)
    return same & recent
