"""Pure-PyTorch reference for the FlashLOB sparse attention.

This is the CORRECTNESS ORACLE for the Triton kernel — slow, O(L^2), never used
in training. The kernel must match this to FP16 tolerance.

Attention pattern (per query q, causal):
    * local : keys in the causal sliding window  [q - window + 1, q]
    * hot   : a per-sequence set of high-Hawkes-excitation key indices
              (build_hot_index), visible to every query (causal-clipped)
    * (optional) type-local: same-event-type keys within the window
Hawkes bias (A * log_size_k * exp(-D * dt)) is ADDED to the QK^T logits before
softmax over the union of admitted keys.

Type-local is OFF by default so the kernel (which implements local + hot + bias)
can be validated for exact parity; enable include_type to study its effect.
"""

from __future__ import annotations

import math

import torch

from src.hawkes.bias import build_hot_index
from src.model import masks as M


def flashlob_reference(
    Q: torch.Tensor,            # [B, H, L, d]
    K: torch.Tensor,
    V: torch.Tensor,
    event_types: torch.Tensor,  # [B, L] int64
    log_sizes: torch.Tensor,    # [B, L] float32
    times: torch.Tensor,        # [B, L] float32
    hawkes_params: dict,
    window_size: int = 256,
    n_global_hot: int = 64,
    include_type: bool = False,
    use_bias: bool = True,
) -> torch.Tensor:
    B, H, L, d = Q.shape
    device = Q.device
    scale = 1.0 / math.sqrt(d)

    # --- build the boolean keep-mask [B,1,L,L] ---
    keep = M.sliding_window_mask(L, window_size, device, batch=B)            # local
    hot_idx = build_hot_index(log_sizes, event_types, hawkes_params, times,
                              n_hot=n_global_hot)                            # [B,n_hot]
    keep = keep | M.hot_index_global_mask(hot_idx, L)
    if include_type:
        keep = keep | M.type_local_mask(event_types, window_size)
    # enforce causality (no self, no future) to match the kernel
    idx = torch.arange(L, device=device)
    causal = (idx.view(L, 1) - idx.view(1, L)) > 0          # strictly past
    keep = keep & causal.view(1, 1, L, L)

    # --- logits + Hawkes bias ---
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,L,L]
    if use_bias:
        A = torch.tensor(hawkes_params["adjacency"], device=device, dtype=torch.float32)
        D = torch.tensor(hawkes_params["decays"], device=device, dtype=torch.float32)
        nt = A.shape[0]
        et = event_types.clamp(0, nt - 1)
        dt = (times[:, :, None] - times[:, None, :]).clamp(min=0)            # [B,L,L]
        tq = et[:, :, None].expand(B, L, L)
        tk = et[:, None, :].expand(B, L, L)
        flat = (tq * nt + tk).reshape(B, -1)
        a = A.reshape(-1)[flat].reshape(B, L, L)
        dd = D.reshape(-1)[flat].reshape(B, L, L)
        bias = a * log_sizes[:, None, :] * torch.exp(-dd * dt)
        bias = torch.where(dt > 0, bias, torch.zeros_like(bias)).clamp(-5, 5)
        scores = scores + bias.unsqueeze(1)

    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~keep, neg_inf)
    weights = torch.softmax(scores, dim=-1)
    # Fully-masked rows (e.g. q=0) softmax to *uniform* (all-equal -inf), not NaN;
    # force their weights to zero so they contribute nothing, matching the kernel.
    weights = weights.masked_fill(~keep, 0.0)
    return torch.matmul(weights, V)                # [B,H,L,d]


def hawkes_bias_tables(hawkes_params: dict, device, dtype=torch.float32):
    """Return (A, D) [n_types, n_types] float tensors for the kernel."""
    A = torch.tensor(hawkes_params["adjacency"], device=device, dtype=dtype)
    D = torch.tensor(hawkes_params["decays"], device=device, dtype=dtype)
    return A.contiguous(), D.contiguous()
