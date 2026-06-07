"""Vectorized Hawkes attention bias, added to QK^T before softmax.

Given fitted params (A, D) and per-event (type, log_size, time), produces a
[B, 1, L, L] additive bias (broadcast across heads):

    bias[b,0,q,k] = A[tq, tk] * log_size_k * exp(-D[tq, tk] * (t_q - t_k))

clamped to [-5, 5], zeroed for non-causal / self pairs. No Python loops over L.

Note: ``log_size`` here is the dataset's log1p(size) channel (meta['log_sizes']),
so the formula multiplies by it directly rather than re-applying log1p.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HawkesBias(nn.Module):
    def __init__(self, hawkes_params: dict, clamp: float = 5.0):
        super().__init__()
        A = torch.tensor(hawkes_params["adjacency"], dtype=torch.float32)
        D = torch.tensor(hawkes_params["decays"], dtype=torch.float32)
        self.register_buffer("A", A)         # [n_types, n_types]
        self.register_buffer("D", D)
        self.n_types = A.shape[0]
        self.clamp = clamp

    def forward(self, meta: dict) -> torch.Tensor:
        et = meta["event_types"].long()                 # [B, L]
        log_sz = meta["log_sizes"].float()              # [B, L]
        times = meta["times"].float()                   # [B, L]
        B, L = et.shape
        device = self.A.device
        et = et.to(device).clamp(0, self.n_types - 1)
        log_sz = log_sz.to(device)
        times = times.to(device)

        delta_t = times[:, :, None] - times[:, None, :]  # [B,L,L]  (t_q - t_k)
        delta_t = delta_t.clamp(min=0.0)

        tq = et[:, :, None].expand(B, L, L)             # query type
        tk = et[:, None, :].expand(B, L, L)             # key type
        flat = (tq * self.n_types + tk).reshape(B, -1)  # [B, L*L]
        A = self.A.reshape(-1)[flat].reshape(B, L, L)
        D = self.D.reshape(-1)[flat].reshape(B, L, L)

        log_sz_k = log_sz[:, None, :]                   # [B,1,L]
        bias = A * log_sz_k * torch.exp(-D * delta_t)

        # zero out non-causal (delta_t==0 includes self and future) entries
        causal = (delta_t > 0)
        bias = torch.where(causal, bias, torch.zeros_like(bias))
        bias = bias.clamp(-self.clamp, self.clamp)
        return bias.unsqueeze(1)                         # [B,1,L,L]


def build_hot_index(log_sizes: torch.Tensor, event_types: torch.Tensor,
                    hawkes_params: dict, times: torch.Tensor,
                    n_hot: int = 64) -> torch.Tensor:
    """Top-n_hot events per sequence by worst-case Hawkes excitation potential.

    score[k] = max_i A[i, type_k] * log_size_k   (max over query types).
    Runs on whatever device the inputs are on; O(L) + a topk. Returns [B, n_hot]
    long indices (sorted ascending so kernels can stream them in order).
    """
    A = torch.tensor(hawkes_params["adjacency"], dtype=torch.float32,
                     device=log_sizes.device)
    n_types = A.shape[0]
    et = event_types.long().clamp(0, n_types - 1)        # [B,L]
    # max over query type i of A[i, type_k]: gather columns then max over rows.
    col_max = A.max(dim=0).values                        # [n_types]
    excite = col_max[et]                                 # [B,L]
    score = excite * log_sizes.to(excite.dtype)          # [B,L]
    k = min(n_hot, score.shape[1])
    idx = score.topk(k, dim=1).indices                   # [B,k]
    return idx.sort(dim=1).values
