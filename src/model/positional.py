"""Time-aware Rotary Position Encoding.

Events are irregularly spaced in time. Integer-index RoPE ignores this. We rotate
Q/K by an angle proportional to each event's ACTUAL (normalized) timestamp:

    theta_i(t) = t * inv_freq_i,   inv_freq_i = 1 / (base ^ (2i / d))

so two events close in time get similar rotations regardless of how many events
fell between them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TimeAwareRoPE(nn.Module):
    """Apply time-conditioned rotary encoding to per-head Q/K tensors.

    Args:
        d_head: per-head dimension (must be even).
        base: RoPE base (default 10000).
        time_scale: multiplies normalized time before forming angles. Normalized
            time lives in [0,1]; scaling spreads it across a useful angular range.
    """

    def __init__(self, d_head: int, base: float = 10000.0, time_scale: float = 1000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        self.time_scale = time_scale
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _angles(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # times: [B, L] -> angles [B, L, d_head] (cos/sin), repeat-interleaved.
        t = times.float() * self.time_scale
        freqs = torch.einsum("bl,f->blf", t, self.inv_freq)  # [B, L, d_head/2]
        emb = freqs.repeat_interleave(2, dim=-1)              # [B, L, d_head]
        return emb.cos(), emb.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor, times: torch.Tensor):
        """Rotate q, k by time-derived angles.

        q, k: [B, H, L, d_head].  times: [B, L].
        Returns rotated (q, k) with the same shapes.
        """
        cos, sin = self._angles(times)               # [B, L, d_head]
        cos = cos.unsqueeze(1).to(q.dtype)           # [B, 1, L, d_head]
        sin = sin.unsqueeze(1).to(q.dtype)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot
