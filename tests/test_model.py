"""Model tests: forward shapes, param count, gradient flow, RoPE time-sensitivity."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.positional import TimeAwareRoPE  # noqa: E402
from src.model.transformer import LOBTransformer  # noqa: E402


def _inputs(B=2, L=32, device="cpu"):
    events = torch.randn(B, L, 8, device=device)
    times = torch.sort(torch.rand(B, L, device=device), dim=1).values
    meta = {"event_types": torch.randint(0, 4, (B, L), device=device),
            "log_sizes": torch.rand(B, L, device=device) * 5, "times": times}
    return events, times, meta


def test_forward_shapes():
    events, times, meta = _inputs()
    m = LOBTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, context_length=32,
                       attention_backend="standard")
    ld, lv = m(events, times, meta)
    assert ld.shape == (2, 3) and lv.shape == (2, 1)
    assert torch.isfinite(ld).all() and torch.isfinite(lv).all()


def test_attention_weights_returned():
    events, times, meta = _inputs()
    m = LOBTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, context_length=32,
                       attention_backend="standard")
    ld, lv, attns = m(events, times, meta, return_attn=True)
    assert len(attns) == 2 and attns[0].shape == (2, 4, 32, 32)


def test_gradient_flow():
    events, times, meta = _inputs()
    m = LOBTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, context_length=32,
                       attention_backend="standard")
    (m(events, times, meta)[0].sum() + m(events, times, meta)[1].sum()).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_rope_time_sensitive():
    rope = TimeAwareRoPE(d_head=16)
    q = torch.randn(1, 2, 8, 16); k = torch.randn(1, 2, 8, 16)
    t1 = torch.sort(torch.rand(1, 8), dim=1).values
    t2 = t1 + 0.5
    q1, _ = rope(q, k, t1)
    q2, _ = rope(q, k, t2)
    assert (q1 - q2).abs().max() > 1e-4


def test_param_count_stable():
    m = LOBTransformer(d_model=128, n_heads=8, n_layers=6, d_ff=512, context_length=128)
    # architecture is context-independent; param count must not depend on ctx
    m2 = LOBTransformer(d_model=128, n_heads=8, n_layers=6, d_ff=512, context_length=2048)
    assert m.num_params() == m2.num_params()
