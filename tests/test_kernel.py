"""Triton FlashLOB kernel correctness tests (skipped without CUDA/Triton)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kernel.flash_lob import HAS_TRITON, flashlob_attention  # noqa: E402
from src.kernel.reference import flashlob_reference  # noqa: E402

cuda = pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()),
                          reason="requires CUDA + Triton")

PARAMS = {
    "baseline": [0.5, 0.2, 0.1, 0.15],
    "adjacency": [[0.01, 0.03, 0.09, 0.34], [0.04, 0.01, 0.004, 0.02],
                  [0.003, 0.03, 0.19, 0.20], [0.03, 0.02, 0.02, 0.15]],
    "decays": [[3.8, 1.6, 0.12, 0.56], [1.1, 1.1, 9.8, 1.3],
               [1.7, 3.0, 0.21, 0.23], [1.5, 1.4, 0.51, 1.6]],
}


def _inputs(L, B=2, H=4, d=16, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    K = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    V = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    et = torch.randint(0, 4, (B, L), device=device, generator=g)
    ls = torch.rand(B, L, device=device, generator=g) * 5
    tm = torch.sort(torch.rand(B, L, device=device, generator=g), dim=1).values
    return Q, K, V, et, ls, tm


@cuda
@pytest.mark.parametrize("L", [512, 1024, 2048])
def test_flashlob_correctness(L):
    Q, K, V, et, ls, tm = _inputs(L)
    ref = flashlob_reference(Q.float(), K.float(), V.float(), et, ls, tm,
                             PARAMS, window_size=256, n_global_hot=64).half()
    out = flashlob_attention(Q, K, V, et, ls, tm, PARAMS, 256, 64)
    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() < 1e-2
    assert diff.mean().item() < 1e-3


@cuda
def test_flashlob_causal():
    """Output at position i must not depend on positions j > i."""
    L = 512
    Q, K, V, et, ls, tm = _inputs(L)
    out1 = flashlob_attention(Q, K, V, et, ls, tm, PARAMS, 256, 64)
    i = 100
    # perturb everything strictly after i; positions <= i must be unchanged
    K2 = K.clone(); V2 = V.clone()
    K2[:, :, i + 1:] += 3.0
    V2[:, :, i + 1:] += 3.0
    out2 = flashlob_attention(Q, K2, V2, et, ls, tm, PARAMS, 256, 64)
    assert torch.allclose(out1[:, :, :i + 1], out2[:, :, :i + 1], atol=1e-2)


@cuda
def test_flashlob_hot_events():
    """A high-Hawkes-score event outside the window must influence the output."""
    L = 1024
    Q, K, V, et, ls, tm = _inputs(L)
    # make event 5 a huge large-trade so build_hot_index selects it
    et[:, 5] = 3
    ls[:, 5] = 50.0
    # give it a distinctive V so its contribution is detectable far away
    V[:, :, 5] = 9.0
    out_with = flashlob_attention(Q, K, V, et, ls, tm, PARAMS, window_size=64, n_global_hot=64)
    # without the hot stream (n_hot=0 -> only local window, event 5 far outside)
    out_local = flashlob_attention(Q, K, V, et, ls, tm, PARAMS, window_size=64, n_global_hot=1)
    # query at the end (far from event 5, outside window=64) should differ
    q = L - 1
    assert (out_with[:, :, q] - out_local[:, :, q]).abs().max().item() > 1e-3
