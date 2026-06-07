"""Hawkes tests: calibration sanity, vectorized bias == reference loop, hot index."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hawkes.bias import HawkesBias, build_hot_index  # noqa: E402
from src.hawkes.calibrate import HawkesCalibrator  # noqa: E402


def _synthetic_events(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(0.3, n))
    types = rng.integers(0, 4, n)
    return times, types


def test_calibration_sane_params():
    times, types = _synthetic_events()
    cal = HawkesCalibrator()
    p = cal.fit(times, types)
    A = np.array(p["adjacency"]); D = np.array(p["decays"]); mu = np.array(p["baseline"])
    assert A.shape == (4, 4) and D.shape == (4, 4) and mu.shape == (4,)
    assert np.isfinite(A).all() and np.isfinite(D).all() and np.isfinite(mu).all()
    assert (A >= 0).all() and (D > 0).all() and (mu >= 0).all()


def test_bias_matches_reference_loop():
    params = {"baseline": [0.5, 0.2, 0.1, 0.15],
              "adjacency": [[0.1, 0.2, 0.3, 0.4]] * 4,
              "decays": [[1.0, 2.0, 0.5, 1.5]] * 4}
    B, L = 2, 16
    torch.manual_seed(0)
    meta = {"event_types": torch.randint(0, 4, (B, L)),
            "log_sizes": torch.rand(B, L) * 5,
            "times": torch.sort(torch.rand(B, L), dim=1).values}
    bias = HawkesBias(params)(meta).numpy()

    A = np.array(params["adjacency"]); D = np.array(params["decays"])
    ref = np.zeros((B, 1, L, L))
    for b in range(B):
        et = meta["event_types"][b].numpy(); ls = meta["log_sizes"][b].numpy()
        tm = meta["times"][b].numpy()
        for q in range(L):
            for k in range(L):
                dt = tm[q] - tm[k]
                if dt > 0:
                    ref[b, 0, q, k] = np.clip(A[et[q], et[k]] * ls[k] * np.exp(-D[et[q], et[k]] * dt), -5, 5)
    assert np.abs(ref - bias).max() < 1e-4


def test_build_hot_index():
    params = {"baseline": [0.1] * 4, "adjacency": [[0.5, 0.1, 0.1, 0.1]] * 4,
              "decays": [[1.0] * 4] * 4}
    B, L = 2, 64
    ls = torch.rand(B, L)
    ls[:, 7] = 100.0  # a dominant event
    et = torch.randint(0, 4, (B, L))
    tm = torch.sort(torch.rand(B, L), dim=1).values
    hot = build_hot_index(ls, et, params, tm, n_hot=8)
    assert hot.shape == (B, 8)
    assert (hot == 7).any()  # the dominant event must be selected
