"""Multivariate exponential Hawkes calibration.

Produces the parameters the attention prior needs:
    baseline  mu[i]        [4]
    adjacency A[i, j]      [4, 4]   excitation of type-i by type-j
    decays    D[i, j]      [4, 4]   exponential decay rate

Backend: tries `tick` (HawkesSumExpKern) if installed; otherwise uses a
self-contained MLE fitted with scipy. The MLE uses the standard O(N) recursive
evaluation of the exponential-kernel log-likelihood (Ogata 1981), so the rest of
the pipeline is independent of `tick` (which often has no wheel for Python 3.12).

Kernel form:  phi_{ij}(t) = A[i,j] * exp(-D[i,j] * t),  t >= 0
Intensity:    lambda_i(t) = mu_i + sum_j sum_{t_m^j < t} A[i,j] exp(-D[i,j](t - t_m^j))
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

N_TYPES = 4


def _split_by_type(times: np.ndarray, types: np.ndarray, n_types: int):
    """Return list of sorted arrays of event times per type."""
    order = np.argsort(times, kind="stable")
    times, types = times[order], types[order]
    return [times[types == k] for k in range(n_types)], times, types


def _neg_loglik(params, times, types, T, n_types):
    """Negative log-likelihood of the multivariate exp-Hawkes (log-parameterized).

    params packs [log mu (n), log A (n*n), log D (n*n)]. Uses the recursive
    exponential structure for the summation term -> O(N * n) per evaluation.
    """
    n = n_types
    mu = np.exp(params[:n])
    A = np.exp(params[n:n + n * n]).reshape(n, n)
    D = np.exp(params[n + n * n:]).reshape(n, n)

    N = len(times)
    # Recursive state R[i,j] = sum_{m<current, type j} exp(-D[i,j](t_cur - t_m)).
    # Per step: decay existing events to the current time, evaluate the intensity
    # just before the event, then inject the current event.
    R = np.zeros((n, n))
    last_t = times[0] if N else 0.0
    ll = 0.0
    for idx in range(N):
        t = times[idx]
        dt = t - last_t
        if dt > 0:
            R *= np.exp(-D * dt)
        i = types[idx]
        lam_i = mu[i] + (A[i] * R[i]).sum()
        if lam_i <= 0:
            return 1e12
        ll += np.log(lam_i)
        R[:, i] += 1.0
        last_t = t

    # Compensator: integral of intensity over [0, T].
    comp = mu.sum() * T
    for j in range(n):
        tj = times[types == j]
        if tj.size == 0:
            continue
        for i in range(n):
            d = D[i, j]
            comp += (A[i, j] / d) * np.sum(1.0 - np.exp(-d * (T - tj)))
    ll -= comp
    return -ll


class HawkesCalibrator:
    def __init__(self, n_types: int = N_TYPES, kernel: str = "exp",
                 penalty_C: float = 1000.0, max_events: int = 20000):
        self.n_types = n_types
        self.kernel = kernel
        self.penalty_C = penalty_C
        self.max_events = max_events
        self.params: dict | None = None

    # ------------------------------------------------------------------- fit
    def fit(self, times: np.ndarray, types: np.ndarray) -> dict:
        """Fit on event (time, type) arrays. Returns the params dict."""
        times = np.asarray(times, dtype=np.float64)
        types = np.asarray(types, dtype=np.int64)
        if len(times) > self.max_events:
            # contiguous head keeps the temporal structure intact
            times, types = times[: self.max_events], types[: self.max_events]
        times = times - times.min()

        params = self._fit_tick(times, types)
        if params is None:
            params = self._fit_mle(times, types)
        self.params = params
        return params

    def _fit_tick(self, times, types):
        try:
            from tick.hawkes import HawkesSumExpKern
        except Exception:
            return None
        try:
            decays = np.array([1.0, 10.0, 100.0])
            model = HawkesSumExpKern(decays, C=self.penalty_C, penalty="l2")
            per_type = [np.ascontiguousarray(times[types == k])
                        for k in range(self.n_types)]
            model.fit([per_type])
            mu = np.asarray(model.baseline, dtype=np.float64)
            # adjacency: sum over decays; effective decay = magnitude-weighted mean
            adj = np.asarray(model.adjacency, dtype=np.float64)  # [n,n,n_decays]
            A = adj.sum(axis=2)
            wsum = (adj * decays[None, None, :]).sum(axis=2)
            D = np.divide(wsum, A, out=np.full_like(A, decays.mean()), where=A > 0)
            return self._pack(mu, A, D)
        except Exception:
            return None

    def _fit_mle(self, times, types):
        n = self.n_types
        T = float(times[-1] - times[0]) if len(times) > 1 else 1.0
        T = max(T, 1e-6)
        # init: mu = empirical rate / 2, A small, D moderate
        rate = np.array([(types == k).sum() for k in range(n)], dtype=np.float64)
        rate = np.maximum(rate / T, 1e-6)
        x0 = np.concatenate([
            np.log(rate * 0.5),
            np.log(np.full(n * n, 0.1)),
            np.log(np.full(n * n, 1.0)),
        ])
        res = minimize(
            _neg_loglik, x0, args=(times, types, T, n),
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 2000},
        )
        x = res.x
        mu = np.exp(x[:n])
        A = np.exp(x[n:n + n * n]).reshape(n, n)
        D = np.exp(x[n + n * n:]).reshape(n, n)
        return self._pack(mu, A, D)

    def _pack(self, mu, A, D):
        # guard against pathological values
        A = np.nan_to_num(A, nan=0.0, posinf=5.0, neginf=0.0).clip(0, 50)
        D = np.nan_to_num(D, nan=1.0, posinf=100.0, neginf=1e-3).clip(1e-3, 1000)
        mu = np.nan_to_num(mu, nan=1e-6).clip(1e-9, None)
        return {"baseline": mu.tolist(), "adjacency": A.tolist(), "decays": D.tolist()}

    # ------------------------------------------------------- attention prior
    def compute_attention_prior(self, t_query, t_keys, type_keys, size_keys,
                                type_query):
        """Hawkes bias for one query over its keys (numpy, used by reference/tests).

        bias(k) = A[tq, tk] * log1p(size_k) * exp(-D[tq, tk] * (t_q - t_k)), causal.
        """
        assert self.params is not None
        A = np.asarray(self.params["adjacency"])
        D = np.asarray(self.params["decays"])
        t_keys = np.asarray(t_keys, dtype=np.float64)
        type_keys = np.asarray(type_keys, dtype=np.int64)
        size_keys = np.asarray(size_keys, dtype=np.float64)
        dt = t_query - t_keys
        a = A[type_query, type_keys]
        d = D[type_query, type_keys]
        bias = a * np.log1p(size_keys) * np.exp(-d * dt)
        bias = np.where(dt > 0, bias, 0.0)
        return bias

    # ------------------------------------------------------------- io
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    @staticmethod
    def load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)


def _cli():
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.data.lobster import LOBSTERLoader  # noqa: E402

    ap = argparse.ArgumentParser(description="Calibrate Hawkes params on LOBSTER")
    ap.add_argument("--message", default="data/lobster/sample_message.csv")
    ap.add_argument("--book", default="data/lobster/sample_orderbook.csv")
    ap.add_argument("--output", default="data/processed/hawkes_params.json")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    loader = LOBSTERLoader()
    df = loader.load(args.message, args.book)
    times = df["time"].to_numpy(dtype=float)
    types = loader.hawkes_types(df)

    cal = HawkesCalibrator()
    params = cal.fit(times, types)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cal.save(args.output)
    print(f"Saved Hawkes params -> {args.output}")
    print("baseline:", np.round(params["baseline"], 5))
    print("adjacency:\n", np.round(params["adjacency"], 4))
    print("decays:\n", np.round(params["decays"], 4))

    if args.validate:
        from src.hawkes.validate import compensator_residuals, qq_plot
        res = compensator_residuals(times[:20000], types[:20000], params)
        fig = qq_plot(res, "paper/figures/hawkes_qq.png")
        print("QQ-plot ->", fig)


if __name__ == "__main__":
    _cli()
