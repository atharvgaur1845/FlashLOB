"""Goodness-of-fit for a fitted Hawkes model via the random time-change theorem.

Under a correctly specified model, the integrated intensity between consecutive
events of a type (the "compensator increments") are i.i.d. unit-Exponential. We
compute these residuals and a QQ-plot against Exp(1).
"""

from __future__ import annotations

import numpy as np


def compensator_residuals(times: np.ndarray, types: np.ndarray,
                          params: dict, n_types: int = 4) -> dict:
    """Return {type: residual array} of compensator increments per type.

    For type i, the compensator is Lambda_i(t) = integral_0^t lambda_i.
    Residuals tau_k = Lambda_i(t_k^i) - Lambda_i(t_{k-1}^i) ~ Exp(1) if well-fit.
    """
    times = np.asarray(times, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    order = np.argsort(times)
    times, types = times[order], types[order]
    times = times - times[0]

    mu = np.asarray(params["baseline"])
    A = np.asarray(params["adjacency"])
    D = np.asarray(params["decays"])

    # Cumulative compensator per type, integrating piecewise between events.
    Lambda = np.zeros(n_types)
    R = np.zeros((n_types, n_types))  # R[i,j] decayed count state
    last_t = 0.0
    prev_type = -1
    last_event_lambda = {i: 0.0 for i in range(n_types)}
    residuals = {i: [] for i in range(n_types)}

    for idx in range(len(times)):
        t = times[idx]
        dt = t - last_t
        # integral of each kernel contribution over [last_t, t], using R as it
        # stood at last_t (events strictly before this interval)
        if dt > 0:
            for i in range(n_types):
                Lambda[i] += mu[i] * dt
                for j in range(n_types):
                    d = D[i, j]
                    Lambda[i] += A[i, j] * R[i, j] * (1 - np.exp(-d * dt)) / d
            R *= np.exp(-D * dt)
        i = types[idx]
        residuals[i].append(Lambda[i] - last_event_lambda[i])
        last_event_lambda[i] = Lambda[i]
        R[:, i] += 1.0          # inject current event after accounting for it
        last_t = t
        prev_type = i

    return {i: np.asarray(v) for i, v in residuals.items()}


def qq_plot(residuals: dict, save_path: str):
    """QQ-plot of pooled residuals vs unit Exponential quantiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = np.concatenate([v for v in residuals.values() if len(v)])
    pooled = np.sort(pooled[np.isfinite(pooled)])
    n = len(pooled)
    if n == 0:
        return save_path
    theo = -np.log(1 - (np.arange(1, n + 1) - 0.5) / n)  # Exp(1) quantiles

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(theo, pooled, s=4, alpha=0.5)
    lim = max(theo.max(), pooled.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Exp(1) theoretical quantiles")
    ax.set_ylabel("residual quantiles")
    ax.set_title("Hawkes residual QQ-plot")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
