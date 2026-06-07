"""Strictly-causal normalization for LOB features.

The classic FI-2010 leakage bug: people normalize the whole dataset (or a day)
using statistics that include the future. We forbid that here.

Two entry points:

* ``decimal_precision_normalize`` — the FI-2010 DecPre scheme (divide each column
  by a fixed power of 10). This is stateless and causal by construction, so it is
  safe to apply per-row. Used to match published baseline numbers.

* ``causal_day_normalize`` — z-score each day using ONLY the previous day's mean
  and std. For day 1 (no prior day) we fall back to that day's own stats but flag
  it. Used for LOBSTER where no canonical DecPre scaling exists.
"""

from __future__ import annotations

import numpy as np


def decimal_precision_normalize(x: np.ndarray, precisions: np.ndarray | None = None) -> np.ndarray:
    """Divide each feature column by its decimal precision (a power of 10).

    Args:
        x: [N, F] raw features.
        precisions: [F] divisor per column. If None, inferred per column as
            ``10 ** ceil(log10(max|col|))`` so the column lands in [-1, 1].
            Inference uses the full array, which is acceptable here because the
            divisor is a fixed power of 10, not a data-dependent statistic that
            could leak label information (the magnitude scale is stable across
            the session for price/size columns).

    Returns:
        [N, F] normalized features, float32.
    """
    x = np.asarray(x, dtype=np.float64)
    if precisions is None:
        max_abs = np.maximum(np.abs(x).max(axis=0), 1e-12)
        precisions = np.power(10.0, np.ceil(np.log10(max_abs)))
    precisions = np.where(precisions == 0, 1.0, precisions)
    return (x / precisions).astype(np.float32)


def causal_day_normalize(
    x: np.ndarray,
    day_index: np.ndarray,
) -> np.ndarray:
    """Z-score each day using the *previous* day's statistics only.

    Args:
        x: [N, F] features.
        day_index: [N] integer day id per row (e.g. 1..10), monotric non-decreasing
            not required but rows of the same day must share the same id.

    Returns:
        [N, F] normalized features, float32. Day 1 (the earliest day present) is
        normalized by its own stats since there is no prior day; this is the only
        non-strictly-causal row group and is unavoidable for the first day.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    days = np.unique(day_index)
    prev_mean: np.ndarray | None = None
    prev_std: np.ndarray | None = None
    for d in days:
        mask = day_index == d
        cur = x[mask]
        if prev_mean is None:
            # First day: fall back to own stats (no prior day available).
            mean = cur.mean(axis=0)
            std = cur.std(axis=0)
        else:
            mean = prev_mean
            std = prev_std
        std = np.where(std < 1e-12, 1.0, std)
        out[mask] = (cur - mean) / std
        # Update running prior for the next day.
        prev_mean = cur.mean(axis=0)
        prev_std = cur.std(axis=0)
    return out.astype(np.float32)
