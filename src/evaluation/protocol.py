"""Locked evaluation protocol — single source of truth.

Import ``PROTOCOL`` (and the helpers) everywhere; never redefine these constants
inline. Every baseline and every FlashLOB run must read from here so Table 3 is a
fair apples-to-apples comparison.

Rationale for each choice is documented inline; the values match the FI-2010
convention used by DeepLOB / TransLOB / TLOB so reproduced numbers are comparable.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# THE PROTOCOL. Do not edit casually — changing any value invalidates Table 3.
# ---------------------------------------------------------------------------
PROTOCOL = {
    # --- FI-2010 day splits (standard convention) ---
    # Days are 1..10. Train on 1-7, test on 8-10, hold out 20% of train as val.
    "fi2010_train_days": [1, 2, 3, 4, 5, 6, 7],
    "fi2010_test_days": [8, 9, 10],
    "fi2010_val_frac": 0.2,

    # --- Normalization ---
    # FI-2010 "decimal precision" (DecPre): each feature divided by its decimal
    # level (a power of 10). This is what DeepLOB/TransLOB/TLOB use.
    # NOT z-score, NOT min-max. For LOBSTER we apply a strictly-causal variant
    # (normalize each day from the *previous* day's stats) — see fi2010_normalize.py.
    "normalization": "decimal_precision",

    # --- Prediction horizons (k events ahead) ---
    "horizons": [10, 20, 30, 50, 100],
    "primary_horizon": 10,  # headline number in Table 3; others go to the appendix

    # --- Label convention (mid-price direction, 3-class) ---
    #   up    (2): mid[t+k] >  mid[t] * (1 + alpha)
    #   down  (0): mid[t+k] <  mid[t] * (1 - alpha)
    #   stable(1): otherwise
    # Class index order is {0: down, 1: stable, 2: up} throughout the codebase.
    "label_alpha": 0.00002,  # 0.002% — standard FI-2010 threshold

    # --- Metrics (report all; bold the best per column) ---
    "metrics": ["accuracy", "f1_weighted", "f1_per_class", "n_params"],

    # --- Seeds: run each model 3x, report mean +/- std ---
    "seeds": [42, 123, 456],
}

# Canonical class indices used by labels.py, metrics.py and every baseline adapter.
CLASS_DOWN = 0
CLASS_STABLE = 1
CLASS_UP = 2
CLASS_NAMES = ["down", "stable", "up"]
N_CLASSES = 3


def label_from_return(forward_return: float, alpha: float | None = None) -> int:
    """Map a fractional mid-price return to a 3-class label per PROTOCOL.

    forward_return is (mid[t+k] - mid[t]) / mid[t].
    """
    if alpha is None:
        alpha = PROTOCOL["label_alpha"]
    if forward_return > alpha:
        return CLASS_UP
    if forward_return < -alpha:
        return CLASS_DOWN
    return CLASS_STABLE


def validate_protocol() -> None:
    """Cheap invariants so an accidental edit fails loudly in CI."""
    assert sorted(PROTOCOL["fi2010_train_days"]) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted(PROTOCOL["fi2010_test_days"]) == [8, 9, 10]
    assert PROTOCOL["primary_horizon"] in PROTOCOL["horizons"]
    assert 0.0 < PROTOCOL["fi2010_val_frac"] < 1.0
    assert PROTOCOL["label_alpha"] > 0.0
    assert len(PROTOCOL["seeds"]) == 3
