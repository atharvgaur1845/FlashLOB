"""Gate check between phases. Reads a phase's summary.json and prints a STOP/GO
verdict, exiting non-zero on STOP so run_paper1.sh can halt the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def phase0_gate(results_dir: str, min_improvement: float) -> bool:
    """Pass if volatility MAE improves >= min_improvement from ctx 500 -> 2048
    (or the nearest available pair of short vs long contexts)."""
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)
    ctxs = sorted(int(k) for k in summary)

    def nearest(target):
        return min(ctxs, key=lambda c: abs(c - target))

    short, long = nearest(500), nearest(2048)
    mae_short = summary[str(short)]["volatility_mae"][0]
    mae_long = summary[str(long)]["volatility_mae"][0]
    rel = (mae_short - mae_long) / mae_short if mae_short > 0 else 0.0

    print("=" * 60)
    print(f"PHASE 0 GATE: volatility MAE {short}->{long}")
    print(f"  MAE@{short}={mae_short:.6f}  MAE@{long}={mae_long:.6f}")
    print(f"  relative improvement = {rel:.2%} (threshold {min_improvement:.0%})")
    ok = rel >= min_improvement
    print(f"  >>> {'GO — proceed to Phase 1' if ok else 'STOP — no context signal'}")
    print("=" * 60)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--min_improvement_volatility", type=float, default=0.03)
    args = ap.parse_args()

    if args.phase == 0:
        ok = phase0_gate(args.results_dir, args.min_improvement_volatility)
    else:
        raise SystemExit(f"no gate defined for phase {args.phase}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
