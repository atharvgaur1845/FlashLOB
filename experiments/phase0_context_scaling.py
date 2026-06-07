"""Phase 0 — context-scaling empirical study.

Sweeps CONTEXT_LENGTHS x SEEDS, training the dense transformer on both tasks
jointly, recording accuracy / F1 / volatility-MAE / peak-VRAM / throughput per run
to results/phase0/{ctx}_{seed}.json, then writes an aggregated summary.json
(mean +/- std over seeds) consumed by the table/figure generators.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

from experiments.common import load_tokens, run_single
from src.utils.config import apply_overrides, load_config
from src.utils.device import auto_budget


METRIC_KEYS = ["accuracy", "f1_weighted", "volatility_mae",
               "peak_vram_mb", "tokens_per_second"]


def aggregate(results: list[dict]) -> dict:
    """Group by context_length; mean/std over seeds for each metric."""
    by_ctx: dict[int, list[dict]] = {}
    for r in results:
        by_ctx.setdefault(r["context_length"], []).append(r)
    summary = {}
    for ctx, runs in sorted(by_ctx.items()):
        summary[ctx] = {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in runs]
            summary[ctx][k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
    return summary


def run(cfg: dict):
    p = cfg["phase0"]
    out_dir = p["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    budget = auto_budget(cfg["model"]["d_model"], cfg["model"]["n_layers"])
    contexts = budget.filter_contexts(p["context_lengths"], sparse=False)
    seeds = cfg["runtime"]["seeds"]

    toks, ldir, lvol, htypes = load_tokens(cfg)
    results = []
    for ctx in contexts:
        for seed in seeds:
            tag = f"p0_ctx{ctx}_s{seed}"
            print(f"[phase0] context={ctx} seed={seed}")
            m = run_single(cfg, toks, ldir, lvol, htypes, ctx, seed, tag)
            results.append(m)
            with open(os.path.join(out_dir, f"{ctx}_{seed}.json"), "w") as f:
                json.dump(m, f, indent=2)

    summary = aggregate(results)
    # JSON keys must be strings
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"[phase0] wrote summary over {len(contexts)} contexts x {len(seeds)} seeds")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phase0.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run(cfg)


if __name__ == "__main__":
    main()
