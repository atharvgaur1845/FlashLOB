"""Run all baselines under the unified evaluation protocol.

Saves results/baselines/{model}_{seed}_{horizon}.json and an aggregated
summary.json (mean +/- std over seeds), which feeds Table 3.

Usage:
    python baselines/run_all_baselines.py --dataset fi2010 \
        --models deeplob tlob --horizons 10 --seeds 42 123 456
    python baselines/run_all_baselines.py --dataset lobster \
        --models deeplob tlob dense flashlob --horizons 10
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

from baselines.runners import get_runner
from src.evaluation.protocol import PROTOCOL
from src.utils.config import apply_overrides, load_config


def run_all(args):
    cfg = apply_overrides(load_config(args.config), args.set)
    cfg["data"]["dataset"] = args.dataset
    cfg["baseline_context"] = args.context
    out_dir = "results/baselines"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name in args.models:
        per = []
        for seed in args.seeds:
            for h in args.horizons:
                runner = get_runner(name, cfg)
                try:
                    r = runner.run(seed=seed, horizon=h)
                except Exception as e:  # keep going; record the failure
                    r = {"model": name, "seed": seed, "horizon": h, "error": str(e)}
                    print(f"[baselines] {name} s{seed} h{h} FAILED: {e}")
                per.append(r)
                with open(os.path.join(out_dir, f"{name}_{seed}_{h}.json"), "w") as f:
                    json.dump(r, f, indent=2)
                if "accuracy" in r:
                    print(f"[baselines] {name} s{seed} h{h}: acc={r['accuracy']:.4f} "
                          f"f1={r['f1_weighted']:.4f} params={r.get('n_params')}")
        results[name] = _aggregate(per, args.primary_horizon)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[baselines] wrote summary for {list(results)}")
    return results


def _aggregate(per, primary_horizon):
    rows = [r for r in per if r.get("horizon") == primary_horizon and "accuracy" in r]
    if not rows:
        return {"note": "no successful runs at primary horizon"}
    agg = {}
    for k in ("accuracy", "f1_weighted", "n_params", "context_length"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
    agg["model"] = rows[0]["model"]
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--dataset", default="lobster", choices=["fi2010", "lobster"])
    ap.add_argument("--models", nargs="+",
                    default=["deeplob", "tlob", "dense", "flashlob"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[10])
    ap.add_argument("--primary_horizon", type=int, default=PROTOCOL["primary_horizon"])
    ap.add_argument("--seeds", nargs="+", type=int, default=PROTOCOL["seeds"])
    ap.add_argument("--context", type=int, default=100)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
