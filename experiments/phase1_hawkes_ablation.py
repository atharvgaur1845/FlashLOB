"""Phase 1 — Hawkes attention-bias ablation.

Compares four attention variants at a fixed context length (default 2048) using
PyTorch masking (no Triton kernel yet):

    dense_fp16        full causal attention
    sliding_512       causal sliding window (W=512)
    sliding_threshold sliding + global to top volume-percentile events
    sliding_hawkes    sliding + global to Hawkes top-k events + Hawkes bias (ours)

Acceptance: sliding_hawkes accuracy must beat sliding_threshold by >= hawkes_min_gain.
Results (mean +/- std over seeds) -> results/phase1/.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

import torch
from torch.utils.data import DataLoader

from experiments.common import load_tokens, make_loaders
from src.data.dataset import collate
from src.hawkes.bias import HawkesBias
from src.hawkes.calibrate import HawkesCalibrator
from src.model.transformer import LOBTransformer
from src.train.trainer import Trainer
from src.utils.config import apply_overrides, load_config
from src.utils.device import auto_budget, get_device, resolve, seed_everything


def build_variant(cfg, ctx, variant, hawkes_params):
    """Return a LOBTransformer configured for the given ablation variant."""
    m = cfg["model"]
    p = cfg["phase1"]
    common = dict(d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"],
                  d_ff=m["d_ff"], dropout=m["dropout"], context_length=ctx)
    W = p["sliding_window"]

    if variant == "dense_fp16":
        return LOBTransformer(attention_backend="flash2", **common)
    if variant == "sliding_512":
        return LOBTransformer(attention_backend="flashlob",
                              attn_spec={"window": W}, **common)
    if variant == "sliding_threshold":
        return LOBTransformer(attention_backend="flashlob",
                              attn_spec={"window": W, "global": "volume_p90",
                                         "volume_pct": p["volume_percentile"]}, **common)
    if variant == "sliding_hawkes":
        return LOBTransformer(attention_backend="flashlob",
                              attn_spec={"window": W, "global": "hawkes_topk",
                                         "n_hot": p["n_global_hot"]},
                              hawkes_bias=HawkesBias(hawkes_params),
                              hawkes_params=hawkes_params, **common)
    raise ValueError(variant)


def run(cfg):
    p = cfg["phase1"]
    out_dir = p["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    budget = auto_budget(cfg["model"]["d_model"], cfg["model"]["n_layers"])
    ctx = min(p["context_length"], budget.max_sparse_context)
    batch = resolve(cfg["train"]["batch_size"], budget.max_batch)

    # Hawkes params (calibrate if missing).
    hp_path = p["hawkes_params"]
    if not os.path.exists(hp_path):
        hp_path = "data/processed/hawkes_params.json"
    if os.path.exists(hp_path):
        hawkes_params = HawkesCalibrator.load(hp_path)
    else:
        from src.data.lobster import LOBSTERLoader
        loader = LOBSTERLoader()
        df = loader.load(cfg["data"]["lobster_message"], cfg["data"]["lobster_book"])
        cal = HawkesCalibrator()
        hawkes_params = cal.fit(df["time"].to_numpy(float), loader.hawkes_types(df))

    toks, ldir, lvol, htypes = load_tokens(cfg)
    results = {}
    for variant in p["variants"]:
        per_seed = []
        for seed in cfg["runtime"]["seeds"]:
            seed_everything(seed)
            tr, va, te = make_loaders(toks, ldir, lvol, htypes, ctx,
                                      cfg["data"]["stride"], batch,
                                      cfg["runtime"].get("num_workers", 2))
            model = build_variant(cfg, ctx, variant, hawkes_params)
            tag = f"p1_{variant}_s{seed}"
            print(f"[phase1] {variant} seed={seed} ctx={ctx}")
            trainer = Trainer(model, device, lr=cfg["train"]["lr"],
                              weight_decay=cfg["train"]["weight_decay"],
                              warmup_steps=cfg["train"]["warmup_steps"],
                              grad_clip=cfg["train"]["grad_clip"],
                              loss_dir_weight=cfg["train"]["loss_dir_weight"],
                              loss_vol_weight=cfg["train"]["loss_vol_weight"],
                              amp=cfg["train"]["amp"], ckpt_dir=cfg["train"]["ckpt_dir"],
                              log_every=cfg["train"]["log_every"], tag=tag,
                              csv_log=f"results/logs/{tag}.csv")
            trainer.fit(tr, va, epochs=cfg["train"]["epochs"],
                        patience=cfg["train"].get("patience", 5))
            mtr = trainer.evaluate(te)
            mtr["peak_vram_mb"] = round(trainer.peak_train_mb, 1)
            per_seed.append(mtr)
            with open(os.path.join(out_dir, f"{variant}_s{seed}.json"), "w") as f:
                json.dump(mtr, f, indent=2)

        agg = {}
        for k in ("accuracy", "f1_weighted", "volatility_mae", "peak_vram_mb"):
            vals = [r[k] for r in per_seed]
            agg[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
        results[variant] = agg

    verdict = check_acceptance(results, p["hawkes_min_gain"])
    summary = {"context_length": ctx, "results": results, "acceptance": verdict}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_acceptance(verdict)
    return summary


def check_acceptance(results, min_gain):
    if "sliding_hawkes" in results and "sliding_threshold" in results:
        gain = (results["sliding_hawkes"]["accuracy"][0]
                - results["sliding_threshold"]["accuracy"][0])
    else:
        gain = 0.0
    return {"hawkes_gain_over_threshold": round(gain, 4),
            "min_gain": min_gain, "passed": bool(gain >= min_gain)}


def print_acceptance(v):
    print("\n" + "=" * 60)
    print("PHASE 1 ACCEPTANCE")
    print(f"  Hawkes acc gain over volume-threshold: {v['hawkes_gain_over_threshold']:+.4f}")
    print(f"  threshold: >= {v['min_gain']}")
    if v["passed"]:
        print("  >>> PASS — Hawkes bias beats heuristic threshold.")
    else:
        print("  >>> FAIL — fall back to adaptive percentile (report as such).")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phase1.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run(cfg)


if __name__ == "__main__":
    main()
