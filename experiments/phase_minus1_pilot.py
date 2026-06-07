"""Phase -1 — PILOT / KILL-SWITCH.

The cheapest possible de-risking experiment, run BEFORE committing to the full
project. Dense transformer, single stock, single seed, 5 epochs, contexts
[100,250,500,1000,2048]. Answers three go/no-go questions:

    Q1. Does volatility MAE improve with context?      -> proceed signal
    Q2. Does attention cluster on event types?         -> Hawkes-bias signal
    Q3. Does memory explode with context?              -> kernel motivation

Verdict logic (thresholds in configs/phase_minus1.yaml):
    GO    if Q1 (vol MAE drops >= vol_improvement_min from shortest to longest ctx)
    Otherwise NO-GO -> the prompt's instruction: "throw away 80% of the project",
    i.e. pivot to an empirical-only note.

Usage:
    python experiments/phase_minus1_pilot.py            # uses synthetic if no real data
    python experiments/phase_minus1_pilot.py --config phase_minus1.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from experiments.common import (build_model, load_tokens, make_loaders, run_single)
from src.analysis.attention_viz import (attention_by_event_type, cluster_score,
                                        plot_attention_heatmap)
from src.data.dataset import collate
from src.utils.config import apply_overrides, load_config
from src.utils.device import auto_budget, get_device, resolve


def run(cfg: dict):
    p = cfg["phase_minus1"]
    out_dir = p["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    budget = auto_budget(cfg["model"]["d_model"], cfg["model"]["n_layers"])
    contexts = budget.filter_contexts(p["context_lengths"], sparse=False)
    seed = p["seed"]

    toks, ldir, lvol, htypes = load_tokens(cfg)
    results = []
    for ctx in contexts:
        tag = f"pm1_ctx{ctx}_s{seed}"
        print(f"[phase-1] training context={ctx} seed={seed} ...")
        m = run_single(cfg, toks, ldir, lvol, htypes, ctx, seed, tag,
                       epochs=cfg["train"]["epochs"])
        results.append(m)
        with open(os.path.join(out_dir, f"ctx{ctx}_s{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"   acc={m['accuracy']:.4f} volMAE={m['volatility_mae']:.6f} "
              f"peakVRAM={m['peak_vram_mb']}MB")

    # --- Q2: attention clustering on the longest feasible context ---
    ctx = contexts[-1]
    batch = resolve(cfg["train"]["batch_size"], budget.max_batch)
    _, _, te = make_loaders(toks, ldir, lvol, htypes, ctx, cfg["data"]["stride"],
                            batch, cfg["runtime"].get("num_workers", 2))
    model = build_model(cfg, ctx).to(device)
    # load best checkpoint for this ctx if present
    ckpt = os.path.join(cfg["train"]["ckpt_dir"], f"pm1_ctx{ctx}_s{seed}_{cfg['train']['epochs']-1}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    abt = attention_by_event_type(model, te, device, n_batches=20)
    cscore = cluster_score(abt)
    fig_path = os.path.join(out_dir, "attention_heatmap.png")
    try:
        plot_attention_heatmap(abt, fig_path)
    except Exception as e:
        fig_path = f"(figure failed: {e})"

    verdict = decide(results, cscore, cfg)
    summary = {
        "contexts": contexts,
        "results": results,
        "cluster_score": cscore,
        "attention_heatmap": fig_path,
        "verdict": verdict,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_verdict(verdict)
    return summary


def decide(results, cscore, cfg) -> dict:
    p = cfg["phase_minus1"]
    by_ctx = {r["context_length"]: r for r in results}
    ctxs = sorted(by_ctx)
    short, long = ctxs[0], ctxs[-1]
    mae_short = by_ctx[short]["volatility_mae"]
    mae_long = by_ctx[long]["volatility_mae"]
    rel_impr = (mae_short - mae_long) / mae_short if mae_short > 0 else 0.0

    vram_short = by_ctx[short]["peak_vram_mb"]
    vram_long = by_ctx[long]["peak_vram_mb"]
    vram_growth = vram_long / max(vram_short, 1e-9)

    q1 = rel_impr >= p["vol_improvement_min"]
    q2 = cscore >= p["attention_cluster_min"]
    q3 = vram_growth >= 2.0  # memory grows materially with context

    go = bool(q1)  # primary gate per the user's instruction
    return {
        "q1_volatility_improves": q1, "vol_rel_improvement": round(rel_impr, 4),
        "q2_attention_clusters": q2, "cluster_score": round(cscore, 4),
        "q3_memory_grows": q3, "vram_growth_x": round(vram_growth, 2),
        "GO": go,
    }


def print_verdict(v: dict):
    print("\n" + "=" * 60)
    print("PHASE -1 VERDICT")
    print("=" * 60)
    print(f"  Q1 volatility improves: {v['q1_volatility_improves']} "
          f"(rel improvement {v['vol_rel_improvement']:.1%})")
    print(f"  Q2 attention clusters : {v['q2_attention_clusters']} "
          f"(score {v['cluster_score']})")
    print(f"  Q3 memory grows       : {v['q3_memory_grows']} "
          f"({v['vram_growth_x']}x)")
    if v["GO"]:
        print("\n  >>> GO. Proceed to Phase 0 with confidence.")
    else:
        print("\n  >>> NO-GO. Volatility shows no context benefit.")
        print("      Per plan: pivot to an empirical-only note (drop ~80%).")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phase_minus1.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides a.b=c")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run(cfg)


if __name__ == "__main__":
    main()
