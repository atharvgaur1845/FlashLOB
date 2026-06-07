"""Generate paper figures from experiment result JSON.

  Figure 1: context-scaling curve (accuracy + volatility MAE)   [phase0]
  Figure 2: attention heatmap by event type                     [phase -1/0 artifact]
  Figure 3: memory vs context (FA2 / dense+bias / FlashLOB)      [phase2]
"""

from __future__ import annotations

import argparse
import json
import os


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def figure1_context_curve(results_dir, out):
    s = _load(os.path.join(results_dir, "phase0", "summary.json"))
    if not s:
        print("fig1: no phase0 summary; skip")
        return
    summary = {int(k): {m: tuple(v[m]) for m in ("accuracy", "volatility_mae")}
               for k, v in s.items()}
    from src.analysis.context_curve import plot_context_curve
    plot_context_curve(summary, out)
    print("wrote", out)


def figure3_memory_curve(results_dir, out):
    rows = _load(os.path.join(results_dir, "phase2", "kernel_benchmark.json"))
    if not rows:
        print("fig3: no phase2 benchmark; skip")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = [r["context"] for r in rows]
    def series(key):
        xs, ys = [], []
        for r in rows:
            if r[key] != "OOM":
                xs.append(r["context"]); ys.append(r[key])
        return xs, ys

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for key, label, color in [("fa2_vram_mb", "FA2 (dense, FlashAttn)", "tab:blue"),
                              ("dense_bias_vram_mb", "Dense + Hawkes bias", "tab:red"),
                              ("flashlob_vram_mb", "FlashLOB (ours)", "tab:orange")]:
        xs, ys = series(key)
        if xs:
            ax.plot(xs, ys, "-o", label=label, color=color)
    # O(L^2) reference, anchored at the first dense+bias point
    db = series("dense_bias_vram_mb")
    ref_at = {}
    if db[0]:
        x0, y0 = db[0][0], db[1][0]
        ref = [y0 * (c / x0) ** 2 for c in ctx]
        ref_at = dict(zip(ctx, ref))
        ax.plot(ctx, ref, "--", color="gray", alpha=0.6, label="$O(L^2)$ ref")
    # mark OOM points for dense+bias at the extrapolated O(L^2) height
    for r in rows:
        if r["dense_bias_vram_mb"] == "OOM" and r["context"] in ref_at:
            ax.scatter([r["context"]], [ref_at[r["context"]]],
                       marker="x", color="tab:red", s=80, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("context length (events)"); ax.set_ylabel("peak VRAM (MB)")
    ax.set_title("Memory vs. context length")
    ax.legend(fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    figure1_context_curve(args.results_dir, os.path.join(args.out_dir, "fig1_context.pdf"))
    figure3_memory_curve(args.results_dir, os.path.join(args.out_dir, "fig3_memory.pdf"))
    # figure 2 (attention heatmap) is emitted by the phase -1/0 runs directly into
    # results/, since it needs a trained model; copy it here if present.
    src = os.path.join(args.results_dir, "phase_minus1", "attention_heatmap.png")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(args.out_dir, "fig2_attention.png"))
        print("copied fig2 attention heatmap")


if __name__ == "__main__":
    main()
