"""Generate LaTeX tables from experiment result JSON. No hand-entered numbers.

Tables:
  1. context scaling      (results/phase0/summary.json)
  2. Hawkes ablation      (results/phase1/summary.json)
  2b. kernel efficiency   (results/phase2/kernel_benchmark.json)
  3. financial baselines  (results/baselines/summary.json)

Each generator tolerates missing files (emits a stub) so the pipeline never hard
-fails before all phases have run.
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


def _ms(pair, fmt="{:.4f}"):
    """Format a (mean, std) pair."""
    if not isinstance(pair, (list, tuple)):
        return str(pair)
    return f"{fmt.format(pair[0])} $\\pm$ {fmt.format(pair[1])}"


def table1_context_scaling(results_dir):
    s = _load(os.path.join(results_dir, "phase0", "summary.json"))
    rows = []
    if s:
        for ctx in sorted(s, key=lambda k: int(k)):
            d = s[ctx]
            rows.append(f"{ctx} & {_ms(d['accuracy'])} & {_ms(d['f1_weighted'])} & "
                        f"{_ms(d['volatility_mae'], '{:.5f}')} & "
                        f"{d['peak_vram_mb'][0]:.0f} & {d['tokens_per_second'][0]:.0f} \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{6}{c}{(run phase 0)} \\\\"
    return r"""\begin{table}[t]
\centering
\caption{Context length vs.\ prediction quality on LOBSTER. Mean $\pm$ std over 3 seeds.}
\label{tab:context}
\begin{tabular}{lccccc}
\toprule
Context & Dir Acc & Dir F1 & Vol MAE & VRAM (MB) & Tok/s \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table2_hawkes_ablation(results_dir):
    s = _load(os.path.join(results_dir, "phase1", "summary.json"))
    rows = []
    label = {"dense_fp16": "Dense (baseline)", "sliding_512": "Sliding $W{=}512$",
             "sliding_threshold": "Sliding + Vol P90", "sliding_hawkes": "Sliding + Hawkes (ours)"}
    if s:
        for v, r in s.get("results", {}).items():
            rows.append(f"{label.get(v, v)} & {_ms(r['accuracy'])} & {_ms(r['f1_weighted'])} & "
                        f"{_ms(r['volatility_mae'], '{:.5f}')} \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{4}{c}{(run phase 1)} \\\\"
    ctx = s.get("context_length", "?") if s else "?"
    return rf"""\begin{{table}}[t]
\centering
\caption{{Attention-bias ablation at context={ctx}. Mean $\pm$ std over 3 seeds.}}
\label{{tab:ablation}}
\begin{{tabular}}{{lccc}}
\toprule
Attention variant & Dir Acc & Dir F1 & Vol MAE \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table2b_kernel(results_dir):
    rows_data = _load(os.path.join(results_dir, "phase2", "kernel_benchmark.json"))
    rows = []
    if rows_data:
        for r in rows_data:
            rows.append(f"{r['context']} & {r['fa2_vram_mb']} & {r['dense_bias_vram_mb']} & "
                        f"{r['flashlob_vram_mb']} & {r.get('speedup_x_vs_fa2','--')} \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{5}{c}{(run phase 2)} \\\\"
    return r"""\begin{table}[t]
\centering
\caption{Kernel efficiency: peak VRAM (MB) and FlashLOB speedup vs.\ FA2. OOM = out of memory.}
\label{tab:kernel}
\begin{tabular}{lcccc}
\toprule
Context & FA2 VRAM & Dense+Bias VRAM & FlashLOB VRAM & Speedup$\times$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table3_financial_baselines(results_dir):
    s = _load(os.path.join(results_dir, "baselines", "summary.json"))
    rows = []
    if s:
        for name, d in s.items():
            if "accuracy" not in d:
                continue
            ctx = d.get("context_length", ("--",))[0] if isinstance(d.get("context_length"), (list, tuple)) else d.get("context_length", "--")
            npar = d.get("n_params", ("--",))[0] if isinstance(d.get("n_params"), (list, tuple)) else "--"
            rows.append(f"{d.get('model', name)} & {ctx:.0f} & {_ms(d['accuracy'])} & "
                        f"{_ms(d['f1_weighted'])} & {npar/1e3:.0f}K \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{5}{c}{(run baselines)} \\\\"
    return r"""\begin{table}[t]
\centering
\caption{Financial baselines under the unified protocol. Snapshot models (DeepLOB,
TLOB) vs.\ event models (Dense, FlashLOB). Mean $\pm$ std over 3 seeds.}
\label{tab:baselines}
\begin{tabular}{lcccc}
\toprule
Model & Context & Dir Acc & Dir F1 & Params \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gens = {
        "table1_context.tex": table1_context_scaling,
        "table2_ablation.tex": table2_hawkes_ablation,
        "table2b_kernel.tex": table2b_kernel,
        "table3_baselines.tex": table3_financial_baselines,
    }
    for fname, fn in gens.items():
        tex = fn(args.results_dir)
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(tex + "\n")
        print(f"wrote {os.path.join(args.out_dir, fname)}")


if __name__ == "__main__":
    main()
