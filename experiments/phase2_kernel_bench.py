"""Phase 2 — FlashLOB kernel benchmark driver.

Sweeps context lengths, comparing dense FA2 vs the FlashLOB Triton kernel on
throughput and peak VRAM, recording the OOM crossover. Writes Table 2 inputs to
results/phase2/kernel_benchmark.json.
"""

from __future__ import annotations

import argparse

from src.kernel.benchmark import run_benchmark
from src.utils.config import apply_overrides, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phase2.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    p = cfg["phase2"]
    run_benchmark(
        context_lengths=tuple(p["context_lengths"]),
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        batch_size=p["batch_size"], window=p["window_size"], n_hot=p["n_global_hot"],
        n_warmup=p["n_warmup"], n_bench=p["n_bench"],
        out_path=f"{p['output_dir']}/kernel_benchmark.json",
    )


if __name__ == "__main__":
    main()
