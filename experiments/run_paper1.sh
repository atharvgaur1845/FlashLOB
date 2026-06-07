#!/bin/bash
# FlashLOB Paper 1 — full gated pipeline.
# Requires data (run scripts/download_data.sh first, or use the synthetic fixture:
#   python scripts/make_synthetic_lobster.py --out data/lobster --n 200000)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=== FlashLOB Paper 1: Full Pipeline ==="

echo "--- Phase -1: PILOT / kill-switch ---"
python experiments/phase_minus1_pilot.py --config phase_minus1.yaml
# (Phase -1 prints its own GO/NO-GO. Inspect before continuing on real data.)

echo "--- Phase 0: Context Scaling ---"
python experiments/phase0_context_scaling.py --config phase0.yaml

echo "--- Gate check (Phase 0) ---"
python experiments/gate_check.py --phase 0 \
    --results_dir results/phase0 --min_improvement_volatility 0.03 \
  || { echo "Phase 0 gate FAILED — stopping (pivot to empirical-only)."; exit 1; }

echo "--- Hawkes calibration ---"
python src/hawkes/calibrate.py \
    --message data/lobster/sample_message.csv \
    --book data/lobster/sample_orderbook.csv \
    --output data/processed/hawkes_params.json --validate

echo "--- Phase 1: Hawkes Ablation ---"
python experiments/phase1_hawkes_ablation.py --config phase1.yaml

echo "--- Phase 2: Kernel Benchmark ---"
python experiments/phase2_kernel_bench.py --config phase2.yaml

echo "--- Baselines (DeepLOB, TLOB, Dense, FlashLOB) ---"
python baselines/run_all_baselines.py --dataset lobster \
    --models deeplob tlob dense flashlob --horizons 10 --context 100

echo "--- Generating paper assets ---"
python paper/generate_tables.py --results_dir results
python paper/generate_figures.py --results_dir results

echo "=== Done. Tables in paper/tables/, figures in paper/figures/ ==="
