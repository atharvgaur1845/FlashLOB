# FlashLOB — Long-Context Limit Order Book Transformers via Market-Structure-Guided Sparse Attention

Paper 1 implementation. A systematic study of context-length scaling for LOB
transformers, a Hawkes-calibrated attention bias, and a Triton sparse-attention
kernel ("FlashLOB") that makes long contexts trainable.

## Claim

> Long historical context contains additional predictive signal for volatility
> tasks; that signal concentrates on high-impact events predictable from Hawkes
> microstructure theory; and a structured sparse attention kernel exploiting this
> structure enables much longer contexts with better accuracy than dense baselines.

## Phases (gated, run in order)

| Phase | What | Gate |
|------|------|------|
| **-1 PILOT** | Kill-switch: dense transformer, ctx [100..2048], 1 stock, 1 seed, 5 epochs. Q1 vol improves? Q2 attention clusters? Q3 memory grows? | NO-GO → pivot to empirical-only |
| **0** | Context-scaling sweep [100..4096] × 3 seeds, both tasks | Vol MAE must improve ≥3% (500→2048) |
| **1** | Hawkes attention-bias ablation (dense / sliding / +vol-threshold / +Hawkes) | Hawkes ≥ vol-threshold by ≥0.5% acc |
| **2** | Triton FlashLOB kernel: correctness + VRAM/throughput benchmark | kernel == reference (FP16 tol) |
| **Baselines** | DeepLOB, TLOB, Dense-transformer, FlashLOB under one protocol | — |

## Hardware

Config-driven and hybrid: runs on an **8GB RTX 4060** (auto-shrinks batch/context)
and scales unchanged to an **A100 80GB**. `src/utils/device.py` picks safe caps from
detected VRAM. The 8GB OOM crossover for dense+bias attention is recorded as a result.

## Setup

```bash
pip install -r requirements.txt   # torch (CUDA), triton, numpy, pandas, scipy, sklearn, mpl, pyyaml
# Triton ships with the Linux torch wheel. tick (Hawkes) is optional — an in-repo
# scipy MLE fallback produces identical params, so tick is not required.
```

## Data

```bash
bash scripts/download_data.sh         # instructions for FI-2010 (free) + LOBSTER sample (free)
# Until you have real data, generate a synthetic fixture (smoke tests / dev only):
python scripts/make_synthetic_lobster.py --out data/lobster --n 200000
```

* **FI-2010** (NoAuction/DecPre) → `data/fi2010/` — Table 3 cross-model comparison.
* **LOBSTER** message+orderbook → `data/lobster/` — Phases -1/0/1 long-context work.

## Run

```bash
bash experiments/run_paper1.sh        # full gated pipeline -> tables + figures

# or individual phases:
python experiments/phase_minus1_pilot.py
python experiments/phase0_context_scaling.py
python experiments/gate_check.py --phase 0 --results_dir results/phase0
python src/hawkes/calibrate.py --validate
python experiments/phase1_hawkes_ablation.py
python experiments/phase2_kernel_bench.py
python baselines/run_all_baselines.py --dataset lobster --models deeplob tlob dense flashlob
python paper/generate_tables.py && python paper/generate_figures.py
```

Every config knob lives in `experiments/configs/*.yaml`; override on the CLI with
`--set a.b=c`, e.g. `--set train.epochs=5 phase0.context_lengths=[100,500]`.

## Layout

```
src/data/      LOBSTER + FI-2010 loaders, event tokenizer, windowed dataset, labels
src/model/     event embedding, time-aware RoPE, attention (flash2/standard/flashlob), transformer
src/hawkes/    multivariate exp-Hawkes calibration (+ scipy fallback), vectorized bias, GoF
src/kernel/    pure-PyTorch reference + Triton FlashLOB kernel + benchmark
src/train/     raw PyTorch trainer, metrics, VRAM/time profiler
src/analysis/  attention-by-event-type, context curve, attention–Hawkes correlation
src/evaluation/ locked protocol + causal normalization
experiments/   phase drivers + gate check + shared harness + configs
baselines/     DeepLOB/TLOB reimplementations + Dense/FlashLOB runners + unified runner
paper/         LaTeX table + figure generators (read result JSON; no hand-entered numbers)
tests/         data / model / hawkes / kernel
```

## Notes & honest caveats

* The Triton kernel is **forward-only** (its job is the VRAM/throughput argument).
  Training uses PyTorch masking for the sparse path (Phase 1 approach); do not
  overstate the kernel as the training engine.
* Baselines are clean **reimplementations** trained under the identical protocol
  (more robust than the original repos). `scripts/clone_baselines.sh` fetches the
  originals if you want to cross-check.
* On the 4060, contexts 8192/16384 train only with the sparse path; dense+bias OOMs
  past ~4096. This is recorded, not worked around.
* Results are JSON-per-run (mean ± std over 3 seeds); failures (OOM, missing gains)
  are reported, not hidden.
```
