# FlashLOB — Project Roadmap & Task Tracker

> **Living document.** Update result cells as experiments complete.
> Every hypothesis has a result slot. Fill it. Never leave it blank.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Scientific Hypothesis Hierarchy](#2-scientific-hypothesis-hierarchy)
3. [Result Targets by Table](#3-result-targets-by-table)
4. [Phase 0 — Context Scaling Empirical Study](#4-phase-0)
5. [Phase 1 — Hawkes Attention Bias](#5-phase-1)
6. [Phase 2 — Triton FlashLOB Kernel](#6-phase-2)
7. [Baseline Reproduction](#7-baseline-reproduction)
8. [Analysis & Interpretability](#8-analysis--interpretability)
9. [Paper Writing Tasks](#9-paper-writing-tasks)
10. [Risk Register](#10-risk-register)
11. [Timeline](#11-timeline)
12. [Decision Log](#12-decision-log)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | FlashLOB |
| **Paper title** | FlashLOB: Long-Context Limit Order Book Transformers via Market-Structure-Guided Sparse Attention |
| **Target venue (P1)** | NeurIPS 2026 Workshop on Efficient ML or ML for Finance |
| **Target venue (P2)** | ICML 2027 or NeurIPS 2026 main track |
| **Primary dataset** | LOBSTER AAPL 2012-06-21 (free sample) |
| **Secondary dataset** | FI-2010 (benchmark, for baseline comparison) |
| **Hardware** | Single A100 80GB or RTX 4090 24GB |
| **Language** | Python 3.10+, PyTorch 2.x, Triton 2.x |
| **Status** | [ ] Not started / [ ] In progress / [ ] Complete |

### Core Claim (one sentence)

> We show for the first time that long historical context (8k–16k events) improves
> limit order book volatility prediction, that this signal is concentrated on
> high-impact events predictable from Hawkes microstructure theory, and that a
> domain-aware Triton attention kernel exploiting this structure enables such
> contexts on a single GPU.

### What this paper is NOT

- Not a paper about quantization (that is Paper 2)
- Not a paper about LOB architecture design (not competing with TLOB on FI-2010 accuracy)
- Not a paper about Hawkes processes (Hawkes is a tool, not the contribution)
- Not a systems paper (the kernel enables science, it is not the science)

---

## 2. Scientific Hypothesis Hierarchy

Every task in this roadmap maps to one of these hypotheses.
A task only matters if it helps prove or disprove a hypothesis.

### H0 — The Gate Hypothesis (must be true for the paper to exist)

> **H0**: Long historical context beyond 500 events contains additional predictive
> signal for LOB tasks. Specifically, realized volatility prediction improves
> monotonically from context=500 to context≥2048 on LOBSTER AAPL.

**Why this is the gate**: If H0 is false, there is no scientific motivation for long
context, and the kernel is just an optimization for a problem that does not exist.

| Sub-hypothesis | Claim | Expected result | Actual result |
|---|---|---|---|
| **H0a** | Direction accuracy saturates at ~500 events | Plateau between context=500 and context=1000 | _fill_ |
| **H0b** | Volatility MAE improves past 500 events | ≥3% MAE reduction from context=500→2048 | _fill_ |
| **H0c** | Beyond a saturation point, accuracy degrades (noise) | Possible degradation at context=16384 without sparse attention | _fill_ |

**If H0 is FALSE**: Pivot. Write the paper as:
"We present the first systematic empirical study of context length scaling in LOB
transformers and find that signal saturates at [K] events — this negative result
has direct implications for future LOB model design."
This is still publishable at ICAIF or finance-focused workshops.

---

### H1 — The Structure Hypothesis

> **H1**: The predictive signal in long LOB history is concentrated on high-impact
> events (large trades, spread jumps) as predicted by Hawkes microstructure theory,
> not uniformly distributed across all events.

| Sub-hypothesis | Claim | Expected result | Actual result |
|---|---|---|---|
| **H1a** | Trained attention heads preferentially attend to trade events | Mean attention weight to trades > to cancellations in later layers | _fill_ |
| **H1b** | Top-attended events correlate with ex-post price impact | Spearman rank corr(attn, impact) > 0.3 | _fill_ |
| **H1c** | Hawkes-calibrated selection outperforms volume-threshold | Hawkes top-K recall of "true hot events" > P90 volume recall | _fill_ |

**If H1 is FALSE** (attention is uniform, no correlation with impact):
"Surprisingly, attention weights are uniformly distributed across event types.
This suggests that LOB transformers learn statistical patterns beyond microstructure
theory, and motivates future interpretability work." Still publishable — negative
interpretability results are valuable.

---

### H2 — The Efficiency Hypothesis

> **H2**: Structured sparse attention (sliding window + Hawkes-selected global events)
> achieves equal or better accuracy than dense attention at long contexts,
> using a fraction of the memory.

| Sub-hypothesis | Claim | Expected result | Actual result |
|---|---|---|---|
| **H2a** | Sparse outperforms dense at context=2048 on volatility | FlashLOB-sparse vol. MAE ≤ dense vol. MAE | _fill_ |
| **H2b** | VRAM scales as O(L·W) not O(L²) | Peak VRAM at context=8192 ≤ 2× peak VRAM at context=4096 | _fill_ |
| **H2c** | Hawkes bias improves over pure sliding window | sliding+hawkes vol. MAE < sliding-only vol. MAE by ≥0.5% | _fill_ |
| **H2d** | Sliding+Hawkes outperforms dense at context=4096 | Despite seeing only ~7% of tokens, accuracy does not degrade | _fill_ |

---

### H3 — The Hawkes Prior Hypothesis

> **H3**: A Hawkes-process–calibrated attention prior is a more effective
> inductive bias than a naive volume-based threshold for identifying
> high-information events.

| Sub-hypothesis | Claim | Expected result | Actual result |
|---|---|---|---|
| **H3a** | Hawkes prior correctly ranks event importance | Precision@K: Hawkes > volume P90 > uniform | _fill_ |
| **H3b** | Hawkes prior improves final accuracy | sliding+hawkes_bias better than sliding+volume_threshold | _fill_ |
| **H3c** | Hawkes parameters are stable across days | Fitted A, D matrices have CV < 20% across training days | _fill_ |

---

### H4 — The Kernel Hypothesis

> **H4**: A Triton implementation of FlashLOB sparse attention enables
> context lengths that are infeasible with standard FlashAttention 2,
> without compromising numerical correctness.

| Sub-hypothesis | Claim | Expected result | Actual result |
|---|---|---|---|
| **H4a** | FA2 OOMs at some context on target hardware | FA2 OOMs at context≤8192 on RTX 4090 | _fill_ |
| **H4b** | FlashLOB kernel correct vs. reference | Max abs error < 1e-2 vs PyTorch reference | _fill_ |
| **H4c** | FlashLOB enables OOM context | FlashLOB runs at context where FA2 OOMs | _fill_ |
| **H4d** | FlashLOB is competitive in speed | FlashLOB throughput ≥ 50% of FA2 at context=2048 | _fill_ |

---

## 3. Result Targets by Table

### Table 1 — Context Scaling (FlashLOB model only, LOBSTER AAPL)

**Purpose**: Proves H0 and H2b. The paper's core empirical finding.

| Context | Dir Acc (%) | Dir F1 | Vol MAE | Peak VRAM (MB) | Tok/s | Notes |
|---|---|---|---|---|---|---|
| 100 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FA2 dense |
| 250 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FA2 dense |
| 500 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FA2 dense |
| 1000 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FA2 dense |
| 2048 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FA2 dense |
| 4096 | OOM | — | — | — | — | FA2 OOMs here |
| 4096 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FlashLOB kernel |
| 8192 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FlashLOB kernel |
| 16384 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | FlashLOB kernel |

**Target pattern**: Vol MAE should decrease from 100→best_context (proves H0b).
Dir Acc should plateau around 500-1000 (proves H0a).

---

### Table 2 — Attention Bias Ablation (context=2048, LOBSTER AAPL)

**Purpose**: Proves H2c and H3b. Shows Hawkes is better than heuristic.

| Variant | Dir Acc (%) ± std | Dir F1 ± std | Vol MAE ± std | Description |
|---|---|---|---|---|
| Dense FA2 | _fill_ | _fill_ | _fill_ | All-pairs, no mask |
| Sliding W=512 | _fill_ | _fill_ | _fill_ | Local only |
| Sliding + Vol P90 | _fill_ | _fill_ | _fill_ | Local + high-volume global |
| Sliding + Hawkes | _fill_ | _fill_ | _fill_ | **Ours**: local + Hawkes-selected global |

**Target pattern**: Sliding+Hawkes ≥ Sliding+VolP90 > Sliding-only > Dense
(Dense is last because dense at context=2048 has too much noise in long history
without structured selection.)

---

### Table 3 — Financial Baselines Comparison (FI-2010, horizon k=10)

**Purpose**: Locates FlashLOB in the LOB literature. Proves you beat prior work.

| Model | Context | Dir Acc (%) ± std | Dir F1 ± std | #Params | VRAM | Source |
|---|---|---|---|---|---|---|
| DeepLOB (2019) | 100 | _fill_ | _fill_ | ~300K | ~100MB | reproduced |
| TransLOB (2020) | 100 | _fill_ | _fill_ | ~1.2M | ~200MB | reproduced |
| MLPLOB (2025) | 500 | _fill_ | _fill_ | _fill_ | _fill_ | reproduced |
| TLOB (2025) | 500 | _fill_ | _fill_ | ~2M | ~500MB | reproduced |
| LOBERT (2025) | 500 | _fill_ | _fill_ | — | — | †paper numbers |
| FA2+Transformer | 2048 | _fill_ | _fill_ | ~2M | _fill_ | reproduced |
| FA2+Transformer | 4096 | OOM | OOM | ~2M | OOM | OOM on hardware |
| **FlashLOB (ours)** | **4096** | _fill_ | _fill_ | ~2M | _fill_ | **ours** |
| **FlashLOB (ours)** | **8192** | _fill_ | _fill_ | ~2M | _fill_ | **ours** |
| **FlashLOB (ours)** | **16384** | _fill_ | _fill_ | ~2M | _fill_ | **ours** |

†LOBERT: no public code; numbers from arXiv:2511.12563.

---

### Table 4 — Kernel Benchmarks (speed + memory)

**Purpose**: Proves H4. Shows the kernel is fast enough to be practical.

| Context | FA2 VRAM (MB) | FlashLOB VRAM (MB) | VRAM Δ | FA2 Tok/s | FlashLOB Tok/s | Speedup |
|---|---|---|---|---|---|---|
| 512 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| 1024 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| 2048 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| 4096 | OOM | _fill_ | — | OOM | _fill_ | — |
| 8192 | OOM | _fill_ | — | OOM | _fill_ | — |
| 16384 | OOM | _fill_ | — | OOM | _fill_ | — |

---

## 4. Phase 0

### Scientific goal

Prove or disprove **H0**, **H0a**, **H0b**, **H0c**.
Generate **Table 1** rows for context=100 through context=2048.
Generate **Figure 1** (context scaling curve).
Produce **attention analysis** data for H1a.

This phase has no kernel work. Pure PyTorch + FA2 only.

---

### 4.1 Data Pipeline

#### Task D1 — FI-2010 Loader

**File**: `src/data/fi2010.py`
**Input**: `data/fi2010/NoAuction/3.NoAuction_DecPre/` (decimal precision normalized CSVs)
**Output**: `(tokens: FloatTensor [N, 40], labels: LongTensor [N], mid_prices: FloatTensor [N])`

**Implementation details**:
- FI-2010 has 5 stocks, 10 days each, 149 rows per timestamp
- Rows 0–39: LOB features (10 levels × 4: ask_price, ask_size, bid_price, bid_size)
- Rows 40–143: hand-crafted features (NOT used — use only first 40 columns)
- Rows 144–148: 5 prediction labels at horizons k=10,20,30,50,100
- Label encoding in FI-2010: 1=up, 2=stable, 3=down → remap to 2=up, 1=stable, 0=down
- Split: days 1–7 train, day 7 last 20% → val, days 8–10 test
- Each "day" for each stock is a separate CSV file

**Acceptance test** (`tests/test_data.py::test_fi2010_loader`):
```
assert tokens.shape[1] == 40
assert labels.max() == 2 and labels.min() == 0
assert labels.float().mean() between 0.28 and 0.38  # roughly balanced
```

---

#### Task D2 — LOBSTER Loader

**File**: `src/data/lobster.py`
**Input**: LOBSTER message CSV + orderbook CSV
**Output**: `pd.DataFrame` with columns: `[time, type, size, price, direction, mid_price, spread, best_bid, best_ask, bid_vol_total, ask_vol_total, vol_imbalance]`

**Implementation details**:
- Message file columns: Time, Type, OrderID, Size, Price, Direction
- Price in LOBSTER is stored as integer × 10000. Convert: `price_float = price_raw / 10000`
- Type mapping: 1=limit_submit, 2=partial_cancel, 3=cancel, 4=exec_visible, 5=exec_hidden, 7=halt
- Mid price: `(best_ask_price + best_bid_price) / 2` from book file
- Spread: `best_ask_price - best_bid_price`
- Filter: remove rows where Type=7 (trading halt) and where spread ≤ 0
- Filter: keep only normal trading hours — seconds 34200 (9:30 AM) to 57600 (4:00 PM)
- `vol_imbalance = (bid_vol_total - ask_vol_total) / (bid_vol_total + ask_vol_total)`

**Acceptance test** (`tests/test_data.py::test_lobster_loader`):
```
assert df['spread'].min() > 0        # no crossed book
assert df['time'].between(34200, 57600).all()   # trading hours only
assert df['type'].isin([1,2,3,4,5]).all()       # no halts after filter
assert not df.isnull().any().any()              # no NaN
```

---

#### Task D3 — Event Tokenizer

**File**: `src/data/lobster.py::LOBSTERLoader.tokenize()`
**Input**: LOBSTER DataFrame
**Output**: `FloatTensor [N_events, 8]`

**Feature engineering** (each feature, its formula, and its range):

| Index | Name | Formula | Range | Rationale |
|---|---|---|---|---|
| 0 | `time_norm` | `(time - 34200) / 23400` | [0, 1] | Intra-day position |
| 1 | `log_size` | `log1p(size_shares)` | [0, ~12] | Size in log scale; used by Hawkes bias |
| 2 | `price_tick` | `(price - mid_price_prev) / tick_size` | [-20, +20] | Signed ticks from mid |
| 3 | `direction` | `direction ∈ {+1, -1}` | {+1, -1} | Buy vs sell |
| 4 | `event_type` | int ∈ {0,1,2,3,4,5} | {0..5} | Mapped from LOBSTER types |
| 5 | `spread_norm` | `spread / median_spread_today` | [0.5, 10] | Relative spread |
| 6 | `vol_imbalance` | `(bid-ask)/(bid+ask)` | [-1, +1] | Order book pressure |
| 7 | `level` | book level of limit order (0=best, 9=deepest); -1 for trades | {-1..9} | Depth info |

**Critical**: `price_tick` must use `mid_price_prev` (mid price at the PREVIOUS event),
not the current mid. Using current mid creates a subtle look-ahead leak.

**Acceptance test**:
```
tokens = loader.tokenize(df)
assert tokens.shape == (len(df), 8)
assert tokens[:, 0].between(0, 1).all()         # time_norm in [0,1]
assert tokens[:, 3].abs().eq(1).all()           # direction ∈ {+1,-1}
assert tokens[:, 4].between(0, 5).all()         # event type valid
assert not tokens.isnan().any()
```

---

#### Task D4 — Label Generation

**File**: `src/data/labels.py`

**Mid-price direction** (`mid_price_direction`):
```
smoothed_mid[t] = mean(mid_price[t : t+k])   # k = horizon_events
label[t] = 2 (up)     if smoothed_mid[t] > mid_price[t] * (1 + 0.00002)
label[t] = 0 (down)   if smoothed_mid[t] < mid_price[t] * (1 - 0.00002)
label[t] = 1 (stable) otherwise
```

Note: use SMOOTHED mid (average of next k mid-prices), not single-point mid at t+k.
This is the FI-2010 convention. Single-point is noisier and gives different numbers.

**Realized volatility** (`realized_volatility`):
```
log_returns = log(mid_price[t+1:t+W]) - log(mid_price[t:t+W-1])
  where W = all events within next window_seconds seconds
vol[t] = std(log_returns) * sqrt(annualization_factor)
```
Use `window_seconds=60.0` as default. Report in basis points (multiply by 10000).

**Acceptance test**:
```
dir_labels = mid_price_direction(df, horizon_events=10)
# class balance should be roughly 30/40/30 (some stable dominance is OK)
counts = dir_labels.bincount()
assert counts.max() / counts.sum() < 0.6   # no single class > 60%

vol_labels = realized_volatility(df, window_seconds=60.0)
assert (vol_labels >= 0).all()
assert vol_labels.mean() between 0.0001 and 0.01  # reasonable bps range
```

---

#### Task D5 — PyTorch Dataset

**File**: `src/data/dataset.py::LOBEventDataset`

**Key design**: The `context_length` argument is the single variable swept in Phase 0.
The same code must work for context=100 and context=16384 without modification.

**Stride selection**: Default stride=50 (sample every 50th event as a query point).
At stride=50, LOBSTER AAPL 1 day (~45k events) gives ~900 training windows.
For training: increase to stride=1 for more data, stride=100 for faster iteration.

**Padding**: If sequence start has fewer than context_length events, pad with zeros
and create a mask (mask=0 for padded positions). Pass mask to attention.

**`meta` dictionary** (required for Hawkes bias and attention analysis):
```python
meta = {
    'event_types': LongTensor [context_length],     # column 4 of token tensor
    'log_sizes':   FloatTensor [context_length],    # column 1
    'times':       FloatTensor [context_length],    # column 0 (raw seconds, not normalized)
    'is_trade':    BoolTensor [context_length],     # event_type ∈ {4, 5}
}
```

---

### 4.2 Baseline Model

#### Task M1 — Event Embedding

**File**: `src/model/embedding.py::EventEmbedding`

Architecture:
```
input: FloatTensor [B, L, 8]
→ Linear(8, d_model)      → LayerNorm(d_model) → GELU
→ Linear(d_model, d_model)
output: FloatTensor [B, L, d_model]
```

No tokenization. No discretization. Raw continuous features → embeddings directly.
LayerNorm after first linear (not before) — this prevents gradient issues with
heterogeneous feature scales (time_norm is [0,1], log_size is [0,12]).

**Acceptance test**:
```
emb = EventEmbedding(d_model=128)
x = torch.randn(2, 512, 8)
out = emb(x)
assert out.shape == (2, 512, 128)
assert not out.isnan().any()
```

---

#### Task M2 — Time-Aware Rotary Positional Encoding

**File**: `src/model/positional.py::TimeAwareRoPE`

**Why not integer RoPE**: LOB events are irregularly spaced. Event 500 may be
1 millisecond after event 499, or 10 seconds after. Integer RoPE treats them
identically. TimeAwareRoPE encodes actual wall-clock time.

**Implementation**:
```
Given: times FloatTensor [B, L] in seconds from midnight

theta_dim_i = 1.0 / (10000 ^ (2i / d_head))   for i in 0..d_head//2
angles[b, t, i] = times[b, t] * theta_dim_i

cos_vals = cos(angles).repeat_interleave(2, dim=-1)  # [B, L, d_head]
sin_vals = sin(angles).repeat_interleave(2, dim=-1)

def rotate(x):
    # x: [B, H, L, d_head]
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos_vals[:, None] + x_rot * sin_vals[:, None]

Apply rotate() to Q and K before computing attention.
```

**Acceptance test**:
```
rope = TimeAwareRoPE(d_head=16)
q = torch.randn(2, 8, 100, 16)
times = torch.linspace(0, 1, 100).unsqueeze(0).expand(2, -1)
q_rot = rope(q, times)
assert q_rot.shape == q.shape
# test: two queries at same time should have same rotation
times2 = times.clone(); times2[:, 50] = times2[:, 49]  # same time
q2_rot = rope(q, times2)
assert torch.allclose(q2_rot[:, :, 49], q2_rot[:, :, 50], atol=1e-5)
```

---

#### Task M3 — LOB Transformer

**File**: `src/model/transformer.py::LOBTransformer`

**Architecture** (fixed for all experiments — do not change d_model between phases):
```
d_model   = 128
n_heads   = 8
n_layers  = 6
d_ff      = 512
d_head    = 16   (= d_model / n_heads)
dropout   = 0.1
```

**Forward pass**:
```
events [B, L, 8]
→ EventEmbedding                    → [B, L, 128]
→ TimeAwareRoPE applied to Q, K in each layer
→ 6× TransformerEncoderLayer
    (causal: each position attends only to past positions)
→ Take last position's hidden state: h = output[:, -1, :]  [B, 128]
→ FC head → logits_direction [B, 3]
→ FC head → logits_volatility [B, 1]
```

**Attention backend switching** (needed across phases):
```python
class LOBTransformer(nn.Module):
    def __init__(self, ..., attention_backend='flash2'):
        # 'flash2'    → F.scaled_dot_product_attention(is_causal=True) [Phase 0]
        # 'standard'  → manual QK^T + softmax + mask [for Phase 1 masking experiments]
        # 'flashlob'  → custom Triton kernel [Phase 2]
```

**Parameter count**: ~2.1M params (consistent with TLOB for fair comparison).

---

#### Task M4 — Prediction Heads

**File**: `src/model/head.py`

```python
class DirectionHead(nn.Module):
    # Linear(d_model, 3)
    # Used with CrossEntropyLoss

class VolatilityHead(nn.Module):
    # Linear(d_model, 1) → squeeze
    # Used with MSELoss on log-scale volatility
    # Predict log1p(vol) and inverse-transform at evaluation
```

**Combined loss** (used in trainer):
```
loss = 0.7 * F.cross_entropy(logits_dir, labels_dir)
     + 0.3 * F.mse_loss(pred_vol.squeeze(), labels_vol.log1p())
```

Rationale for 0.7/0.3 split: direction task dominates (classification signal is cleaner);
volatility task provides regularization and is the scientifically interesting metric.

---

### 4.3 Training Infrastructure

#### Task T1 — Trainer

**File**: `src/train/trainer.py::Trainer`

**Training protocol** (identical for all models across all phases):
```
Optimizer:   AdamW, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999)
Schedule:    CosineAnnealingLR with warmup
Warmup:      500 steps linear (lr 0 → 3e-4)
After warmup: cosine decay to lr_min=1e-5
Max epochs:  50
Early stop:  patience=10 epochs on val vol MAE
Grad clip:   max_norm=1.0
AMP:         torch.autocast('cuda', dtype=torch.float16)
Batch size:  32 (reduce to 16 at context=4096+ if needed)
```

**Logging** (every 100 steps):
```python
log = {
    'step': int,
    'train_loss': float,
    'train_loss_dir': float,
    'train_loss_vol': float,
    'peak_vram_mb': torch.cuda.max_memory_allocated() / 1e6,
    'wall_clock_s': time.time() - step_start,
    'tokens_per_second': (batch_size * context_length) / wall_clock_s,
}
```

---

#### Task T2 — Memory Profiler

**File**: `src/train/profiler.py::MemoryProfiler`

```python
class MemoryProfiler:
    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.peak_mb = torch.cuda.max_memory_allocated() / 1e6

# Usage pattern in every experiment:
with MemoryProfiler() as prof:
    output = model(events, times, meta)
    loss = compute_loss(output, labels)
    loss.backward()
peak_mb = prof.peak_mb
```

---

#### Task T3 — Metrics

**File**: `src/train/metrics.py`

```python
def compute_metrics(
    logits_dir: Tensor,    # [B, 3]
    labels_dir: Tensor,    # [B]
    pred_vol: Tensor,      # [B, 1]
    labels_vol: Tensor,    # [B]
) -> dict:
    return {
        'accuracy': (logits_dir.argmax(-1) == labels_dir).float().mean().item(),
        'f1_weighted': sklearn.metrics.f1_score(
            labels_dir.cpu(), logits_dir.argmax(-1).cpu(), average='weighted'
        ),
        'f1_up':     f1_score(..., labels=[2]),
        'f1_stable': f1_score(..., labels=[1]),
        'f1_down':   f1_score(..., labels=[0]),
        'vol_mae':   F.l1_loss(pred_vol.exp() - 1, labels_vol).item(),  # inverse log1p
        'vol_mse':   F.mse_loss(pred_vol.exp() - 1, labels_vol).item(),
    }
```

---

### 4.4 Phase 0 Experiment

#### Task E0 — Context Scaling Sweep

**File**: `experiments/phase0_context_scaling.py`

**Sweep grid**:
```
context_lengths = [100, 250, 500, 1000, 2048, 4096]
seeds = [42, 123, 456]
tasks = ['direction', 'volatility']  # both trained simultaneously
```

**Per-run output** (`results/phase0/{ctx}_{seed}.json`):
```json
{
    "context_length": 2048,
    "seed": 42,
    "val_accuracy": 0.652,
    "val_f1_weighted": 0.638,
    "val_vol_mae": 0.00312,
    "test_accuracy": 0.649,
    "test_f1_weighted": 0.635,
    "test_vol_mae": 0.00318,
    "peak_vram_mb": 4821.3,
    "tokens_per_second": 18420.0,
    "train_time_total_min": 47.2,
    "epochs_trained": 31,
    "stopped_early": true,
    "oom": false
}
```

**If OOM** (`oom: true`): record the exact context at which OOM occurred and the
VRAM usage at the last successful step. This is Table 1's "OOM" entry.

---

### 4.5 Phase 0 Analysis

#### Task A0 — Attention Heatmap

**File**: `src/analysis/attention_viz.py`

**Method**: After Phase 0 training, load the best checkpoint for context=2048.
Run on 50 random test batches with `return_attn=True`.
For each layer and head, compute mean attention weight from each query event type
to each key event type.

**Output**: `results/attention/attn_by_type_{ctx}.json`
```json
{
    "layer_0_head_0": [[0.21, 0.18, 0.41, 0.20], ...],  // [4 query types × 4 key types]
    ...
    "layer_5_head_7": [[...], ...]
}
```

**Figure 2**: 4×4 heatmap averaged across all heads in layer 5 (last layer).
Expected pattern: column for "trade" events should be brightest (highest attention).
If it isn't, record which column is brightest and what event type it corresponds to.
Either way, this is a publishable finding.

**Gate check for H1a**:
```python
# Mean attention weight to trades (type=2) across all heads, layer 5
trade_attn_mean = attn_matrix[:, 2].mean()  # mean of column 2 (trades)
all_types_mean = attn_matrix.mean()
if trade_attn_mean > all_types_mean * 1.2:
    print("H1a SUPPORTED: trades receive 20%+ more attention than average")
else:
    print("H1a NOT SUPPORTED: attention is approximately uniform")
```

---

### 4.6 Phase 0 Gate

**Gate condition**: proceed to Phase 1 only if BOTH:
1. Volatility MAE improves by ≥3% from context=500 to context=2048
2. At least one context (500–2048) shows better vol MAE than context=100

**Gate script** (`experiments/gate_check.py --phase 0`):
```python
results = load_phase0_results()
vol_500 = results['context_500']['test_vol_mae']
vol_2048 = results['context_2048']['test_vol_mae']
improvement = (vol_500 - vol_2048) / vol_500

if improvement >= 0.03:
    print(f"✓ GATE PASSED: vol MAE improved by {improvement:.1%} (500→2048)")
    print("→ Proceed to Phase 1")
else:
    print(f"✗ GATE FAILED: vol MAE improvement only {improvement:.1%}")
    print("→ Pivot: write empirical-study-only paper")
    print("→ Do NOT start Phase 1 without discussion")
```

---

## 5. Phase 1

### Scientific goal

Prove **H1**, **H2c**, **H3**.
Generate **Table 2** (attention bias ablation).
Calibrate Hawkes model for use in Phase 2 kernel.

---

### 5.1 Hawkes Calibration

#### Task H1 — Fit Hawkes Model

**File**: `src/hawkes/calibrate.py::HawkesCalibrator`

**Library**: `tick` (Bacry et al.). Install: `pip install tick`.

**Event types for Hawkes model** (simplified from 6 LOBSTER types to 4):
```
Type 0: limit order submit   ← LOBSTER type 1
Type 1: cancellation         ← LOBSTER types 2, 3
Type 2: trade (small)        ← LOBSTER types 4, 5 with size ≤ 90th percentile
Type 3: trade (large)        ← LOBSTER types 4, 5 with size > 90th percentile
```

**Fitting procedure**:
```python
from tick.hawkes import HawkesSumExpKern

# Prepare input: list of lists of timestamps per event type
timestamps = [
    times_type0,   # FloatArray of event times in seconds
    times_type1,
    times_type2,
    times_type3,
]

learner = HawkesSumExpKern(
    decays=[0.1, 1.0, 10.0],   # three exponential decay rates (multi-scale)
    C=1000,                     # L2 regularization strength
    n_threads=4,
    max_iter=300,
    tol=1e-5,
    verbose=True,
)
learner.fit(timestamps, end_times=[end_time])

# Extract fitted parameters:
baseline = learner.baseline   # FloatArray [4] — base intensity per type
adjacency = learner.adjacency # FloatArray [4, 4] — excitation matrix A[i,j]
decays = learner.decays       # FloatArray [3] — the input decays (fixed)
# Note: tick uses shared decays across all pairs; compute effective per-pair
# decay as argmax decay dimension for each (i,j) pair
```

**Goodness of fit** (`src/hawkes/validate.py::check_fit_quality`):
- Compute time-rescaled residuals: for each event sequence, rescale time by
  the integral of the fitted intensity. Under correct model, residuals should
  be i.i.d. unit exponential.
- Plot QQ plot vs. unit exponential. Save to `results/hawkes/qq_plot.pdf`.
- Compute KS statistic. If KS p-value < 0.05, the fit is poor. Report anyway —
  imperfect fit is honest and shows the model is an inductive bias, not a truth.

**Output** (`data/processed/hawkes_params.json`):
```json
{
    "baseline": [0.12, 0.34, 0.08, 0.02],
    "adjacency": [[0.21, 0.18, 0.41, 0.61], [...],...],
    "decays": [0.1, 1.0, 10.0],
    "fit_ks_pvalue": 0.12,
    "fit_date": "2012-06-21",
    "n_events": 45231,
    "note": "Fitted on LOBSTER AAPL single day"
}
```

**Stability check** (`src/hawkes/validate.py::check_param_stability`):
If you have multiple LOBSTER days, fit the Hawkes model on each day separately.
Compute coefficient of variation (CV = std/mean) for each entry in adjacency matrix.
If CV < 20% for the large entries (>0.1), parameters are stable → use the mean.
If CV > 30%, report instability as a limitation and use the median instead.

---

#### Task H2 — Hawkes Attention Bias Module

**File**: `src/hawkes/bias.py::HawkesBias`

**Formula implemented**:
```
bias(q, k) = A[type_q, type_k] * log1p(size_k) * Σ_d w_d * exp(-decay_d * Δt)

where:
  Δt       = time_q - time_k   (must be ≥ 0 for causal — clamp negative to 0)
  A        = adjacency matrix from calibrated Hawkes model (not learned)
  decay_d  = the three decay rates [0.1, 1.0, 10.0]
  w_d      = mixture weights (use uniform 1/3 for each decay rate)
```

**Vectorized GPU implementation** (no Python loops):
```python
def forward(self, meta: dict) -> FloatTensor:
    times = meta['times']              # [B, L]
    types = meta['event_types']        # [B, L] int64
    log_sz = meta['log_sizes']         # [B, L]

    delta_t = times[:, :, None] - times[:, None, :]   # [B, L, L]
    delta_t = delta_t.clamp(min=0)                     # causal
    delta_t_self = (delta_t == 0)                      # diagonal mask

    A_mat = self.adjacency[types[:, :, None], types[:, None, :]]  # [B, L, L]
    # Multi-scale exponential: sum over 3 decay rates
    decay_kernel = sum(
        exp(-decay * delta_t) / 3.0
        for decay in self.decays
    )                                                  # [B, L, L]

    log_sz_k = log_sz[:, None, :]                      # [B, 1, L]
    bias = A_mat * log_sz_k * decay_kernel             # [B, L, L]
    bias = bias.masked_fill(delta_t_self, 0.0)         # no self-boost
    bias = bias.clamp(-5.0, 5.0)                       # numerical stability
    return bias.unsqueeze(1)                           # [B, 1, L, L]
```

**Memory warning**: The [B, L, L] intermediate tensor is large.
At B=32, L=2048: 32 × 2048 × 2048 × 4 bytes = 512 MB.
This is acceptable for Phase 1 (context=2048 only).
At context=4096, this would be 2 GB just for the bias — hence Phase 2 kernel
computes bias on-the-fly in tiles, never materializing the full [L, L] matrix.

---

### 5.2 Attention Mask Variants

#### Task M5 — Attention Mask Factory

**File**: `src/model/attention.py::build_attention_mask`

Implement four mask variants as boolean tensors of shape [B, L, L]
(True = compute attention, False = mask out):

```python
def build_attention_mask(
    mask_type: str,
    seq_len: int,
    meta: dict,
    hawkes_params: dict = None,
    window: int = 512,
    n_hot: int = 64,
    vol_percentile: float = 0.90,
) -> BoolTensor:  # [B, L, L]

    # Causal base mask (always applied):
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    if mask_type == 'dense':
        return causal.expand(B, -1, -1)

    elif mask_type == 'sliding':
        window_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            window_mask[i, max(0, i-window):i+1] = True
        return (causal & window_mask).expand(B, -1, -1)

    elif mask_type == 'sliding_threshold':
        # sliding + global attention to events above volume P90
        vol_threshold = meta['log_sizes'].quantile(vol_percentile, dim=1)  # [B]
        is_hot = meta['log_sizes'] > vol_threshold[:, None]  # [B, L]
        hot_mask = is_hot[:, None, :].expand(-1, seq_len, -1)  # any query → hot key
        sliding = build_attention_mask('sliding', seq_len, meta, window=window)
        return (sliding | (causal & hot_mask))

    elif mask_type == 'sliding_hawkes':
        # sliding + global attention to top-K events by Hawkes prior score
        # Hawkes prior score for each event k (query-independent):
        #   score[k] = max_{type_q} A[type_q, type_k] * log_size[k]
        A = torch.tensor(hawkes_params['adjacency'])   # [4, 4]
        types_k = meta['event_types']                  # [B, L]
        log_sz = meta['log_sizes']                     # [B, L]
        score = A[:, types_k].max(dim=0).values * log_sz  # [B, L]
        _, hot_idx = score.topk(n_hot, dim=1)          # [B, n_hot]
        hot_mask = torch.zeros(B, L, L, dtype=torch.bool)
        hot_mask.scatter_(2, hot_idx.unsqueeze(1).expand(-1, L, -1), True)
        sliding = build_attention_mask('sliding', seq_len, meta, window=window)
        return (sliding | (causal & hot_mask))
```

---

### 5.3 Phase 1 Experiment

#### Task E1 — Hawkes Ablation

**File**: `experiments/phase1_hawkes_ablation.py`

**Fixed parameters**: context_length=2048, all other model params same as Phase 0.
**Variable**: mask_type ∈ {dense, sliding, sliding_threshold, sliding_hawkes}

**For sliding_hawkes variant**: add HawkesBias to QK^T in the attention computation:
```python
# In attention layer forward:
logits = (Q @ K.transpose(-2, -1)) / sqrt(d_head)
if self.hawkes_bias is not None:
    logits = logits + self.hawkes_bias(meta)  # [B, 1, L, L] broadcasts over heads
logits = logits.masked_fill(~mask, float('-inf'))
weights = softmax(logits, dim=-1)
```

**Per-variant output** (`results/phase1/{variant}_{seed}.json`):
```json
{
    "variant": "sliding_hawkes",
    "seed": 42,
    "context_length": 2048,
    "window_size": 512,
    "n_hot": 64,
    "val_accuracy": 0.661,
    "val_f1_weighted": 0.648,
    "val_vol_mae": 0.00298,
    "test_accuracy": 0.658,
    "test_vol_mae": 0.00301,
    "peak_vram_mb": 5240.0,
    "tokens_per_second": 12100.0
}
```

---

### 5.4 Phase 1 Gate

**Gate condition**: proceed to Phase 2 only if:
`sliding_hawkes test_vol_mae < sliding_threshold test_vol_mae` (at 2σ confidence)

**Script** (`experiments/gate_check.py --phase 1`):
```python
hawkes_maes = [load_result('sliding_hawkes', seed)['test_vol_mae'] for seed in seeds]
thresh_maes = [load_result('sliding_threshold', seed)['test_vol_mae'] for seed in seeds]

t_stat, p_val = scipy.stats.ttest_ind(hawkes_maes, thresh_maes)
if p_val < 0.05 and np.mean(hawkes_maes) < np.mean(thresh_maes):
    print(f"✓ GATE PASSED: Hawkes (MAE={np.mean(hawkes_maes):.5f}) "
          f"< threshold (MAE={np.mean(thresh_maes):.5f}), p={p_val:.3f}")
else:
    print(f"✗ GATE: Hawkes not significantly better (p={p_val:.3f})")
    print("→ Use sliding_threshold variant in Phase 2 (fallback)")
    print("→ Note in paper: 'Hawkes calibration did not significantly outperform "
          "volume-threshold selection; we therefore use adaptive P90 threshold.'")
    print("→ Phase 2 still proceeds — kernel is valuable regardless.")
```

---

## 6. Phase 2

### Scientific goal

Prove **H4**.
Generate **Table 4** (kernel benchmarks).
Enable long-context training for Table 1 rows at context=4096, 8192, 16384.

---

### 6.1 Reference Implementation

#### Task K0 — PyTorch Reference

**File**: `src/kernel/reference.py::flashlob_reference`

This is ONLY for correctness validation. Never used in training.
Implements the exact computation the Triton kernel should replicate.

```python
def flashlob_reference(Q, K, V, meta, hawkes_params,
                        window_size=256, n_hot=64) -> Tensor:
    B, H, L, d = Q.shape
    # Build full attention mask [B, L, L]
    mask = build_attention_mask('sliding_hawkes', L, meta, hawkes_params,
                                window=window_size, n_hot=n_hot)
    # Build Hawkes bias [B, 1, L, L]
    bias = HawkesBias(hawkes_params)(meta)
    # Compute full attention (this OOMs at L>4096 — OK for reference only)
    logits = (Q @ K.transpose(-2,-1)) / math.sqrt(d)     # [B, H, L, L]
    logits = logits + bias                                # broadcast over heads
    logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
    weights = F.softmax(logits, dim=-1)
    return weights @ V   # [B, H, L, d]
```

---

### 6.2 Triton Kernel

#### Task K1 — Stage A: Tiled FlashAttention (FP16, Dense)

**File**: `src/kernel/flash_lob.py` (Stage A only)

**What it does**: Replicate standard FlashAttention 2 tiled algorithm in Triton.
This is the foundation. Every subsequent stage adds to it.

**Tile sizes**:
```
BLOCK_M = 64   # query block size (number of query positions per CTA)
BLOCK_N = 64   # key block size
HEAD_DIM = 16  # d_head (= d_model / n_heads = 128 / 8)
```

**Algorithm** (FlashAttention 2 forward, online softmax):
```
for each query block q_block:
    m = -inf, l = 0, o = zeros   # running max, sum, output
    for each key block k_block:
        # 1. Load Q[q_block], K[k_block], V[k_block] from HBM
        # 2. Compute S = Q @ K^T / sqrt(d)                [BLOCK_M, BLOCK_N]
        # 3. m_new = max(m, rowmax(S))
        # 4. P = exp(S - m_new)
        # 5. l_new = exp(m - m_new) * l + rowsum(P)
        # 6. o = exp(m - m_new) * o + P @ V[k_block]
        # 7. m = m_new, l = l_new
    # 8. o = o / l   (normalize)
    # 9. Store o to HBM
```

**Acceptance test** (`tests/test_kernel.py::test_stage_a`):
```python
torch.manual_seed(42)
B, H, L, d = 2, 8, 512, 16
Q = torch.randn(B, H, L, d, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, L, d, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, L, d, dtype=torch.float16, device='cuda')

ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
out = flashlob_stage_a(Q, K, V, is_causal=True)

assert (ref - out).abs().max() < 1e-2   # FP16 tolerance
assert (ref - out).abs().mean() < 1e-3
```

**Speed check**: kernel must achieve ≥80% of FA2 speed at L=1024.
If slower than 80% of FA2, profile and fix before adding Stage B.

---

#### Task K2 — Stage B: Causal Sliding Window Mask

**Adds to Stage A**: skip key blocks entirely outside the sliding window.

**Key logic inside kernel** (before loading K, V for a key block):
```python
# query_start = start position of current query block
# key_start   = start position of current key block
# WINDOW_SIZE is a tl.constexpr

key_end = key_start + BLOCK_N
query_start = pid_m * BLOCK_M

# Case 1: Entire key block is outside window (too far in the past) → SKIP
if key_end < query_start - WINDOW_SIZE:
    continue   # early return, no memory load

# Case 2: Entire key block is within window → compute normally, no per-element mask
# Case 3: Partial overlap → compute with per-element positional mask
#   mask[m, n] = (key_start + n) >= (query_start + m - WINDOW_SIZE)
#   and (key_start + n) < (query_start + m)   (causal)
```

**Expected speedup at L=4096, W=256**: the inner key loop now iterates over
only W/BLOCK_N = 256/64 = 4 blocks instead of L/BLOCK_N = 64 blocks.
Theoretical speedup: 16×. Practical: 8–12× due to global hot-event overhead.

**Acceptance test** (`tests/test_kernel.py::test_stage_b`):
```python
# Reference uses the sliding mask from reference.py
ref = flashlob_reference(Q, K, V, meta, hawkes_params,
                          window_size=256, n_hot=0)  # n_hot=0: local only
out = flashlob_stage_b(Q, K, V, meta, window_size=256)
assert (ref - out).abs().max() < 1e-2
```

---

#### Task K3 — Stage C: Global Hot-Event Stream

**Adds to Stage B**: a second attention pass over n_hot hot events.

**Preprocessing** (`src/kernel/flash_lob.py::build_hot_index`):
```python
def build_hot_index(meta, hawkes_params, n_hot=64) -> LongTensor:
    # [B, n_hot] — indices of hot events per sequence
    A = torch.tensor(hawkes_params['adjacency'])   # [4, 4]
    types = meta['event_types']                    # [B, L]
    log_sz = meta['log_sizes']                     # [B, L]
    # Query-agnostic score: max excitation potential across all query types
    score = A[:, types].max(dim=0).values * log_sz  # [B, L]
    _, hot_idx = score.topk(n_hot, dim=1)           # [B, n_hot]
    return hot_idx.contiguous()                      # must be contiguous for kernel
```

**Inside kernel** (after Stage B local window loop):
```python
# Second pass: global hot events
for i in range(N_HOT):   # N_HOT is tl.constexpr = 64
    hot_k = tl.load(hot_index_ptr + bid * N_HOT + i)   # scalar index
    # Load K[hot_k], V[hot_k]
    k_hot = tl.load(K_ptr + bid*stride_kb + hid*stride_kh + hot_k*stride_kn + offs_d)
    v_hot = tl.load(V_ptr + ...)
    # Compute QK^T for this single key
    qk_hot = tl.sum(q * k_hot[None, :], dim=1) * sm_scale   # [BLOCK_M]
    # Update online softmax accumulators (m, l, o) from Stage B
    m_new = tl.maximum(m, qk_hot)
    l_new = tl.exp(m - m_new) * l + tl.exp(qk_hot - m_new)
    o = tl.exp(m - m_new)[:, None] * o + tl.exp(qk_hot - m_new)[:, None] * v_hot[None, :]
    m, l = m_new, l_new
```

**Acceptance test** (`tests/test_kernel.py::test_stage_c`):
```python
# Construct sequence where one event has extremely high Hawkes score
# and is outside the local window. Verify its contribution appears in output.
meta['log_sizes'][0, 100] = 15.0   # giant event at position 100
meta['event_types'][0, 100] = 3    # large trade type
hot_idx = build_hot_index(meta, hawkes_params, n_hot=64)
assert 100 in hot_idx[0]           # must be selected as hot event

ref = flashlob_reference(Q, K, V, meta, hawkes_params, window_size=256, n_hot=64)
out = flashlob_stage_c(Q, K, V, meta, hawkes_params, window_size=256, n_hot=64)
assert (ref - out).abs().max() < 1e-2
```

---

#### Task K4 — Stage D: Hawkes Bias in Kernel

**Adds to Stage C**: compute Hawkes bias on-the-fly inside the kernel tiles.
Never materialize the full [B, L, L] bias matrix — compute [BLOCK_M, BLOCK_N] at a time.

**Kernel arguments added**:
```python
# New arguments to the Triton kernel:
Meta_ptr,           # [B, L, 3]: packed (event_type, log_size, time)
A_table_ptr,        # [4, 4] float32 adjacency matrix (16 values, tiny)
D_table_ptr,        # [3] float32 decay rates (3 values)
stride_meta_b, stride_meta_l,
```

**Bias computation inside kernel** (per tile):
```python
# Load meta for query block [BLOCK_M, 3]
meta_q = tl.load(Meta_ptr + q_start*stride_meta_l + offs_meta)
time_q  = meta_q[:, 2]  # [BLOCK_M]
type_q  = meta_q[:, 0].to(tl.int32)  # [BLOCK_M]

# Load meta for key block [BLOCK_N, 3]
meta_k = tl.load(Meta_ptr + k_start*stride_meta_l + offs_meta_n)
time_k  = meta_k[:, 2]  # [BLOCK_N]
type_k  = meta_k[:, 0].to(tl.int32)  # [BLOCK_N]
log_sz_k = meta_k[:, 1]  # [BLOCK_N]

# Compute delta_t [BLOCK_M, BLOCK_N]
delta_t = time_q[:, None] - time_k[None, :]
delta_t = tl.maximum(delta_t, 0.0)

# Load A values: A[type_q, type_k] — 4×4 lookup
# In Triton: A_table is passed as a 16-element flat array
a_idx = type_q[:, None] * 4 + type_k[None, :]   # [BLOCK_M, BLOCK_N]
A_vals = tl.load(A_table_ptr + a_idx)            # [BLOCK_M, BLOCK_N]

# Multi-scale decay: sum over 3 decay rates
d0, d1, d2 = tl.load(D_table_ptr), tl.load(D_table_ptr+1), tl.load(D_table_ptr+2)
decay_kernel = (tl.exp(-d0*delta_t) + tl.exp(-d1*delta_t) + tl.exp(-d2*delta_t)) / 3.0

bias = A_vals * log_sz_k[None, :] * decay_kernel  # [BLOCK_M, BLOCK_N]
bias = tl.where(delta_t > 0, bias, 0.0)
bias = tl.clamp(bias, -5.0, 5.0)

# Add to logits before online softmax
qk = qk + bias
```

**Acceptance test** (`tests/test_kernel.py::test_stage_d`):
```python
ref = flashlob_reference(Q, K, V, meta, hawkes_params, window_size=256, n_hot=64)
out = flashlob_stage_d(Q, K, V, meta, hawkes_params, window_size=256, n_hot=64)
assert (ref - out).abs().max() < 1e-2
assert (ref - out).abs().mean() < 1e-3
```

---

### 6.3 Kernel Benchmarking

#### Task K5 — Throughput and Memory Benchmarks

**File**: `src/kernel/benchmark.py::run_benchmark`

**Benchmark protocol**:
```
For each context_length in [512, 1024, 2048, 4096, 8192, 16384]:
    For each backend in ['fa2_dense', 'flashlob']:
        try:
            warmup: run 10 forward passes, discard results
            benchmark: run 50 forward passes, record:
                - wall_clock_ms per pass (CUDA events, not time.time())
                - peak_vram_mb (torch.cuda.max_memory_allocated())
                - tokens_per_second = (B * L) / wall_clock_s
        except torch.cuda.OutOfMemoryError:
            record OOM, continue
```

**CUDA event timing** (not wall clock — eliminates Python overhead):
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output = model_forward(...)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

---

## 7. Baseline Reproduction

### Scientific goal

Populate **Table 3** with reproduced numbers.
Ensure every row is comparable (same protocol, same dataset split, same labels).

---

### 7.1 Evaluation Protocol (Locked)

```
Dataset:     FI-2010 (NoAuction, DecimalPrecision normalization)
Train split: stocks × days 1–7
Val split:   last 20% of train, per stock, temporally
Test split:  stocks × days 8–10
Horizon:     k=10 (primary), also report k=20,30,50,100 in appendix
Labels:      smoothed mid-price direction (alpha=0.00002)
Seeds:       42, 123, 456
Report:      mean ± std over 3 seeds
Metrics:     accuracy, weighted-F1, per-class F1
```

**Normalization detail**: FI-2010 "decimal precision" means each feature is divided
by a power of 10 equal to its decimal order of magnitude. Do NOT use z-score.
Pre-normalized files are already in the FI-2010 download package.
Use folder: `NoAuction/3.NoAuction_DecPre/`.

---

### 7.2 Baseline Tasks

#### Task B1 — DeepLOB

**Repo**: `github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books`
**Use**: PyTorch version in `jupyter_pytorch/`

**Adapter** (`baselines/deeplob_runner.py`):
```python
# Clone repo, add to sys.path, import model
# Override: seed setting, data path, logging
# DO NOT modify the model architecture — compare as-published
# Report the PyTorch version numbers (TensorFlow version has label bug)

def extract_metrics(model, test_loader, seed) -> dict:
    # Run inference on test set
    # Use YOUR metrics.py for consistency — not DeepLOB's own evaluation
```

**Expected results** (FI-2010, k=10, decimal precision):
```
Accuracy:    64–66%
Weighted F1: 0.62–0.65
```
If your reproduction is outside these ranges: check normalization first, then label threshold.

---

#### Task B2 — TLOB

**Repo**: `github.com/LeonardoBerti00/TLOB`

```bash
git clone https://github.com/LeonardoBerti00/TLOB
cd TLOB
pip install -r requirements.txt

# CRITICAL config settings for fair comparison:
# In config file, set:
#   labeling_method: standard    # NOT their new method — that inflates numbers
#   all_features: false          # FI-2010 only (no extra computed features)
#   hidden_dim: 40               # their recommended setting for FI-2010

python main.py +model=tlob +dataset=fi2010 hydra.job.chdir=False
python main.py +model=mlplob +dataset=fi2010 hydra.job.chdir=False
python main.py +model=deeplob +dataset=fi2010 hydra.job.chdir=False  # cross-check vs B1
```

**Expected results** (TLOB, FI-2010, k=10, standard labels):
```
Accuracy:    66–68%
Weighted F1: 0.65–0.67
```

---

#### Task B3 — TransLOB

**Repo**: `github.com/zelinwang123/translob` (PyTorch reimplementation)

```bash
git clone https://github.com/zelinwang123/translob
# Edit data path in train.py
python train.py --data_path data/fi2010 --horizon 10 --seed 42
```

**Expected results** (TransLOB, FI-2010, k=10):
```
Accuracy:    64–66%
Weighted F1: 0.63–0.65
```
Note: TransLOB is typically the second-worst after DeepLOB in benchmark studies.
If TransLOB beats TLOB in your run: check TLOB config (may be using wrong label method).

---

#### Task B4 — LOBERT (Paper Numbers)

No public code. Record from Table 3 of arXiv:2511.12563.
Add footnote `†` to Table 3: "LOBERT numbers from Linna et al. 2025.
Not directly comparable — different dataset and preprocessing."

---

#### Task B5 — FlashLOB Short Context (Internal Ablation)

**Critical for paper**: Run your own FlashLOB at context=100 and context=500 on FI-2010.

This separates two effects:
- Effect A: better architecture (event messages vs. snapshots, TimeAwareRoPE)
- Effect B: longer context

If FlashLOB@100 > TLOB@500: Effect A is real, mention it.
If FlashLOB@100 ≈ TLOB@500: improvement is purely from context. The cleaner story.

---

#### Task B6 — LOBSTER AAPL Snapshot Conversion

**File**: `src/data/lobster_to_snapshot.py`

Convert LOBSTER message+book data to FI-2010-compatible snapshot format.
Run DeepLOB, TLOB, TransLOB on LOBSTER AAPL using these snapshots.
This gives you a same-dataset comparison (Table 1 footnote or supplementary table).

---

#### Task B7 — Sanity Checks

Run after every baseline:
```python
sanity_check_baseline(model_name, result):
    assert 58 < result['accuracy'] < 80
    assert 0.55 < result['f1_weighted'] < 0.80
    # Stable class should not completely dominate
    assert result['f1_stable'] < 0.70
    # Variance check
    if 'std_accuracy' in result:
        assert result['std_accuracy'] < 3.0
```

---

## 8. Analysis & Interpretability

### Task I1 — Attention Weight vs. Event Type

**File**: `src/analysis/attention_viz.py`
**When**: After Phase 0 training at context=2048

Compute mean attention weight matrix [4×4] for each (layer, head).
The expected finding: in later layers, trades (type 2 and 3) receive
disproportionately high attention regardless of the query event type.

**Result slot**:
```
Attention to trades (type 2+3) vs. all events — layer 5 average:
  Expected: >1.2× mean
  Actual:   _fill_
  H1a: SUPPORTED / NOT SUPPORTED
```

---

### Task I2 — Attention vs. Market Impact Correlation

**File**: `src/analysis/hawkes_corr.py`
**When**: After Phase 0 training, during Phase 1

For each query event in test set, extract the K events with highest total attention
(summed across heads and layers). For each of those events, compute the ex-post
price impact: |mid_price[t+60s] - mid_price[t]| / mid_price[t].

Compute Spearman rank correlation between attention rank and price impact.

```
H1b result slot:
  Spearman rank corr (attention → price impact): _fill_
  p-value: _fill_
  H1b: SUPPORTED (ρ > 0.3) / WEAKLY SUPPORTED (ρ > 0.1) / NOT SUPPORTED
```

---

### Task I3 — Hawkes Parameter Stability

**File**: `src/hawkes/validate.py::check_param_stability`
**When**: During Phase 1 (if multiple LOBSTER days available)

```
H3c result slot:
  CV of adjacency matrix entries > 0.1: _fill_
  Interpretation: STABLE (<20%) / UNSTABLE (>30%)
  If UNSTABLE: use adaptive per-day calibration and note as limitation
```

---

### Task I4 — Failure Mode Analysis

**File**: `src/analysis/failure_modes.py`
**When**: After Phase 2

Find the context lengths where accuracy DEGRADES (if any).
Hypothesize: at very long context (16384), dense attention has too much noise
from distant low-impact events. Sparse FlashLOB should not degrade.

Compare:
- FA2 dense at context=2048 vs. 4096 (where feasible)
- FlashLOB at context=4096 vs. 8192 vs. 16384

Plot: accuracy vs. context separately for high-volatility and low-volatility test windows.
Expected: degradation is worse in low-volatility windows (more noise, less signal).

---

## 9. Paper Writing Tasks

### Task W1 — Section 1: Introduction

**Draft trigger**: After Phase 0 results exist (even preliminary).
**Requires**: OOM context length from Table 1, vol MAE at context=100 and best context.

**Key numbers to insert** (fill from results):
```
"[...] standard FlashAttention 2 runs out of memory at context=_____ events 
 on an A100 GPU, consuming _____ MB."
"Volatility MAE improves from _____ at 100 events to _____ at _____ events (___% reduction)."
```

---

### Task W2 — Section 3: Method

**Draft trigger**: After Phase 1 implementation is complete.
**Requires**: Hawkes formula (fixed), kernel architecture (fixed).

**The one equation in the main paper**:
```
attention_bias(q, k) = A[τ_q, τ_k] · log(1 + v_k) · Σ_d (1/3) · e^{-δ_d · Δt_{qk}}
```
where τ = event type, v = volume, δ_d = decay rates, Δt = time elapsed.
This is the paper's methodological core. Every other equation goes in the appendix.

---

### Task W3 — Section 4: Experiments

**Draft trigger**: After all three tables exist (even with placeholder numbers).
**Rule**: Write one sentence per result, then fill in numbers.

Template:
```
"Direction accuracy plateaus at context=_____, while volatility MAE 
 continues improving to context=_____ (Table 1, Figure 1)."
"The Hawkes-derived bias improves volatility MAE by ___% over volume-threshold
 selection (Table 2, p=_____)."
"FlashLOB reduces peak VRAM from ___MB to ___MB at context=4096,
 enabling training contexts 8× beyond the FA2 limit (Table 4)."
```

---

### Task W4 — Section 5: Analysis

**Draft trigger**: After interpretability tasks I1–I4 complete.
**Two acceptable framings** depending on findings:

If H1a supported: "Consistent with market microstructure theory, trained attention
heads preferentially weight trade events (Figure 2)..."

If H1a not supported: "Contrary to our hypothesis, attention weights are approximately
uniform across event types. This suggests LOB transformers learn statistical regularities
beyond microstructure theory, motivating future interpretability work..."

---

### Task W5 — Figures

| Figure | Content | Source | Status |
|---|---|---|---|
| Fig 1 | Context scaling curve (accuracy + vol MAE vs. context, log x-axis) | Phase 0 | _fill_ |
| Fig 2 | Attention heatmap (query type × key type, layer 5) | Phase 0 analysis | _fill_ |
| Fig 3 | Memory curve (FA2 dense vs. FlashLOB vs. O(L²) reference) | Phase 2 kernel bench | _fill_ |

All figures: PDF vector format, 11pt font (matches LaTeX body), no rasterization.
Use matplotlib with `rcParams['font.size'] = 11`.

---

### Task W6 — LaTeX Tables

All tables auto-generated by `paper/generate_tables.py`.
Template files in `paper/templates/`.
Bold the best number per column with `\textbf{}`.
Mark ± with `\pm`.
Mark OOM with `\textsc{oom}`.

---

## 10. Risk Register

| Risk | Probability | Impact | Mitigation | Outcome |
|---|---|---|---|---|
| H0 false: no signal past 500 events | 20% | High | Pivot to empirical study paper | _fill_ |
| H0 false for direction AND volatility | 5% | Very High | Full pivot to negative result paper | _fill_ |
| H1a false: attention not on trades | 40% | Medium | Report as finding, still publishable | _fill_ |
| H1b false: no impact correlation | 40% | Low | Drop from paper; focus on H2/H4 | _fill_ |
| H3 false: Hawkes not better than P90 | 30% | Low | Use P90 as fallback; note in paper | _fill_ |
| H4a false: FA2 doesn't OOM | 15% | High | Try smaller GPU (RTX 4090 24GB) | _fill_ |
| Triton kernel numerically unstable | 25% | Medium | Use PyTorch sparse masking as fallback for workshop paper | _fill_ |
| TLOB repo broken / not reproducible | 10% | Medium | Use their reported paper numbers with † footnote | _fill_ |
| LOBSTER data insufficient | 5% | High | Supplement with ABIDES synthetic data | _fill_ |
| Phase 0+1 takes >8 weeks | 20% | Medium | Submit workshop paper with Phases 0+1 only (no kernel) | _fill_ |

---

## 11. Timeline

```
Week  1–2 │ D1–D5: Data pipeline complete. Unit tests pass.
Week  3   │ M1–M4: Baseline model. Training loop. Memory profiler.
Week  4–5 │ E0: Phase 0 experiments running (overnight sweeps).
           │ B1–B3: Baselines running in parallel on FI-2010.
Week  6   │ Phase 0 results in. Gate check. Attention analysis.
           │ Table 1 rows 1–5 populated. Figure 1 generated.
           │ Table 3 populated with baseline numbers.
           ├─ GATE 0 ─ proceed or pivot ─────────────────────────────
Week  7   │ H1: Hawkes calibration. QQ plot. Stability check.
Week  8   │ H2, M5: Bias module + mask factory. Unit tests.
Week  9   │ E1: Phase 1 ablation running.
           │ Table 2 populated. Gate check.
           ├─ GATE 1 ─ proceed or fallback ───────────────────────────
Week 10   │ K0: Reference implementation complete.
Week 11   │ K1: Triton Stage A. Correctness + speed test.
Week 12   │ K2: Stage B (sliding window). Speed test.
Week 13   │ K3: Stage C (hot events). Correctness test.
Week 14   │ K4: Stage D (Hawkes bias in kernel). Full correctness.
Week 15   │ K5: Benchmark all context lengths. Table 4.
           │ Table 1 rows 6–9 populated (4096, 8192, 16384).
           ├─ Paper 1 data complete ──────────────────────────────────
Week 16   │ W1–W4: Paper draft. All sections.
Week 17   │ I1–I4: Interpretability analysis. Figures 1–3 finalized.
Week 18   │ W5–W6: LaTeX tables. PDF figures. Full paper.
           │ Internal review. Fix errors.
Week 19   │ Submit to workshop.
           ├─ Paper 1 submitted ─────────────────────────────────────
Week 20+  │ Reviewer feedback. Begin Paper 2 (precision routing).
```

---

## 12. Decision Log

*Record every significant decision here. Date + rationale.*

| Date | Decision | Rationale | Alternatives rejected |
|---|---|---|---|
| _fill_ | Use context=2048 for Phase 1 ablation (not 4096) | Dense masking OOMs at 4096 before kernel; need comparable conditions | Could downsample batch |
| _fill_ | TimeAwareRoPE over integer RoPE | Events are irregularly spaced; integer RoPE ignores timing structure | Integer RoPE: ignores timing. Absolute PE: doesn't generalize. |
| _fill_ | 0.7/0.3 direction/volatility loss weighting | Volatility is scientifically more interesting for long-context hypothesis; direction gives training signal | 0.5/0.5: too much MSE noise early in training |
| _fill_ | Use decimal-precision FI-2010 normalization | Matches DeepLOB, TLOB published protocol; z-score gives different numbers | Z-score: incomparable to published numbers |
| _fill_ | Fit Hawkes with exponential kernel (not power-law) | Faster to calibrate; power-law available as extension for Paper 2 | Power-law: more accurate but 10× slower to fit |