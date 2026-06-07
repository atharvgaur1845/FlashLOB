"""Throughput + peak-VRAM benchmark: dense FlashAttention-2 vs FlashLOB sparse.

Records OOM where it happens (expected on the 8GB 4060 well before an A100), which
is itself a headline result for the paper (Figure 3 / Table 2).
"""

from __future__ import annotations

import gc
import json
import os

import torch
import torch.nn.functional as F

from src.hawkes.calibrate import HawkesCalibrator
from src.kernel.flash_lob import flashlob_attention
from src.train.profiler import MemoryProfiler, Timer


def _rand_inputs(B, H, L, d, device):
    g = torch.Generator(device=device).manual_seed(0)
    Q = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    K = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    V = torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=g)
    et = torch.randint(0, 4, (B, L), device=device, generator=g)
    ls = torch.rand(B, L, device=device, generator=g) * 5
    tm = torch.sort(torch.rand(B, L, device=device, generator=g), dim=1).values
    return Q, K, V, et, ls, tm


def _bench_fn(fn, n_warmup, n_bench):
    """Return (tokens_per_sec_factor placeholder, ms_per_call, peak_mb) or OOM."""
    try:
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        with MemoryProfiler() as mp, Timer() as t:
            for _ in range(n_bench):
                fn()
        return t.elapsed / n_bench * 1000.0, mp.peak_mb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def run_benchmark(context_lengths=(512, 1024, 2048, 4096, 8192, 16384),
                  d_model=128, n_heads=8, batch_size=4,
                  window=256, n_hot=64, n_warmup=10, n_bench=50,
                  hawkes_params=None, out_path="results/phase2/kernel_benchmark.json"):
    device = torch.device("cuda")
    d = d_model // n_heads
    if hawkes_params is None:
        hp = "data/processed/hawkes_params.json"
        hawkes_params = HawkesCalibrator.load(hp) if os.path.exists(hp) else \
            {"baseline": [0.1] * 4, "adjacency": [[0.1] * 4] * 4, "decays": [[1.0] * 4] * 4}

    rows = []
    for L in context_lengths:
        Q, K, V, et, ls, tm = _rand_inputs(batch_size, n_heads, L, d, device)
        tokens = batch_size * L

        scale = 1.0 / (d ** 0.5)
        A = torch.tensor(hawkes_params["adjacency"], device=device, dtype=torch.float32)
        D = torch.tensor(hawkes_params["decays"], device=device, dtype=torch.float32)
        nt = A.shape[0]

        def dense_fa2():
            # memory-efficient FlashAttention-2: O(L) memory, the speed baseline.
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        def dense_bias():
            # naive domain-specific attention: materializes [B,H,L,L] scores + the
            # [B,1,L,L] Hawkes bias -> O(L^2) memory. This is what OOMs.
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
            dt = (tm[:, :, None] - tm[:, None, :]).clamp(min=0)
            etc = et.clamp(0, nt - 1)
            flat = (etc[:, :, None] * nt + etc[:, None, :]).reshape(B := Q.shape[0], -1)
            a = A.reshape(-1)[flat].reshape(Q.shape[0], L, L)
            dd = D.reshape(-1)[flat].reshape(Q.shape[0], L, L)
            bias = (a * ls[:, None, :] * torch.exp(-dd * dt)).clamp(-5, 5)
            scores = scores + bias.unsqueeze(1)
            scores = scores.masked_fill(
                ~torch.ones(L, L, device=device, dtype=torch.bool).tril(), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            return torch.matmul(w, V.float())

        def sparse():
            return flashlob_attention(Q, K, V, et, ls, tm, hawkes_params, window, n_hot)

        fa_ms, fa_mb = _bench_fn(dense_fa2, n_warmup, n_bench)
        db_ms, db_mb = _bench_fn(dense_bias, max(1, n_warmup // 2), max(2, n_bench // 5))
        s_ms, s_mb = _bench_fn(sparse, n_warmup, n_bench)
        del Q, K, V, et, ls, tm
        gc.collect(); torch.cuda.empty_cache()

        row = {
            "context": L,
            "fa2_ms": round(fa_ms, 3) if fa_ms else "OOM",
            "fa2_vram_mb": round(fa_mb, 1) if fa_mb else "OOM",
            "dense_bias_ms": round(db_ms, 3) if db_ms else "OOM",
            "dense_bias_vram_mb": round(db_mb, 1) if db_mb else "OOM",
            "flashlob_ms": round(s_ms, 3) if s_ms else "OOM",
            "flashlob_vram_mb": round(s_mb, 1) if s_mb else "OOM",
            "flashlob_tok_s": round(tokens / (s_ms / 1000), 1) if s_ms else "OOM",
        }
        if db_mb and s_mb:
            row["vram_reduction_x_vs_densebias"] = round(db_mb / s_mb, 2)
        if fa_ms and s_ms:
            row["speedup_x_vs_fa2"] = round(fa_ms / s_ms, 2)
        rows.append(row)
        print(f"L={L}: FA2 {row['fa2_vram_mb']}MB/{row['fa2_ms']}ms | "
              f"dense+bias {row['dense_bias_vram_mb']}MB/{row['dense_bias_ms']}ms | "
              f"FlashLOB {row['flashlob_vram_mb']}MB/{row['flashlob_ms']}ms")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows
