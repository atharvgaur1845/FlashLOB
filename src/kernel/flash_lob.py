"""Triton FlashLOB kernel — causal sparse attention with three streams merged via
online softmax:

    (A) tiled FlashAttention core (fp16 in, fp32 accumulate)
    (B) causal sliding window of WINDOW keys
    (C) global hot-event stream (per-sequence top-k Hawkes-excitation events),
        de-duplicated against the window so each key is counted once
    (D) Hawkes bias  A[tq,tk] * log_size_k * exp(-D[tq,tk]*dt)  added to logits

The Python entry point ``flashlob_attention`` matches src.kernel.reference
.flashlob_reference (include_type=False) to FP16 tolerance.

The four stages from the design doc are realized inside one kernel: the window
loop is stage A/B, the hot loop is stage C, and the bias computed in both loops is
stage D. Comments mark each.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flashlob_fwd(
        Q, K, V, Out,
        Types, Times, LogSz,           # [B, L]
        HotIdx,                        # [B, N_HOT] int32
        Atbl, Dtbl,                    # [NT*NT] fp32
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_mb, stride_ml,          # meta strides (B, L)
        stride_hb, stride_hn,          # hot idx strides
        B, H, L,
        WINDOW: tl.constexpr,
        N_HOT: tl.constexpr,
        NT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CLAMP: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        m_valid = offs_m < L

        # base pointers for this (b, h)
        q_ptr = Q + b * stride_qb + h * stride_qh
        k_ptr = K + b * stride_kb + h * stride_kh
        v_ptr = V + b * stride_vb + h * stride_vh
        o_ptr = Out + b * stride_ob + h * stride_oh
        meta_b = b * stride_mb

        # load Q block [BLOCK_M, HEAD_DIM]
        q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None], other=0.0).to(tl.float32)
        q_type = tl.load(Types + meta_b + offs_m * stride_ml, mask=m_valid, other=0).to(tl.int32)
        q_time = tl.load(Times + meta_b + offs_m * stride_ml, mask=m_valid, other=0.0)

        scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)
        NEG = -1e30  # finite sentinel: avoids nan in exp(m_i - m_new) for empty rows
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEG
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q_min = pid_m * BLOCK_M
        q_max = q_min + BLOCK_M - 1

        # =============== (A/B) causal sliding-window pass ===============
        lo = q_min - WINDOW + 1
        if lo < 0:
            lo = 0
        lo = (lo // BLOCK_N) * BLOCK_N
        for start_n in range(lo, q_max + 1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < L
            k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=n_valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                        mask=n_valid[:, None], other=0.0).to(tl.float32)
            k_type = tl.load(Types + meta_b + offs_n * stride_ml, mask=n_valid, other=0).to(tl.int32)
            k_time = tl.load(Times + meta_b + offs_n * stride_ml, mask=n_valid, other=0.0)
            k_lsz = tl.load(LogSz + meta_b + offs_n * stride_ml, mask=n_valid, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale            # [BLOCK_M, BLOCK_N]

            # (D) Hawkes bias
            dt = q_time[:, None] - k_time[None, :]
            tbl_idx = q_type[:, None] * NT + k_type[None, :]
            a_val = tl.load(Atbl + tbl_idx)
            d_val = tl.load(Dtbl + tbl_idx)
            bias = a_val * k_lsz[None, :] * tl.exp(-d_val * dt)
            bias = tl.where(dt > 0, bias, 0.0)
            bias = tl.minimum(tl.maximum(bias, -CLAMP), CLAMP)
            qk = qk + bias

            diff = offs_m[:, None] - offs_n[None, :]
            keep = (diff > 0) & (diff < WINDOW) & n_valid[None, :] & m_valid[:, None]
            qk = tl.where(keep, qk, NEG)

            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(keep, p, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # =============== (C) global hot-event pass ===============
        for start_h in range(0, N_HOT, BLOCK_N):
            offs_h = start_h + tl.arange(0, BLOCK_N)
            h_valid = offs_h < N_HOT
            hidx = tl.load(HotIdx + b * stride_hb + offs_h * stride_hn,
                           mask=h_valid, other=0).to(tl.int32)
            in_range = hidx < L
            k = tl.load(k_ptr + hidx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=(h_valid & in_range)[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr + hidx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                        mask=(h_valid & in_range)[:, None], other=0.0).to(tl.float32)
            k_type = tl.load(Types + meta_b + hidx * stride_ml, mask=h_valid & in_range, other=0).to(tl.int32)
            k_time = tl.load(Times + meta_b + hidx * stride_ml, mask=h_valid & in_range, other=0.0)
            k_lsz = tl.load(LogSz + meta_b + hidx * stride_ml, mask=h_valid & in_range, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            dt = q_time[:, None] - k_time[None, :]
            tbl_idx = q_type[:, None] * NT + k_type[None, :]
            a_val = tl.load(Atbl + tbl_idx)
            d_val = tl.load(Dtbl + tbl_idx)
            bias = a_val * k_lsz[None, :] * tl.exp(-d_val * dt)
            bias = tl.where(dt > 0, bias, 0.0)
            bias = tl.minimum(tl.maximum(bias, -CLAMP), CLAMP)
            qk = qk + bias

            diff = offs_m[:, None] - hidx[None, :]
            in_window = (diff > 0) & (diff < WINDOW)        # already counted locally
            keep = (diff > 0) & h_valid[None, :] & in_range[None, :] & (~in_window) & m_valid[:, None]
            qk = tl.where(keep, qk, NEG)

            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(keep, p, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # finalize (rows with no admitted key -> l_i==0 -> output 0)
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        tl.store(o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc.to(Out.dtype.element_ty), mask=m_valid[:, None])


def flashlob_attention(
    Q, K, V, event_types, log_sizes, times, hawkes_params,
    window_size: int = 256, n_global_hot: int = 64,
    block_m: int = 64, block_n: int = 64,
):
    """Triton FlashLOB forward. Q,K,V: [B,H,L,d]. Returns [B,H,L,d].

    Falls back to the pure-PyTorch reference if Triton/CUDA is unavailable.
    """
    B, H, L, d = Q.shape
    if not (HAS_TRITON and Q.is_cuda):
        from src.kernel.reference import flashlob_reference
        return flashlob_reference(Q, K, V, event_types, log_sizes, times,
                                  hawkes_params, window_size, n_global_hot)

    from src.hawkes.bias import build_hot_index
    nt = len(hawkes_params["baseline"])
    A = torch.tensor(hawkes_params["adjacency"], device=Q.device, dtype=torch.float32).reshape(-1).contiguous()
    D = torch.tensor(hawkes_params["decays"], device=Q.device, dtype=torch.float32).reshape(-1).contiguous()
    hot_idx = build_hot_index(log_sizes, event_types, hawkes_params, times,
                              n_hot=n_global_hot).to(torch.int32).contiguous()

    types = event_types.to(torch.int32).contiguous()
    times = times.to(torch.float32).contiguous()
    logsz = log_sizes.to(torch.float32).contiguous()
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
    Out = torch.empty_like(Q)

    grid = (triton.cdiv(L, block_m), B * H)
    _flashlob_fwd[grid](
        Q, K, V, Out, types, times, logsz, hot_idx, A, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        types.stride(0), types.stride(1),
        hot_idx.stride(0), hot_idx.stride(1),
        B, H, L,
        WINDOW=window_size, N_HOT=hot_idx.shape[1], NT=nt,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d, CLAMP=5.0,
    )
    return Out
