"""Attention analysis: do trained heads preferentially attend to certain event
types (e.g. trades)? Answers Phase -1/0 question 2 (does attention cluster?)."""

from __future__ import annotations

import numpy as np
import torch

N_TYPES = 4  # 0 limit, 1 cancel, 2 trade, 3 large_trade


@torch.no_grad()
def attention_by_event_type(model, dataloader, device, n_batches: int = 50) -> dict:
    """Mean attention weight from each query event-type to each key event-type.

    Returns {(layer_idx, head_idx): np.ndarray [4,4]} where entry [i,j] is the
    mean attention mass that type-i queries place on type-j keys (row-normalized).
    Requires the model's attention backend to return weights (use 'standard').
    """
    model.eval()
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], np.ndarray] = {}

    for b, batch in enumerate(dataloader):
        if b >= n_batches:
            break
        events, _, _, meta = batch
        events = events.to(device)
        times = meta["times"].to(device)
        etypes = meta["event_types"].to(device)
        meta_d = {k: v.to(device) for k, v in meta.items()}
        _, _, attns = model(events, times, meta_d, return_attn=True)
        if attns[0] is None:
            raise RuntimeError("model returned no attention weights; use backend='standard'")

        B, H = attns[0].shape[0], attns[0].shape[1]
        for li, w in enumerate(attns):              # w: [B,H,L,L]
            w = w.float().cpu().numpy()
            et = etypes.cpu().numpy()               # [B,L]
            for h in range(H):
                key = (li, h)
                if key not in sums:
                    sums[key] = np.zeros((N_TYPES, N_TYPES))
                    counts[key] = np.zeros((N_TYPES, N_TYPES))
                for bi in range(B):
                    tq = et[bi]                     # [L]
                    wh = w[bi, h]                   # [L,L]
                    for qi in range(N_TYPES):
                        qmask = tq == qi
                        if not qmask.any():
                            continue
                        rows = wh[qmask]            # [nq, L]
                        for kj in range(N_TYPES):
                            kmask = tq == kj
                            if kmask.any():
                                sums[key][qi, kj] += rows[:, kmask].sum()
                                counts[key][qi, kj] += rows.shape[0]

    out = {}
    for key in sums:
        m = sums[key] / np.maximum(counts[key], 1e-9)
        row = m.sum(axis=1, keepdims=True)
        out[key] = m / np.maximum(row, 1e-9)        # row-normalized
    return out


def cluster_score(attn_by_type: dict) -> float:
    """Single scalar: how far the average attention is from uniform-over-types.

    For each head we take, per query row, the max attended-type mass; average over
    heads/types and subtract the uniform baseline (1/N_TYPES). >0 means clustering.
    """
    if not attn_by_type:
        return 0.0
    peaks = []
    for m in attn_by_type.values():
        peaks.append(m.max(axis=1).mean())          # avg over query types of row-max
    return float(np.mean(peaks) - 1.0 / N_TYPES)


def plot_attention_heatmap(attn_by_type: dict, save_path: str, layer: int | None = None):
    """Heatmap of mean attention (query type x key type), averaged across heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({k[0] for k in attn_by_type})
    if layer is None:
        layer = layers[-1]
    mats = [m for (li, _h), m in attn_by_type.items() if li == layer]
    avg = np.mean(mats, axis=0)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(avg, cmap="plasma", aspect="auto")
    labels = ["limit", "cancel", "trade", "lg_trade"]
    ax.set_xticks(range(N_TYPES)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(N_TYPES)); ax.set_yticklabels(labels)
    ax.set_xlabel("key event type"); ax.set_ylabel("query event type")
    ax.set_title(f"Mean attention by event type (layer {layer})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
