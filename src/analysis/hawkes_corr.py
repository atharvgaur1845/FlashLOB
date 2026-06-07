"""Correlate learned attention weights with the Hawkes excitation prior.

If trained attention puts mass where Hawkes theory predicts excitation, this
supports the paper's central claim that the predictive signal concentrates on
Hawkes-predictable high-impact events.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


@torch.no_grad()
def attention_hawkes_correlation(model, hawkes_bias, dataloader, device,
                                 n_batches: int = 20) -> dict:
    """Correlation between per-(q,k) attention mass and Hawkes bias magnitude.

    Returns {'pearson':..., 'spearman':..., 'n':...} pooled over heads/batches.
    """
    model.eval()
    attn_vals, bias_vals = [], []
    for b, batch in enumerate(dataloader):
        if b >= n_batches:
            break
        events, _, _, meta = batch
        events = events.to(device)
        times = meta["times"].to(device)
        meta_d = {k: v.to(device) for k, v in meta.items()}
        _, _, attns = model(events, times, meta_d, return_attn=True)
        if attns[-1] is None:
            raise RuntimeError("need attention weights; use backend='standard'")
        bias = hawkes_bias({k: meta_d[k] for k in ("event_types", "log_sizes", "times")})
        w = attns[-1].mean(dim=1)                 # [B,L,L] mean over heads, last layer
        bias = bias.squeeze(1)                    # [B,L,L]
        # causal lower-triangle entries only
        L = w.shape[-1]
        tri = torch.tril(torch.ones(L, L, device=device), diagonal=-1).bool()
        attn_vals.append(w[:, tri].flatten().float().cpu().numpy())
        bias_vals.append(bias[:, tri].flatten().float().cpu().numpy())

    a = np.concatenate(attn_vals); h = np.concatenate(bias_vals)
    # subsample to keep correlation cheap
    if a.size > 200_000:
        idx = np.random.default_rng(0).choice(a.size, 200_000, replace=False)
        a, h = a[idx], h[idx]
    pear = float(pearsonr(a, h)[0]) if a.std() > 0 and h.std() > 0 else 0.0
    spear = float(spearmanr(a, h)[0]) if a.std() > 0 and h.std() > 0 else 0.0
    return {"pearson": pear, "spearman": spear, "n": int(a.size)}
