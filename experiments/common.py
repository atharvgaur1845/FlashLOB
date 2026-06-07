"""Shared helpers for the training experiments (Phase -1, Phase 0, ablation).

Builds tokenized data + labels once, then constructs context-windowed datasets and
runs a single (context_length, seed) training, returning a metrics dict.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LOBEventDataset, collate
from src.data.labels import mid_price_direction, realized_volatility
from src.data.lobster import LOBSTERLoader
from src.model.transformer import LOBTransformer
from src.train.trainer import Trainer
from src.utils.device import auto_budget, get_device, resolve, seed_everything


def load_tokens(cfg: dict):
    """Load + tokenize LOBSTER data per cfg. Returns (tokens, ldir, lvol, htypes)."""
    d = cfg["data"]
    loader = LOBSTERLoader()
    df = loader.load(d["lobster_message"], d["lobster_book"])
    tokens = loader.tokenize(df)
    ldir = mid_price_direction(df, horizon_events=d["horizon_events"])
    lvol = realized_volatility(df, window_seconds=d["vol_window_seconds"])
    htypes = torch.from_numpy(loader.hawkes_types(df))
    return tokens, ldir, lvol, htypes


def chrono_split(n: int, val=0.15, test=0.15):
    """Chronological index ranges (no shuffling) to avoid leakage."""
    tr = int(n * (1 - val - test))
    va = int(n * (1 - test))
    return (0, tr), (tr, va), (va, n)


def make_loaders(tokens, ldir, lvol, htypes, context_length, stride, batch_size,
                 num_workers=2):
    n = tokens.size(0)
    (a, b), (c, d), (e, f) = chrono_split(n)
    def ds(lo, hi):
        return LOBEventDataset(tokens[lo:hi], ldir[lo:hi], lvol[lo:hi],
                               context_length=context_length, stride=stride,
                               hawkes_types=htypes[lo:hi])
    common = dict(batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    return (DataLoader(ds(a, b), shuffle=True, **common),
            DataLoader(ds(c, d), shuffle=False, **common),
            DataLoader(ds(e, f), shuffle=False, **common))


def build_model(cfg: dict, context_length: int):
    m = cfg["model"]
    attn_spec = None
    hawkes_bias = None
    hawkes_params = None
    if m["attention_backend"] == "flashlob":
        # Sparse sliding-window + Hawkes-top-k + Hawkes bias (PyTorch masking).
        import os
        from src.hawkes.bias import HawkesBias
        from src.hawkes.calibrate import HawkesCalibrator
        attn_spec = m.get("attn_spec", {"window": 512, "global": "hawkes_topk", "n_hot": 64})
        hp = m.get("hawkes_params", "data/processed/hawkes_params.json")
        if os.path.exists(hp):
            hawkes_params = HawkesCalibrator.load(hp)
            hawkes_bias = HawkesBias(hawkes_params)
    return LOBTransformer(
        d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"],
        d_ff=m["d_ff"], dropout=m["dropout"], context_length=context_length,
        attention_backend=m["attention_backend"],
        attn_spec=attn_spec, hawkes_bias=hawkes_bias, hawkes_params=hawkes_params,
    )


def run_single(cfg, tokens, ldir, lvol, htypes, context_length, seed, tag,
               epochs=None):
    """Train one (context, seed); return metrics + resource numbers."""
    seed_everything(seed)
    device = get_device()
    budget = auto_budget(cfg["model"]["d_model"], cfg["model"]["n_layers"])
    batch_size = resolve(cfg["train"]["batch_size"], budget.max_batch)
    epochs = epochs or cfg["train"]["epochs"]

    tr, va, te = make_loaders(tokens, ldir, lvol, htypes, context_length,
                              cfg["data"]["stride"], batch_size,
                              cfg["runtime"].get("num_workers", 2))

    model = build_model(cfg, context_length)
    csv_log = os.path.join("results", "logs", f"{tag}.csv")
    trainer = Trainer(
        model, device,
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        warmup_steps=cfg["train"]["warmup_steps"], grad_clip=cfg["train"]["grad_clip"],
        loss_dir_weight=cfg["train"]["loss_dir_weight"],
        loss_vol_weight=cfg["train"]["loss_vol_weight"],
        amp=cfg["train"]["amp"], ckpt_dir=cfg["train"]["ckpt_dir"],
        log_every=cfg["train"]["log_every"], tag=tag, csv_log=csv_log,
    )
    trainer.fit(tr, va, epochs=epochs, patience=cfg["train"].get("patience", 5))
    metrics = trainer.evaluate(te)
    metrics.update({
        "context_length": context_length, "seed": seed,
        "peak_vram_mb": round(trainer.peak_train_mb, 1),
        "tokens_per_second": round(trainer.tokens_per_sec, 1),
        "n_params": model.num_params(),
        "batch_size": batch_size,
    })
    return metrics
