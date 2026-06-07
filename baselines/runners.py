"""Unified baseline runners under the locked evaluation protocol.

Four models (per user): DeepLOB, TLOB (snapshot-based) and Dense-transformer,
FlashLOB (our event-based model). Each runner exposes .run(seed, horizon) and
returns a metrics dict in a common schema so Table 3 is apples-to-apples.

Snapshot baselines train direction-only (classification). The event models reuse
the joint-task pipeline but Table 3 reports their direction metrics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baselines.models import DeepLOB, TLOB, count_params
from src.train.metrics import compute_all
from src.utils.device import auto_budget, get_device, seed_everything


# --------------------------------------------------------------------------- data
class SnapshotWindowDataset(Dataset):
    """Length-L windows of snapshots [40] ending at t; label is direction at t."""

    def __init__(self, snapshots: torch.Tensor, labels: torch.Tensor,
                 context_length: int, stride: int = 10):
        self.x = snapshots.float()
        self.y = labels.long()
        self.L = context_length
        self.ends = list(range(context_length - 1, len(snapshots), stride))

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        e = self.ends[i]
        return self.x[e - self.L + 1:e + 1], self.y[e]


def _snapshot_loaders(snaps, labels, ctx, stride, batch, val=0.15, test=0.15):
    n = len(snaps)
    a = int(n * (1 - val - test)); b = int(n * (1 - test))
    def mk(lo, hi, sh):
        ds = SnapshotWindowDataset(snaps[lo:hi], labels[lo:hi], ctx, stride)
        return DataLoader(ds, batch_size=batch, shuffle=sh)
    return mk(0, a, True), mk(a, b, False), mk(b, n, False)


def load_snapshots(cfg: dict, horizon: int):
    """Return (snapshots [N,40], labels [N]) for the configured dataset."""
    dataset = cfg["data"]["dataset"]
    if dataset == "fi2010":
        from src.data.fi2010 import load_fi2010
        feats, labels = load_fi2010(cfg["data"]["fi2010_dir"], "train", horizon)
        return feats, labels
    # LOBSTER -> snapshots
    from src.data.lobster import LOBSTERLoader
    from src.data.lobster_to_snapshot import compute_fi2010_labels, messages_to_snapshots
    loader = LOBSTERLoader()
    df = loader.load(cfg["data"]["lobster_message"], cfg["data"]["lobster_book"])
    snaps = messages_to_snapshots(merged_df=df, stride=1)
    labels = compute_fi2010_labels(df["mid_price"].to_numpy(), horizon=horizon, stride=1)
    return torch.from_numpy(snaps), labels


# ----------------------------------------------------------------------- training
def _train_classifier(model, tr, va, te, device, epochs, lr=1e-3, patience=5,
                      amp=True):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best, best_state, bad = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        acc = _eval_acc(model, va, device)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return _eval_full(model, te, device)


@torch.no_grad()
def _eval_acc(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _eval_full(model, loader, device):
    model.eval(); logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu()); ys.append(y)
    logits = torch.cat(logits); ys = torch.cat(ys)
    dummy = torch.zeros(len(ys))
    m = compute_all(logits, dummy, ys, dummy)
    return {k: m[k] for k in ("accuracy", "f1_weighted", "f1_down", "f1_stable", "f1_up")}


# ------------------------------------------------------------------------ runners
class _SnapshotRunner:
    name = "snapshot"
    def __init__(self, cfg):
        self.cfg = cfg
    def _build(self, ctx):
        raise NotImplementedError
    def run(self, seed: int, horizon: int = 10) -> dict:
        seed_everything(seed)
        device = get_device()
        budget = auto_budget()
        cfg = self.cfg
        snaps, labels = load_snapshots(cfg, horizon)
        ctx = min(cfg.get("baseline_context", 100), len(snaps) - 1)
        batch = budget.max_batch * 4
        tr, va, te = _snapshot_loaders(snaps, labels, ctx,
                                       cfg["data"].get("baseline_stride", 10), batch)
        model = self._build(ctx)
        metrics = _train_classifier(model, tr, va, te, device,
                                    epochs=cfg["train"]["epochs"])
        metrics.update({"model": self.name, "seed": seed, "horizon": horizon,
                        "context_length": ctx, "n_params": count_params(model)})
        return metrics


class DeepLOBRunner(_SnapshotRunner):
    name = "DeepLOB"
    def _build(self, ctx):
        return DeepLOB()


class TLOBRunner(_SnapshotRunner):
    name = "TLOB"
    def _build(self, ctx):
        return TLOB(seq_len=ctx)


class _EventRunner:
    """Dense-transformer and FlashLOB share the event pipeline (common.run_single)."""
    name = "event"
    backend = "flash2"
    attn_spec = None
    def __init__(self, cfg):
        self.cfg = cfg
    def run(self, seed: int, horizon: int = 10) -> dict:
        from experiments.common import load_tokens, run_single
        cfg = dict(self.cfg)
        cfg["model"] = dict(cfg["model"], attention_backend=self.backend)
        cfg["data"] = dict(cfg["data"], horizon_events=horizon)
        toks, ldir, lvol, htypes = load_tokens(cfg)
        ctx = cfg.get("baseline_context", 100)
        m = run_single(cfg, toks, ldir, lvol, htypes, ctx, seed,
                       tag=f"{self.name}_s{seed}_h{horizon}")
        m.update({"model": self.name, "horizon": horizon})
        return m


class DenseTransformerRunner(_EventRunner):
    name = "Dense"
    backend = "flash2"


class FlashLOBRunner(_EventRunner):
    name = "FlashLOB"
    backend = "flashlob"
    def run(self, seed: int, horizon: int = 10) -> dict:
        # FlashLOB uses sparse attention; set a default attn_spec via config
        self.cfg.setdefault("model", {})
        return super().run(seed, horizon)


RUNNERS = {
    "deeplob": DeepLOBRunner,
    "tlob": TLOBRunner,
    "dense": DenseTransformerRunner,
    "flashlob": FlashLOBRunner,
}


def get_runner(name: str, cfg: dict):
    return RUNNERS[name](cfg)
