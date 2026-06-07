"""Raw PyTorch training loop. No Lightning, no HF Trainer — we need exact memory
and timing numbers.

Features (per spec):
  * AdamW(lr=3e-4, wd=0.01), cosine schedule with 500-step warmup
  * fp16 autocast + GradScaler
  * grad clipping max_norm=1.0
  * joint loss 0.7 * CE(direction) + 0.3 * MSE(volatility)
  * peak VRAM (MB) + tokens/sec logged every ``log_every`` steps
  * per-epoch checkpoints {tag}_{epoch}.pt
  * wandb if WANDB_API_KEY set, else CSV log

Volatility targets are tiny (~1e-4); we standardize them with train-set stats so
the MSE term contributes meaningfully, and invert the standardization when
reporting MAE in original units.
"""

from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.metrics import compute_all
from src.train.profiler import MemoryProfiler, Timer


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(prog, 1.0)))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        loss_dir_weight: float = 0.7,
        loss_vol_weight: float = 0.3,
        amp: bool = True,
        ckpt_dir: str = "results/checkpoints",
        log_every: int = 100,
        tag: str = "run",
        csv_log: str | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.w_dir = loss_dir_weight
        self.w_vol = loss_vol_weight
        self.amp = amp and device.type == "cuda"
        self.ckpt_dir = ckpt_dir
        self.log_every = log_every
        self.tag = tag

        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.vol_mean = 0.0
        self.vol_std = 1.0
        self.peak_train_mb = 0.0
        self.tokens_per_sec = 0.0

        os.makedirs(ckpt_dir, exist_ok=True)
        self._wandb = self._init_wandb()
        self.csv_log = csv_log
        if csv_log:
            os.makedirs(os.path.dirname(csv_log) or ".", exist_ok=True)
            self._csv_init = False

    # ---------------------------------------------------------------- logging
    def _init_wandb(self):
        if not os.environ.get("WANDB_API_KEY"):
            return None
        try:
            import wandb
            wandb.init(project="flashlob", name=self.tag, reinit=True)
            return wandb
        except Exception:
            return None

    def _log(self, row: dict):
        if self._wandb:
            self._wandb.log(row)
        if self.csv_log:
            write_header = not self._csv_init
            with open(self.csv_log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                    self._csv_init = True
                w.writerow(row)

    # ------------------------------------------------------------ vol scaling
    def fit_vol_stats(self, loader):
        vals = []
        for _, _, yv, _ in loader:
            vals.append(yv)
        v = torch.cat(vals)
        self.vol_mean = float(v.mean())
        self.vol_std = float(v.std()) or 1.0

    def _norm_vol(self, v):
        return (v - self.vol_mean) / self.vol_std

    def _denorm_vol(self, v):
        return v * self.vol_std + self.vol_mean

    # -------------------------------------------------------------- one batch
    def _step(self, batch):
        events, y_dir, y_vol, meta = batch
        events = events.to(self.device, non_blocking=True)
        times = meta["times"].to(self.device, non_blocking=True)
        y_dir = y_dir.to(self.device)
        y_vol = self._norm_vol(y_vol.to(self.device))
        meta = {k: v.to(self.device) for k, v in meta.items()}

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits_dir, logits_vol = self.model(events, times, meta)
            loss_dir = F.cross_entropy(logits_dir, y_dir)
            loss_vol = F.mse_loss(logits_vol.squeeze(-1), y_vol)
            loss = self.w_dir * loss_dir + self.w_vol * loss_vol
        return loss, loss_dir.detach(), loss_vol.detach(), events.numel() // 8

    # -------------------------------------------------------------------- fit
    def fit(self, train_loader, val_loader, epochs=20, patience=5):
        self.fit_vol_stats(train_loader)
        total_steps = epochs * max(1, len(train_loader))
        step = 0
        best_metric = float("inf")  # track val volatility MAE
        best_state = None
        bad = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_tokens = 0
            with Timer() as etimer:
                for batch in train_loader:
                    lr = cosine_warmup(step, self.warmup_steps, total_steps, self.lr)
                    for g in self.opt.param_groups:
                        g["lr"] = lr

                    self.opt.zero_grad(set_to_none=True)
                    if step % self.log_every == 0:
                        with MemoryProfiler() as mp, Timer() as t:
                            loss, ld, lv, ntok = self._step(batch)
                            self.scaler.scale(loss).backward()
                        self.peak_train_mb = max(self.peak_train_mb, mp.peak_mb)
                        self.tokens_per_sec = ntok / t.elapsed if t.elapsed > 0 else 0.0
                        self._log({"step": step, "epoch": epoch, "lr": lr,
                                   "loss": float(loss.detach()), "loss_dir": float(ld),
                                   "loss_vol": float(lv), "peak_mb": self.peak_train_mb,
                                   "tok_s": self.tokens_per_sec})
                    else:
                        loss, ld, lv, ntok = self._step(batch)
                        self.scaler.scale(loss).backward()

                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    step += 1
                    epoch_tokens += ntok

            self.epoch_time = etimer.elapsed
            val = self.evaluate(val_loader)
            self._log({"epoch": epoch, "val_accuracy": val["accuracy"],
                       "val_f1_weighted": val["f1_weighted"],
                       "val_volatility_mae": val["volatility_mae"],
                       "epoch_time_s": etimer.elapsed})

            torch.save({"model": self.model.state_dict(), "epoch": epoch,
                        "vol_mean": self.vol_mean, "vol_std": self.vol_std},
                       os.path.join(self.ckpt_dir, f"{self.tag}_{epoch}.pt"))

            if val["volatility_mae"] < best_metric - 1e-9:
                best_metric = val["volatility_mae"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_ld, all_pv, all_yd, all_yv = [], [], [], []
        for batch in loader:
            events, y_dir, y_vol, meta = batch
            events = events.to(self.device)
            times = meta["times"].to(self.device)
            meta = {k: v.to(self.device) for k, v in meta.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits_dir, logits_vol = self.model(events, times, meta)
            all_ld.append(logits_dir.float().cpu())
            all_pv.append(self._denorm_vol(logits_vol.float().cpu()))
            all_yd.append(y_dir)
            all_yv.append(y_vol)
        return compute_all(torch.cat(all_ld), torch.cat(all_pv),
                           torch.cat(all_yd), torch.cat(all_yv))
