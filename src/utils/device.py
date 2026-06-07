"""Device + VRAM-budget helpers.

These make the codebase hybrid: the same config runs on an 8GB RTX 4060 (where it
auto-shrinks batch/context) and on an 80GB A100 (where it uses the full sweep).
Nothing about model *architecture* changes — only batch size and which context
lengths are attempted — so results stay comparable where memory allows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_total_gb() -> float:
    """Total VRAM of the current CUDA device, in GiB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / (1024 ** 3)


def gpu_free_gb() -> float:
    """Currently free VRAM in GiB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


@dataclass
class DeviceBudget:
    """Safe defaults derived from detected VRAM.

    Heuristics are deliberately conservative — better to run than to OOM. They can
    always be overridden explicitly in a config file.
    """

    total_gb: float
    max_batch: int
    # Largest context length we will *attempt* to train with a dense (FA2) model.
    max_dense_context: int
    # Largest context length we will attempt with the sparse FlashLOB kernel.
    max_sparse_context: int
    device: str

    def filter_contexts(self, contexts: list[int], sparse: bool = False) -> list[int]:
        """Drop context lengths that exceed the budget for this backend."""
        cap = self.max_sparse_context if sparse else self.max_dense_context
        return [c for c in contexts if c <= cap]


def auto_budget(d_model: int = 128, n_layers: int = 6) -> DeviceBudget:
    """Pick conservative batch/context caps from the detected GPU.

    The caps are tuned for d_model=128, 6 layers, fp16. Larger models scale the
    caps down proportionally. Tiers are coarse on purpose.
    """
    total = gpu_total_gb()
    scale = (128 * 6) / max(d_model * n_layers, 1)  # >1 => model smaller than ref

    if total == 0.0:  # CPU
        return DeviceBudget(0.0, max_batch=2, max_dense_context=256,
                            max_sparse_context=512, device="cpu")
    if total < 12:  # ~8GB laptop 4060 tier
        budget = DeviceBudget(total, max_batch=8, max_dense_context=2048,
                              max_sparse_context=8192, device="cuda")
    elif total < 32:  # ~24GB 4090 tier
        budget = DeviceBudget(total, max_batch=16, max_dense_context=4096,
                              max_sparse_context=16384, device="cuda")
    else:  # A100 80GB tier
        budget = DeviceBudget(total, max_batch=32, max_dense_context=8192,
                              max_sparse_context=16384, device="cuda")

    budget.max_batch = max(1, int(budget.max_batch * min(scale, 4)))
    return budget


def resolve(value, fallback):
    """Return ``fallback`` when value is None or the string 'auto', else value."""
    if value is None or value == "auto":
        return fallback
    return value


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
