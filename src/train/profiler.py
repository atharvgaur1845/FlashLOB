"""Peak-VRAM + wall-clock profiling helpers (raw CUDA APIs, no framework)."""

from __future__ import annotations

import time

import torch


class MemoryProfiler:
    """Context manager reporting peak CUDA memory (MB) during the block.

    Usage:
        with MemoryProfiler() as p:
            out = model(x)
        print(p.peak_mb)
    On CPU it reports 0.0 and never raises.
    """

    def __init__(self):
        self.peak_mb = 0.0
        self._cuda = torch.cuda.is_available()

    def __enter__(self):
        if self._cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        if self._cuda:
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return False


class Timer:
    """Wall-clock timer that synchronizes CUDA so GPU time is included."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        return False
