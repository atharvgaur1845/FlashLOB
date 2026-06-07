"""Tiny YAML config loader with base-merging and dotted overrides.

Configs live in experiments/configs/. A phase config may set ``_base_: base.yaml``
to inherit and override. Keeps all sweep knobs in one place so the same code runs
unchanged across hardware tiers.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "configs")


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config, resolving a single ``_base_`` include."""
    if not os.path.isabs(path):
        path = os.path.join(CONFIG_DIR, path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        base = load_config(base_name)
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` CLI overrides (values parsed as YAML scalars)."""
    cfg = deepcopy(cfg)
    for ov in overrides:
        key, _, raw = ov.partition("=")
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
