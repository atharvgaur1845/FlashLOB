"""Data pipeline tests on a tiny hand-built synthetic fixture."""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import LOBEventDataset, collate  # noqa: E402
from src.data.labels import mid_price_direction, realized_volatility  # noqa: E402
from src.data.lobster import LOBSTERLoader  # noqa: E402


def _fixture(n=200, n_levels=10):
    rng = np.random.default_rng(0)
    t = 34200 + np.cumsum(rng.exponential(0.5, n))
    types = rng.choice([1, 3, 4], n)
    sizes = rng.integers(1, 500, n)
    mid = 1_000_000 + np.cumsum(rng.normal(0, 30, n)).astype(int)
    direction = rng.choice([1, -1], n)
    price = mid + direction * rng.integers(0, 5, n) * 100
    msg = pd.DataFrame({"time": t, "type": types, "order_id": np.arange(n),
                        "size": sizes, "price": price, "direction": direction})
    book = {}
    for lvl in range(1, n_levels + 1):
        book[f"ask_px_{lvl}"] = mid + lvl * 100
        book[f"ask_sz_{lvl}"] = rng.integers(50, 500, n)
        book[f"bid_px_{lvl}"] = mid - lvl * 100
        book[f"bid_sz_{lvl}"] = rng.integers(50, 500, n)
    return msg, pd.DataFrame(book)


def test_tokenize_shapes_and_dtype():
    msg, book = _fixture()
    loader = LOBSTERLoader()
    df = loader.load_from_frames(msg, book)
    tok = loader.tokenize(df)
    assert tok.shape == (len(df), 8)
    assert tok.dtype == torch.float32
    assert torch.isfinite(tok).all()
    # time normalized to [0,1]; vol imbalance in [-1,1]
    assert tok[:, 0].min() >= 0 and tok[:, 0].max() <= 1
    assert tok[:, 6].min() >= -1 and tok[:, 6].max() <= 1


def test_labels_classes_and_vol():
    msg, book = _fixture()
    loader = LOBSTERLoader()
    df = loader.load_from_frames(msg, book)
    ld = mid_price_direction(df, horizon_events=5)
    lv = realized_volatility(df, window_seconds=10.0)
    assert ld.shape == (len(df),)
    assert set(ld.tolist()) <= {0, 1, 2}
    assert lv.shape == (len(df),)
    assert (lv >= 0).all()


def test_dataset_windowing_causal():
    msg, book = _fixture()
    loader = LOBSTERLoader()
    df = loader.load_from_frames(msg, book)
    tok = loader.tokenize(df)
    ld = mid_price_direction(df, 5)
    lv = realized_volatility(df, 10.0)
    ds = LOBEventDataset(tok, ld, lv, context_length=20, stride=5)
    ev, yd, yv, meta = ds[0]
    assert ev.shape == (20, 8)
    # times within a window must be non-decreasing (causal ordering)
    assert (meta["times"][1:] - meta["times"][:-1] >= -1e-6).all()
    batch = collate([ds[0], ds[1]])
    assert batch[0].shape == (2, 20, 8)


def test_hawkes_types_range():
    msg, book = _fixture()
    loader = LOBSTERLoader()
    df = loader.load_from_frames(msg, book)
    ht = loader.hawkes_types(df)
    assert set(np.unique(ht)).issubset({0, 1, 2, 3})
