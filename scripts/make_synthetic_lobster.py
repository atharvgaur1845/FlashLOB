"""Generate a synthetic LOBSTER-format message + orderbook pair.

For SMOKE TESTS ONLY — not for paper numbers. Events are drawn with mild Hawkes-style
self-excitation (trades cluster, and volatility rises after trade bursts) so the
pipeline's "does volatility improve / does attention cluster" checks have *some*
structure to find. Output matches the real LOBSTER CSV layout, so swapping in real
files later requires no code change.

Usage:
    python scripts/make_synthetic_lobster.py --out data/lobster --n 20000 --seed 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def generate(n: int, seed: int = 0, n_levels: int = 10):
    rng = np.random.default_rng(seed)

    # --- latent excitation drives event clustering + volatility bursts ---
    excite = 0.0
    base_rate = 1.0
    times = np.zeros(n)
    types = np.zeros(n, dtype=int)
    t = 34200.0  # 09:30 in seconds from midnight (LOBSTER convention)

    mid = 1_000_000  # price * 10000 => $100.00
    mids = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=int)
    prices = np.zeros(n, dtype=np.int64)
    directions = np.zeros(n, dtype=int)

    for i in range(n):
        rate = base_rate + excite
        dt = rng.exponential(1.0 / rate)
        t += dt
        times[i] = t

        # event type: more trades when excited
        p_trade = 0.15 + 0.4 * (excite / (1 + excite))
        r = rng.random()
        if r < p_trade:
            etype = 4  # exec_visible (trade)
        elif r < p_trade + 0.45:
            etype = 1  # limit submit
        else:
            etype = 3  # cancel
        types[i] = etype

        size = int(rng.lognormal(mean=4.0, sigma=1.0))
        size = max(size, 1)
        sizes[i] = size

        direction = 1 if rng.random() < 0.5 else -1
        directions[i] = direction

        # trades excite future activity + move the mid (volatility burst)
        if etype == 4:
            excite += 0.6 * np.log1p(size) / 5.0
            mid += int(direction * rng.normal(0, 1) * 50)
        excite *= 0.97  # exponential decay of excitation

        mids[i] = mid
        # price near mid: limits sit a few ticks off, trades hit best quote
        offset = rng.integers(0, n_levels) * 100
        prices[i] = mid + (offset if direction == -1 else -offset)

    # --- message file: Time, Type, OrderID, Size, Price, Direction ---
    order_ids = np.arange(1, n + 1)
    message = np.column_stack([times, types, order_ids, sizes, prices, directions])

    # --- orderbook file: 10 levels of ask px/sz, bid px/sz around the mid ---
    book = np.zeros((n, 4 * n_levels), dtype=np.int64)
    for lvl in range(n_levels):
        ask_px = mids + (lvl + 1) * 100
        bid_px = mids - (lvl + 1) * 100
        ask_sz = rng.integers(50, 500, size=n)
        bid_sz = rng.integers(50, 500, size=n)
        book[:, 4 * lvl + 0] = ask_px
        book[:, 4 * lvl + 1] = ask_sz
        book[:, 4 * lvl + 2] = bid_px
        book[:, 4 * lvl + 3] = bid_sz

    return message, book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lobster")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    message, book = generate(args.n, args.seed)

    msg_path = os.path.join(args.out, "sample_message.csv")
    book_path = os.path.join(args.out, "sample_orderbook.csv")
    np.savetxt(msg_path, message, delimiter=",",
               fmt=["%.9f", "%d", "%d", "%d", "%d", "%d"])
    np.savetxt(book_path, book, delimiter=",", fmt="%d")
    print(f"wrote {msg_path} and {book_path} ({args.n} events)")


if __name__ == "__main__":
    main()
