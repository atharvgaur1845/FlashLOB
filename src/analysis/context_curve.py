"""Plot accuracy + volatility-MAE vs. context length (the paper's key figure)."""

from __future__ import annotations

import numpy as np


def plot_context_curve(summary: dict, save_path: str):
    """Two-panel figure: direction accuracy and volatility MAE vs context.

    summary: {context_length: {'accuracy': (mean,std), 'volatility_mae': (mean,std)}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        pass

    ctx = sorted(summary.keys())
    acc_m = [summary[c]["accuracy"][0] for c in ctx]
    acc_s = [summary[c]["accuracy"][1] for c in ctx]
    mae_m = [summary[c]["volatility_mae"][0] for c in ctx]
    mae_s = [summary[c]["volatility_mae"][1] for c in ctx]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.6))

    axL.plot(ctx, acc_m, "-o", color="tab:blue")
    axL.fill_between(ctx, np.array(acc_m) - np.array(acc_s),
                     np.array(acc_m) + np.array(acc_s), alpha=0.2, color="tab:blue")
    axL.set_xscale("log"); axL.set_xlabel("context length (events)")
    axL.set_ylabel("direction accuracy"); axL.set_title("Direction")
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    axR.plot(ctx, mae_m, "-o", color="tab:orange")
    axR.fill_between(ctx, np.array(mae_m) - np.array(mae_s),
                     np.array(mae_m) + np.array(mae_s), alpha=0.2, color="tab:orange")
    axR.set_xscale("log"); axR.set_xlabel("context length (events)")
    axR.set_ylabel("volatility MAE"); axR.set_title("Volatility")
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
