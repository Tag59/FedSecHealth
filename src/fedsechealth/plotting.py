"""Static figures for the README and reports (matplotlib, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated all-pairs)
NEUTRAL = "#8a8983"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "lines.linewidth": 2,
        "font.size": 10,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training(result: dict, path: Path) -> Path:
    """Global-model accuracy per round (mean ± std over seeds) vs. baselines."""
    acc = np.array([[r["accuracy"] for r in h] for h in result["federated"]])
    rounds = np.arange(1, acc.shape[1] + 1)
    mean, std = acc.mean(0), acc.std(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, mean, color=SERIES[0], label="Federated (FedAvg)")
    ax.fill_between(rounds, mean - std, mean + std, color=SERIES[0], alpha=0.15, linewidth=0)
    central = np.mean([r["accuracy"] for r in result["centralized"]])
    local = np.mean([np.mean([c["accuracy"] for c in run]) for run in result["local_only"]])
    ax.axhline(
        central, color=SERIES[1], linestyle="--", label=f"Centralized (pooled) {central:.3f}"
    )
    ax.axhline(local, color=SERIES[2], linestyle=":", label=f"Local only (mean) {local:.3f}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(result.get("title", "Federated training"))
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_tradeoff(tradeoff: dict, path: Path) -> Path:
    """Two stacked panels sharing the epsilon axis: utility (top) and attack success (bottom)."""
    eps = tradeoff["epsilons"]
    x = list(range(len(eps))) + [len(eps)]
    labels = [f"{e:g}" for e in eps] + ["no DP"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    acc_m = tradeoff["accuracy_mean"]
    acc_s = tradeoff["accuracy_std"]
    ax1.errorbar(x, acc_m, yerr=acc_s, color=SERIES[0], marker="o", markersize=6, capsize=3)
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Utility: global model accuracy")

    for i, (name, rates) in enumerate(tradeoff["attack_success"].items()):
        ax2.plot(x, rates, color=SERIES[i], marker="o", markersize=6, label=name)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Reconstruction success rate")
    ax2.set_title("Privacy: gradient inversion attacks")
    ax2.set_xticks(x, labels)
    ax2.set_xlabel("Privacy budget ε (smaller = stronger privacy)")
    ax2.legend(loc="center left")
    return _save(fig, path)
