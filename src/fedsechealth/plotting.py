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


METRIC_LABELS = {"accuracy": "Test accuracy", "balanced_accuracy": "Test balanced accuracy"}
ATTACK_LABELS = {
    "analytic": "Analytic (linear layer)",
    "idlg": "iDLG",
    "dlg": "DLG",
    "ig": "Inverting Gradients",
}


def plot_training(result: dict, path: Path) -> Path:
    """Global-model utility per round (mean ± std over seeds) vs. baselines."""
    metric = result.get("metric", "accuracy")
    acc = np.array([[r[metric] for r in h] for h in result["federated"]])
    rounds = np.arange(1, acc.shape[1] + 1)
    mean, std = acc.mean(0), acc.std(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, mean, color=SERIES[0], label="Federated (FedAvg)")
    ax.fill_between(rounds, mean - std, mean + std, color=SERIES[0], alpha=0.15, linewidth=0)
    central = np.mean([r[metric] for r in result["centralized"]])
    local = np.mean([np.mean([c[metric] for c in run]) for run in result["local_only"]])
    ax.axhline(
        central, color=SERIES[1], linestyle="--", label=f"Centralized (pooled) {central:.3f}"
    )
    ax.axhline(local, color=SERIES[2], linestyle=":", label=f"Local only (mean) {local:.3f}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(METRIC_LABELS[metric])
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
    ax1.set_ylabel(METRIC_LABELS[tradeoff.get("metric", "accuracy")])
    ax1.set_title("Utility: global model")

    for i, (name, rates) in enumerate(tradeoff["attack_success"].items()):
        label = ATTACK_LABELS.get(name, name)
        ax2.plot(x, rates, color=SERIES[i], marker="o", markersize=6, label=label)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Reconstruction success rate")
    ax2.set_title("Privacy: gradient inversion attacks")
    ax2.set_xticks(x, labels)
    ax2.set_xlabel("Privacy budget ε (smaller = stronger privacy)")
    ax2.legend(loc="center left")
    return _save(fig, path)


def _setting_order(names: list[str], settings: dict) -> list[str]:
    """Weakest to strongest protection: no DP, clip only, then increasing noise."""
    head = [n for n in ("no DP", "clip only") if n in names]
    rest = [n for n in names if n not in head]
    rest.sort(key=lambda n: settings[n]["noise_multiplier"] or 0.0)
    return head + rest


def _setting_label(name: str, settings: dict) -> str:
    s = settings[name]
    if s["noise_multiplier"] is None or s["noise_multiplier"] == 0:
        return name
    eps = s["epsilon"]
    eps_txt = "ε=∞" if not np.isfinite(eps) else f"ε={eps:.3g}"
    sigma_txt = f"σ={s['noise_multiplier']:.3g}"
    # Budgets chosen as epsilon (realistic DP) are listed epsilon-first.
    return f"{eps_txt}\n{sigma_txt}" if name.startswith("eps=") else f"{sigma_txt}\n{eps_txt}"


def _show(ax, img: np.ndarray) -> None:
    if img.shape[0] == 1:
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_gallery(
    originals: np.ndarray, rows: dict[str, list], settings: dict, method: str, path: Path
) -> Path:
    """Grid: real images (top row), then the attacker's reconstructions per defense setting.

    Each reconstruction is annotated with its SSIM to the real image.
    """
    import torch

    from .attacks.metrics import ssim

    order = _setting_order(list(rows), settings)
    n = len(originals)
    width = max(1.35 * n + 1.3, 6.5)
    fig, axes = plt.subplots(
        len(order) + 1, n, figsize=(width, 1.35 * (len(order) + 1) + 0.5), squeeze=False
    )
    for j in range(n):
        _show(axes[0, j], originals[j])
    axes[0, 0].set_ylabel("real", rotation=0, ha="right", va="center", color=TEXT)
    ref = torch.from_numpy(originals)
    for i, name in enumerate(order, start=1):
        recs = np.stack(rows[name])
        scores = ssim(torch.from_numpy(recs).float(), ref.float()).numpy()
        for j in range(n):
            _show(axes[i, j], recs[j])
            axes[i, j].set_title(f"{scores[j]:.2f}", fontsize=7, color=TEXT_2, pad=2)
        label = _setting_label(name, settings)
        axes[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", color=TEXT, fontsize=8)
    for ax in axes.flat:
        ax.grid(False)
    title = f"{ATTACK_LABELS.get(method, method)}: reconstructions (SSIM above each)"
    fig.suptitle(title, color=TEXT, fontweight="bold")
    return _save(fig, path)


def plot_noise_sweep(result: dict, modality: str, path: Path) -> Path:
    """Attack quality as the DP noise multiplier grows (one line per attack)."""
    settings = result["settings"]
    order = _setting_order(list(settings), settings)
    key, ylabel = (
        ("median_ssim", "Median SSIM (1 = perfect)")
        if modality == "image"
        else ("success_rate", "Reconstruction success rate")
    )
    methods = list(settings[order[0]]["attacks"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(order)), 4))
    for i, m in enumerate(methods):
        y = [settings[n]["attacks"][m][key] for n in order]
        ax.plot(x, y, color=SERIES[i], marker="o", markersize=6, label=ATTACK_LABELS.get(m, m))
    ax.set_xticks(x, [_setting_label(n, settings) for n in order], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Defense (left: none, right: stronger noise)")
    title = f"Where gradient inversion breaks ({result['dataset']}, batch {result['batch_size']})"
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)
