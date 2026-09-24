"""End-to-end experiments: training, attacks, and the privacy/utility trade-off."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attacks import analytic_linear_attack, dlg_attack, idlg_attack
from .config import ExperimentConfig
from .data import FederatedDataset, load_federated_breast_cancer
from .fl import run_centralized, run_federated, run_local_only
from .models import build_model
from .plotting import plot_tradeoff, plot_training
from .privacy import dp_gradient, noise_multiplier_for
from .utils import get_device, save_json, set_seed

Log = Callable[[str], None]


def _dataset(cfg: ExperimentConfig, seed: int) -> FederatedDataset:
    f = cfg.fl
    return load_federated_breast_cancer(f.n_clients, f.partition, f.alpha, seed=seed)


def _model_fn(fds: FederatedDataset, cfg: ExperimentConfig) -> Callable[[], nn.Module]:
    return lambda: build_model(fds.n_features, fds.n_classes, cfg.fl.hidden, cfg.fl.activation)


def _out(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_dir) / cfg.name


# --------------------------------------------------------------------------- training


def experiment_train(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Federated training vs. centralized and local-only baselines, over several seeds."""
    device = get_device(cfg.fl.device)
    result: dict = {"federated": [], "centralized": [], "local_only": []}
    for seed in cfg.seeds:
        set_seed(seed)
        fl_cfg = replace(cfg.fl, seed=seed)
        fds = _dataset(cfg, seed)
        mf = _model_fn(fds, cfg)
        _, hist = run_federated(fds, mf, fl_cfg, device)
        result["federated"].append(hist)
        result["centralized"].append(run_centralized(fds, mf, fl_cfg, device))
        result["local_only"].append(run_local_only(fds, mf, fl_cfg, device))
        log(
            f"seed {seed}: federated={hist[-1]['accuracy']:.4f} "
            f"centralized={result['centralized'][-1]['accuracy']:.4f} "
            f"local={np.mean([r['accuracy'] for r in result['local_only'][-1]]):.4f}"
            + (f" eps={hist[-1]['epsilon']:.2f}" if "epsilon" in hist[-1] else "")
        )
    dp = cfg.fl.dp
    split = "IID" if cfg.fl.partition == "iid" else f"non-IID (Dirichlet α={cfg.fl.alpha:g})"
    privacy = f"DP ε={dp.target_epsilon:g}" if dp.enabled else "no DP"
    result["title"] = f"{cfg.fl.n_clients} hospitals, {split}, {privacy}"
    out = _out(cfg)
    save_json(result, out / "train.json")
    plot_training(result, out / "train.png")
    log(f"saved to {out}")
    return result


# --------------------------------------------------------------------------- attacks


def dp_noise_multiplier(cfg: ExperimentConfig, n_client_samples: int, epsilon: float) -> float:
    """Noise multiplier a client uses under the training schedule of ``cfg`` (as Opacus does)."""
    f = cfg.fl
    sample_rate = f.batch_size / n_client_samples
    steps = int(f.rounds * f.local_epochs / sample_rate)
    return noise_multiplier_for(epsilon, f.dp.target_delta, sample_rate, steps)


def _run_attacks(model, grads, x, label, cfg: ExperimentConfig, seed: int) -> list[dict]:
    a = cfg.attack
    kw = {"success_threshold": a.success_threshold}
    results = [analytic_linear_attack(model, grads, x, label, **kw)]
    results.append(idlg_attack(model, grads, x, label, iterations=a.iterations, seed=seed, **kw))
    results.append(
        dlg_attack(model, grads, x, label, n_classes=2, iterations=a.iterations, seed=seed, **kw)
    )
    return [r.summary() for r in results]


def experiment_attack(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Honest-but-curious server inverting hospital gradients, with and without DP.

    For each target batch drawn from hospital 0, the server observes one
    gradient computed on a freshly initialised model (the DLG setting) and runs
    the analytic, iDLG and DLG attacks.
    """
    device = get_device(cfg.fl.device)
    a = cfg.attack
    seed = cfg.seeds[0]
    set_seed(seed)
    fds = _dataset(cfg, seed)
    mf = _model_fn(fds, cfg)
    client = fds.clients[0]
    rng = np.random.default_rng(seed)
    loss_fn = nn.CrossEntropyLoss()

    # (name, epsilon, noise multiplier); sigma=None means the raw, unprotected gradient.
    # "clip only" isolates the effect of per-sample clipping without Gaussian noise.
    settings: list[tuple[str, float | None, float | None]] = [
        (f"eps={e:g}", e, dp_noise_multiplier(cfg, len(client), e)) for e in a.epsilons
    ]
    settings += [("no DP", None, None), ("clip only", None, 0.0)]
    per_setting: dict[str, dict] = {}
    for name, eps, sigma in settings:
        records = []
        for t in range(a.n_targets):
            idx = rng.choice(len(client), size=a.batch_size, replace=False)
            x = torch.from_numpy(client.x[idx]).to(device)
            y = torch.from_numpy(client.y[idx]).to(device)
            torch.manual_seed(seed * 10_000 + t)
            model = mf().to(device)
            if sigma is None:
                loss = loss_fn(model(x), y)
                grads = [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]
            else:
                gen = torch.Generator().manual_seed(seed * 10_000 + t)
                grads = dp_gradient(
                    model, x, y, loss_fn, cfg.fl.dp.max_grad_norm, sigma, generator=gen
                )
            label = int(y[0]) if a.batch_size == 1 else None
            records.extend(_run_attacks(model, grads, x, label, cfg, seed + t))
        summary = {}
        for attack in ("analytic", "idlg", "dlg"):
            rs = [r for r in records if r["attack"] == attack]
            summary[attack] = {
                "success_rate": float(np.mean([r["success"] for r in rs])),
                "median_rel_error": float(np.median([r["rel_error"] for r in rs])),
                "median_cosine": float(np.median([r["cosine"] for r in rs])),
            }
            labels = [r["label_correct"] for r in rs if r["label_correct"] is not None]
            if labels:
                summary[attack]["label_accuracy"] = float(np.mean(labels))
        per_setting[name] = {"epsilon": eps, "noise_multiplier": sigma, "attacks": summary}
        log(
            f"{name:>8} (sigma={sigma if sigma is None else round(sigma, 3)}): "
            + "  ".join(f"{k}={v['success_rate']:.0%}" for k, v in summary.items())
        )
    result = {"batch_size": a.batch_size, "n_targets": a.n_targets, "settings": per_setting}
    save_json(result, _out(cfg) / "attack.json")
    return result


# --------------------------------------------------------------------------- trade-off


def experiment_tradeoff(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Sweep the privacy budget: global accuracy vs. attack success rate."""
    device = get_device(cfg.fl.device)
    eps_list = list(cfg.attack.epsilons)
    acc_mean, acc_std = [], []
    for eps in [*eps_list, None]:
        accs = []
        for seed in cfg.seeds:
            set_seed(seed)
            dp = replace(cfg.fl.dp, enabled=eps is not None, target_epsilon=eps or 0.0)
            fl_cfg = replace(cfg.fl, seed=seed, dp=dp)
            fds = _dataset(cfg, seed)
            _, hist = run_federated(fds, _model_fn(fds, cfg), fl_cfg, device)
            accs.append(hist[-1]["accuracy"])
        acc_mean.append(float(np.mean(accs)))
        acc_std.append(float(np.std(accs)))
        eps_label = "inf" if eps is None else f"{eps:g}"
        log(f"eps={eps_label:>5}: accuracy {acc_mean[-1]:.4f} ± {acc_std[-1]:.4f}")

    attack = experiment_attack(cfg, log)
    names = {"analytic": "Analytic (linear layer)", "idlg": "iDLG", "dlg": "DLG"}
    plotted = [f"eps={e:g}" for e in eps_list] + ["no DP"]
    success = {
        label: [attack["settings"][s]["attacks"][key]["success_rate"] for s in plotted]
        for key, label in names.items()
    }
    result = {
        "epsilons": eps_list,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "attack_success": success,
        "attack": attack,
    }
    out = _out(cfg)
    save_json(result, out / "tradeoff.json")
    plot_tradeoff(result, out / "tradeoff.png")
    log(f"saved to {out}")
    return result


# --------------------------------------------------------------------------- demo


def demo_reconstruction(cfg: ExperimentConfig, epsilon: float = 5.0, n_features: int = 6) -> str:
    """Markdown table: one patient's record vs. what the server reconstructs, in clinical units."""
    device = get_device(cfg.fl.device)
    seed = cfg.seeds[0]
    set_seed(seed)
    fds = _dataset(cfg, seed)
    model = _model_fn(fds, cfg)().to(device)
    client = fds.clients[0]
    x = torch.from_numpy(client.x[:1]).to(device)
    y = torch.from_numpy(client.y[:1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    clean = [
        g.detach() for g in torch.autograd.grad(loss_fn(model(x), y), list(model.parameters()))
    ]
    sigma = dp_noise_multiplier(cfg, len(client), epsilon)
    noisy = dp_gradient(
        model,
        x,
        y,
        loss_fn,
        cfg.fl.dp.max_grad_norm,
        sigma,
        generator=torch.Generator().manual_seed(seed),
    )
    rec_clean = idlg_attack(model, clean, x, int(y), iterations=cfg.attack.iterations)
    rec_dp = idlg_attack(model, noisy, x, int(y), iterations=cfg.attack.iterations)

    true = fds.unscale(x.cpu().numpy())[0]
    rc = fds.unscale(rec_clean.x_rec.numpy())[0]
    rd = fds.unscale(rec_dp.x_rec.numpy())[0]
    diag = {0: "malignant", 1: "benign"}
    rows = [
        f"| Feature | Real patient | Reconstructed (no DP) | Reconstructed (DP, ε={epsilon:g}) |",
        "|---|---:|---:|---:|",
        f"| diagnosis | {diag[int(y)]} | {diag[rec_clean.label_rec]} | {diag[rec_dp.label_rec]} |",
    ]
    for i in range(n_features):
        rows.append(f"| {fds.feature_names[i]} | {true[i]:.3f} | {rc[i]:.3f} | {rd[i]:.3f} |")
    table = "\n".join(rows)
    out = _out(cfg) / "reconstruction.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    return table
