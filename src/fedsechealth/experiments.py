"""End-to-end experiments: training, attacks, and the privacy/utility trade-off."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attacks import (
    Adversary,
    Reconstruction,
    analytic_linear_attack,
    backdoor_asr,
    dlg_attack,
    idlg_attack,
    image_metrics,
    inverting_gradients_attack,
    match_batch,
    tabular_metrics,
)
from .config import ExperimentConfig
from .data import ClientData, FederatedDataset, load_federated_dataset
from .defenses import Aggregator
from .fl import run_centralized, run_federated, run_local_only
from .models import build_model
from .plotting import (
    plot_gallery,
    plot_noise_sweep,
    plot_robustness_heatmap,
    plot_robustness_sweep,
    plot_tradeoff,
    plot_training,
)
from .privacy import DPConfig, dp_gradient, epsilon_for, noise_multiplier_for
from .utils import get_device, save_json, set_seed

Log = Callable[[str], None]

DEFAULT_METHODS = {"tabular": ["analytic", "idlg", "dlg"], "image": ["idlg", "dlg", "ig"]}


def _dataset(cfg: ExperimentConfig, seed: int) -> FederatedDataset:
    return load_federated_dataset(cfg.data, seed)


def _model_fn(fds: FederatedDataset, cfg: ExperimentConfig) -> Callable[[], nn.Module]:
    f = cfg.fl
    return lambda: build_model(fds.input_shape, fds.n_classes, f.model, f.hidden, f.activation)


def _out(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_dir) / cfg.name


def _metric(fds: FederatedDataset) -> str:
    """Headline utility metric: balanced accuracy for (imbalanced) multi-class imaging."""
    return "balanced_accuracy" if fds.modality == "image" else "accuracy"


# --------------------------------------------------------------------------- training


def experiment_train(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Federated training vs. centralized and local-only baselines, over several seeds."""
    device = get_device(cfg.fl.device)
    result: dict = {"federated": [], "centralized": [], "local_only": []}
    for seed in cfg.seeds:
        set_seed(seed)
        fl_cfg = replace(cfg.fl, seed=seed)
        fds = _dataset(cfg, seed)
        metric = _metric(fds)
        mf = _model_fn(fds, cfg)
        _, hist = run_federated(fds, mf, fl_cfg, device)
        result["federated"].append(hist)
        result["centralized"].append(run_centralized(fds, mf, fl_cfg, device))
        result["local_only"].append(run_local_only(fds, mf, fl_cfg, device))
        local = np.mean([r[metric] for r in result["local_only"][-1]])
        log(
            f"seed {seed} ({metric}): federated={hist[-1][metric]:.4f} "
            f"centralized={result['centralized'][-1][metric]:.4f} local={local:.4f}"
            + (f" eps={hist[-1]['epsilon']:.2f}" if "epsilon" in hist[-1] else "")
        )
    d, dp = cfg.data, cfg.fl.dp
    split = "IID" if d.partition == "iid" else f"non-IID (Dirichlet α={d.alpha:g})"
    privacy = f"DP ε={dp.target_epsilon:g}" if dp.enabled else "no DP"
    result["title"] = f"{d.name}: {d.n_clients} hospitals, {split}, {privacy}"
    result["metric"] = metric
    out = _out(cfg)
    save_json(result, out / "train.json")
    plot_training(result, out / "train.png")
    log(f"saved to {out}")
    return result


# --------------------------------------------------------------------------- attacks


def _schedule(cfg: ExperimentConfig, n_client_samples: int) -> tuple[float, int]:
    """(sample rate, number of steps) of one hospital's DP-SGD schedule, as Opacus computes it."""
    f = cfg.fl
    sample_rate = f.batch_size / n_client_samples
    return sample_rate, int(f.rounds * f.local_epochs / sample_rate)


def dp_noise_multiplier(cfg: ExperimentConfig, n_client_samples: int, epsilon: float) -> float:
    """Noise multiplier reaching ``epsilon`` under the training schedule of ``cfg``."""
    q, steps = _schedule(cfg, n_client_samples)
    return noise_multiplier_for(epsilon, cfg.fl.dp.target_delta, q, steps)


def _attack_settings(cfg: ExperimentConfig, n: int) -> list[tuple[str, float | None, float | None]]:
    """(name, epsilon, noise multiplier). sigma=None: raw gradient; sigma=0: clipping only."""
    q, steps = _schedule(cfg, n)
    delta = cfg.fl.dp.target_delta
    a = cfg.attack
    settings = [(f"eps={e:g}", e, dp_noise_multiplier(cfg, n, e)) for e in a.epsilons]
    settings += [(f"sigma={s:g}", epsilon_for(s, q, steps, delta), s) for s in a.noise_multipliers]
    settings += [("no DP", None, None), ("clip only", None, 0.0)]
    return settings


def _run_method(
    name: str,
    model: nn.Module,
    grads: list[torch.Tensor],
    shape: tuple[int, ...],
    y: torch.Tensor,
    fds: FederatedDataset,
    cfg: ExperimentConfig,
    seed: int,
) -> Reconstruction:
    a = cfg.attack
    if name == "analytic":
        return analytic_linear_attack(model, grads)
    if name == "idlg":
        return idlg_attack(model, grads, shape, iterations=a.iterations, seed=seed)
    if name == "dlg":
        return dlg_attack(model, grads, shape, fds.n_classes, iterations=a.iterations, seed=seed)
    if name == "ig":
        bounds = fds.bounds()
        if bounds is not None:
            bounds = tuple(torch.from_numpy(b) for b in bounds)
        return inverting_gradients_attack(
            model,
            grads,
            shape,
            labels=None if shape[0] == 1 else y,  # batch > 1: known-label assumption
            iterations=a.ig_iterations,
            lr=a.ig_lr,
            tv_weight=a.ig_tv_weight,
            bounds=bounds,
            restarts=a.ig_restarts,
            seed=seed,
        )
    raise ValueError(f"Unknown attack {name!r}")


def _to_pixels(x: torch.Tensor, fds: FederatedDataset) -> torch.Tensor:
    mean, std = torch.from_numpy(fds.feature_mean), torch.from_numpy(fds.feature_std)
    return x.detach().cpu() * std + mean


def _score(rec: Reconstruction, x: torch.Tensor, y: torch.Tensor, fds, cfg) -> dict:
    a = cfg.attack
    if fds.modality == "image":
        m = image_metrics(_to_pixels(rec.x, fds), _to_pixels(x, fds), a.ssim_threshold)
    else:
        m = tabular_metrics(rec.x.cpu(), x.cpu(), a.success_threshold)
    if rec.labels is not None:  # labels are recovered as a multiset
        m["label_correct"] = sorted(rec.labels.tolist()) == sorted(y.tolist())
    return m


def _summarise(records: list[dict], modality: str) -> dict:
    keys = ["psnr", "ssim"] if modality == "image" else ["rel_error", "cosine"]
    out = {"success_rate": float(np.mean([r["success"] for r in records]))}
    for k in keys:
        out[f"median_{k}"] = float(np.median([r[k] for r in records]))
    labels = [r["label_correct"] for r in records if "label_correct" in r]
    if labels:
        out["label_accuracy"] = float(np.mean(labels))
    return out


def experiment_attack(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Honest-but-curious server inverting hospital gradients, with and without DP.

    The same target batches (drawn from hospital 0) are attacked under every
    setting, so settings are compared on identical patients. The server
    observes one gradient per batch, computed either on a freshly initialised
    model (the DLG benchmark setting) or on the global model after
    ``attack.trained_rounds`` rounds of FedAvg.
    """
    device = get_device(cfg.fl.device)
    a = cfg.attack
    seed = cfg.seeds[0]
    set_seed(seed)
    fds = _dataset(cfg, seed)
    mf = _model_fn(fds, cfg)
    methods = a.methods or DEFAULT_METHODS[fds.modality]
    client = fds.clients[0]
    loss_fn = nn.CrossEntropyLoss()

    trained_state = None
    if a.trained_rounds > 0:
        pre = replace(cfg.fl, rounds=a.trained_rounds, dp=DPConfig(enabled=False), seed=seed)
        global_model, hist = run_federated(fds, mf, pre, device)
        trained_state = global_model.state_dict()
        log(
            f"attacking the global model after {a.trained_rounds} rounds "
            f"({_metric(fds)}={hist[-1][_metric(fds)]:.3f})"
        )

    rng = np.random.default_rng(seed)
    targets = [
        rng.choice(len(client), size=a.batch_size, replace=False) for _ in range(a.n_targets)
    ]
    shape = (a.batch_size, *fds.input_shape)

    per_setting: dict[str, dict] = {}
    gallery: dict[str, dict[str, list]] = {m: {} for m in methods}
    originals: list[np.ndarray] = []
    for name, eps, sigma in _attack_settings(cfg, len(client)):
        records: dict[str, list[dict]] = {m: [] for m in methods}
        for t, idx in enumerate(targets):
            x = torch.from_numpy(client.x[idx]).to(device)
            y = torch.from_numpy(client.y[idx]).to(device)
            torch.manual_seed(seed * 10_000 + t)
            model = mf().to(device)
            if trained_state is not None:
                model.load_state_dict(trained_state)
            if sigma is None:
                loss = loss_fn(model(x), y)
                grads = [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]
            else:
                gen = torch.Generator().manual_seed(seed * 10_000 + t)
                grads = dp_gradient(
                    model, x, y, loss_fn, cfg.fl.dp.max_grad_norm, sigma, generator=gen
                )
            collect = fds.modality == "image" and t * a.batch_size < a.gallery_size
            if collect and name == "no DP":
                originals.extend(_to_pixels(x, fds).numpy())
            for m in methods:
                rec = _run_method(m, model, grads, shape, y, fds, cfg, seed + t)
                score = _score(rec, x, y, fds, cfg)
                records[m].append(score)
                if collect:
                    px = _to_pixels(rec.x, fds)[match_batch(rec.x.cpu(), x.cpu())].clamp(0, 1)
                    gallery[m].setdefault(name, []).extend(px.numpy())
        summary = {m: _summarise(rs, fds.modality) for m, rs in records.items()}
        per_setting[name] = {"epsilon": eps, "noise_multiplier": sigma, "attacks": summary}
        key = "median_ssim" if fds.modality == "image" else "median_rel_error"
        eps_txt = "" if eps is None else f", eps={eps:.3g}"
        log(
            f"{name:>12} (sigma={sigma}{eps_txt}): "
            + "  ".join(
                f"{m}={s['success_rate']:.0%} ({key[7:]}={s[key]:.3f})" for m, s in summary.items()
            )
        )

    result = {
        "dataset": cfg.data.name,
        "model": cfg.fl.model,
        "batch_size": a.batch_size,
        "n_targets": a.n_targets,
        "trained_rounds": a.trained_rounds,
        "settings": per_setting,
    }
    out = _out(cfg)
    save_json(result, out / "attack.json")
    if fds.modality == "image":
        n = min(a.gallery_size, len(originals))
        for m in methods:
            rows = {k: v[:n] for k, v in gallery[m].items()}
            plot_gallery(np.stack(originals[:n]), rows, per_setting, m, out / f"gallery_{m}.png")
    if a.noise_multipliers:
        plot_noise_sweep(result, fds.modality, out / "noise_sweep.png")
    log(f"saved to {out}")
    return result


# --------------------------------------------------------------------------- trade-off


def experiment_tradeoff(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Sweep the privacy budget: global model utility vs. attack success rate."""
    device = get_device(cfg.fl.device)
    eps_list = list(cfg.attack.epsilons)
    acc_mean, acc_std = [], []
    metric = "accuracy"
    for eps in [*eps_list, None]:
        accs = []
        for seed in cfg.seeds:
            set_seed(seed)
            dp = replace(cfg.fl.dp, enabled=eps is not None, target_epsilon=eps or 0.0)
            fl_cfg = replace(cfg.fl, seed=seed, dp=dp)
            fds = _dataset(cfg, seed)
            metric = _metric(fds)
            _, hist = run_federated(fds, _model_fn(fds, cfg), fl_cfg, device)
            accs.append(hist[-1][metric])
        acc_mean.append(float(np.mean(accs)))
        acc_std.append(float(np.std(accs)))
        eps_label = "inf" if eps is None else f"{eps:g}"
        log(f"eps={eps_label:>5}: {metric} {acc_mean[-1]:.4f} ± {acc_std[-1]:.4f}")

    attack = experiment_attack(replace(cfg, attack=replace(cfg.attack, noise_multipliers=[])), log)
    plotted = [f"eps={e:g}" for e in eps_list] + ["no DP"]
    methods = next(iter(attack["settings"].values()))["attacks"].keys()
    success = {
        m: [attack["settings"][s]["attacks"][m]["success_rate"] for s in plotted] for m in methods
    }
    result = {
        "epsilons": eps_list,
        "metric": metric,
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


# --------------------------------------------------------------------------- robustness


def split_root(fds: FederatedDataset, size: int, seed: int) -> tuple[ClientData, FederatedDataset]:
    """Carve the server's small trusted dataset (FLTrust) out of the test set.

    Every aggregator is then evaluated on the same remaining test samples.
    """
    rng = np.random.default_rng(seed + 12345)
    idx = rng.permutation(len(fds.y_test))
    root_idx, test_idx = np.sort(idx[:size]), np.sort(idx[size:])
    root = ClientData(-1, fds.x_test[root_idx], fds.y_test[root_idx])
    return root, replace(fds, x_test=fds.x_test[test_idx], y_test=fds.y_test[test_idx])


def _robust_runs(cfg: ExperimentConfig):
    """(attack, n_malicious) pairs; the clean run is done once, with no attacker."""
    g = cfg.robustness
    if "none" in g.attacks:
        yield "none", 0
    for n_mal in g.n_malicious:
        for attack in g.attacks:
            if attack != "none" and n_mal > 0:
                yield attack, n_mal


def summarise_robustness(rows: list[dict]) -> list[dict]:
    """Mean ± std over seeds for each (attack, n_malicious, aggregator)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["attack"], r["n_malicious"], r["aggregator"]), []).append(r)
    out = []
    for (attack, n_mal, agg), rs in groups.items():
        s = {"attack": attack, "n_malicious": n_mal, "aggregator": agg, "n_seeds": len(rs)}
        for k in ("accuracy", "balanced_accuracy", "backdoor_asr", "malicious_weight"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                s[k], s[f"{k}_std"] = float(np.mean(vals)), float(np.std(vals))
        out.append(s)
    return out


def experiment_robustness(cfg: ExperimentConfig, log: Log = print) -> dict:
    """Malicious hospitals vs. aggregation rules: final utility and backdoor success."""
    device = get_device(cfg.fl.device)
    g, adv_base = cfg.robustness, cfg.adversary
    # The defender configures Krum for the worst case it wants to tolerate.
    f = cfg.aggregator.n_byzantine
    f = max(g.n_malicious) if f is None else f
    rows = []
    for seed in cfg.seeds:
        set_seed(seed)
        root, fds = split_root(_dataset(cfg, seed), cfg.aggregator.root_size, seed)
        metric = _metric(fds)
        mf = _model_fn(fds, cfg)
        fl_cfg = replace(cfg.fl, seed=seed)
        for attack, n_mal in _robust_runs(cfg):
            adv_cfg = replace(adv_base, attack=attack, n_malicious=n_mal)
            for agg_name in g.aggregators:
                adversary = Adversary(adv_cfg, len(fds.clients), fds.n_classes, seed)
                agg_cfg = replace(cfg.aggregator, name=agg_name, n_byzantine=f)
                model, hist = run_federated(
                    fds,
                    mf,
                    fl_cfg,
                    device,
                    aggregator=Aggregator(agg_cfg),
                    adversary=adversary,
                    root=root,
                )
                last = hist[-1]
                asr = last.get("backdoor_asr")
                if asr is None and "backdoor" in g.attacks:  # baseline rate of the clean model
                    asr = backdoor_asr(
                        model, fds, device, adv_base.backdoor_target, adv_base.trigger_size
                    )
                mw = [h["malicious_weight"] for h in hist if "malicious_weight" in h]
                rows.append(
                    {
                        "seed": seed,
                        "attack": attack,
                        "n_malicious": n_mal,
                        "aggregator": agg_name,
                        "accuracy": last["accuracy"],
                        "balanced_accuracy": last["balanced_accuracy"],
                        "backdoor_asr": asr,
                        "malicious_weight": float(np.mean(mw)) if mw else None,
                    }
                )
                log(
                    f"seed {seed} {attack:>10} x{n_mal} {agg_name:>12}: {metric}={last[metric]:.3f}"
                    + (f" asr={asr:.3f}" if asr is not None else "")
                    + (f" mal_w={np.mean(mw):.2f}" if mw else "")
                )
    summary = summarise_robustness(rows)
    result = {
        "dataset": cfg.data.name,
        "n_clients": cfg.data.n_clients,
        "metric": metric,
        "krum_f": f,
        "rows": rows,
        "summary": summary,
    }
    out = _out(cfg)
    save_json(result, out / "robustness.json")
    for n_mal in g.n_malicious:
        plot_robustness_heatmap(result, n_mal, out / f"heatmap_{n_mal}mal.png")
    if len(g.n_malicious) > 1:
        for attack in [a for a in g.attacks if a != "none"]:
            key = "backdoor_asr" if attack == "backdoor" else metric
            plot_robustness_sweep(result, attack, key, out / f"sweep_{attack}.png")
            if attack == "backdoor":
                plot_robustness_sweep(result, attack, metric, out / f"sweep_{attack}_{metric}.png")
    log(f"saved to {out}")
    return result


# --------------------------------------------------------------------------- demo


def demo_reconstruction(cfg: ExperimentConfig, epsilon: float = 5.0, n_features: int = 6) -> str:
    """Markdown table: one patient's record vs. what the server reconstructs, in clinical units."""
    device = get_device(cfg.fl.device)
    seed = cfg.seeds[0]
    set_seed(seed)
    fds = _dataset(cfg, seed)
    if fds.modality != "tabular":
        raise ValueError("The demo table is for tabular data; image attacks produce galleries.")
    model = _model_fn(fds, cfg)().to(device)
    client = fds.clients[0]
    x = torch.from_numpy(client.x[:1]).to(device)
    y = torch.from_numpy(client.y[:1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(model(x), y)
    clean = [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]
    sigma = dp_noise_multiplier(cfg, len(client), epsilon)
    gen = torch.Generator().manual_seed(seed)
    noisy = dp_gradient(model, x, y, loss_fn, cfg.fl.dp.max_grad_norm, sigma, generator=gen)
    it = cfg.attack.iterations
    rec_clean = idlg_attack(model, clean, tuple(x.shape), iterations=it)
    rec_dp = idlg_attack(model, noisy, tuple(x.shape), iterations=it)

    true = fds.unscale(x.cpu().numpy())[0]
    rc = fds.unscale(rec_clean.x.cpu().numpy())[0]
    rd = fds.unscale(rec_dp.x.cpu().numpy())[0]
    names = fds.class_names
    diag = [names[int(v)] for v in (y, rec_clean.labels, rec_dp.labels)]
    rows = [
        f"| Feature | Real patient | Reconstructed (no DP) | Reconstructed (DP, ε={epsilon:g}) |",
        "|---|---:|---:|---:|",
        f"| diagnosis | {diag[0]} | {diag[1]} | {diag[2]} |",
    ]
    for i in range(n_features):
        rows.append(f"| {fds.feature_names[i]} | {true[i]:.3f} | {rc[i]:.3f} | {rd[i]:.3f} |")
    table = "\n".join(rows)
    out = _out(cfg) / "reconstruction.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    return table
