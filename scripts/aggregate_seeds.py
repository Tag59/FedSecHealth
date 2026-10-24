"""Merge multi-seed runs and report mean, std and 95 % confidence intervals.

Experiments write one JSON per run; seeds were added over time (e.g. seeds 0-1
first, then 2-4), so this script merges every result directory belonging to
the same experiment, de-duplicates identical (seed, setting) rows, and writes
``results/consolidated/`` plus re-plotted figures in ``docs/figures``.

    uv run python scripts/aggregate_seeds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from fedsechealth.experiments import summarise_robustness
from fedsechealth.plotting import plot_robustness_heatmap, plot_robustness_sweep

R, OUT, FIG = Path("results"), Path("results/consolidated"), Path("docs/figures")


def ci95(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "std": sd, "ci95": half, "n": n}


def merge_robustness(dirs: list[str]) -> dict:
    base, rows, seen = None, [], set()
    for d in dirs:
        path = R / d / "robustness.json"
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        r = json.loads(path.read_text())
        base = base or r
        for row in r["rows"]:
            key = (row["seed"], row["attack"], row["n_malicious"], row["aggregator"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
    summary = summarise_robustness(rows)
    for s in summary:
        group = [
            r
            for r in rows
            if (r["attack"], r["n_malicious"], r["aggregator"])
            == (s["attack"], s["n_malicious"], s["aggregator"])
        ]
        for k in ("balanced_accuracy", "backdoor_asr"):
            vals = [r[k] for r in group if r.get(k) is not None]
            if vals:
                s[f"{k}_ci95"] = ci95(vals)["ci95"]
    return {**base, "rows": rows, "summary": summary, "seeds": sorted({r["seed"] for r in rows})}


def merge_attack(dirs: list[str], method: str = "ig") -> dict:
    keys = ("success_rate", "median_ssim", "median_psnr")
    per_setting: dict[str, dict[str, list[float]]] = {}
    epsilons: dict[str, float | None] = {}
    for d in dirs:
        path = R / d / "attack.json"
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        for name, s in json.loads(path.read_text())["settings"].items():
            a = s["attacks"].get(method)
            if a is None:
                continue
            acc = per_setting.setdefault(name, {k: [] for k in keys})
            for k in keys:
                acc[k].append(a[k])
            epsilons.setdefault(name, s["epsilon"])
    return {
        name: {"epsilon": epsilons[name], **{k: ci95(v) for k, v in acc.items()}}
        for name, acc in per_setting.items()
    }


def merge_membership(dirs: list[str]) -> dict:
    """Per defense and view: utility, LiRA AUC / TPR at 1 % FPR, best attack's TPR at 1 % FPR."""
    acc: dict[str, dict[str, list[float]]] = {}
    for d in dirs:
        path = R / d / "membership.json"
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        for row in json.loads(path.read_text())["results"]:
            a = acc.setdefault(row["defense"], {})
            a.setdefault("balanced_accuracy", []).append(row["balanced_accuracy"])
            for view, v in row["views"].items():
                atk = v["attacks"]
                a.setdefault(f"{view}_lira_auc", []).append(atk["lira"]["auc"])
                a.setdefault(f"{view}_lira_tpr1", []).append(atk["lira"]["tpr_at_1fpr"])
                a.setdefault(f"{view}_best_tpr1", []).append(
                    max(x["tpr_at_1fpr"] for x in atk.values())
                )
    return {d: {k: ci95(v) for k, v in m.items()} for d, m in acc.items()}


def fmt(c: dict, pct: bool = False) -> str:
    if pct:
        return f"{100 * c['mean']:.1f} % ± {100 * c['ci95']:.1f}"
    return f"{c['mean']:.3f} ± {c['ci95']:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Consolidated results (mean ± 95 % CI over seeds)\n"]

    attack = merge_attack(["pneumonia_attack"] + [f"pneumonia_attack_seed{k}" for k in range(1, 5)])
    (OUT / "pneumonia_attack_ig.json").write_text(json.dumps(attack, indent=2))
    md += [
        "## Inverting Gradients on PneumoniaMNIST\n",
        "| Setting | Seeds | Median SSIM | Success rate |",
        "|---|---|---|---|",
    ]
    for name in ["no DP", "clip only", "sigma=0.001", "sigma=0.01", "sigma=0.1", "eps=8"]:
        if name in attack:
            a = attack[name]
            md.append(
                f"| {name} | {a['median_ssim']['n']} | {fmt(a['median_ssim'])} "
                f"| {fmt(a['success_rate'], True)} |"
            )

    for title, dirs, name, plots in [
        (
            "Medical backdoor (PneumoniaMNIST)",
            ["pneumonia_backdoor", "pneumonia_backdoor_s234"],
            "pneumonia_backdoor",
            "backdoor",
        ),
        (
            "Aggregator x attack grid (BloodMNIST, 2 malicious)",
            ["bloodmnist_byzantine", "bloodmnist_byzantine_s234"],
            "bloodmnist_byzantine",
            "grid",
        ),
        (
            "Sweeps over the number of attackers (BloodMNIST)",
            [
                "bloodmnist_byzantine_sweep",
                "bloodmnist_byzantine_sweep_alie_s1234",
                "bloodmnist_byzantine_sweep_sign_flip_s1234",
            ],
            "bloodmnist_byzantine_sweep",
            "sweep",
        ),
    ]:
        merged = merge_robustness(dirs)
        (OUT / f"{name}.json").write_text(json.dumps(merged, indent=2))
        md += [
            f"\n## {title} (seeds {merged['seeds']})\n",
            "| Attack | Malicious | Aggregator | Balanced accuracy | Backdoor success |",
            "|---|---|---|---|---|",
        ]
        for s in sorted(
            merged["summary"], key=lambda s: (s["attack"], s["n_malicious"], s["aggregator"])
        ):
            ba_ci = s.get("balanced_accuracy_ci95", float("nan"))
            ba = f"{s['balanced_accuracy']:.3f} ± {ba_ci:.3f}"
            asr = (
                f"{100 * s['backdoor_asr']:.1f} % ± {100 * s.get('backdoor_asr_ci95', np.nan):.1f}"
                if "backdoor_asr" in s
                else ""
            )
            md.append(f"| {s['attack']} | {s['n_malicious']} | {s['aggregator']} | {ba} | {asr} |")
        if plots == "backdoor":
            plot_robustness_sweep(
                merged, "backdoor", "backdoor_asr", FIG / "backdoor_sweep_pneumonia.png"
            )
        elif plots == "grid":
            plot_robustness_heatmap(merged, 2, FIG / "byzantine_heatmap_bloodmnist.png")
        else:
            for a in ("alie", "sign_flip"):
                plot_robustness_sweep(
                    merged, a, "balanced_accuracy", FIG / f"byzantine_sweep_{a}.png"
                )

    mia = merge_membership([f"dermamnist_membership_overfit_tuned_seed{k}" for k in range(5)])
    if mia:
        (OUT / "dermamnist_membership.json").write_text(json.dumps(mia, indent=2))
        md += [
            "\n## Membership inference (DermaMNIST, tuned DP)\n",
            "| Defense | Seeds | Balanced accuracy | LiRA AUC (global) | Best TPR@1% FPR (global) "
            "| Best TPR@1% FPR (local) |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in mia.items():
            md.append(
                f"| {name} | {m['balanced_accuracy']['n']} | {fmt(m['balanced_accuracy'])} "
                f"| {fmt(m['global_lira_auc'])} | {fmt(m['global_best_tpr1'], True)} "
                f"| {fmt(m['local_best_tpr1'], True)} |"
            )

    (OUT / "headline.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
