"""Tune DP-SGD hyper-parameters for a dataset at a fixed privacy budget.

DP-SGD needs its own hyper-parameters (Ponomareva et al., 2023): larger
batches, a higher learning rate and fewer steps usually beat the non-private
settings at the same epsilon. Candidates are ranked on the MedMNIST
*validation* split, never on the test set, so the test numbers reported in the
README stay unbiased. Note: hyper-parameter search itself consumes privacy in
a strict accounting; like most of the literature we report the budget of the
final run only.

    uv run python scripts/tune_dp.py -c configs/dermamnist_membership.yaml --epsilon 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fedsechealth.config import load_config
from fedsechealth.experiments import _dataset, _model_fn
from fedsechealth.fl import evaluate, run_federated
from fedsechealth.utils import set_seed

# (batch size, learning rate, local epochs, rounds, max grad norm)
GRID = [
    (32, 0.05, 1, 30, 1.0),
    (64, 0.1, 1, 30, 1.0),
    (128, 0.2, 1, 30, 1.0),
    (128, 0.5, 1, 30, 1.0),
    (256, 0.5, 1, 30, 1.0),
    (256, 1.0, 1, 30, 1.0),
    (128, 0.2, 2, 30, 1.0),
    (256, 0.5, 1, 60, 1.0),
    (128, 0.5, 1, 30, 0.5),
    (128, 0.5, 1, 30, 2.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg.seeds[0]
    set_seed(seed)
    fds = _dataset(cfg, seed)
    if fds.x_aux is None:
        raise SystemExit("tuning needs a validation split (MedMNIST)")
    device = torch.device("cpu")
    rows = []
    for bs, lr, ep, rounds, clip in GRID:
        dp = replace(cfg.fl.dp, enabled=True, target_epsilon=args.epsilon, max_grad_norm=clip)
        fl = replace(cfg.fl, batch_size=bs, lr=lr, local_epochs=ep, rounds=rounds, dp=dp, seed=seed)
        t0 = time.time()
        model, hist = run_federated(fds, _model_fn(fds, cfg), fl, device)
        val = evaluate(model, fds.x_aux, fds.y_aux, device)
        row = {
            "batch_size": bs,
            "lr": lr,
            "local_epochs": ep,
            "rounds": rounds,
            "max_grad_norm": clip,
            "epsilon": hist[-1]["epsilon"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "test_balanced_accuracy": hist[-1]["balanced_accuracy"],
            "minutes": (time.time() - t0) / 60,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = max(rows, key=lambda r: r["val_balanced_accuracy"])
    print("BEST (by validation):", json.dumps(best), flush=True)
    out = Path(args.out or f"results/dp_tuning/{cfg.data.name}_eps{args.epsilon:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"epsilon": args.epsilon, "rows": rows, "best": best}, indent=2))
    median = np.median([r["val_balanced_accuracy"] for r in rows])
    print(f"saved to {out}; median val bacc {median:.3f}")


if __name__ == "__main__":
    main()
