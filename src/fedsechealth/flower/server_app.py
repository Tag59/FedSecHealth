"""Flower ServerApp: the aggregation server.

Runs ``num-server-rounds`` rounds with the aggregation rule chosen in the run
config (``strategy``: fedavg, median, trimmed_mean, krum, multi_krum, i.e.
Flower's built-in counterparts of the rules studied in v0.3), waits for all
``num-hospitals`` SuperNodes, evaluates the global model after each round and
writes the history to ``output`` (JSON).

Evaluating on a central test set is for monitoring the experiment; it plays the
role of an independent validation authority, not of the aggregation server.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedMedian, FedTrimmedAvg, Krum, MultiKrum

from ..fl import evaluate
from .common import dataset, model_fn, set_threads

app = ServerApp()


def build_strategy(name: str, n: int, n_byzantine: int, trim: float):
    common = {
        "fraction_train": 1.0,
        "fraction_evaluate": 0.0,
        "min_train_nodes": n,
        "min_available_nodes": n,
        "min_evaluate_nodes": 0,
    }
    if name == "fedavg":
        return FedAvg(**common)
    if name == "median":
        return FedMedian(**common)
    if name == "trimmed_mean":
        return FedTrimmedAvg(beta=trim, **common)
    if name == "krum":
        return Krum(num_malicious_nodes=n_byzantine, **common)
    if name == "multi_krum":
        return MultiKrum(
            num_malicious_nodes=n_byzantine, num_nodes_to_select=n - n_byzantine, **common
        )
    raise ValueError(f"Unknown strategy {name!r}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    set_threads()
    run = context.run_config
    n = int(run["num-hospitals"])
    rounds = int(run["num-server-rounds"])
    fds = dataset(run, n)
    torch.manual_seed(int(run.get("seed", 0)))
    model = model_fn(run, fds)()
    history: list[dict] = []

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        m = evaluate(model, fds.x_test, fds.y_test, torch.device("cpu"))
        history.append({"round": server_round, **m})
        print(
            f"[round {server_round}/{rounds}] accuracy={m['accuracy']:.4f} "
            f"balanced_accuracy={m['balanced_accuracy']:.4f}",
            flush=True,
        )
        return MetricRecord(m)

    strategy = build_strategy(
        str(run.get("strategy", "fedavg")),
        n,
        int(run.get("n-byzantine", 0)),
        float(run.get("trim", 0.2)),
    )
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=rounds,
        evaluate_fn=global_evaluate,
    )
    out = Path(str(run.get("output", "results/flower_deployment/history.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_config": dict(run), "history": history}, indent=2, default=str))
    print(f"history written to {out}", flush=True)
