"""Flower ClientApp: one hospital.

Each SuperNode is started with ``--node-config "partition-id=<i> num-partitions=<n>"``
and only ever touches its own shard. It receives the global model, trains
locally with ``fl.Hospital`` (optionally with DP-SGD) and returns the new
weights plus metrics; raw data never leaves the node.

DP accounting survives across rounds even if Flower starts a fresh ClientApp
process for every message: the number of DP-SGD steps already taken is kept in
the node's persistent ``context.state`` and restored into Opacus' accountant.
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..fl import Hospital, evaluate
from .common import dataset, fl_config, model_fn, set_threads

app = ClientApp()
_STATE_KEY = "fedsechealth"


def _hospital(context: Context) -> tuple[Hospital, int]:
    run, node = context.run_config, context.node_config
    pid, n = int(node["partition-id"]), int(node["num-partitions"])
    fds = dataset(run, n)
    cfg = fl_config(run, seed=int(run.get("seed", 0)))
    hospital = Hospital(fds.clients[pid], model_fn(run, fds), cfg, torch.device("cpu"))
    return hospital, pid


def _restore_accountant(hospital: Hospital, context: Context) -> None:
    if hospital.privacy_engine is None or _STATE_KEY not in context.state:
        return
    steps = int(context.state[_STATE_KEY]["dp-steps"])
    if steps:
        opt = hospital.optimizer
        hospital.privacy_engine.accountant.history = [
            (opt.noise_multiplier, opt.expected_batch_size / hospital.n_samples, steps)
        ]


def _save_accountant(hospital: Hospital, context: Context) -> None:
    if hospital.privacy_engine is None:
        return
    steps = sum(s for _, _, s in hospital.privacy_engine.accountant.history)
    context.state[_STATE_KEY] = ConfigRecord({"dp-steps": int(steps)})


def local_train(state: dict, context: Context) -> tuple[dict, dict]:
    """One round on this hospital: returns the new weights and the metrics to report.

    Kept free of Flower messaging so it can be tested without a running federation.
    """
    set_threads()
    hospital, pid = _hospital(context)
    _restore_accountant(hospital, context)
    new_state = hospital.fit(state)
    _save_accountant(hospital, context)

    run, node = context.run_config, context.node_config
    client = dataset(run, int(node["num-partitions"])).clients[pid]
    local = evaluate(hospital.base_model, client.x, client.y, torch.device("cpu"))
    print(f"[hospital {pid}] local accuracy={local['accuracy']:.4f}", flush=True)
    metrics = {
        "num-examples": hospital.n_samples,
        "train-accuracy": local["accuracy"],
        "train-loss": local["loss"],
    }
    if hospital.privacy_engine is not None:
        metrics["epsilon"] = hospital.epsilon()
    return new_state, metrics


@app.train()
def train(msg: Message, context: Context) -> Message:
    new_state, metrics = local_train(msg.content["arrays"].to_torch_state_dict(), context)
    content = RecordDict({"arrays": ArrayRecord(new_state), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
