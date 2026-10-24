"""Configuration shared by the Flower ClientApp and ServerApp.

Everything comes from Flower's run config (``[tool.flwr.app.config]``,
overridable with ``flwr run -c``) and node config (``--node-config`` on each
SuperNode: ``partition-id`` and ``num-partitions``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data import DataConfig, FederatedDataset, load_federated_dataset
from ..fl import FLConfig
from ..models import build_model
from ..privacy import DPConfig

_DATASETS: dict[tuple, FederatedDataset] = {}


def fl_config(run: Mapping[str, Any], seed: int) -> FLConfig:
    dp = DPConfig(
        enabled=bool(run.get("dp-enabled", False)),
        target_epsilon=float(run.get("dp-epsilon", 8.0)),
        target_delta=float(run.get("dp-delta", 1e-5)),
        max_grad_norm=float(run.get("dp-max-grad-norm", 1.0)),
    )
    return FLConfig(
        rounds=int(run["num-server-rounds"]),
        local_epochs=int(run.get("local-epochs", 1)),
        batch_size=int(run.get("batch-size", 32)),
        lr=float(run.get("learning-rate", 0.05)),
        model=str(run.get("model", "auto")),
        seed=seed,
        device="cpu",
        dp=dp,
    )


def dataset(run: Mapping[str, Any], num_partitions: int) -> FederatedDataset:
    """Load and partition the dataset deterministically (cached per process).

    In this simulated deployment every hospital derives its own shard from the
    same seed; in a real one each hospital would read its own files instead.
    """
    cfg = DataConfig(
        name=str(run.get("dataset", "pneumoniamnist")),
        n_clients=num_partitions,
        partition=str(run.get("partition", "iid")),
        alpha=float(run.get("alpha", 0.5)),
    )
    seed = int(run.get("seed", 0))
    key = (cfg.name, cfg.n_clients, cfg.partition, cfg.alpha, seed)
    if key not in _DATASETS:
        _DATASETS[key] = load_federated_dataset(cfg, seed)
    return _DATASETS[key]


def model_fn(run: Mapping[str, Any], fds: FederatedDataset):
    name = str(run.get("model", "auto"))
    return lambda: build_model(fds.input_shape, fds.n_classes, name)


def set_threads() -> None:
    # Several hospitals may share one machine: keep each process modest.
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
