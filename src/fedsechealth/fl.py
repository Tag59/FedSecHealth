"""Federated learning simulation engine (FedAvg, McMahan et al., 2017).

A lightweight, deterministic in-process simulator: every hospital is a
``Hospital`` object owning its private data, and the server only ever sees
model updates. Keeping the engine in plain PyTorch makes it easy to plug in
attacks (malicious clients, curious server) and defenses (DP, robust
aggregation). A Flower-based deployment reusing the same client logic is
planned (see ROADMAP.md).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ClientData, FederatedDataset
from .privacy import DPConfig

StateDict = dict[str, torch.Tensor]

# Harmless Opacus/PyTorch hook warning: inputs never require grad during training.
warnings.filterwarnings("ignore", message="Full backward hook is firing")


@dataclass
class FLConfig:
    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.1
    model: str = "auto"  # auto | mlp | cnn | lenet
    hidden: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    seed: int = 0
    device: str = "auto"
    dp: DPConfig = field(default_factory=DPConfig)


class Hospital:
    """A federated client: holds private data and trains locally on request."""

    def __init__(
        self,
        data: ClientData,
        model_fn: Callable[[], nn.Module],
        cfg: FLConfig,
        device: torch.device,
    ) -> None:
        self.id = data.client_id
        self.n_samples = len(data)
        self.device = device
        self.dp = cfg.dp
        self.local_epochs = cfg.local_epochs
        self.loss_fn = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.y))
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + self.id)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=gen)
        model = model_fn().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

        self.privacy_engine: PrivacyEngine | None = None
        if cfg.dp.enabled:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Opacus warns about secure RNG in simulation
                self.privacy_engine = PrivacyEngine(accountant="rdp")
                model, optimizer, loader = self.privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    target_epsilon=cfg.dp.target_epsilon,
                    target_delta=cfg.dp.target_delta,
                    epochs=cfg.rounds * cfg.local_epochs,
                    max_grad_norm=cfg.dp.max_grad_norm,
                )
        self.model, self.optimizer, self.loader = model, optimizer, loader

    @property
    def base_model(self) -> nn.Module:
        """The underlying nn.Module (unwrapped from Opacus' GradSampleModule)."""
        return getattr(self.model, "_module", self.model)

    def fit(self, global_state: StateDict) -> StateDict:
        """Load the global model, train ``local_epochs`` epochs, return the new weights."""
        self.base_model.load_state_dict(global_state)
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                if len(y) == 0:  # Poisson sampling (DP) can yield empty batches
                    continue
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                self.loss_fn(self.model(x), y).backward()
                self.optimizer.step()
        return {k: v.detach().clone() for k, v in self.base_model.state_dict().items()}

    def epsilon(self) -> float:
        if self.privacy_engine is None:
            return float("inf")
        return self.privacy_engine.get_epsilon(self.dp.target_delta)


def fedavg(states: list[StateDict], weights: list[float]) -> StateDict:
    """Sample-size weighted average of client models."""
    total = float(sum(weights))
    return {
        k: sum(s[k].float() * (w / total) for s, w in zip(states, weights, strict=True))
        for k in states[0]
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 1024
) -> dict:
    """Accuracy, balanced accuracy (mean per-class recall) and loss on a held-out set."""
    model.eval()
    logits = torch.cat(
        [
            model(torch.from_numpy(x[i : i + batch_size]).to(device))
            for i in range(0, len(x), batch_size)
        ]
    )
    yt = torch.from_numpy(y).to(device)
    pred = logits.argmax(1)
    recalls = [(pred[yt == c] == c).float().mean().item() for c in torch.unique(yt)]
    return {
        "accuracy": (pred == yt).float().mean().item(),
        "balanced_accuracy": float(np.mean(recalls)),
        "loss": nn.functional.cross_entropy(logits, yt).item(),
    }


def run_federated(
    fds: FederatedDataset,
    model_fn: Callable[[], nn.Module],
    cfg: FLConfig,
    device: torch.device,
    log: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Run ``cfg.rounds`` rounds of FedAvg and return the global model and per-round history."""
    torch.manual_seed(cfg.seed)
    global_model = model_fn().to(device)
    hospitals = [Hospital(c, model_fn, cfg, device) for c in fds.clients]
    weights = [h.n_samples for h in hospitals]

    history = []
    for rnd in range(1, cfg.rounds + 1):
        state = copy.deepcopy(global_model.state_dict())
        updates = [h.fit(state) for h in hospitals]
        global_model.load_state_dict(fedavg(updates, weights))
        metrics = evaluate(global_model, fds.x_test, fds.y_test, device)
        metrics["round"] = rnd
        if cfg.dp.enabled:
            metrics["epsilon"] = max(h.epsilon() for h in hospitals)
        history.append(metrics)
        if log:
            log(metrics)
    return global_model, history


def run_local_only(
    fds: FederatedDataset,
    model_fn: Callable[[], nn.Module],
    cfg: FLConfig,
    device: torch.device,
) -> list[dict]:
    """Baseline: each hospital trains alone, without collaboration."""
    results = []
    for c in fds.clients:
        torch.manual_seed(cfg.seed)
        h = Hospital(c, model_fn, cfg, device)
        state = model_fn().to(device).state_dict()
        for _ in range(cfg.rounds):
            state = h.fit(state)
        results.append(
            {"client": c.client_id, **evaluate(h.base_model, fds.x_test, fds.y_test, device)}
        )
    return results


def run_centralized(
    fds: FederatedDataset,
    model_fn: Callable[[], nn.Module],
    cfg: FLConfig,
    device: torch.device,
) -> dict:
    """Upper-bound baseline: all data pooled in one place (what FL tries to avoid)."""
    pooled = ClientData(
        0,
        np.concatenate([c.x for c in fds.clients]),
        np.concatenate([c.y for c in fds.clients]),
    )
    torch.manual_seed(cfg.seed)
    h = Hospital(pooled, model_fn, cfg, device)
    state = model_fn().to(device).state_dict()
    for _ in range(cfg.rounds):
        state = h.fit(state)
    return evaluate(h.base_model, fds.x_test, fds.y_test, device)
