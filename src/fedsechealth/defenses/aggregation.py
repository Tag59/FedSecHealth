"""Server-side aggregation rules, from plain FedAvg to Byzantine-robust ones.

Every rule receives the clients' *updates* (local model minus global model),
flattened into vectors, and returns the aggregated update plus, when the rule
selects or weights clients, the weight it gave to each one (used to measure
how much influence malicious hospitals kept).

Implemented rules
-----------------
* ``fedavg`` - sample-size weighted mean (McMahan et al., 2017). Not robust:
  a single client can move the model arbitrarily.
* ``median`` - coordinate-wise median (Yin et al., ICML 2018).
* ``trimmed_mean`` - coordinate-wise mean after dropping the ``beta`` largest
  and smallest values (Yin et al., ICML 2018).
* ``krum`` / ``multi_krum`` - pick the update(s) closest to their n - f - 2
  nearest neighbours (Blanchard et al., NeurIPS 2017).
* ``norm_clip`` - clip every update to a norm bound, then average (Sun et al.,
  2019, "Can you really backdoor federated learning?").
* ``fltrust`` - the server trains on a small trusted root dataset and weights
  each client by the cosine similarity of its update to the server's own,
  after rescaling it to the server update's norm (Cao et al., NDSS 2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

StateDict = dict[str, torch.Tensor]


def flatten(state: StateDict) -> torch.Tensor:
    return torch.cat([v.detach().float().flatten() for v in state.values()])


def unflatten(vec: torch.Tensor, like: StateDict) -> StateDict:
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[i : i + n].view_as(v).to(v.dtype)
        i += n
    return out


@dataclass
class AggregatorConfig:
    name: str = "fedavg"
    trim_ratio: float = 0.2  # trimmed_mean: fraction removed on *each* side
    n_byzantine: int | None = None  # krum's f; None -> the true number of attackers
    clip_norm: float | None = None  # norm_clip bound; None -> median update norm
    root_size: int = 100  # fltrust: size of the server's trusted dataset


@dataclass
class Aggregation:
    update: torch.Tensor
    client_weights: torch.Tensor | None = None  # influence given to each client (sums to 1)


def _weighted_mean(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w / w.sum()
    return w @ u


def fedavg(updates: torch.Tensor, weights: torch.Tensor) -> Aggregation:
    w = weights / weights.sum()
    return Aggregation(w @ updates, w)


def coordinate_median(updates: torch.Tensor) -> Aggregation:
    s = updates.sort(dim=0).values
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return Aggregation(med)


def trimmed_mean(updates: torch.Tensor, beta: float) -> Aggregation:
    n = len(updates)
    k = int(beta * n)
    if 2 * k >= n:
        raise ValueError(f"trim_ratio={beta} removes every client (n={n})")
    s = updates.sort(dim=0).values
    return Aggregation(s[k : n - k].mean(0))


def krum_scores(updates: torch.Tensor, f: int) -> torch.Tensor:
    """Sum of squared distances to the n - f - 2 closest other updates (lower = more central)."""
    n = len(updates)
    k = max(1, n - f - 2)
    d = torch.cdist(updates, updates) ** 2
    d.fill_diagonal_(float("inf"))
    return d.topk(k, dim=1, largest=False).values.sum(1)


def multi_krum(updates: torch.Tensor, f: int, m: int) -> Aggregation:
    scores = krum_scores(updates, f)
    chosen = scores.topk(m, largest=False).indices
    w = torch.zeros(len(updates), device=updates.device)
    w[chosen] = 1.0 / m
    return Aggregation(w @ updates, w)


def norm_clip(updates: torch.Tensor, weights: torch.Tensor, bound: float | None) -> Aggregation:
    norms = updates.norm(dim=1)
    bound = float(norms.median()) if bound is None else bound
    factors = torch.clamp(bound / (norms + 1e-12), max=1.0)
    return Aggregation(_weighted_mean(updates * factors[:, None], weights), weights / weights.sum())


def fltrust(updates: torch.Tensor, server_update: torch.Tensor) -> Aggregation:
    s_norm = server_update.norm()
    cos = (updates @ server_update) / (updates.norm(dim=1) * s_norm + 1e-12)
    trust = torch.relu(cos)
    if trust.sum() == 0:
        return Aggregation(torch.zeros_like(server_update), trust)
    rescaled = updates * (s_norm / (updates.norm(dim=1) + 1e-12))[:, None]
    w = trust / trust.sum()
    return Aggregation(w @ rescaled, w)


AGGREGATORS = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "norm_clip", "fltrust"]


class Aggregator:
    """Callable wrapper selected by name from an ``AggregatorConfig``."""

    def __init__(self, cfg: AggregatorConfig, n_malicious: int = 0) -> None:
        if cfg.name not in AGGREGATORS:
            raise ValueError(f"Unknown aggregator {cfg.name!r}; choose from {AGGREGATORS}")
        self.cfg = cfg
        self.f = cfg.n_byzantine if cfg.n_byzantine is not None else n_malicious

    @property
    def needs_server_update(self) -> bool:
        return self.cfg.name == "fltrust"

    def __call__(
        self,
        updates: torch.Tensor,
        weights: torch.Tensor,
        server_update: torch.Tensor | None = None,
    ) -> Aggregation:
        name, n = self.cfg.name, len(updates)
        f = min(self.f, max(0, (n - 3) // 2))  # Krum requires n > 2f + 2
        if name == "fedavg":
            return fedavg(updates, weights)
        if name == "median":
            return coordinate_median(updates)
        if name == "trimmed_mean":
            return trimmed_mean(updates, self.cfg.trim_ratio)
        if name == "krum":
            return multi_krum(updates, f, 1)
        if name == "multi_krum":
            return multi_krum(updates, f, n - f)
        if name == "norm_clip":
            return norm_clip(updates, weights, self.cfg.clip_norm)
        if server_update is None:
            raise ValueError("fltrust needs the server's root-dataset update")
        return fltrust(updates, server_update)
