"""Dataset loading and partitioning across simulated hospitals.

Only the tabular Breast Cancer Wisconsin (Diagnostic) dataset is supported for
now; imaging datasets (MedMNIST) are planned for v0.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass
class ClientData:
    """Private data held by one hospital."""

    client_id: int
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class FederatedDataset:
    clients: list[ClientData]
    x_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def unscale(self, x: np.ndarray) -> np.ndarray:
        """Map standardised features back to their original clinical units."""
        return x * self.feature_std + self.feature_mean


def iid_partition(y: np.ndarray, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle indices and split them into ``n_clients`` equal-sized shards."""
    return [np.sort(s) for s in np.array_split(rng.permutation(len(y)), n_clients)]


def dirichlet_partition(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_size: int = 10,
    max_tries: int = 100,
) -> list[np.ndarray]:
    """Label-skewed non-IID split (Hsu et al., 2019).

    For each class, the share of samples given to each client is drawn from
    Dir(alpha). Small ``alpha`` => very heterogeneous hospitals.
    """
    classes = np.unique(y)
    for _ in range(max_tries):
        shards: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(y == c))
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for shard, part in zip(shards, np.split(idx, cuts), strict=True):
                shard.extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.sort(np.array(s)) for s in shards]
    raise ValueError(
        f"Could not draw a Dirichlet(alpha={alpha}) partition with >= {min_size} "
        f"samples per client after {max_tries} tries; increase alpha or reduce n_clients."
    )


def federated_standardize(
    clients: list[ClientData], x_test: np.ndarray
) -> tuple[list[ClientData], np.ndarray, np.ndarray, np.ndarray]:
    """Standardise features with statistics computed *federatedly*.

    Each hospital only shares (count, sum, sum of squares) per feature, never
    raw records; the server aggregates them into a global mean/std. This avoids
    the common shortcut of fitting a scaler on pooled data, which would be
    impossible in a real deployment.
    """
    n = sum(len(c) for c in clients)
    s = sum(c.x.sum(axis=0) for c in clients)
    sq = sum((c.x**2).sum(axis=0) for c in clients)
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 1e-12))
    scaled = [ClientData(c.client_id, (c.x - mean) / std, c.y) for c in clients]
    return scaled, (x_test - mean) / std, mean, std


def load_federated_breast_cancer(
    n_clients: int = 3,
    partition: str = "iid",
    alpha: float = 0.5,
    test_size: float = 0.2,
    seed: int = 0,
) -> FederatedDataset:
    """Load Breast Cancer Wisconsin and distribute it across ``n_clients`` hospitals.

    A stratified global test set is held out to evaluate the shared model.
    """
    ds = load_breast_cancer()
    x = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    rng = np.random.default_rng(seed)
    if partition == "iid":
        parts = iid_partition(y_train, n_clients, rng)
    elif partition == "dirichlet":
        parts = dirichlet_partition(y_train, n_clients, alpha, rng)
    else:
        raise ValueError(f"Unknown partition scheme: {partition!r}")

    clients = [ClientData(i, x_train[p], y_train[p]) for i, p in enumerate(parts)]
    clients, x_test, mean, std = federated_standardize(clients, x_test)
    clients = [ClientData(c.client_id, c.x.astype(np.float32), c.y) for c in clients]
    return FederatedDataset(
        clients=clients,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
        n_features=x.shape[1],
        n_classes=2,
        feature_names=list(ds.feature_names),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
    )
