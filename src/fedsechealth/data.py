"""Dataset loading and partitioning across simulated hospitals.

Supported datasets
------------------
* ``breast_cancer`` - Breast Cancer Wisconsin (Diagnostic), 30 tabular features.
* ``pneumoniamnist`` - paediatric chest X-rays, normal vs. pneumonia (1x28x28).
* ``bloodmnist`` - blood cell microscopy, 8 cell types (3x28x28).
* ``dermamnist`` - dermatoscopic images of skin lesions, 7 classes (3x28x28).

MedMNIST (Yang et al., Scientific Data 2023) is downloaded on first use to
``$MEDMNIST_ROOT`` (default ``~/.medmnist``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

MEDMNIST = {
    "pneumoniamnist": "PneumoniaMNIST",
    "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST",
}
DATASETS = ["breast_cancer", *MEDMNIST]


@dataclass
class DataConfig:
    name: str = "breast_cancer"
    n_clients: int = 3
    partition: str = "iid"
    alpha: float = 0.5
    test_size: float = 0.2  # tabular only; MedMNIST uses its official test split
    max_train: int | None = None  # subsample the training pool (speed)


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
    n_classes: int
    input_shape: tuple[int, ...]
    modality: str  # "tabular" | "image"
    feature_mean: np.ndarray
    feature_std: np.ndarray
    class_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    def unscale(self, x: np.ndarray) -> np.ndarray:
        """Map standardised inputs back to original units (clinical units or [0, 1] pixels)."""
        return x * self.feature_std + self.feature_mean

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Valid input range in standardised space (images live in [0, 1] pixel space)."""
        if self.modality != "image":
            return None
        return (0 - self.feature_mean) / self.feature_std, (
            1 - self.feature_mean
        ) / self.feature_std


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
    """Standardise inputs with statistics computed *federatedly*.

    Each hospital only shares (count, sum, sum of squares) per feature (tabular)
    or per channel (images), never raw records; the server aggregates them into
    a global mean/std. This avoids the common shortcut of fitting a scaler on
    pooled data, which would be impossible in a real deployment.
    """
    image = clients[0].x.ndim == 4
    axes = (0, 2, 3) if image else (0,)
    per_sample = int(np.prod([clients[0].x.shape[a] for a in axes[1:]])) if image else 1
    n = sum(len(c) for c in clients) * per_sample
    s = sum(c.x.sum(axis=axes, dtype=np.float64) for c in clients)
    sq = sum((c.x.astype(np.float64) ** 2).sum(axis=axes) for c in clients)
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 1e-12))
    if image:  # broadcast over (C, H, W)
        mean, std = mean[:, None, None], std[:, None, None]
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    scaled = [
        ClientData(c.client_id, ((c.x - mean) / std).astype(np.float32), c.y) for c in clients
    ]
    return scaled, ((x_test - mean) / std).astype(np.float32), mean, std


def _partition(y: np.ndarray, cfg: DataConfig, rng: np.random.Generator) -> list[np.ndarray]:
    if cfg.partition == "iid":
        return iid_partition(y, cfg.n_clients, rng)
    if cfg.partition == "dirichlet":
        return dirichlet_partition(y, cfg.n_clients, cfg.alpha, rng)
    raise ValueError(f"Unknown partition scheme: {cfg.partition!r}")


def _load_breast_cancer(cfg: DataConfig, seed: int):
    ds = load_breast_cancer()
    x = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    meta = {
        "modality": "tabular",
        "class_names": ["malignant", "benign"],
        "feature_names": list(ds.feature_names),
    }
    return x_train, y_train, x_test, y_test, meta


def _load_medmnist(cfg: DataConfig):
    import medmnist

    root = Path(os.environ.get("MEDMNIST_ROOT", Path.home() / ".medmnist"))
    root.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, MEDMNIST[cfg.name])

    def load(split: str):
        d = cls(split=split, download=True, size=28, root=str(root))
        imgs = d.imgs.astype(np.float32) / 255.0
        imgs = imgs[:, None] if imgs.ndim == 3 else imgs.transpose(0, 3, 1, 2)
        return imgs, d.labels.reshape(-1).astype(np.int64), d.info

    x_train, y_train, info = load("train")
    x_test, y_test, _ = load("test")
    labels = info["label"]
    meta = {"modality": "image", "class_names": [labels[str(i)] for i in range(len(labels))]}
    return x_train, y_train, x_test, y_test, meta


def load_federated_dataset(cfg: DataConfig, seed: int = 0) -> FederatedDataset:
    """Load a dataset and distribute its training pool across ``cfg.n_clients`` hospitals.

    A global test set is held out to evaluate the shared model.
    """
    if cfg.name == "breast_cancer":
        x_train, y_train, x_test, y_test, meta = _load_breast_cancer(cfg, seed)
    elif cfg.name in MEDMNIST:
        x_train, y_train, x_test, y_test, meta = _load_medmnist(cfg)
    else:
        raise ValueError(f"Unknown dataset {cfg.name!r}; choose from {DATASETS}")

    rng = np.random.default_rng(seed)
    if cfg.max_train is not None and cfg.max_train < len(y_train):
        keep = np.sort(rng.choice(len(y_train), size=cfg.max_train, replace=False))
        x_train, y_train = x_train[keep], y_train[keep]

    parts = _partition(y_train, cfg, rng)
    clients = [ClientData(i, x_train[p], y_train[p]) for i, p in enumerate(parts)]
    clients, x_test, mean, std = federated_standardize(clients, x_test)
    return FederatedDataset(
        clients=clients,
        x_test=x_test,
        y_test=y_test,
        n_classes=len(meta["class_names"]),
        input_shape=tuple(x_train.shape[1:]),
        feature_mean=mean,
        feature_std=std,
        **meta,
    )


def load_federated_breast_cancer(
    n_clients: int = 3,
    partition: str = "iid",
    alpha: float = 0.5,
    test_size: float = 0.2,
    seed: int = 0,
) -> FederatedDataset:
    """Backward-compatible shortcut for the tabular dataset."""
    cfg = DataConfig("breast_cancer", n_clients, partition, alpha, test_size)
    return load_federated_dataset(cfg, seed)
