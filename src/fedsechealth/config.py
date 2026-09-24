"""YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .data import DataConfig
from .fl import FLConfig
from .privacy import DPConfig


@dataclass
class AttackConfig:
    # None -> default for the modality: tabular [analytic, idlg, dlg], image [idlg, dlg, ig]
    methods: list[str] | None = None
    n_targets: int = 30
    batch_size: int = 1
    iterations: int = 300  # L-BFGS iterations for DLG / iDLG
    ig_iterations: int = 1000
    ig_lr: float = 0.1
    ig_tv_weight: float = 1e-2
    ig_restarts: int = 1
    # Defended settings: DP budgets (noise calibrated to the training schedule)
    # and/or raw noise multipliers (to locate where attacks break).
    epsilons: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    noise_multipliers: list[float] = field(default_factory=list)
    success_threshold: float = 0.1  # tabular: relative L2 error below this
    ssim_threshold: float = 0.6  # images: SSIM at or above this
    trained_rounds: int = 0  # >0: attack the global model after this many FedAvg rounds
    gallery_size: int = 6


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "results"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    data: DataConfig = field(default_factory=DataConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) if path else {}
    raw = raw or {}
    for key, value in (overrides or {}).items():  # dotted overrides, e.g. {"fl.rounds": 5}
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value

    fl_raw = dict(raw.pop("fl", {}) or {})
    dp = _build(DPConfig, fl_raw.pop("dp", {}) or {})
    if "hidden" in fl_raw:
        fl_raw["hidden"] = tuple(fl_raw["hidden"])
    fl = _build(FLConfig, {**fl_raw, "dp": dp})
    data = _build(DataConfig, raw.pop("data", {}) or {})
    attack = _build(AttackConfig, raw.pop("attack", {}) or {})
    return _build(ExperimentConfig, {**raw, "data": data, "fl": fl, "attack": attack})
