"""YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .fl import FLConfig
from .privacy import DPConfig


@dataclass
class AttackConfig:
    n_targets: int = 30
    batch_size: int = 1
    iterations: int = 300
    epsilons: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    success_threshold: float = 0.1


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "results"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
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
    attack = _build(AttackConfig, raw.pop("attack", {}) or {})
    return _build(ExperimentConfig, {**raw, "fl": fl, "attack": attack})
