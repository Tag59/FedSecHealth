"""Small shared helpers: seeding, device selection, result I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Return the requested device.

    ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API,
    so ``"cuda"`` covers both NVIDIA (CUDA) and AMD (ROCm/HIP).
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path
