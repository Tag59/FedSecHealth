"""Model architectures.

Models avoid BatchNorm so they stay compatible with Opacus per-sample
gradients (DP-SGD).
"""

from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Multi-layer perceptron for tabular clinical features."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: tuple[int, ...] = (64, 32),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    n_features: int, n_classes: int, hidden: tuple[int, ...] = (64, 32), activation: str = "relu"
) -> nn.Module:
    return MLP(n_features, n_classes, tuple(hidden), activation)
