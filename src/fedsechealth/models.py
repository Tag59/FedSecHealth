"""Model architectures.

Models avoid BatchNorm (GroupNorm instead) so they stay compatible with
Opacus per-sample gradients (DP-SGD).
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


class CNN(nn.Module):
    """Small CNN for 28x28 medical images (two conv blocks + MLP head)."""

    def __init__(
        self, in_channels: int, n_classes: int, width: int = 32, activation: str = "relu"
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GroupNorm(8, width),
            act(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GroupNorm(8, 2 * width),
            act(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width * 7 * 7, 128),
            act(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class LeNet(nn.Module):
    """LeNet variant used as the reference target in DLG / iDLG (sigmoid, strided convs)."""

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 12, 5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, padding=2, stride=1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(12 * 7 * 7, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(
    input_shape: tuple[int, ...] | int,
    n_classes: int,
    name: str = "auto",
    hidden: tuple[int, ...] = (64, 32),
    activation: str = "relu",
) -> nn.Module:
    """Build a model for inputs of ``input_shape`` (features,) or (C, H, W).

    ``name="auto"`` picks an MLP for tabular inputs and a CNN for images.
    """
    shape = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
    if name == "auto":
        name = "mlp" if len(shape) == 1 else "cnn"
    if name == "mlp":
        return MLP(int(torch.tensor(shape).prod()), n_classes, tuple(hidden), activation)
    if name == "cnn":
        return CNN(shape[0], n_classes, activation=activation)
    if name == "lenet":
        return LeNet(shape[0], n_classes)
    raise ValueError(f"Unknown model {name!r}")
