"""Differential privacy utilities (DP-SGD, Abadi et al., 2016).

Training uses Opacus directly (see ``fl.py``). This module additionally
exposes the *same* Gaussian mechanism as a pure function so that the attack
experiments can simulate exactly what an honest-but-curious server observes
when a DP client sends an update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch import nn


@dataclass
class DPConfig:
    enabled: bool = False
    target_epsilon: float = 5.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0


def noise_multiplier_for(
    target_epsilon: float, target_delta: float, sample_rate: float, steps: int
) -> float:
    """Smallest noise multiplier sigma reaching (epsilon, delta) after ``steps`` steps (RDP)."""
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant="rdp",
    )


def per_sample_gradients(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> list[list[torch.Tensor]]:
    """Per-sample gradients via a simple loop (fine for the small batches used in attacks)."""
    grads = []
    for xi, yi in zip(x, y, strict=True):
        loss = loss_fn(model(xi.unsqueeze(0)), yi.unsqueeze(0))
        grads.append([g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))])
    return grads


def dp_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    max_grad_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Gradient released by one DP-SGD step: per-sample clipping + Gaussian noise, averaged.

    Mirrors Opacus: g = (sum_i clip(g_i, C) + N(0, sigma^2 C^2 I)) / B.
    """
    samples = per_sample_gradients(model, x, y, loss_fn)
    batch = len(samples)
    summed = [torch.zeros_like(p) for p in model.parameters()]
    for g in samples:
        norm = torch.sqrt(sum((t**2).sum() for t in g))
        factor = torch.clamp(max_grad_norm / (norm + 1e-6), max=1.0)
        for acc, t in zip(summed, g, strict=True):
            acc += t * factor
    noisy = []
    for acc in summed:
        noise = torch.randn(acc.shape, generator=generator).to(acc)
        noisy.append((acc + noise * noise_multiplier * max_grad_norm) / batch)
    return noisy
