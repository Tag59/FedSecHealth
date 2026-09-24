"""Gradient inversion attacks: reconstructing patient data from shared gradients.

Threat model: an *honest-but-curious server* (or anyone intercepting updates)
observes the gradient a hospital computes on a small batch of patients, knows
the model architecture and current weights, and tries to recover the inputs.
Attacks only receive what the server sees (gradients, weights, input shape);
the ground truth is used afterwards, for evaluation only (see ``metrics.py``).

Implemented attacks
-------------------
* ``analytic_linear_attack`` - closed-form recovery through a first fully
  connected layer (Phong et al., 2017): for a single sample,
  dL/dW_j = (dL/db_j) * x, hence x = (dL/dW_j) / (dL/db_j).
* ``dlg_attack`` - Deep Leakage from Gradients (Zhu et al., NeurIPS 2019):
  jointly optimises a dummy input and dummy soft label (L-BFGS, L2 gradient distance).
* ``idlg_attack`` - improved DLG (Zhao et al., 2020): infers the label from the
  last layer's bias gradient, then optimises the input only.
* ``inverting_gradients_attack`` - Geiping et al., NeurIPS 2020: cosine
  gradient distance + total-variation image prior, signed Adam, box constraints.
  The strongest of the four on images.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Grads = list[torch.Tensor]


@dataclass
class Reconstruction:
    attack: str
    x: torch.Tensor  # reconstructed input(s), standardised space, shape (B, *input_shape)
    labels: torch.Tensor | None  # recovered labels, if the attack recovers them


def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise ValueError("Model has no nn.Linear layer.")
    return layers


def _grad_of(model: nn.Module, grads: Grads, layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (weight grad, bias grad) of ``layer`` from the flat list of parameter grads."""
    params = list(model.parameters())
    iw = next(i for i, p in enumerate(params) if p is layer.weight)
    ib = next(i for i, p in enumerate(params) if p is layer.bias)
    return grads[iw], grads[ib]


def infer_label_from_bias_grad(model: nn.Module, grads: Grads) -> int:
    """iDLG label inference.

    With cross-entropy and a single sample, dL/db_out = softmax(z) - onehot(y):
    the only negative entry is at the true class.
    """
    _, gb = _grad_of(model, grads, _linear_layers(model)[-1])
    return int(torch.argmin(gb).item())


def analytic_linear_attack(model: nn.Module, grads: Grads) -> Reconstruction:
    """Closed-form reconstruction through the first layer, if it is linear (exact for batch 1)."""
    first = next(m for m in model.modules() if isinstance(m, nn.Linear | nn.Conv2d))
    if not isinstance(first, nn.Linear):
        raise ValueError("The analytic attack requires a model whose first layer is nn.Linear.")
    gw, gb = _grad_of(model, grads, first)
    j = int(torch.argmax(gb.abs()).item())
    return Reconstruction("analytic", (gw[j] / gb[j]).unsqueeze(0).detach(), None)


def _gradient_matching(
    model: nn.Module,
    target: Grads,
    dummy_x: torch.Tensor,
    label_fn,
    params_to_opt: list[torch.Tensor],
    iterations: int,
    lr: float,
) -> None:
    """DLG-style L2 gradient matching with L-BFGS."""
    loss_fn = nn.CrossEntropyLoss()
    weights = list(model.parameters())
    opt = torch.optim.LBFGS(
        params_to_opt, lr=lr, max_iter=20, history_size=100, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = loss_fn(model(dummy_x), label_fn())
        dummy_grads = torch.autograd.grad(loss, weights, create_graph=True)
        diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, target, strict=True))
        diff.backward()
        return diff

    for _ in range(max(1, iterations // 20)):
        opt.step(closure)


def _randn(shape, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(device)


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def dlg_attack(
    model: nn.Module,
    grads: Grads,
    shape: tuple[int, ...],
    n_classes: int,
    iterations: int = 300,
    lr: float = 1.0,
    seed: int = 0,
) -> Reconstruction:
    """Deep Leakage from Gradients: optimise dummy inputs *and* dummy soft labels.

    ``shape`` is the batch shape (B, *input_shape).
    """
    dev = _device(model)
    dummy_x = _randn(shape, seed, dev).requires_grad_(True)
    dummy_y = _randn((shape[0], n_classes), seed + 1, dev).requires_grad_(True)
    target = [t.detach() for t in grads]
    _gradient_matching(
        model,
        target,
        dummy_x,
        lambda: torch.softmax(dummy_y, -1),
        [dummy_x, dummy_y],
        iterations,
        lr,
    )
    return Reconstruction("dlg", dummy_x.detach(), dummy_y.detach().argmax(-1))


def idlg_attack(
    model: nn.Module,
    grads: Grads,
    shape: tuple[int, ...],
    iterations: int = 300,
    lr: float = 1.0,
    seed: int = 0,
) -> Reconstruction:
    """iDLG: infer the label analytically (batch of 1), then optimise the input only."""
    if shape[0] != 1:
        raise ValueError("iDLG label inference only applies to a batch of one sample.")
    dev = _device(model)
    label = torch.tensor([infer_label_from_bias_grad(model, grads)], device=dev)
    dummy_x = _randn(shape, seed, dev).requires_grad_(True)
    target = [t.detach() for t in grads]
    _gradient_matching(model, target, dummy_x, lambda: label, [dummy_x], iterations, lr)
    return Reconstruction("idlg", dummy_x.detach(), label)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a batch of images (B, C, H, W)."""
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


def inverting_gradients_attack(
    model: nn.Module,
    grads: Grads,
    shape: tuple[int, ...],
    labels: torch.Tensor | None = None,
    iterations: int = 1000,
    lr: float = 0.1,
    tv_weight: float = 1e-2,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    restarts: int = 1,
    seed: int = 0,
) -> Reconstruction:
    """Inverting Gradients (Geiping et al., 2020).

    Minimises 1 - cos(grad(x'), grad_observed) + tv_weight * TV(x') with signed
    Adam and a step-decay schedule, projecting x' onto the valid pixel range.
    Labels are inferred iDLG-style for a batch of one; for larger batches they
    must be provided (the usual "known labels" assumption, i.e. a stronger attacker).
    """
    dev = _device(model)
    if labels is None:
        if shape[0] != 1:
            raise ValueError("Provide labels for batches larger than one.")
        labels = torch.tensor([infer_label_from_bias_grad(model, grads)], device=dev)
    labels = labels.to(dev)
    target = [t.detach() for t in grads]
    target_norm = torch.sqrt(sum((t**2).sum() for t in target))
    weights = list(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    lo, hi = (b.to(dev) for b in bounds) if bounds is not None else (None, None)

    best_x, best_loss = None, float("inf")
    for r in range(restarts):
        x = _randn(shape, seed + 1000 * r, dev).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        milestones = [int(iterations * f) for f in (3 / 8, 5 / 8, 7 / 8)]
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
        for _ in range(iterations):
            opt.zero_grad()
            loss = loss_fn(model(x), labels)
            dg = torch.autograd.grad(loss, weights, create_graph=True)
            dot = sum((a * b).sum() for a, b in zip(dg, target, strict=True))
            dg_norm = torch.sqrt(sum((a**2).sum() for a in dg))
            rec_loss = 1 - dot / (dg_norm * target_norm + 1e-12)
            total = rec_loss + tv_weight * total_variation(x)
            total.backward()
            x.grad.sign_()
            opt.step()
            sched.step()
            if lo is not None:
                with torch.no_grad():
                    x.data = torch.maximum(torch.minimum(x.data, hi), lo)
        if rec_loss.item() < best_loss:
            best_loss, best_x = rec_loss.item(), x.detach().clone()
    return Reconstruction("ig", best_x, labels)
