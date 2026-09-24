"""Gradient inversion attacks: reconstructing patient records from shared gradients.

Threat model: an *honest-but-curious server* (or anyone intercepting updates)
observes the gradient a hospital computes on a small batch of patients, knows
the model architecture and current weights, and tries to recover the inputs.

Implemented attacks
-------------------
* ``analytic_linear_attack`` - closed-form recovery through the first fully
  connected layer (Phong et al., 2017; Geiping et al., 2020): for a single
  sample, dL/dW_j = (dL/db_j) * x, hence x = (dL/dW_j) / (dL/db_j).
* ``dlg_attack`` - Deep Leakage from Gradients (Zhu et al., NeurIPS 2019):
  jointly optimises a dummy input and dummy soft label so that their gradient
  matches the observed one.
* ``idlg_attack`` - improved DLG (Zhao et al., 2020): first infers the true
  label analytically from the last layer's bias gradient, then optimises the
  input only. More stable than DLG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

Grads = list[torch.Tensor]


@dataclass
class AttackResult:
    attack: str
    x_rec: torch.Tensor
    label_rec: int | None
    mse: float
    rel_error: float
    cosine: float
    label_correct: bool | None
    success: bool

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("x_rec")
        return d


def reconstruction_metrics(x_rec: torch.Tensor, x_true: torch.Tensor) -> dict:
    """Compare a reconstruction with the ground truth (standardised feature space)."""
    x_rec, x_true = x_rec.flatten().float(), x_true.flatten().float()
    mse = torch.mean((x_rec - x_true) ** 2).item()
    rel = (torch.linalg.norm(x_rec - x_true) / (torch.linalg.norm(x_true) + 1e-12)).item()
    cos = nn.functional.cosine_similarity(x_rec, x_true, dim=0).item()
    return {"mse": mse, "rel_error": rel, "cosine": cos}


def _result(
    name: str,
    x_rec: torch.Tensor,
    x_true: torch.Tensor,
    label_rec: int | None,
    label_true: int | None,
    success_threshold: float,
) -> AttackResult:
    m = reconstruction_metrics(x_rec, x_true)
    label_ok = None if label_rec is None or label_true is None else label_rec == label_true
    return AttackResult(
        attack=name,
        x_rec=x_rec.detach().cpu(),
        label_rec=label_rec,
        label_correct=label_ok,
        success=m["rel_error"] < success_threshold,
        **m,
    )


def _first_linear(model: nn.Module) -> nn.Linear:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m
    raise ValueError("Model has no nn.Linear layer.")


def _last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("Model has no nn.Linear layer.")
    return last


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
    _, gb = _grad_of(model, grads, _last_linear(model))
    return int(torch.argmin(gb).item())


def analytic_linear_attack(
    model: nn.Module,
    grads: Grads,
    x_true: torch.Tensor,
    label_true: int | None = None,
    success_threshold: float = 0.1,
) -> AttackResult:
    """Closed-form reconstruction through the first linear layer (exact when batch size = 1)."""
    gw, gb = _grad_of(model, grads, _first_linear(model))
    j = int(torch.argmax(gb.abs()).item())
    x_rec = (gw[j] / gb[j]).unsqueeze(0)
    return _result("analytic", x_rec, x_true, None, label_true, success_threshold)


def _gradient_matching(
    model: nn.Module,
    target: Grads,
    dummy_x: torch.Tensor,
    label_fn,
    params_to_opt: list[torch.Tensor],
    iterations: int,
    lr: float,
) -> None:
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


def dlg_attack(
    model: nn.Module,
    grads: Grads,
    x_true: torch.Tensor,
    label_true: int | None = None,
    n_classes: int = 2,
    iterations: int = 300,
    lr: float = 1.0,
    seed: int = 0,
    success_threshold: float = 0.1,
) -> AttackResult:
    """Deep Leakage from Gradients: optimise dummy input *and* dummy soft label."""
    g = torch.Generator().manual_seed(seed)
    device = x_true.device
    dummy_x = torch.randn(x_true.shape, generator=g).to(device).requires_grad_(True)
    dummy_y = torch.randn((x_true.shape[0], n_classes), generator=g).to(device).requires_grad_(True)
    target = [t.detach() for t in grads]

    def soft_label():
        return torch.softmax(dummy_y, dim=-1)

    _gradient_matching(model, target, dummy_x, soft_label, [dummy_x, dummy_y], iterations, lr)
    label_rec = int(dummy_y.argmax(-1)[0].item())
    return _result("dlg", dummy_x.detach(), x_true, label_rec, label_true, success_threshold)


def idlg_attack(
    model: nn.Module,
    grads: Grads,
    x_true: torch.Tensor,
    label_true: int | None = None,
    iterations: int = 300,
    lr: float = 1.0,
    seed: int = 0,
    success_threshold: float = 0.1,
) -> AttackResult:
    """iDLG: infer the label analytically, then optimise the input only."""
    g = torch.Generator().manual_seed(seed)
    device = x_true.device
    label_rec = infer_label_from_bias_grad(model, grads)
    label = torch.tensor([label_rec], device=device)
    dummy_x = torch.randn(x_true.shape, generator=g).to(device).requires_grad_(True)
    target = [t.detach() for t in grads]

    _gradient_matching(model, target, dummy_x, lambda: label, [dummy_x], iterations, lr)
    return _result("idlg", dummy_x.detach(), x_true, label_rec, label_true, success_threshold)
