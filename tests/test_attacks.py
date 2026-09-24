import torch
from torch import nn

from fedsechealth.attacks import (
    analytic_linear_attack,
    idlg_attack,
    infer_label_from_bias_grad,
)
from fedsechealth.models import build_model
from fedsechealth.privacy import dp_gradient


def _setup(label: int = 1):
    torch.manual_seed(0)
    model = build_model(10, 2, hidden=(16,))
    x = torch.randn(1, 10)
    y = torch.tensor([label])
    grads = torch.autograd.grad(nn.functional.cross_entropy(model(x), y), list(model.parameters()))
    return model, x, y, grads


def test_analytic_attack_is_exact_without_defense():
    model, x, y, grads = _setup()
    res = analytic_linear_attack(model, grads, x, int(y))
    assert res.success and res.rel_error < 1e-4


def test_label_inference_from_bias_gradient():
    for label in (0, 1):
        model, _, _, grads = _setup(label)
        assert infer_label_from_bias_grad(model, grads) == label


def test_idlg_recovers_input_without_defense():
    model, x, y, grads = _setup()
    res = idlg_attack(model, grads, x, int(y), iterations=200)
    assert res.label_correct and res.success


def test_dp_gradient_defeats_analytic_attack():
    model, x, y, _ = _setup()
    noisy = dp_gradient(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        max_grad_norm=1.0,
        noise_multiplier=2.0,
        generator=torch.Generator().manual_seed(0),
    )
    res = analytic_linear_attack(model, noisy, x, int(y))
    assert not res.success


def test_dp_gradient_without_noise_is_clipped():
    model, x, y, _ = _setup()
    g = dp_gradient(model, x, y, nn.CrossEntropyLoss(), max_grad_norm=1e-3, noise_multiplier=0.0)
    norm = torch.sqrt(sum((t**2).sum() for t in g))
    assert norm <= 1e-3 + 1e-6
