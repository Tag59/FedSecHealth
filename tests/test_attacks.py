import torch
from torch import nn

from fedsechealth.attacks import (
    analytic_linear_attack,
    idlg_attack,
    infer_label_from_bias_grad,
    inverting_gradients_attack,
    tabular_metrics,
)
from fedsechealth.models import build_model
from fedsechealth.privacy import dp_gradient


def _grads(model, x, y):
    loss = nn.functional.cross_entropy(model(x), y)
    return [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]


def _tabular(label: int = 1):
    torch.manual_seed(0)
    model = build_model(10, 2, hidden=(16,))
    x = torch.randn(1, 10)
    y = torch.tensor([label])
    return model, x, y, _grads(model, x, y)


def test_analytic_attack_is_exact_without_defense():
    model, x, _, grads = _tabular()
    rec = analytic_linear_attack(model, grads)
    m = tabular_metrics(rec.x, x)
    assert m["success"] and m["rel_error"] < 1e-4


def test_label_inference_from_bias_gradient():
    for label in (0, 1):
        model, _, _, grads = _tabular(label)
        assert infer_label_from_bias_grad(model, grads) == label


def test_idlg_recovers_input_without_defense():
    model, x, y, grads = _tabular()
    rec = idlg_attack(model, grads, tuple(x.shape), iterations=200)
    assert int(rec.labels) == int(y)
    assert tabular_metrics(rec.x, x)["success"]


def test_dp_gradient_defeats_analytic_attack():
    model, x, y, _ = _tabular()
    noisy = dp_gradient(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        max_grad_norm=1.0,
        noise_multiplier=2.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert not tabular_metrics(analytic_linear_attack(model, noisy).x, x)["success"]


def test_dp_gradient_without_noise_is_clipped():
    model, x, y, _ = _tabular()
    g = dp_gradient(model, x, y, nn.CrossEntropyLoss(), max_grad_norm=1e-3, noise_multiplier=0.0)
    norm = torch.sqrt(sum((t**2).sum() for t in g))
    assert norm <= 1e-3 + 1e-6


def test_inverting_gradients_improves_over_random_init():
    torch.manual_seed(0)
    model = build_model((1, 28, 28), 2, name="lenet")
    x = torch.rand(1, 1, 28, 28)
    y = torch.tensor([1])
    grads = _grads(model, x, y)
    rec = inverting_gradients_attack(model, grads, (1, 1, 28, 28), iterations=150, tv_weight=0.0)
    start = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    assert int(rec.labels) == 1
    assert ((rec.x - x) ** 2).mean() < 0.5 * ((start - x) ** 2).mean()


def test_analytic_attack_rejects_conv_models():
    model = build_model((1, 28, 28), 2, name="cnn")
    x, y = torch.rand(1, 1, 28, 28), torch.tensor([0])
    try:
        analytic_linear_attack(model, _grads(model, x, y))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
