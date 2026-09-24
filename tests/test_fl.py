import numpy as np
import torch

from fedsechealth.config import load_config
from fedsechealth.data import ClientData, FederatedDataset, load_federated_breast_cancer
from fedsechealth.fl import FLConfig, evaluate, fedavg, run_federated
from fedsechealth.models import build_model
from fedsechealth.privacy import DPConfig, epsilon_for, noise_multiplier_for


def test_fedavg_is_weighted_mean():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([4.0, 8.0])}
    out = fedavg([a, b], [3, 1])
    torch.testing.assert_close(out["w"], torch.tensor([1.0, 2.0]))


def _run(dp: DPConfig):
    fds = load_federated_breast_cancer(n_clients=3, seed=0)
    cfg = FLConfig(rounds=5, dp=dp)
    return run_federated(fds, lambda: build_model(30, 2), cfg, torch.device("cpu"))


def test_federated_training_learns():
    _, hist = _run(DPConfig(enabled=False))
    assert hist[-1]["accuracy"] > 0.9


def test_dp_training_respects_budget():
    _, hist = _run(DPConfig(enabled=True, target_epsilon=2.0))
    assert hist[-1]["epsilon"] <= 2.0 + 1e-3
    assert hist[-1]["accuracy"] > 0.7


def test_dp_training_on_images_runs():
    """CNN + GroupNorm must be Opacus-compatible (per-sample gradients)."""
    rng = np.random.default_rng(0)
    clients = [
        ClientData(
            i, rng.standard_normal((40, 1, 28, 28)).astype(np.float32), rng.integers(0, 2, 40)
        )
        for i in range(2)
    ]
    fds = FederatedDataset(
        clients=clients,
        x_test=clients[0].x,
        y_test=clients[0].y,
        n_classes=2,
        input_shape=(1, 28, 28),
        modality="image",
        feature_mean=np.zeros((1, 1, 1), np.float32),
        feature_std=np.ones((1, 1, 1), np.float32),
    )
    cfg = FLConfig(rounds=2, batch_size=8, dp=DPConfig(enabled=True, target_epsilon=5.0))
    _, hist = run_federated(fds, lambda: build_model((1, 28, 28), 2), cfg, torch.device("cpu"))
    assert 0 <= hist[-1]["balanced_accuracy"] <= 1 and hist[-1]["epsilon"] <= 5.0 + 1e-3


def test_balanced_accuracy_on_imbalanced_labels():
    model = torch.nn.Linear(1, 2)
    with torch.no_grad():  # always predicts class 1
        model.weight.zero_()
        model.bias.copy_(torch.tensor([0.0, 1.0]))
    x = np.zeros((10, 1), np.float32)
    y = np.array([1] * 9 + [0])
    m = evaluate(model, x, y, torch.device("cpu"))
    assert abs(m["accuracy"] - 0.9) < 1e-6 and abs(m["balanced_accuracy"] - 0.5) < 1e-6


def test_epsilon_accounting_round_trip():
    sigma = noise_multiplier_for(3.0, 1e-5, sample_rate=0.1, steps=200)
    assert abs(epsilon_for(sigma, 0.1, 200, 1e-5) - 3.0) < 0.1
    assert epsilon_for(0.0, 0.1, 200, 1e-5) == float("inf")


def test_config_overrides():
    cfg = load_config(
        None,
        {"fl.rounds": 3, "fl.dp.enabled": True, "attack.n_targets": 2, "data.name": "bloodmnist"},
    )
    assert cfg.fl.rounds == 3 and cfg.fl.dp.enabled and cfg.attack.n_targets == 2
    assert cfg.data.name == "bloodmnist"
