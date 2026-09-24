import torch

from fedsechealth.config import load_config
from fedsechealth.data import load_federated_breast_cancer
from fedsechealth.fl import FLConfig, fedavg, run_federated
from fedsechealth.models import build_model
from fedsechealth.privacy import DPConfig


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


def test_config_overrides():
    cfg = load_config(None, {"fl.rounds": 3, "fl.dp.enabled": True, "attack.n_targets": 2})
    assert cfg.fl.rounds == 3 and cfg.fl.dp.enabled and cfg.attack.n_targets == 2
