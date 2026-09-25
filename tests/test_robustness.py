import numpy as np
import torch

from fedsechealth.attacks import Adversary, AdversaryConfig, alie_z, apply_trigger
from fedsechealth.data import ClientData, load_federated_breast_cancer
from fedsechealth.defenses import (
    Aggregator,
    AggregatorConfig,
    coordinate_median,
    flatten,
    fltrust,
    krum_scores,
    multi_krum,
    norm_clip,
    trimmed_mean,
    unflatten,
)
from fedsechealth.fl import FLConfig, run_federated
from fedsechealth.models import build_model


def _honest_plus_outlier(n_honest=6, d=5, outlier=100.0):
    torch.manual_seed(0)
    honest = 1.0 + 0.1 * torch.randn(n_honest, d)
    return torch.cat([honest, torch.full((1, d), outlier)]), honest


def test_flatten_roundtrip():
    m = build_model(10, 2, hidden=(4,))
    s = m.state_dict()
    back = unflatten(flatten(s), s)
    for k in s:
        torch.testing.assert_close(back[k], s[k])


def test_median_and_trimmed_mean_ignore_one_outlier():
    u, honest = _honest_plus_outlier()
    assert (coordinate_median(u).update - honest.median(0).values).abs().max() < 0.2
    assert (trimmed_mean(u, 0.2).update - 1.0).abs().max() < 0.2


def test_krum_never_selects_the_outlier():
    u, _ = _honest_plus_outlier()
    assert krum_scores(u, f=1).argmin().item() != 6
    agg = multi_krum(u, f=1, m=5)
    assert agg.client_weights[6] == 0 and abs(agg.client_weights.sum().item() - 1) < 1e-6


def test_norm_clip_bounds_influence():
    u, _ = _honest_plus_outlier()
    agg = norm_clip(u, torch.ones(7), bound=None)
    assert agg.update.norm() < 3.0


def test_fltrust_zeroes_opposite_updates():
    server = torch.ones(5)
    u = torch.stack([torch.ones(5), 2 * torch.ones(5), -torch.ones(5)])
    agg = fltrust(u, server)
    assert agg.client_weights[2] == 0
    torch.testing.assert_close(agg.update, server)  # rescaled to the server norm


def test_fedavg_equivalence_with_plain_training():
    """The robust path with FedAvg and no attacker must match the plain FedAvg path."""
    fds = load_federated_breast_cancer(n_clients=3, seed=0)
    cfg = FLConfig(rounds=3)
    mf = lambda: build_model(30, 2)  # noqa: E731
    _, h1 = run_federated(fds, mf, cfg, torch.device("cpu"))
    _, h2 = run_federated(
        fds, mf, cfg, torch.device("cpu"), aggregator=Aggregator(AggregatorConfig("fedavg"))
    )
    assert abs(h1[-1]["accuracy"] - h2[-1]["accuracy"]) < 1e-6


def test_label_flip_poisons_only_malicious_clients():
    fds = load_federated_breast_cancer(n_clients=4, seed=0)
    adv = Adversary(AdversaryConfig("label_flip", n_malicious=1), 4, 2)
    poisoned = adv.poison_clients(fds.clients, fds)
    assert (poisoned[0].y == 1 - fds.clients[0].y).all()
    assert (poisoned[1].y == fds.clients[1].y).all()


def test_backdoor_trigger_and_poisoning():
    fds = load_federated_breast_cancer(n_clients=3, seed=0)
    x = apply_trigger(fds.clients[0].x[:5], fds, size=3)
    assert (x[:, :3] == 4.0).all() and (x[:, 3:] == fds.clients[0].x[:5, 3:]).all()
    cfg = AdversaryConfig("backdoor", n_malicious=1, backdoor_target=1, poison_fraction=0.5)
    poisoned = Adversary(cfg, 3, 2).poison_clients(fds.clients, fds)[0]
    triggered = (poisoned.x[:, :3] == 4.0).all(axis=1)
    assert triggered.sum() == len(poisoned) // 2 and (poisoned.y[triggered] == 1).all()


def test_image_trigger_is_white_square():
    from fedsechealth.data import FederatedDataset

    x = np.zeros((2, 3, 8, 8), np.float32)
    fds = FederatedDataset(
        clients=[ClientData(0, x, np.zeros(2, np.int64))],
        x_test=x,
        y_test=np.zeros(2, np.int64),
        n_classes=2,
        input_shape=(3, 8, 8),
        modality="image",
        feature_mean=np.full((3, 1, 1), 0.5, np.float32),
        feature_std=np.full((3, 1, 1), 0.25, np.float32),
    )
    t = apply_trigger(x, fds, size=2)
    assert np.allclose(t[:, :, 5:7, 5:7], 2.0) and t.sum() == 2 * 3 * 4 * 2.0


def test_model_poisoning_rows():
    u = torch.ones(4, 3)
    sf = Adversary(AdversaryConfig("sign_flip", n_malicious=1, scale=2.0), 4, 2).corrupt(u)
    assert (sf[0] == -2).all() and (sf[1:] == 1).all()
    u = torch.tensor([[0.0], [0.0], [1.0], [2.0], [3.0]])
    alie = Adversary(AdversaryConfig("alie", n_malicious=2, alie_z=1.0), 5, 2).corrupt(u)
    assert torch.allclose(alie[:2], torch.tensor([[1.0], [1.0]]))  # mean 2 - 1 * std 1


def test_alie_z_is_positive_for_minority_attackers():
    assert alie_z(10, 2) > 0 and alie_z(50, 10) > 0


def test_krum_f_is_clamped_for_small_federations():
    agg = Aggregator(AggregatorConfig("krum", n_byzantine=10))
    u, _ = _honest_plus_outlier()
    out = agg(u, torch.ones(7))
    assert out.client_weights.sum() == 1
