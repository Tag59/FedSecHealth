import numpy as np
import pytest
import torch

from fedsechealth.attacks.membership import (
    entropy_scores,
    lira_offline_scores,
    loss_scores,
    mia_metrics,
    scaled_logit,
)
from fedsechealth.data import load_federated_breast_cancer
from fedsechealth.defenses import Aggregator, AggregatorConfig
from fedsechealth.defenses.aggregation import dp_fedavg
from fedsechealth.defenses.secure_aggregation import (
    SecAggClient,
    decode,
    encode,
    secure_sum,
    shamir_reconstruct,
    shamir_share,
)
from fedsechealth.fl import FLConfig, run_federated
from fedsechealth.models import build_model

# --------------------------------------------------------------------------- secure aggregation


def test_shamir_any_t_shares_reconstruct():
    secret = 2**255 + 12345
    shares = shamir_share(secret, n=7, t=4)
    assert shamir_reconstruct(shares[:4]) == secret
    assert shamir_reconstruct([shares[i] for i in (1, 3, 5, 6)]) == secret
    assert shamir_reconstruct(shares[:3]) != secret  # below threshold: wrong value


def test_fixed_point_roundtrip():
    x = np.array([-1.5, 0.0, 3e-4, 12.25])
    np.testing.assert_allclose(decode(encode(x)), x, atol=1e-7)


@pytest.mark.parametrize("dropped", [set(), {1}, {0, 4}])
def test_secure_sum_matches_plain_sum_with_dropouts(dropped):
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=1000) for _ in range(6)]
    total, stats = secure_sum(xs, dropped)
    expected = sum(x for i, x in enumerate(xs) if i not in dropped)
    np.testing.assert_allclose(total, expected, atol=1e-6)
    assert stats.n_dropped == len(dropped)


def test_secure_sum_aborts_below_threshold():
    xs = [np.ones(10) for _ in range(5)]
    with pytest.raises(RuntimeError):
        secure_sum(xs, dropped={0, 1, 2}, threshold=3)


def test_masked_upload_hides_the_update():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    a, b = SecAggClient(0), SecAggClient(1)
    masked = decode(a.masked_input(x, {0: a.pk, 1: b.pk}))
    assert abs(np.corrcoef(masked, x)[0, 1]) < 0.05


def test_secure_fedavg_equals_plain_fedavg():
    fds = load_federated_breast_cancer(n_clients=4, seed=0)
    cfg = FLConfig(rounds=3)
    mf = lambda: build_model(30, 2)  # noqa: E731
    dev = torch.device("cpu")
    _, plain = run_federated(fds, mf, cfg, dev, aggregator=Aggregator(AggregatorConfig()))
    _, sec = run_federated(fds, mf, cfg, dev, aggregator=Aggregator(AggregatorConfig(secure=True)))
    assert abs(plain[-1]["accuracy"] - sec[-1]["accuracy"]) < 1e-6
    assert "secagg_seconds" in sec[-1]


def test_secure_aggregation_rejects_robust_rules():
    with pytest.raises(ValueError):
        Aggregator(AggregatorConfig("median", secure=True))


def test_return_local_gives_one_state_per_hospital():
    fds = load_federated_breast_cancer(n_clients=3, seed=0)
    out = run_federated(
        fds, lambda: build_model(30, 2), FLConfig(rounds=1), torch.device("cpu"), return_local=True
    )
    assert len(out) == 3 and len(out[2]) == 3


# --------------------------------------------------------------------------- client-level DP


def test_dp_fedavg_clips_and_adds_calibrated_noise():
    u = torch.stack([torch.full((10_000,), 1.0), torch.zeros(10_000)])
    no_noise = dp_fedavg(u, bound=1.0, noise_multiplier=0.0, gen=torch.Generator().manual_seed(0))
    assert no_noise.update.norm() <= 0.5 + 1e-5  # (clipped norm 1 + 0) / 2
    noisy = dp_fedavg(u, bound=1.0, noise_multiplier=2.0, gen=torch.Generator().manual_seed(0))
    noise_std = (noisy.update - no_noise.update).std().item()
    assert abs(noise_std - 2.0 * 1.0 / 2) < 0.05


def test_dp_fedavg_requires_fixed_clip():
    with pytest.raises(ValueError):
        Aggregator(AggregatorConfig("dp_fedavg", noise_multiplier=1.0))


# --------------------------------------------------------------------------- membership inference


def test_mia_metrics_extremes():
    perfect = mia_metrics(np.ones(100), np.zeros(100))
    assert perfect["auc"] == 1.0 and perfect["tpr_at_1fpr"] == 1.0
    rng = np.random.default_rng(0)
    chance = mia_metrics(rng.normal(size=5000), rng.normal(size=5000))
    assert abs(chance["auc"] - 0.5) < 0.03


def test_scores_rank_confident_correct_predictions_higher():
    logits = np.array([[5.0, 0.0], [0.5, 0.0], [0.0, 5.0]])
    y = np.array([0, 0, 0])
    for f in (loss_scores, entropy_scores, scaled_logit):
        s = f(logits, y)
        assert s[0] > s[1] > s[2]


def test_scaled_logit_is_stable_for_extreme_logits():
    logits = np.array([[1000.0, 0.0, -1000.0]])
    assert np.isfinite(scaled_logit(logits, np.array([0])))[0]


def test_lira_normalises_per_example_difficulty():
    # Example 0 is always easy (high phi everywhere): no membership signal.
    # Example 1 is usually hard but the target model is unusually confident: member-like.
    shadows = [np.array([[4.0, 0.0], [0.0 + 0.1 * k, 0.0]]) for k in range(8)]
    target = np.array([[4.0, 0.0], [3.0, 0.0]])
    s = lira_offline_scores(target, np.array([0, 0]), shadows)
    assert s[1] > s[0]
