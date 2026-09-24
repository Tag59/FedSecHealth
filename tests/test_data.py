import numpy as np
import pytest

from fedsechealth.data import dirichlet_partition, iid_partition, load_federated_breast_cancer


def test_iid_partition_covers_all_indices_once():
    y = np.arange(100) % 2
    parts = iid_partition(y, 3, np.random.default_rng(0))
    merged = np.concatenate(parts)
    assert sorted(merged.tolist()) == list(range(100))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_dirichlet_partition_is_disjoint_and_complete():
    y = np.repeat([0, 1], 200)
    parts = dirichlet_partition(y, 4, alpha=0.5, rng=np.random.default_rng(0))
    merged = np.concatenate(parts)
    assert len(merged) == len(set(merged.tolist())) == 400
    assert all(len(p) >= 10 for p in parts)


def test_dirichlet_small_alpha_is_skewed():
    y = np.repeat([0, 1], 500)
    parts = dirichlet_partition(y, 5, alpha=0.1, rng=np.random.default_rng(1), min_size=1)
    ratios = [y[p].mean() for p in parts]
    assert max(ratios) - min(ratios) > 0.5


def test_federated_standardization_matches_pooled_statistics():
    fds = load_federated_breast_cancer(n_clients=3, seed=0)
    pooled = np.concatenate([c.x for c in fds.clients])
    np.testing.assert_allclose(pooled.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(pooled.std(0), 1, atol=1e-3)
    raw = fds.unscale(pooled)
    assert raw[:, 0].min() > 0  # mean radius is positive in original units


def test_unknown_partition_raises():
    with pytest.raises(ValueError):
        load_federated_breast_cancer(partition="nope")
