"""Sanity checks. Run with `pytest`."""

import numpy as np
import pytest

from noise_ceiling.datasets.preprocessing import dct_basis, project_out
from noise_ceiling.datasets.synthetic import ALL_ESTIMATORS, SyntheticDataset
from noise_ceiling.estimators.empirical.estimators import (allen_noise_ceiling,
                                                           leave_one_out_correlation,
                                                           mean_to_one_correlation,
                                                           pearson_corr,
                                                           split_half_correlation)
from noise_ceiling.estimators.empirical.loop import make_split, purged_test_chunks

SMALL = dict(X=4, Y=4, K=40, T=400, n=6, crop=[20, 20], shared_fraction=0.4,
             sigma=0.3, width=1.0, sigma_origin=1.0, within_subject=False,
             gaussian_signals=True)


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert pearson_corr(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-6)


def test_noiseless_trials_give_a_ceiling_of_one():
    signal = np.random.default_rng(0).normal(size=(1, 5, 300))
    identical = np.repeat(signal, 8, axis=0)
    for estimator in (split_half_correlation, leave_one_out_correlation,
                      mean_to_one_correlation, allen_noise_ceiling):
        assert np.allclose(estimator(identical), 1.0, atol=1e-5)


def test_pure_noise_gives_a_ceiling_near_zero():
    noise = np.random.default_rng(0).normal(size=(20, 5, 4000))
    assert abs(split_half_correlation(noise).mean()) < 0.05


def test_dct_basis_is_orthonormal():
    Q = dct_basis(200, tr=1.5, cutoff_s=128.0)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


def test_projection_removes_a_linear_trend():
    # a cosine basis cannot represent a ramp exactly, so a little edge
    # curvature survives; what matters is that almost all the variance goes
    trend = np.linspace(0, 1, 300)[None, :]
    Q = dct_basis(300, tr=1.5, cutoff_s=128.0)
    assert project_out(trend, Q).var() / trend.var() < 1e-3


def test_purged_split_keeps_a_buffer():
    train, test = purged_test_chunks(2000, n_chunks=5, test_fraction=0.25,
                                     buffer_size=50)
    assert not set(train) & set(test)
    assert min(abs(t - v) for t in train for v in [test[0]]) > 0


def test_random_split_covers_every_sample():
    train, test = make_split(500, chunked=False, test_fraction=0.25)
    assert sorted(np.concatenate([train, test])) == list(range(500))


def test_synthetic_estimates_are_finite_and_bounded():
    np.random.seed(0)
    dataset = SyntheticDataset(**SMALL)
    empirical = dataset.estimate(n=6, index=0)
    theory = dataset.theory(6, realized=True)
    for key in ALL_ESTIMATORS:
        assert np.isfinite(empirical[key]) and -1 <= empirical[key] <= 1.2
        assert np.isfinite(theory[key])


def test_sweep_shapes():
    np.random.seed(0)
    dataset = SyntheticDataset(**SMALL)
    result = dataset.sweep_n([2, 4, 6], n_rep=2, realized=True)
    for key in ALL_ESTIMATORS:
        assert result["empirical"][key].shape == (3,)
        assert result["theory"][key].shape == (3,)
