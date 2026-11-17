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
from noise_ceiling.estimators.empirical.loop import (circular_shift, loop_estimate,
                                                    run_folds)

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


def test_run_folds_cover_every_sample():
    data = np.arange(2 * 3 * 100, dtype=float).reshape(2, 3, 100)
    runs, segments = run_folds(data, run_lengths=[40, 60], n_segments=4)
    assert [r.shape[-1] for r in runs] == [40, 60]
    assert np.array_equal(np.concatenate(runs, -1), data)
    assert np.array_equal(np.concatenate(segments[1], -1), runs[1])


def _shared_signal_trials(rng, n=6, units=8, T=400, noise=1.0):
    signal = rng.normal(size=(1, units, T))
    return signal + noise * rng.normal(size=(n, units, T))


def test_loop_recovers_a_shared_signal():
    data = _shared_signal_trials(np.random.default_rng(0))
    runs, segments = run_folds(data)
    assert loop_estimate(runs, segments).mean() > 0.4


def test_loop_is_null_on_shifted_trials():
    rng = np.random.default_rng(1)
    data = circular_shift(_shared_signal_trials(rng), rng=rng)
    runs, segments = run_folds(data)
    assert abs(loop_estimate(runs, segments).mean()) < 0.05


def test_synthetic_estimates_are_finite_and_bounded():
    np.random.seed(0)
    dataset = SyntheticDataset(**SMALL)
    empirical = dataset.estimate(n=6, index=0)
    theory = dataset.theory(6)
    for key in ALL_ESTIMATORS:
        assert np.isfinite(empirical[key]) and -1 <= empirical[key] <= 1.2
        assert np.isfinite(theory[key])


def test_sweep_shapes():
    np.random.seed(0)
    dataset = SyntheticDataset(**SMALL)
    result = dataset.sweep_n([2, 4, 6], n_rep=2)
    for key in ALL_ESTIMATORS:
        assert result["empirical"][key].shape == (3,)
        assert result["theory"][key].shape == (3,)


def test_ridge_score_is_the_pearson_correlation_of_the_prediction():
    import torch
    from noise_ceiling.estimators.empirical.ridge_GPU.ridge_regression import ridge_corr_torch

    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(80, 10)), rng.normal(size=(30, 10))
    weights = rng.normal(size=(10, 3))
    y_train = X_train @ weights + rng.normal(size=(80, 3))
    y_test = X_test @ weights + rng.normal(size=(30, 3))
    alpha = 2.0

    # ridge with penalty alpha^2, the convention of the SVD solver
    beta = np.linalg.solve(X_train.T @ X_train + alpha ** 2 * np.eye(10), X_train.T @ y_train)
    expected = pearson_corr((X_test @ beta).T, y_test.T)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64)

    scores = ridge_corr_torch(tensor(X_train), tensor(X_test), tensor(y_train),
                              tensor(y_test), [alpha]).numpy()[0]
    assert np.allclose(scores, expected, atol=1e-6)
