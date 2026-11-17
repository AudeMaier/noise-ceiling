"""LOOP: leave-one-out prediction.

One trial is predicted from all the others with a ridge encoding model, and the
held-out correlation is the noise ceiling estimate.

The recording is split by run: each run is held out in turn as the test set and
the model is fitted on the others. Every run, and every segment used to choose
the ridge parameter, is preprocessed on its own, so that no test value enters
the preprocessing of the data the model is fitted on. A high-pass fitted on the
whole recording before splitting couples the test samples to the training ones
(their projections on the removed cosines are tied), and a ridge that fits the
training target closely inherits that coupling as a spurious test correlation.

The ridge parameter is chosen per unit by leave-one-segment-out within the
training runs, on a grid relative to the Frobenius norm of the standardised
training predictors, sqrt(n_samples * n_features). It follows the scale of the
singular values as the number of samples and predictors changes, without an SVD.

Two predictor sets are available (`variant`):

full      every unit of every other trial, (n-1) * n_units predictors. It can
          recover the held-out trial's weights from the span of the others.
average   every unit of the average of the other trials, n_units predictors,
          better conditioned but unable to weight the trials individually.
"""

from __future__ import annotations

import numpy as np
import torch

from .ridge_GPU.ridge_regression import ridge_corr_torch

DEFAULT_ALPHA_FRACTIONS = np.logspace(-4, 2, 13)


def run_slices(n_samples, run_lengths=None):
    """One slice per run; two halves if run_lengths is None."""
    if run_lengths is None:
        run_lengths = [n_samples // 2, n_samples - n_samples // 2]
    if sum(run_lengths) != n_samples:
        raise ValueError(f"run_lengths sum to {sum(run_lengths)}, expected {n_samples}")
    bounds = np.cumsum([0] + list(run_lengths))
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def run_folds(data, run_lengths=None, preprocess=None, n_segments=4):
    """Cut a recording into runs, and every run into contiguous segments.

    data        : (..., n_samples).
    run_lengths : lengths of the runs, summing to n_samples; two halves if None.
    preprocess  : applied to every run and every segment on its own, (..., T) to
                  (..., T). None leaves the data unchanged.

    Returns the list of preprocessed runs and, for each run, the list of its
    preprocessed segments.
    """
    preprocess = preprocess or (lambda x: x)
    raw_runs = [data[..., run] for run in run_slices(data.shape[-1], run_lengths)]
    runs = [preprocess(run) for run in raw_runs]
    segments = [[preprocess(segment) for segment in np.array_split(run, n_segments, axis=-1)]
                for run in raw_runs]
    return runs, segments


def circular_shift(data, run_lengths=None, rng=None, min_shift=0.1):
    """Null data: every trial shifted circularly in time, independently within every run.

    The shifts keep the autocorrelation of each trial but break the alignment
    across trials, so a leak-free LOOP gives zero. data : (n_trials, ..., n_samples).
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty_like(data)
    for run in run_slices(data.shape[-1], run_lengths):
        length = run.stop - run.start
        low = max(1, int(min_shift * length))
        for trial in range(data.shape[0]):
            out[trial, ..., run] = np.roll(data[trial, ..., run],
                                           rng.integers(low, length - low + 1), axis=-1)
    return out


def _design(data, index, variant):
    """Predictors (n_samples, n_features) and target (n_samples, n_units)."""
    others = np.delete(data, index, axis=0)
    if variant == "average":
        variables = others.mean(axis=0).T
    else:
        variables = others.reshape(-1, data.shape[-1]).T
    return variables, data[index].T


def _standardize(train, *others, eps=1e-8):
    """Z-score every array using the training statistics only."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return [(a - mu) / sd for a in (train,) + others]


def _test_scores(train, test, index, alphas, variant, device):
    """Test correlation of every unit for every ridge parameter, (n_alphas, n_units)."""
    X_train, y_train = _design(train, index, variant)
    X_test, y_test = _design(test, index, variant)
    X_train, X_test = _standardize(X_train, X_test)
    y_train, y_test = _standardize(y_train, y_test)

    def to_tensor(array):
        return torch.tensor(np.ascontiguousarray(array), dtype=torch.float64, device=device)

    return ridge_corr_torch(to_tensor(X_train), to_tensor(X_test), to_tensor(y_train),
                            to_tensor(y_test), alphas).cpu().numpy()


def _alpha_scale(data, index, variant):
    """Frobenius norm of the standardised predictors: every column has unit variance."""
    n_features = data.shape[1] if variant == "average" else (data.shape[0] - 1) * data.shape[1]
    return np.sqrt(data.shape[-1] * n_features)


def loop_estimate(runs, segments, index=0, alpha_fractions=DEFAULT_ALPHA_FRACTIONS,
                  variant="full", device="cpu", return_alphas=False):
    """LOOP estimate of one trial, every run held out in turn.

    runs     : list of (n_trials, n_units, T_r), each preprocessed on its own.
    segments : for every run, the list of its segments, each preprocessed on its
               own, as returned by `run_folds`.

    For every test run the model is fitted on the other runs, with the ridge
    parameter chosen per unit by leave-one-segment-out within them, and scored
    on the test run. Returns the test correlation per unit averaged over the
    test runs, and the selected ridge parameters (n_runs, n_units) if
    `return_alphas`.
    """
    if variant not in ("full", "average"):
        raise ValueError(f"unknown variant {variant!r}; expected 'full' or 'average'")

    per_run, chosen = [], []
    for test in range(len(runs)):
        train = np.concatenate([run for r, run in enumerate(runs) if r != test], axis=-1)
        alphas = _alpha_scale(train, index, variant) * np.asarray(alpha_fractions)

        folds = [segment for r, run_segments in enumerate(segments) if r != test
                 for segment in run_segments]
        validation = 0
        for k in range(len(folds)):
            fit = np.concatenate([f for j, f in enumerate(folds) if j != k], axis=-1)
            validation = validation + _test_scores(fit, folds[k], index, alphas,
                                                   variant, device) / len(folds)
        best = np.argmax(validation, axis=0)

        scores = _test_scores(train, runs[test], index, alphas, variant, device)
        per_run.append(scores[best, np.arange(scores.shape[1])])
        chosen.append(alphas[best])

    out = np.mean(per_run, axis=0)
    return (out, np.array(chosen)) if return_alphas else out
