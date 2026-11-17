"""Empirical noise ceiling estimators.

Every estimator takes an array whose first axis indexes trials and whose last
axis is the one correlations are computed across (time points for a continuous
stimulus, stimuli for a discrete one), and returns one value per unit.
"""

from __future__ import annotations

import random
from math import comb

import numpy as np


def pearson_corr(data1, data2):
    """Correlation between two arrays along their last axis."""
    data1 = data1 - data1.mean(axis=-1, keepdims=True)
    data2 = data2 - data2.mean(axis=-1, keepdims=True)
    numerator = (data1 * data2).sum(axis=-1)
    denominator = (np.sqrt((data1 ** 2).sum(axis=-1))
                   * np.sqrt((data2 ** 2).sum(axis=-1)) + 1e-8)
    return numerator / denominator


def true_noise_ceiling_mean(data, noiseless_data):
    """NC^mean: correlation between the trial mean and its noiseless counterpart."""
    return pearson_corr(noiseless_data.mean(axis=0), data.mean(axis=0))


def true_noise_ceiling_single(data, noiseless_data, index=0):
    """NC^single: correlation between one trial and its noiseless counterpart."""
    return pearson_corr(noiseless_data[index], data[index])


def random_half_splits(n_trials, n_splits=10):
    """Random splits of the trials, each given by the indices of its first half."""
    return [np.random.permutation(n_trials)[:n_trials // 2] for _ in range(n_splits)]


def split_half_correlation(data, correction=False, n_splits=10, splits=None):
    """SH: correlation between the means of two halves of the trials, averaged
    over splits.

    correction=True gives SHSB: the Spearman-Brown correction 2r / (1 + r) is
    applied to every split, averaged, then rectified and square-rooted.
    splits : indices of the first half of every split, as returned by
             `random_half_splits`. Drawn at random if None.
    """
    if splits is None:
        splits = random_half_splits(data.shape[0], n_splits)

    values = []
    for first_half in splits:
        in_first = np.zeros(data.shape[0], dtype=bool)
        in_first[first_half] = True
        r = pearson_corr(data[in_first].mean(axis=0), data[~in_first].mean(axis=0))
        values.append(2 * r / (1 + r) if correction else r)
    value = np.mean(values, axis=0)

    if correction:
        value = np.sqrt(np.maximum(value, 0))

    return value


def leave_one_out_correlation(data, index=0):
    """LOOC: correlation between trial `index` and the mean of the others."""
    return pearson_corr(np.delete(data, index, axis=0).mean(axis=0), data[index])


def mean_to_one_correlation(data, index=0):
    """TTMC: correlation between trial `index` and the mean of all trials."""
    return pearson_corr(data.mean(axis=0), data[index])


def allen_noise_ceiling(data):
    """ATV: noise ceiling from the signal-to-noise ratio (Allen et al.).

    Every trial is centred over time first, otherwise offsets between trials
    would count as noise. The variance over time uses 1/T and the variance
    across trials 1/(n - 1). The estimate is 0 where the mean has no variance.
    """
    n_trials = data.shape[0]
    data = data - data.mean(axis=-1, keepdims=True)
    sigma_total = data.mean(axis=0).var(axis=-1)
    sigma_noise = data.var(axis=0, ddof=1).mean(axis=-1)

    ratio = np.divide(sigma_total - sigma_noise / n_trials, sigma_total,
                      out=np.zeros_like(sigma_total), where=sigma_total > 0)
    return np.sqrt(np.maximum(ratio, 0))


def get_unique_subsets(data, subset_length, n_bootstraps):
    """Draw `n_bootstraps` distinct subsets of `subset_length` entries."""
    n = len(data)
    max_possible = comb(n, subset_length)
    if n_bootstraps > max_possible:
        print(f"Warning: only {max_possible} unique subsets possible, returning all.")
        n_bootstraps = max_possible

    seen = set()
    subsets = []
    while len(subsets) < n_bootstraps:
        indices = tuple(sorted(random.sample(range(n), subset_length)))
        if indices not in seen:
            seen.add(indices)
            subsets.append([data[i] for i in indices])

    return subsets
