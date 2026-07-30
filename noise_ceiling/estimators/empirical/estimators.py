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


def split_half_correlation(data, correction=False, n_splits=10):
    """SH: correlation between the means of two random halves of the trials.

    correction=True applies the Spearman-Brown correction (SHSB), after
    clipping negative correlations to zero.
    """
    data = data.copy()
    correlation_maps = []
    for _ in range(n_splits):
        np.random.shuffle(data)
        half = data.shape[0] // 2
        correlation_maps.append(pearson_corr(data[:half].mean(axis=0),
                                             data[half:].mean(axis=0)))
    correlation_map = np.array(correlation_maps).mean(axis=0)

    if correction:
        correlation_map[correlation_map < 0] = 0
        correlation_map = np.sqrt((correlation_map * 2) / (1 + correlation_map))

    return correlation_map


def leave_one_out_correlation(data, index=0):
    """LOOC: correlation between trial `index` and the mean of the others."""
    return pearson_corr(np.delete(data, index, axis=0).mean(axis=0), data[index])


def mean_to_one_correlation(data, index=0):
    """TTMC: correlation between trial `index` and the mean of all trials."""
    return pearson_corr(data.mean(axis=0), data[index])


def allen_noise_ceiling(data):
    """SNR: noise ceiling from the signal-to-noise ratio (Allen et al.)."""
    n_trials = data.shape[0]
    sigma_total = data.mean(axis=0).var(axis=-1, ddof=1)
    sigma_noise = data.var(axis=0, ddof=1).mean(axis=-1)

    diff = sigma_total - sigma_noise / n_trials
    diff[diff < 0] = 0
    return np.sqrt(diff / sigma_total)


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
