"""Selecting the units that carry signal.

The split-half correlation of a unit is measured over subsets of increasing
size. A unit carrying signal has a correlation that grows with the number of
trials as a n / (n + c), whereas a unit carrying none has a flat curve. The two
models are compared with AICc, and units whose bootstrapped Delta AIC exceeds a
threshold are kept.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from .estimators.empirical.estimators import split_half_correlation


def _saturating(n, amplitude, half_point):
    return amplitude * n / (n + half_point + 1e-9)


def generate_bootstrap_samples(data, n_samples=100, n_list=range(2, 16, 2)):
    """Split-half correlation over random subsets of each size in `n_list`.

    Returns an array of shape (len(n_list), n_units, n_samples).
    """
    n_list = list(n_list)
    curves = np.zeros((len(n_list), data.shape[1], n_samples))

    for i, n in enumerate(tqdm(n_list, desc="subset size")):
        subsets = list(combinations(range(len(data)), n))
        replace = len(subsets) <= n_samples
        chosen = np.random.choice(np.arange(len(subsets)), size=n_samples,
                                  replace=replace)
        for j, index in enumerate(chosen):
            curves[i, :, j] = split_half_correlation(data[list(subsets[index])],
                                                     correction=False, n_splits=1)

    return curves


def select_curves(curves, n_list=range(2, 16, 2), n_bootstrap=100, aic_threshold=10):
    """Compare the saturating and flat models on each unit's curve.

    Returns the mask of units whose mean Delta AIC exceeds `aic_threshold`, and
    the bootstrap distribution of Delta AIC, of shape (n_units, n_bootstrap).
    Positive values favour the saturating model, i.e. the presence of signal.
    """
    n_values = np.asarray(list(n_list), dtype=float)
    n_pts = len(n_values)
    k_saturating, k_flat = 2, 1

    penalty_saturating = (2 * k_saturating
                          + (2 * k_saturating * (k_saturating + 1))
                          / (n_pts - k_saturating - 1))
    penalty_flat = (2 * k_flat + (2 * k_flat * (k_flat + 1)) / (n_pts - k_flat - 1))

    aic_dist = np.zeros((curves.shape[1], curves.shape[2]))

    for i, curve in enumerate(tqdm(curves.transpose(1, 0, 2), desc="units")):
        for bootstrap in range(n_bootstrap):
            resampled = np.random.choice(range(curve.shape[1]), size=curve.shape[1],
                                         replace=True)
            y_obs = curve[:, resampled].mean(axis=1)

            try:
                popt, _ = curve_fit(_saturating, n_values, y_obs, p0=[0.5, 5.0],
                                    bounds=([0, 0], [1.0, np.inf]))
                rss = max(np.sum((y_obs - _saturating(n_values, *popt)) ** 2), 1e-10)
                aic_saturating = n_pts * np.log(rss / n_pts) + penalty_saturating

                rss_flat = max(np.sum((y_obs - y_obs.mean()) ** 2), 1e-10)
                aic_flat = n_pts * np.log(rss_flat / n_pts) + penalty_flat

                aic_dist[i, bootstrap] = aic_flat - aic_saturating
            except (RuntimeError, ValueError):
                aic_dist[i, bootstrap] = -10.0

    return aic_dist.mean(axis=1) > aic_threshold, aic_dist


def run_aic_procedure(data, n_list=range(2, 16, 2), n_bootstrap=100, n_samples=100,
                      aic_threshold=10):
    """Full selection procedure: bootstrap the curves, then compare the models."""
    curves = generate_bootstrap_samples(data, n_samples=n_samples, n_list=n_list)
    return select_curves(curves, n_list=n_list, n_bootstrap=n_bootstrap,
                         aic_threshold=aic_threshold)
