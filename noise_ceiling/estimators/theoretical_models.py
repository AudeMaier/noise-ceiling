"""Closed-form leading terms of the noise ceiling estimators.

Each function mirrors one estimator in `estimators.empirical.estimators` and
returns, for every unit, the leading term of its large-T expansion. Everything
is conditional on the trials actually used: the weights are those of these
trials, not population moments. The error is O_P(T^-1/2), or O_P(T^-1/4) for
ATV and SHSB at units where the argument of their square root is near zero.

Common arguments
----------------
W         : (n, ..., K) weights of the noise-free signal of every trial on
            centred, orthonormal latent signals, so that the time covariance of
            two noise-free signals is the dot product of their weights.
noise_var : (n, ...) noise variance sigma_l^2 of every trial, averaged over time.
index     : the trial held out by the single-trial estimators.
splits    : list of index arrays A, each defining the split (A, [n] minus A).
"""

from __future__ import annotations

import numpy as np


def signal_var(w):
    """Var_t of a noise-free signal with weights w, sum_k w_k^2."""
    return (w ** 2).sum(axis=-1)


def signal_cov(w_a, w_b):
    """Cov_t of two noise-free signals with weights w_a and w_b."""
    return (w_a * w_b).sum(axis=-1)


def weight_var(W):
    """sum_k Var_l[w_k], the variance of the weights across trials (1/n)."""
    return W.var(axis=0).sum(axis=-1)


def noise_ceiling_mean_th(W, noise_var):
    """NC^mean."""
    n = W.shape[0]
    signal = signal_var(W.mean(axis=0))
    return np.sqrt(signal / (signal + noise_var.mean(axis=0) / n))


def allen_noise_ceiling_th(W, noise_var):
    """ATV. The variance of the weights across trials is counted as noise."""
    n = W.shape[0]
    signal = signal_var(W.mean(axis=0))
    ratio = ((signal - weight_var(W) / (n - 1))
             / (signal + noise_var.mean(axis=0) / n))
    return np.sqrt(np.maximum(ratio, 0))


def _half_averages(W, noise_var, A):
    """Cov_t of the two half averages of the split (A, [n] minus A), and their Var_t."""
    in_A = np.zeros(W.shape[0], dtype=bool)
    in_A[A] = True
    w_A, w_B = W[in_A].mean(axis=0), W[~in_A].mean(axis=0)
    var_A = signal_var(w_A) + noise_var[in_A].sum(axis=0) / in_A.sum() ** 2
    var_B = signal_var(w_B) + noise_var[~in_A].sum(axis=0) / (~in_A).sum() ** 2
    return signal_cov(w_A, w_B), var_A, var_B


def split_half_correlation_th(W, noise_var, splits):
    """SH: correlation of the two half averages, averaged over the splits."""
    correlations = []
    for A in splits:
        cov, var_A, var_B = _half_averages(W, noise_var, A)
        correlations.append(cov / np.sqrt(var_A * var_B))
    return np.mean(correlations, axis=0)


def spearman_brown_noise_ceiling_th(W, noise_var, splits):
    """SHSB: the Spearman-Brown correction is averaged over the splits before
    the rectified square root.

    For halves of equal size, 2 Cov / (Cov + sqrt(Var_A Var_B)) is
    (sum_k wbar_k^2 - Delta^w_AB) / (sum_k wbar_k^2 + sigmabar^2 / n - Delta^sd_AB).
    """
    corrected = []
    for A in splits:
        cov, var_A, var_B = _half_averages(W, noise_var, A)
        corrected.append(2 * cov / (cov + np.sqrt(var_A * var_B)))
    return np.sqrt(np.maximum(np.mean(corrected, axis=0), 0))


def noise_ceiling_single_th(W, noise_var, index=0):
    """NC^single. Depends on trial `index` alone, not on n."""
    signal = signal_var(W[index])
    return np.sqrt(signal / (signal + noise_var[index]))


def leave_one_out_correlation_th(W, noise_var, index=0):
    """LOOC."""
    others = np.delete(np.arange(W.shape[0]), index)
    w_others = W[others].mean(axis=0)
    var_one = signal_var(W[index]) + noise_var[index]
    var_others = signal_var(w_others) + noise_var[others].mean(axis=0) / len(others)
    return signal_cov(W[index], w_others) / np.sqrt(var_one * var_others)


def mean_to_one_correlation_th(W, noise_var, index=0):
    """TTMC. Trial `index` is part of the mean, so its noise is shared."""
    n = W.shape[0]
    w_bar = W.mean(axis=0)
    cov = signal_cov(W[index], w_bar) + noise_var[index] / n
    var_one = signal_var(W[index]) + noise_var[index]
    var_mean = signal_var(w_bar) + noise_var.mean(axis=0) / n
    return cov / np.sqrt(var_one * var_mean)
