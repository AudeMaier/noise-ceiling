"""Closed-form expectations of the noise ceiling estimators.

Each function mirrors one estimator in `estimators.empirical.estimators` and
returns its expected value under the generative model, averaged over units.

Common arguments
----------------
sigma_s2, mean_s : variance and mean of a single signal.
sigma_2          : variance of the additive noise.
M_s, mu_s        : sum over shared sources of E[w^2] and of E[w]^2.
M_i, mu_i        : the same sums over the individual sources.
E_s              : sum over shared sources of E[w] w_0, i.e. the overlap between
                   a trial's weights and those of the held-out trial.
n                : number of trials.
W0               : weights of the held-out trial, (X, Y, K).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gamma as gamma_fn


def estimate_R_phi(phi, max_lag=None):
    """Autocovariance of the signals `phi` (K, T), averaged over signals."""
    _, T = phi.shape
    if max_lag is None:
        max_lag = T - 1
    centered = phi - np.mean(phi, axis=1, keepdims=True)
    full_corr = fftconvolve(centered, centered[:, ::-1], mode="full", axes=1)
    return np.mean(full_corr[:, T - 1:] / T, axis=0)[:max_lag + 1]


def compute_sigma_r2(M_i, M_s, sigma_s2, sigma_2, sigma_corr2, T):
    """Bartlett variance factor of the sample correlation. Depends on T."""
    weight_power = M_i + M_s
    total_var = weight_power * sigma_s2 + sigma_2
    return 1 + 2 * (weight_power * sigma_s2 / total_var) ** 2 * sigma_corr2


def noise_ceiling_mean_th(sigma_s2, sigma_2, M_s, mu_s, M_i, n):
    """NC^mean."""
    return np.sqrt(sigma_s2 * (M_s + (n - 1) * mu_s)
                   / (sigma_s2 * (M_s + (n - 1) * mu_s + M_i) + sigma_2)).mean()


def noise_ceiling_single_th(sigma_s2, sigma_2, W0, shared_number):
    """NC^single."""
    return np.sqrt(sigma_s2 * (W0[:, :, :shared_number] ** 2).sum(axis=-1)
                   / (sigma_s2 * (W0 ** 2).sum(axis=-1) + sigma_2)).mean()


def split_half_correlation_th(sigma_s2, sigma_2, M_s, mu_s, M_i, n):
    """SH."""
    return ((sigma_s2 * n / 2 * mu_s)
            / (sigma_s2 * (M_s + (n / 2 - 1) * mu_s + M_i) + sigma_2)).mean()


def spearman_brown_noise_ceiling_th(sigma_s2, sigma_2, sigma_r2, M_s, mu_s, M_i,
                                    n, T, individual_coords):
    """SHSB, with the rectification correction for units carrying no shared signal."""
    if individual_coords.mean() == 0:
        correction = 0
    else:
        correction = (individual_coords.mean()
                      * ((sigma_r2 * individual_coords).mean()
                         / individual_coords.mean() / (2 * T)) ** 0.25
                      * 1 / np.sqrt(np.pi) * gamma_fn(0.75))
    return np.sqrt((sigma_s2 * n * mu_s)
                   / (sigma_s2 * (M_s + (n - 1) * mu_s + M_i) + sigma_2)).mean() + correction


def leave_one_out_correlation_th(sigma_s2, sigma_2, W0, M_s, mu_s, M_i, E_s, n):
    """LOOC."""
    return ((sigma_s2 * E_s)
            / np.sqrt((sigma_s2 * (M_s + (n - 2) * mu_s + M_i) + sigma_2) / (n - 1)
                      * (sigma_s2 * (W0 ** 2).sum(axis=-1) + sigma_2))).mean()


def mean_to_one_correlation_th(sigma_s2, sigma_2, W0, M_s, mu_s, M_i, E_s, n,
                               shared_number):
    """TTMC."""
    return (((sigma_s2 * (W0 ** 2).sum(axis=-1) + sigma_2) / n
             + sigma_s2 * E_s * (n - 1) / n)
            / np.sqrt((sigma_s2 * (M_s + (n - 1) * mu_s + M_i) + sigma_2) / n
                      * (sigma_s2 * (W0 ** 2).sum(axis=-1) + sigma_2))).mean()


def allen_noise_ceiling_th(sigma_s2, mean_s, sigma_2, sigma_r2, M_s, mu_s, M_i,
                           mu_i, n, T, individual_coords):
    """SNR, with the rectification correction for units carrying no shared signal."""
    if individual_coords.mean() == 0:
        correction = 0
    else:
        correction = (individual_coords.mean()
                      * ((sigma_r2 * individual_coords).mean()
                         / individual_coords.mean() / T) ** 0.25
                      * 1 / np.sqrt(2 * np.pi) * gamma_fn(0.75))
    sigma_total = (sigma_s2 * (M_s + (n - 1) * mu_s + M_i) + sigma_2) / n
    sigma_noise = ((mean_s ** 2 + sigma_s2) * (M_s - mu_s + M_i)
                   - mean_s ** 2 * mu_i + sigma_2)
    rectified = np.maximum(sigma_total - sigma_noise / n, np.zeros_like(sigma_total))
    return np.sqrt(rectified * (1 - individual_coords) / sigma_total).mean() + correction
