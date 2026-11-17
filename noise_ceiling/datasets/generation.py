"""Generative model for the synthetic data.

A set of K latent sources is drawn, some shared across trials and some specific
to each trial. Every unit of a 2-D grid mixes the sources with Gaussian weights
centred on the source positions, and Gaussian noise is added on top.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma


def generate_neural(num_signals, T_total, lambda_=0.1, mu=5.0):
    """Block-design neural sources, before any hemodynamic convolution.

    Events arrive as a Poisson process with rate `lambda_` and last for an
    exponentially distributed duration with mean `mu`.

    Returns (num_signals, T_total), not mean-centred.
    """
    spikes = np.random.poisson(lambda_, (num_signals, T_total))
    underlying = np.zeros((num_signals, T_total + 1))

    rows, cols = np.where(spikes > 0)
    if len(rows) > 0:
        durations = np.random.exponential(mu, size=len(rows)).astype(int)
        end_indices = np.minimum(cols + durations, T_total)
        np.add.at(underlying, (rows, cols), spikes[rows, cols])
        np.add.at(underlying, (rows, end_indices), -spikes[rows, cols])
        underlying = np.cumsum(underlying, axis=1)[:, :T_total]

    return underlying[:, :T_total]


def hrf_kernel(tr=1.5, a1=6, a2=16, b1=1, b2=1, c=1.6, duration=30.0):
    """Double-gamma HRF sampled at `tr`."""
    t_hrf = np.arange(0, duration, tr)
    kernel = gamma.pdf(t_hrf, a=a1, scale=b1) - gamma.pdf(t_hrf, a=a2, scale=b2) / c
    return kernel


def standardize(signals, eps=1e-12):
    """Zero mean and unit variance along time, independently for each signal.

    A flat signal (no event reaching the window) has no variance to rescale and
    is returned as zeros.
    """
    centered = signals - signals.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), eps)


def generate_signals(num_signals, T, lambda_=0.1, mu=5.0, tr=1.5,
                     a1=6, a2=16, b1=1, b2=1, c=1.6, crop=None):
    """Block-design sources convolved with a double-gamma HRF.

    crop : (pad_start, pad_end) samples generated then discarded, so that the
           returned signals are free of convolution edge effects.

    Returns (num_signals, T) signals with zero mean and unit variance along time.
    """
    pad_start = crop[0] if crop is not None else 0
    pad_end = crop[1] if crop is not None else 0
    T_total = T + pad_start + pad_end

    underlying = generate_neural(num_signals, T_total, lambda_=lambda_, mu=mu)
    hrf = hrf_kernel(tr=tr, a1=a1, a2=a2, b1=b1, b2=b2, c=c)
    bold = fftconvolve(underlying, hrf[np.newaxis, :], mode="full", axes=1)[:, :T_total]

    bold = bold[:, pad_start:pad_start + T]

    return standardize(bold)


def generate_gaussian_signals(num_signals, T):
    """White Gaussian sources, the temporally uncorrelated alternative.

    Standardised along time like `generate_signals`.
    """
    return standardize(np.random.normal(0, 1, (num_signals, T)))


def unit_noise_sd(sigma, sigma_spread, X, Y):
    """Noise standard deviation of every unit, fixed across trials.

    sigma_u = sigma * exp(sigma_spread * Z_u) with Z_u standard normal, so that
    sigma is the median over units and sigma_spread the standard deviation of
    log sigma_u. sigma_spread=0 gives the same sigma to every unit.

    Returns (X, Y).
    """
    if sigma_spread == 0:
        return np.full((X, Y), float(sigma))
    return sigma * np.exp(sigma_spread * np.random.normal(0, 1, (X, Y)))


def generate_origins(K, X, Y):
    """Uniformly distributed source positions on the grid."""
    assert X == Y, "Currently only supports square grids"
    return np.random.rand(K, 2) * X


def generate_weights(base_origins, K, X, Y, n, sigma_origin=0.5, width=1.0,
                     within_subject=True):
    """Gaussian mixing weights on a periodic grid, normalised per source.

    within_subject=True gives every trial the same source positions. Otherwise
    each trial's positions are jittered by up to `sigma_origin`, which is what
    makes the shared signal only partially shared across trials.

    Returns W of shape (n, X, Y, K) and the positions used.
    """
    if within_subject:
        all_origins = np.repeat(base_origins[np.newaxis, :, :], n, axis=0)
    else:
        shifts = np.random.uniform(-sigma_origin, sigma_origin, (n, K, 2))
        all_origins = (base_origins + shifts) % X

    x_coords = np.arange(X) + 0.5
    y_coords = np.arange(Y) + 0.5
    ox, oy = all_origins[:, :, 0], all_origins[:, :, 1]

    dx = np.abs(x_coords[None, :, None] - ox[:, None, :])
    dx = np.minimum(dx, X - dx)
    dy = np.abs(y_coords[None, :, None] - oy[:, None, :])
    dy = np.minimum(dy, Y - dy)

    dist_sq = (dx * dx)[:, :, None, :] + (dy * dy)[:, None, :, :]
    W = np.exp(-dist_sq / (2 * width ** 2))
    W /= np.maximum(W.sum(axis=(1, 2), keepdims=True), 1e-8)

    return W, all_origins


def generate_voxel_grids(X, Y, T, K, lambda_=0.1, mu=5.0, tr=1.5, a1=6, a2=16,
                         b1=1, b2=1, c=1.6, n=1, sigma=0.5,
                         shared_fraction=0.5, crop=None, sigma_origin=1,
                         width=1.0, within_subject=True, gaussian_signals=False,
                         sigma_spread=0.0):
    """One realisation of the generative model.

    n                : number of trials.
    K                : number of latent sources, of which `shared_fraction` are
                       common to all trials.
    sigma            : standard deviation of the additive noise, the median
                       over units when sigma_spread > 0.
    sigma_spread     : standard deviation of log sigma_u across units, see
                       `unit_noise_sd`. Each unit keeps its sigma_u in every trial.

    The noise-free signal is the part driven by the shared sources. The
    individual sources are independent across trials, so they belong to the
    noise, whose variance per trial and unit is
    noise_var = sigma_u^2 + sum over individual sources of w^2.

    Returns a dict with the noisy grids (n, X, Y, T), the noise-free grids, the
    sources, the weights and the noise variance (n, X, Y).
    """
    num_shared = int(shared_fraction * K)
    signals = np.zeros((n, K, T))
    signal_kwargs = dict(lambda_=lambda_, mu=mu, tr=tr, a1=a1, a2=a2, b1=b1,
                         b2=b2, c=c, crop=crop)

    if num_shared > 0:
        if gaussian_signals:
            shared_signals = generate_gaussian_signals(num_shared, T)
        else:
            shared_signals = generate_signals(num_shared, T, **signal_kwargs)
    else:
        shared_signals = np.zeros((0, T))

    if K - num_shared > 0:
        if gaussian_signals:
            individual_signals = generate_gaussian_signals(
                n * (K - num_shared), T).reshape(n, K - num_shared, T)
        else:
            individual_signals = np.array(generate_signals(
                n * (K - num_shared), T, **signal_kwargs)).reshape(n, K - num_shared, T)
    else:
        individual_signals = np.zeros((n, 0, T))

    for trial in range(n):
        signals[trial] = np.concatenate((shared_signals, individual_signals[trial]),
                                        axis=0)

    base_origins = generate_origins(K, X, Y)
    W, all_origins = generate_weights(base_origins, K, X, Y, n,
                                      sigma_origin=sigma_origin, width=width,
                                      within_subject=within_subject)

    voxel_grids = np.einsum("nxyk,nkt->nxyt", W, signals)
    noiseless_grids = np.einsum("nxyk,nkt->nxyt", W[:, :, :, :num_shared],
                                signals[:, :num_shared])
    sigma_u = unit_noise_sd(sigma, sigma_spread, X, Y)
    noise_var = sigma_u ** 2 + (W[:, :, :, num_shared:] ** 2).sum(axis=-1)

    if sigma > 0:
        voxel_grids += sigma_u[..., None] * np.random.normal(0, 1, voxel_grids.shape)

    return {
        "voxel_grids": voxel_grids,
        "noiseless_grids": noiseless_grids,
        "signals": signals,
        "W": W,
        "all_origins": all_origins,
        "noise_var": noise_var,
    }
