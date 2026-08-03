from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma
from tqdm import tqdm


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


def hrf_kernel(tr=1.5, a1=6, a2=16, b1=1, b2=1, c=1.6, duration=30.0,
               normalize=False):
    """Double-gamma HRF sampled at `tr`.

    normalize=True rescales the kernel to unit L2 norm, so that changing its
    shape changes the timing and dispersion of the response without changing
    the variance it imparts. Used by the heterogeneous-HRF violation, where a
    shape-dependent gain would confound HRF variability with SNR.
    """
    t_hrf = np.arange(0, duration, tr)
    kernel = gamma.pdf(t_hrf, a=a1, scale=b1) - gamma.pdf(t_hrf, a=a2, scale=b2) / c
    if normalize:
        kernel = kernel / np.sqrt(np.sum(kernel ** 2))
    return kernel


def generate_signals(num_signals, T, lambda_=0.1, mu=5.0, tr=1.5,
                     a1=6, a2=16, b1=1, b2=1, c=1.6, crop=None):
    """Block-design sources convolved with a double-gamma HRF.

    crop : (pad_start, pad_end) samples generated then discarded, so that the
           returned signals are free of convolution edge effects.

    Returns (num_signals, T) mean-centred signals.
    """
    pad_start = crop[0] if crop is not None else 0
    pad_end = crop[1] if crop is not None else 0
    T_total = T + pad_start + pad_end

    underlying = generate_neural(num_signals, T_total, lambda_=lambda_, mu=mu)
    hrf = hrf_kernel(tr=tr, a1=a1, a2=a2, b1=b1, b2=b2, c=c)
    bold = fftconvolve(underlying, hrf[np.newaxis, :], mode="full", axes=1)[:, :T_total]

    if crop is not None:
        bold = bold[:, pad_start:-pad_end]

    return bold - bold.mean(axis=1, keepdims=True)


def generate_gaussian_signals(num_signals, T, sigma=np.sqrt(0.12)):
    """White Gaussian sources, the temporally uncorrelated alternative."""
    return np.random.normal(0, sigma, (num_signals, T))


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


def weight_moments(base_origins, W0, K, X, Y, shared_fraction, sigma_origin,
                   width, within_subject=True, n_iter=100):
    """Monte-Carlo weight moments needed by the closed forms.

    Each iteration draws 1000 fresh trials, which are then split into
    independent groups to estimate products of expectations: two groups of 500
    for E[w]^2, three of 333 for E[w^2] E[w]^2, four of 250 for E[w]^4.

    Returns per-unit sums over sources, of shape (X, Y).
    """
    K_shared = int(shared_fraction * K)
    W0_s = W0[:, :, :K_shared]

    keys = ["M_s", "mu_s", "M_i", "mu_i", "E_s",
            "M_s4", "M_s2_2", "M_s_4", "M_s3_mu", "M_s2_mu2"]
    moments = {key: np.zeros((X, Y)) for key in keys}

    for _ in tqdm(range(n_iter), leave=False):
        W, _ = generate_weights(base_origins, K, X=X, Y=Y, n=1000,
                                sigma_origin=sigma_origin, width=width,
                                within_subject=within_subject)
        W_s, W_i = W[:, :, :, :K_shared], W[:, :, :, K_shared:]
        W2_s = W_s * W_s

        moments["M_s"] += W2_s.sum(axis=-1).mean(axis=0)
        moments["mu_s"] += (W_s[:500] * W_s[500:]).sum(axis=-1).mean(axis=0)
        moments["M_i"] += (W_i * W_i).sum(axis=-1).mean(axis=0)
        moments["mu_i"] += (W_i[:500] * W_i[500:]).sum(axis=-1).mean(axis=0)
        moments["E_s"] += (W_s * W0_s).sum(axis=-1).mean(axis=0)
        moments["M_s4"] += (W2_s * W2_s).sum(axis=-1).mean(axis=0)
        moments["M_s2_2"] += (W2_s[:500] * W2_s[500:]).sum(axis=-1).mean(axis=0)
        moments["M_s_4"] += (W_s[:250] * W_s[250:500]
                             * W_s[500:750] * W_s[750:]).sum(axis=-1).mean(axis=0)
        moments["M_s3_mu"] += (W2_s[:500] * W_s[:500]
                               * W_s[500:]).sum(axis=-1).mean(axis=0)
        moments["M_s2_mu2"] += (W2_s[:333] * W_s[333:666]
                                * W_s[666:999]).sum(axis=-1).mean(axis=0)

    return {key: value / n_iter for key, value in moments.items()}


def generate_voxel_grids(X, Y, T, K, lambda_=0.1, mu=5.0, tr=1.5, a1=6, a2=16,
                         b1=1, b2=1, c=1.6, n=1, sigma=0.5, sigma_of_sigma=0,
                         shared_fraction=0.5, crop=None, sigma_origin=1,
                         width=1.0, within_subject=True,
                         partial_shared_fraction=0, gaussian_signals=False):
    """One realisation of the generative model.

    n                : number of trials.
    K                : number of latent sources, of which `shared_fraction` are
                       common to all trials.
    sigma            : standard deviation of the additive noise; `sigma_of_sigma`
                       makes it vary across units.
    partial_shared_fraction : fraction of the shared sources silenced in each
                       trial, so that they are shared by only some trials.

    Returns a dict with the noisy grids (n, X, Y, T), the grids that would be
    obtained without noise or individual sources, the sources, the weights, the
    weight moments, and a mask of units carrying no shared signal.
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

    for trial in range(n):
        silenced = np.random.choice(num_shared,
                                    size=int(partial_shared_fraction * num_shared),
                                    replace=False)
        W[trial, :, :, silenced] = 0

    moments = weight_moments(base_origins, W[0], K, X, Y, shared_fraction,
                             sigma_origin, width, within_subject=within_subject,
                             n_iter=10)

    voxel_grids = np.einsum("nxyk,nkt->nxyt", W, signals)
    noiseless_grids = np.einsum("nxyk,nkt->nxyt", W[:, :, :, :num_shared],
                                signals[:, :num_shared])
    shared_mask = np.array([1] * num_shared + [0] * (K - num_shared))
    individual_coords = np.where(
        np.einsum("xyk,k->xy", W.mean(axis=0), shared_mask) == 0, 1, 0)

    if sigma > 0:
        if sigma_of_sigma > 0:
            sigma = np.random.normal(sigma, sigma_of_sigma, (n, X, Y))
            noise = np.random.normal(0, sigma[:, :, :, np.newaxis], voxel_grids.shape)
        else:
            noise = np.random.normal(0, sigma, voxel_grids.shape)
        voxel_grids += noise

    return {
        "voxel_grids": voxel_grids,
        "noiseless_grids": noiseless_grids,
        "signals": signals,
        "W": W,
        "all_origins": all_origins,
        "weight_moments": moments,
        "individual_coords": individual_coords,
    }
