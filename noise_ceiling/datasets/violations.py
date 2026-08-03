

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from . import generation
from .preprocessing import dct_basis, project_out
from .synthetic import SyntheticDataset


@dataclass
class Violations:
    """Which assumptions to break, and by how much. All zero/False is a no-op."""

    #heterogeneous HRF
    hrf_subject_sd: float = 0.0
    """SD of the per-trial offset applied to the HRF peak-delay shape a1."""
    hrf_voxel_sd: float = 0.0
    """SD of the per-unit offset applied to a1, as a spatially smooth field."""
    hrf_voxel_smooth: float = 1.5
    """Smoothing (in units of the grid spacing) of the per-unit HRF field."""
    hrf_dispersion_ratio: float = 0.5
    """Dispersion b1 is jittered by this fraction of the a1 jitter."""

    #recorded noise
    real_noise: np.ndarray | None = None
    """(n, X, Y, T) array used as the noise term, z-scored then scaled by sigma.
    Build it with `prepare_real_noise`."""

    #noise shared across trials
    oscillation_amplitude: float = 0.0
    """Amplitude of the slow component, relative to sigma."""
    oscillation_freq_hz: float = 0.008
    """Frequency of the slow component. Below 1/cutoff_s it is removed by the
    high-pass filter, above it survives."""
    oscillation_phase_jitter: float = 0.4
    """Phase spread across trials, in radians. Zero makes the component
    identical in every trial, pi/2 makes it uncorrelated on average."""

    #signal-dependent noise
    shared_individual_corr: float = 0.0
    """Correlation injected between each trial-specific source and a randomly
    paired shared source, in [0, 1)."""

    #heteroscedastic noise across trials
    sigma_trial_sd: float = 0.0
    """SD of the per-trial multiplicative factor on sigma."""

    #applied to data and to v* alike
    highpass_cutoff_s: float | None = None
    """If set, the DCT high-pass of `preprocessing.clean` is applied to the
    grids AND to the noiseless grids, so that v* stays exact. Needed whenever
    the oscillation is used."""

    def active(self):
        """Names of the violations that are switched on."""
        on = []
        if self.hrf_subject_sd or self.hrf_voxel_sd:
            on.append("hrf")
        if self.real_noise is not None:
            on.append("real_noise")
        if self.oscillation_amplitude:
            on.append("oscillation")
        if self.shared_individual_corr:
            on.append("corr")
        if self.sigma_trial_sd:
            on.append("sigma_trial")
        if self.highpass_cutoff_s:
            on.append("highpass")
        return on

    def any(self):
        return bool(self.active())

    def describe(self):
        return ", ".join(self.active()) if self.any() else "none (baseline)"

    @classmethod
    def from_dict(cls, mapping):
        names = {f.name for f in fields(cls)}
        unknown = set(mapping or {}) - names
        if unknown:
            raise ValueError(f"unknown violation(s): {sorted(unknown)}")
        return cls(**(mapping or {}))



def prepare_real_noise(data, n, X, Y, T, rng=None, time_shift=True,
                       scaling="global", return_diagnostics=False):
    """Turn a recorded dataset into a noise array of shape (n, X, Y, T).
"""
    rng = np.random.default_rng() if rng is None else rng
    data = np.asarray(data, dtype=float)
    n_trials, n_units, n_samples = data.shape

    if n_trials < n:
        raise ValueError(f"need {n} trials, dataset has {n_trials}")
    if n_units < X * Y:
        raise ValueError(f"need {X * Y} units, dataset has {n_units}")
    if n_samples < T:
        raise ValueError(f"need {T} samples, dataset has {n_samples}")

    unit_idx = rng.choice(n_units, size=X * Y, replace=False)
    trial_idx = rng.choice(n_trials, size=n, replace=False)
    subset = data[np.ix_(trial_idx, unit_idx)]

    isc_before = _mean_inter_trial_correlation(subset[..., :T])

    if time_shift:
        shifted = np.empty_like(subset)
        for trial in range(n):
            shift = rng.integers(n_samples // 10, n_samples - n_samples // 10)
            shifted[trial] = np.roll(subset[trial], shift, axis=-1)
        subset = shifted

    noise = subset[..., :T]
    isc_after = _mean_inter_trial_correlation(noise)
    noise = noise - noise.mean(axis=-1, keepdims=True)

    unit_sd = noise.std(axis=-1)                      # (n, X * Y)
    if scaling == "global":
        scale = np.full((n, X * Y, 1), noise.std())
    elif scaling == "per_trial":
        scale = np.repeat(noise.std(axis=(1, 2))[:, None], X * Y, 1)[..., None]
    elif scaling == "per_unit":
        scale = unit_sd[..., None]
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    noise = noise / np.where(scale > 0, scale, 1.0)

    trial_sd = noise.std(axis=(1, 2))
    diagnostics = {
        "isc_before": isc_before,
        "isc_after": isc_after,
        "trial_sd": trial_sd,
        #0 means the noise level is identical in every trial, i.e. the homoscedasticity assumption is not violated.
        "trial_sd_spread": float(trial_sd.std() / max(trial_sd.mean(), 1e-12)),
        "unit_sd_spread": float(noise.std(axis=-1).std() / max(noise.std(), 1e-12)),
        "unit_idx": unit_idx,
        "trial_idx": trial_idx,
    }

    noise = noise.reshape(n, X, Y, T)
    return (noise, diagnostics) if return_diagnostics else noise


def _mean_inter_trial_correlation(x):
    """Mean over units and trial pairs of the across-trial correlation."""
    x = x - x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    x = x / np.where(sd > 0, sd, 1.0)
    n = x.shape[0]
    total, count = 0.0, 0
    for a in range(n):
        for b in range(a + 1, n):
            total += float(np.mean(np.mean(x[a] * x[b], axis=-1)))
            count += 1
    return total / max(count, 1)



def _hrf_fields(violations, n, X, Y, a1=6.0, b1=1.0, within_subject=True):
    """Per (trial, unit) HRF parameters a1 and b1.
"""
    if within_subject:
        if violations.hrf_subject_sd:
            warnings.warn(
                "hrf_subject_sd is ignored when within_subject=True: the trials "
                "are repetitions of one subject, so they share its HRF. Vary "
                "hrf_voxel_sd instead.")
        subject = np.zeros((1, 1, 1))
        n_fields = 1
    else:
        subject = np.random.normal(0, violations.hrf_subject_sd, (n, 1, 1))
        n_fields = n

    if violations.hrf_voxel_sd > 0:
        raw = np.random.normal(0, 1.0, (n_fields, X, Y))
        smooth = np.stack([gaussian_filter(raw[i], violations.hrf_voxel_smooth,
                                           mode="wrap") for i in range(n_fields)])
        sd = smooth.std()
        smooth = smooth / (sd if sd > 0 else 1.0) * violations.hrf_voxel_sd
    else:
        smooth = np.zeros((n_fields, X, Y))

    delta = np.broadcast_to(subject + smooth, (n, X, Y))
    a1_field = np.clip(a1 + delta, 1.5, None)
    b1_field = np.clip(b1 + violations.hrf_dispersion_ratio * delta, 0.25, None)
    return a1_field, b1_field


def _convolve_per_unit(neural, a1_field, b1_field, tr, a2, b2, c, T_total):
    """Convolve each unit's neural time course with its own HRF.

    neural : (n, X, Y, T_total). Kernels are L2-normalised, so a change of HRF
    shape changes timing and dispersion but not the variance imparted, which
    keeps HRF heterogeneity from silently moving the SNR.
    """
    n, X, Y, _ = neural.shape
    flat = neural.reshape(n * X * Y, T_total)
    a1_flat = a1_field.reshape(-1)
    b1_flat = b1_field.reshape(-1)

    kernels = np.stack([
        generation.hrf_kernel(tr=tr, a1=a1_flat[i], a2=a2, b1=b1_flat[i], b2=b2,
                              c=c, normalize=True)
        for i in range(flat.shape[0])])

    convolved = fftconvolve(flat, kernels, mode="full", axes=1)[:, :T_total]
    return convolved.reshape(n, X, Y, T_total)


def _correlate_individual_with_shared(individual, shared, rho):
    """Mix a random shared source into each trial-specific source.

    individual : (n, K_i, T), shared : (K_s, T). Both are neural sources.
    Returns sources with Corr(individual, paired shared) ~ rho, which makes the
    noise term correlated with the signal term.
    """
    if rho <= 0 or shared.shape[0] == 0 or individual.shape[1] == 0:
        return individual

    n, K_i, T = individual.shape
    pairing = np.random.randint(0, shared.shape[0], size=K_i)
    partner = shared[pairing]

    def standardise(x):
        x = x - x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        return x / np.where(sd > 0, sd, 1.0)

    z_ind = standardise(individual)
    z_shared = standardise(partner)[None]
    mixed = rho * z_shared + np.sqrt(1 - rho ** 2) * z_ind

    scale = individual.std(axis=-1, keepdims=True)
    return mixed * scale + individual.mean(axis=-1, keepdims=True)


def _shared_oscillation(violations, n, X, Y, T, tr):
    """Slow component shared across trials, with a small phase jitter."""
    t = np.arange(T) * tr
    phases = np.random.uniform(-violations.oscillation_phase_jitter,
                               violations.oscillation_phase_jitter, size=n)
    wave = np.cos(2 * np.pi * violations.oscillation_freq_hz * t[None, :]
                  + phases[:, None])
    return violations.oscillation_amplitude * wave[:, None, None, :] * np.ones((1, X, Y, 1))


def generate_violated_voxel_grids(violations=None, **params):
    """One realisation of the generative model with the requested violations.

    Returns the same dict as `generation.generate_voxel_grids`, so that
    everything downstream is unchanged. With no violations active it delegates
    to the original function, which makes the baseline bit-for-bit identical.
    """
    if violations is None or not violations.any():
        return generation.generate_voxel_grids(**params)

    X, Y, T, K = params["X"], params["Y"], params["T"], params["K"]
    n = params.get("n", 1)
    tr = params.get("tr", 1.5)
    sigma = params.get("sigma", 0.5)
    crop = params.get("crop")
    shared_fraction = params.get("shared_fraction", 0.5)
    num_shared = int(shared_fraction * K)

    if params.get("gaussian_signals", False):
        raise NotImplementedError(
            "the violations are defined on the block-design sources; "
            "set gaussian_signals: false")

    pad_start = crop[0] if crop is not None else 0
    pad_end = crop[1] if crop is not None else 0
    T_total = T + pad_start + pad_end
    time_slice = slice(pad_start, T_total - pad_end) if crop else slice(None)

    # --- neural sources, before any hemodynamic filtering ---------------
    neural_kwargs = dict(lambda_=params.get("lambda_", 0.1), mu=params.get("mu", 5.0))
    shared_neural = (generation.generate_neural(num_shared, T_total, **neural_kwargs)
                     if num_shared > 0 else np.zeros((0, T_total)))
    K_ind = K - num_shared
    individual_neural = (
        generation.generate_neural(n * K_ind, T_total, **neural_kwargs).reshape(n, K_ind, T_total)
        if K_ind > 0 else np.zeros((n, 0, T_total)))

    # B1: correlate the trial-specific sources with the shared ones
    individual_neural = _correlate_individual_with_shared(
        individual_neural, shared_neural, violations.shared_individual_corr)

    neural = np.concatenate(
        [np.broadcast_to(shared_neural, (n,) + shared_neural.shape), individual_neural],
        axis=1)

    base_origins = generation.generate_origins(K, X, Y)
    W, all_origins = generation.generate_weights(
        base_origins, K, X, Y, n,
        sigma_origin=params.get("sigma_origin", 1),
        width=params.get("width", 1.0),
        within_subject=params.get("within_subject", True))

    partial = params.get("partial_shared_fraction", 0)
    if partial and num_shared > 0:
        for trial in range(n):
            silenced = np.random.choice(num_shared, size=int(partial * num_shared),
                                        replace=False)
            W[trial, :, :, silenced] = 0

    mixed_all = np.einsum("nxyk,nkt->nxyt", W, neural)
    mixed_shared = np.einsum("nxyk,nkt->nxyt", W[..., :num_shared], neural[:, :num_shared])

    if violations.hrf_subject_sd or violations.hrf_voxel_sd:
        a1_field, b1_field = _hrf_fields(
            violations, n, X, Y, a1=params.get("a1", 6), b1=params.get("b1", 1),
            within_subject=params.get("within_subject", True))
    else:
        a1_field = np.full((n, X, Y), float(params.get("a1", 6)))
        b1_field = np.full((n, X, Y), float(params.get("b1", 1)))

    conv = dict(tr=tr, a2=params.get("a2", 16), b2=params.get("b2", 1),
                c=params.get("c", 1.6), T_total=T_total)
    voxel_grids = _convolve_per_unit(mixed_all, a1_field, b1_field, **conv)[..., time_slice]
    noiseless_grids = _convolve_per_unit(mixed_shared, a1_field, b1_field, **conv)[..., time_slice]

    voxel_grids = voxel_grids - voxel_grids.mean(axis=-1, keepdims=True)
    noiseless_grids = noiseless_grids - noiseless_grids.mean(axis=-1, keepdims=True)

    if sigma > 0:
        if violations.real_noise is not None:
            noise = np.asarray(violations.real_noise, dtype=float)
            if noise.shape != (n, X, Y, T):
                raise ValueError(f"real_noise has shape {noise.shape}, "
                                 f"expected {(n, X, Y, T)}")
            noise = noise * sigma
        elif params.get("sigma_of_sigma", 0) > 0:
            unit_sigma = np.random.normal(sigma, params["sigma_of_sigma"], (n, X, Y))
            noise = np.random.normal(0, unit_sigma[..., None], voxel_grids.shape)
        else:
            noise = np.random.normal(0, sigma, voxel_grids.shape)

        # B3: noise level varying across trials
        if violations.sigma_trial_sd > 0:
            factor = np.clip(np.random.normal(1.0, violations.sigma_trial_sd, n), 0.1, None)
            noise = noise * factor[:, None, None, None]

        # B2: a slow component shared across trials
        if violations.oscillation_amplitude > 0:
            noise = noise + sigma * _shared_oscillation(violations, n, X, Y, T, tr)

        voxel_grids = voxel_grids + noise

    if violations.highpass_cutoff_s:
        Q = dct_basis(T, tr=tr, cutoff_s=violations.highpass_cutoff_s)
        voxel_grids = project_out(voxel_grids, Q)
        noiseless_grids = project_out(noiseless_grids, Q)

    shared_mask = np.array([1] * num_shared + [0] * (K - num_shared))
    individual_coords = np.where(
        np.einsum("xyk,k->xy", W.mean(axis=0), shared_mask) == 0, 1, 0)

    return {
        "voxel_grids": voxel_grids,
        "noiseless_grids": noiseless_grids,
        "signals": neural,
        "W": W,
        "all_origins": all_origins,
        "weight_moments": None,
        "individual_coords": individual_coords,
        "hrf_a1": a1_field,
        "hrf_b1": b1_field,
    }


class ViolatedDataset(SyntheticDataset):
    """SyntheticDataset with one or more assumptions of Eq. (3) broken.
    """

    def __init__(self, violations=None, signal_stats=None, **params):
        self.violations = violations or Violations()
        self.params = dict(params)
        self._grids = generate_violated_voxel_grids(violations=self.violations, **params)

        if self.violations.any():
            #The moments and the signal statistics feed the closed forms, which do not hold here
            self.stats = signal_stats
        else:
            SyntheticDataset.__init__(self, signal_stats=signal_stats, **params)

    def theory(self, *args, **kwargs):
        if self.violations.any():
            raise NotImplementedError(
                "the closed forms of Sec. 3 assume Eq. (3) and do not hold under "
                f"[{self.violations.describe()}]. Compare the estimators against "
                "the true noise ceiling (NC_mean, NC_single) instead, which is "
                "computed exactly from the noiseless grids.")
        return SyntheticDataset.theory(self, *args, **kwargs)

    def sweep_n(self, n_values, n_rep=10, match_theory_trial=True, realized=False,
                run_loop=False, **loop_kwargs):
        """As the parent, but without the closed forms when a violation is on."""
        if not self.violations.any():
            return SyntheticDataset.sweep_n(
                self, n_values, n_rep=n_rep, match_theory_trial=match_theory_trial,
                realized=realized, run_loop=run_loop, **loop_kwargs)

        from .synthetic import ALL_ESTIMATORS, OPTIONAL_ESTIMATORS

        n_values = list(n_values)
        keys = ALL_ESTIMATORS + (OPTIONAL_ESTIMATORS if run_loop else ())
        empirical = {k: np.zeros(len(n_values)) for k in keys}

        for i, n in enumerate(n_values):
            total = {k: 0.0 for k in keys}
            for _ in range(n_rep):
                if match_theory_trial:
                    others = np.random.choice(np.arange(1, self.n_trials),
                                              size=n - 1, replace=False)
                    indices = np.insert(others, 0, 0)
                else:
                    indices = np.random.choice(self.n_trials, size=n, replace=False)
                estimates = self.estimate(trial_indices=indices, index=0,
                                          run_loop=run_loop, **loop_kwargs)
                for k in keys:
                    total[k] += estimates[k]
            for k in keys:
                empirical[k][i] = total[k] / n_rep

        return {"n": np.array(n_values), "theory": None, "empirical": empirical}
