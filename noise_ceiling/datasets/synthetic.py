"""Synthetic data: one realisation of the generative model, with both the
empirical estimators and their closed forms available on the same object.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..estimators.empirical.estimators import (allen_noise_ceiling,
                                               leave_one_out_correlation,
                                               mean_to_one_correlation,
                                               split_half_correlation,
                                               true_noise_ceiling_mean,
                                               true_noise_ceiling_single)
from ..estimators.empirical.loop import loop_estimate, make_split
from ..estimators.theoretical_models import (allen_noise_ceiling_th,
                                             compute_sigma_r2, estimate_R_phi,
                                             leave_one_out_correlation_th,
                                             mean_to_one_correlation_th,
                                             noise_ceiling_mean_th,
                                             noise_ceiling_single_th,
                                             spearman_brown_noise_ceiling_th,
                                             split_half_correlation_th)
from . import generation

MEAN_ESTIMATORS = ("SNR", "SHSB", "SH", "NC_mean")
SINGLE_ESTIMATORS = ("LOOC", "TTMC", "NC_single")
ALL_ESTIMATORS = MEAN_ESTIMATORS + SINGLE_ESTIMATORS

# LOOP is computed only on request, and has no closed form, so it is kept out of
# ALL_ESTIMATORS.
OPTIONAL_ESTIMATORS = ("LOOP",)

DEFAULT_LOOP_ALPHAS = np.logspace(-3, 10, 7)


class SignalStats:
    """Signal statistics estimated once at high precision.

    The closed forms need the variance, mean and autocorrelation of a single
    source. Estimating them from a fresh draw at every call adds Monte-Carlo
    noise of about the size of the finite-T biases being measured, so they are
    estimated once from `K_ref` sources of length `T_ref`. The variance and mean
    do not depend on T, and the autocorrelation enters only through Bartlett
    weights, which are reapplied per T.
    """

    def __init__(self, T_ref=10000, K_ref=4000, chunk_size=400, seed=0,
                 gaussian_signals=False, **signal_kwargs):
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            var_sum, mean_sum, autocov_sum, n_done = 0.0, 0.0, None, 0
            while n_done < K_ref:
                k = min(chunk_size, K_ref - n_done)
                if gaussian_signals:
                    signals = generation.generate_gaussian_signals(k, T_ref)
                else:
                    signals = generation.generate_signals(k, T_ref, **signal_kwargs)
                var_sum += signals.var(axis=-1, ddof=1).sum()
                mean_sum += signals.mean(axis=-1).sum()
                autocov = estimate_R_phi(signals)
                autocov_sum = autocov * k if autocov_sum is None else autocov_sum + autocov * k
                n_done += k
        finally:
            np.random.set_state(rng_state)

        self.sigma_s2 = float(var_sum / n_done)
        self.mean_s = float(mean_sum / n_done)
        autocov = autocov_sum / n_done
        self._rho = autocov[1:] / autocov[0]
        self.T_ref, self.K_ref, self.seed = T_ref, K_ref, seed

    def sigma_corr2(self, T):
        """sum_tau (1 - tau / T) rho^2(tau), with rho estimated at T_ref."""
        n_lags = min(int(T) - 1, len(self._rho))
        lags = np.arange(1, n_lags + 1)
        return float(np.sum((1 - lags / T) * self._rho[:n_lags] ** 2))

    def as_tuple(self, T):
        return self.sigma_s2, self.sigma_corr2(T), self.mean_s


def realized_weight_moments(W, K_shared, loo_for_E_s=False):
    """Weight moments of the realised trials rather than of the population.

    The Monte-Carlo moments describe 1000 fresh draws, but the data contains
    only n trials. Since the estimators are nonlinear in the weights, the
    mismatch leaves a T-independent offset of order 1/n between theory and
    simulation.

    sum_k E[w_k]^2 is estimated as sum_k (mean_l(w)^2 - s^2 / n) with s^2 the
    unbiased across-trial variance, which is also the exact split-half
    expectation over random equal splits.

    W           : (n, X, Y, K).
    loo_for_E_s : LOOC compares trial 0 against the mean of the others, so E_s
                  must exclude trial 0. TTMC includes it.

    The fourth-order moments are not used by the closed forms and are left None.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    W0 = W[0]
    W_shared, W_individual = W[..., :K_shared], W[..., K_shared:]

    def moments(A):
        second = (A ** 2).mean(0).sum(-1)
        mean = A.mean(0)
        variance = A.var(0, ddof=1) if n > 1 else np.zeros_like(mean)
        return second, (mean ** 2 - variance / n).sum(-1)

    M_s, mu_s = moments(W_shared)
    M_i, mu_i = moments(W_individual)
    mean_shared = W_shared[1:].mean(0) if (loo_for_E_s and n > 1) else W_shared.mean(0)
    E_s = (mean_shared * W0[..., :K_shared]).sum(-1)

    return {"M_s": M_s, "mu_s": mu_s, "M_i": M_i, "mu_i": mu_i, "E_s": E_s,
            "M_s4": None, "M_s2_2": None, "M_s_4": None,
            "M_s3_mu": None, "M_s2_mu2": None}


class SyntheticDataset:
    """One realisation of the generative model, plus its weight moments."""

    _SHARED_STATS = {}

    def __init__(self, signal_stats=None, **params):
        self.params = dict(params)
        self._grids = generation.generate_voxel_grids(**params)

        if signal_stats is not None:
            self.stats = signal_stats
        else:
            kwargs = dict(lambda_=params.get("lambda_", 0.1),
                          mu=params.get("mu", 5.0),
                          tr=params.get("tr", 1.5),
                          crop=params.get("crop"),
                          gaussian_signals=params.get("gaussian_signals", False))
            if kwargs["gaussian_signals"]:
                kwargs = {"gaussian_signals": True}
            key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                               for k, v in kwargs.items()))
            if key not in SyntheticDataset._SHARED_STATS:
                SyntheticDataset._SHARED_STATS[key] = SignalStats(**kwargs)
            self.stats = SyntheticDataset._SHARED_STATS[key]

    @property
    def voxel_grids(self):
        return self._grids["voxel_grids"]

    @property
    def noiseless_grids(self):
        return self._grids["noiseless_grids"]

    @property
    def W(self):
        return self._grids["W"]

    @property
    def signals(self):
        return self._grids["signals"]

    @property
    def moments(self):
        return self._grids["weight_moments"]

    @property
    def individual_coords(self):
        return self._grids["individual_coords"]

    @property
    def n_trials(self):
        return self.params["n"]

    @property
    def T(self):
        return self.params["T"]

    @property
    def sigma2(self):
        return self.params["sigma"] ** 2

    @property
    def K_shared(self):
        return int(self.params["K"] * self.params["shared_fraction"])

    def signal_stats(self, T=None):
        return self.stats.as_tuple(self.T if T is None else T)

    def sigma_r2(self, T=None):
        """Bartlett variance factor. Depends on T, so recompute when T changes."""
        T = self.T if T is None else T
        sigma_s2, sigma_corr2, _ = self.signal_stats(T)
        return compute_sigma_r2(self.moments["M_i"], self.moments["M_s"], sigma_s2,
                                self.sigma2, sigma_corr2, T)

    def realized_moments(self, n=None, loo_for_E_s=False):
        n = self.n_trials if n is None else n
        return realized_weight_moments(self.W[:n], self.K_shared,
                                       loo_for_E_s=loo_for_E_s)

    def theory(self, n, T=None, realized=False):
        """Closed-form value of every estimator at pool size n."""
        T = self.T if T is None else T
        sigma_s2, sigma_corr2, mean_s = self.signal_stats(T)
        sigma_r2 = self.sigma_r2(T)
        W0 = self.W[0]

        if realized:
            m = self.realized_moments(n)
            m_loo = self.realized_moments(n, loo_for_E_s=True)
        else:
            m = m_loo = self.moments

        return {
            "SNR": allen_noise_ceiling_th(
                sigma_s2, mean_s, self.sigma2, sigma_r2, m["M_s"], m["mu_s"],
                m["M_i"], m["mu_i"], n, T, self.individual_coords),
            "SHSB": spearman_brown_noise_ceiling_th(
                sigma_s2, self.sigma2, sigma_r2, m["M_s"], m["mu_s"], m["M_i"],
                n, T, self.individual_coords),
            "SH": split_half_correlation_th(
                sigma_s2, self.sigma2, m["M_s"], m["mu_s"], m["M_i"], n),
            "NC_mean": noise_ceiling_mean_th(
                sigma_s2, self.sigma2, m["M_s"], m["mu_s"], m["M_i"], n),
            "LOOC": leave_one_out_correlation_th(
                sigma_s2, self.sigma2, W0, m_loo["M_s"], m_loo["mu_s"],
                m_loo["M_i"], m_loo["E_s"], n),
            "TTMC": mean_to_one_correlation_th(
                sigma_s2, self.sigma2, W0, m["M_s"], m["mu_s"], m["M_i"],
                m["E_s"], n, self.K_shared),
            "NC_single": noise_ceiling_single_th(
                sigma_s2, self.sigma2, W0, self.K_shared),
        }

    @staticmethod
    def _correlation_estimators(grids, noiseless, index):
        """Every estimator that needs no train/test split, on an already-subset array."""
        return {
            "SNR": allen_noise_ceiling(grids).mean(),
            "SHSB": split_half_correlation(grids, correction=True).mean(),
            "SH": split_half_correlation(grids, correction=False).mean(),
            "NC_mean": true_noise_ceiling_mean(grids, noiseless).mean(),
            "LOOC": leave_one_out_correlation(grids, index).mean(),
            "TTMC": mean_to_one_correlation(grids, index).mean(),
            "NC_single": true_noise_ceiling_single(grids, noiseless, index).mean(),
        }

    def estimate(self, n=None, index=0, trial_indices=None, time_slice=None,
                 run_loop=False, chunked=True, split=None,
                 alphas=DEFAULT_LOOP_ALPHAS, device="cpu", n_validations=10):
        """Empirical estimates on a subset of trials and, optionally, a time window.

        index : held-out trial, as a position within the selected subset. The
                closed forms use W[0], so a comparison against them requires the
                subset to start with trial 0 and index to be 0.

        With run_loop=True a train/test split is drawn and every estimator is
        evaluated on the held-out samples, so that LOOP and the correlation-based
        estimators are not confounded by their different finite-T bias.
        chunked selects purged contiguous chunks over a random split.
        """
        if trial_indices is None:
            n = self.n_trials if n is None else n
            trial_indices = np.arange(n)

        grids = self.voxel_grids[trial_indices]
        noiseless = self.noiseless_grids[trial_indices]
        if time_slice is not None:
            grids, noiseless = grids[..., time_slice], noiseless[..., time_slice]

        if not run_loop:
            return self._correlation_estimators(grids, noiseless, index)

        flat_grids = grids.reshape(grids.shape[0], -1, grids.shape[-1])
        flat_noiseless = noiseless.reshape(noiseless.shape[0], -1, noiseless.shape[-1])
        if split is None:
            split = make_split(flat_grids.shape[-1], chunked=chunked)
        test_idx = split[1]

        out = self._correlation_estimators(
            flat_grids[..., test_idx].reshape(grids.shape[:-1] + (len(test_idx),)),
            flat_noiseless[..., test_idx].reshape(noiseless.shape[:-1] + (len(test_idx),)),
            index)
        out["LOOP"] = float(np.mean(loop_estimate(
            flat_grids, split, alphas, index=index, device=device,
            n_validations=n_validations, val_chunk=chunked)))
        return out

    def sweep_n(self, n_values, n_rep=10, match_theory_trial=True, realized=False,
                run_loop=False, **loop_kwargs):
        """Vary the number of trials. Returns {'n', 'theory', 'empirical'}.

        match_theory_trial keeps trial 0 in every subset and holds it out, so
        that LOOC, TTMC and NC_single are comparable to the W[0]-based theory.
        """
        n_values = list(n_values)
        keys = ALL_ESTIMATORS + (OPTIONAL_ESTIMATORS if run_loop else ())
        empirical = {k: np.zeros(len(n_values)) for k in keys}
        theory = {k: np.zeros(len(n_values)) for k in ALL_ESTIMATORS}

        for i, n in enumerate(n_values):
            theory_values = self.theory(n, realized=realized)
            for k in ALL_ESTIMATORS:
                theory[k][i] = theory_values[k]

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

        return {"n": np.array(n_values), "theory": theory, "empirical": empirical}

    def sweep_T(self, T_values, n=None, n_rep=100, index=0,
                recompute_theory_per_T=True, realized=False, anchor="theory",
                T_ref=None):
        """Vary the number of samples, using contiguous windows.

        anchor : "theory" compares each estimate against the closed forms;
                 "tmax" compares against the estimate at T_ref, which cancels
                 any constant offset of the closed forms. The latter gives
                 measured(T) = c (T^-p - T_ref^-p), so use `fit_power` rather
                 than a log-log slope to recover p.

        recompute_theory_per_T recomputes the Bartlett factor and the
        rectification corrections at every window length.
        """
        n = self.n_trials if n is None else n
        T_values = [int(t) for t in T_values]
        empirical = {k: np.zeros(len(T_values)) for k in ALL_ESTIMATORS}
        theory = {k: np.zeros(len(T_values)) for k in ALL_ESTIMATORS}

        fixed_theory = None if recompute_theory_per_T else self.theory(n, realized=realized)

        reference = None
        if anchor == "tmax":
            T_ref = self.T if T_ref is None else int(T_ref)
            if max(T_values) > T_ref / 2:
                warnings.warn(f"largest T ({max(T_values)}) exceeds T_ref/2 "
                              f"({T_ref / 2:.0f}); use fit_power to recover p")
            reference = self.estimate(n=n, index=index, time_slice=slice(0, T_ref))
        elif anchor != "theory":
            raise ValueError(anchor)

        for i, T in enumerate(T_values):
            theory_values = (self.theory(n, T=T, realized=realized)
                             if recompute_theory_per_T else fixed_theory)
            for k in ALL_ESTIMATORS:
                theory[k][i] = theory_values[k]

            total = {k: 0.0 for k in ALL_ESTIMATORS}
            n_windows = 1 if T >= self.T else n_rep
            for _ in range(n_windows):
                start = 0 if T >= self.T else np.random.randint(0, self.T - T)
                estimates = self.estimate(n=n, index=index,
                                          time_slice=slice(start, start + T))
                for k in ALL_ESTIMATORS:
                    total[k] += estimates[k]
            for k in ALL_ESTIMATORS:
                empirical[k][i] = total[k] / n_windows

        out = {"T": np.array(T_values), "theory": theory, "empirical": empirical}
        if reference is not None:
            out["reference"] = reference
            out["T_ref"] = T_ref
        return out
