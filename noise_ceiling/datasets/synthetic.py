"""Synthetic data: one realisation of the generative model, with both the
empirical estimators and their closed forms available on the same object.
"""

from __future__ import annotations


import numpy as np

from ..estimators.empirical.estimators import (allen_noise_ceiling,
                                               leave_one_out_correlation,
                                               mean_to_one_correlation,
                                               random_half_splits,
                                               split_half_correlation,
                                               true_noise_ceiling_mean,
                                               true_noise_ceiling_single)
from ..estimators.empirical.loop import (DEFAULT_ALPHA_FRACTIONS, circular_shift,
                                         loop_estimate, run_folds, run_slices)
from ..estimators.theoretical_models import (allen_noise_ceiling_th,
                                             leave_one_out_correlation_th,
                                             mean_to_one_correlation_th,
                                             noise_ceiling_mean_th,
                                             noise_ceiling_single_th,
                                             spearman_brown_noise_ceiling_th,
                                             split_half_correlation_th)
from . import generation

MEAN_ESTIMATORS = ("ATV", "SHSB", "SH", "NC_mean")
SINGLE_ESTIMATORS = ("LOOC", "TTMC", "NC_single")
ALL_ESTIMATORS = MEAN_ESTIMATORS + SINGLE_ESTIMATORS


class SyntheticDataset:
    """One realisation of the generative model."""

    def __init__(self, **params):
        self.params = dict(params)
        self._grids = generation.generate_voxel_grids(**params)

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
    def noise_var(self):
        """sigma_l^2 of every trial and unit, (n, X, Y)."""
        return self._grids["noise_var"]

    @property
    def n_trials(self):
        return self.params["n"]

    @property
    def T(self):
        return self.params["T"]

    @property
    def K_shared(self):
        return int(self.params["K"] * self.params["shared_fraction"])

    def _gram_root(self, time_slice=None):
        """G^(1/2), with G the covariance of the shared sources over the window.

        Cached for the full recording, which every repetition of `sweep_n` uses.
        """
        if time_slice is None and getattr(self, "_full_gram_root", None) is not None:
            return self._full_gram_root

        sources = self.signals[0, :self.K_shared]
        if time_slice is not None:
            sources = sources[:, time_slice]
        sources = sources - sources.mean(axis=-1, keepdims=True)
        gram = sources @ sources.T / sources.shape[-1]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T

        if time_slice is None:
            self._full_gram_root = root
        return root

    def latent_weights(self, time_slice=None, trial_indices=None):
        """Weights of the noise-free signals on orthonormal latent signals.

        The shared sources are standardised but not exactly orthogonal at
        finite T. With G their covariance matrix over the time window, the
        weights W G^(1/2) reproduce every time covariance of the noise-free
        signals exactly, for any number of sources and any window length.
        """
        W = self.W if trial_indices is None else self.W[trial_indices]
        return W[..., :self.K_shared] @ self._gram_root(time_slice)

    def theory(self, n=None, index=0, trial_indices=None, time_slice=None,
               half_splits=None):
        """Closed-form value of every estimator, averaged over units.

        The closed forms are conditional on the trials used, so pass the same
        trial_indices, time_slice and half_splits as to `estimate`.
        """
        if trial_indices is None:
            trial_indices = np.arange(self.n_trials if n is None else n)
        if half_splits is None:
            half_splits = random_half_splits(len(trial_indices))

        W = self.latent_weights(time_slice, trial_indices)
        noise_var = self.noise_var[trial_indices]

        values = {
            "ATV": allen_noise_ceiling_th(W, noise_var),
            "SHSB": spearman_brown_noise_ceiling_th(W, noise_var, half_splits),
            "SH": split_half_correlation_th(W, noise_var, half_splits),
            "NC_mean": noise_ceiling_mean_th(W, noise_var),
            "LOOC": leave_one_out_correlation_th(W, noise_var, index),
            "TTMC": mean_to_one_correlation_th(W, noise_var, index),
            "NC_single": noise_ceiling_single_th(W, noise_var, index),
        }
        return {key: float(value.mean()) for key, value in values.items()}

    @staticmethod
    def _correlation_estimators(grids, noiseless, index, half_splits=None):
        """Every estimator that needs no train/test split, on an already-subset array."""
        if half_splits is None:
            half_splits = random_half_splits(grids.shape[0])
        return {
            "ATV": allen_noise_ceiling(grids).mean(),
            "SHSB": split_half_correlation(grids, correction=True,
                                           splits=half_splits).mean(),
            "SH": split_half_correlation(grids, splits=half_splits).mean(),
            "NC_mean": true_noise_ceiling_mean(grids, noiseless).mean(),
            "LOOC": leave_one_out_correlation(grids, index).mean(),
            "TTMC": mean_to_one_correlation(grids, index).mean(),
            "NC_single": true_noise_ceiling_single(grids, noiseless, index).mean(),
        }

    def estimate(self, n=None, index=0, trial_indices=None, time_slice=None,
                 unit_mask=None, half_splits=None):
        """Empirical estimates on a subset of trials and, optionally, a time window.

        index : held-out trial, as a position within the selected subset.

        half_splits : splits used by SH and SHSB, as positions within the
                selected subset. Pass the same ones to `theory`.

        unit_mask : boolean mask over the flattened grid, keeping only some
                units. The closed forms returned by `theory` describe the whole
                grid, so they do not apply to a masked estimate.
        """
        if trial_indices is None:
            n = self.n_trials if n is None else n
            trial_indices = np.arange(n)

        grids = self.voxel_grids[trial_indices]
        noiseless = self.noiseless_grids[trial_indices]
        if time_slice is not None:
            grids, noiseless = grids[..., time_slice], noiseless[..., time_slice]
        if unit_mask is not None:
            grids = grids.reshape(grids.shape[0], -1, grids.shape[-1])[:, unit_mask]
            noiseless = noiseless.reshape(noiseless.shape[0], -1,
                                          noiseless.shape[-1])[:, unit_mask]

        return self._correlation_estimators(grids, noiseless, index, half_splits)

    def loop(self, trial_indices, index=0, run_lengths=None, null=False, rng=None,
             n_segments=4, alpha_fractions=DEFAULT_ALPHA_FRACTIONS, device="cpu"):
        """LOOP averaged over units, every run held out in turn.

        The synthetic recording has no drift, so the runs and segments are used
        as they are. null=True first shifts every trial circularly within each
        run, which a leak-free LOOP maps to zero.
        """
        grids = self.voxel_grids[trial_indices]
        grids = grids.reshape(grids.shape[0], -1, grids.shape[-1])
        if null:
            grids = circular_shift(grids, run_lengths, rng)
        runs, segments = run_folds(grids, run_lengths, n_segments=n_segments)
        return float(np.mean(loop_estimate(runs, segments, index=index,
                                           alpha_fractions=alpha_fractions,
                                           device=device)))

    def sweep_n(self, n_values, n_rep=10, match_theory_trial=True, run_loop=False,
                with_theory=True, run_lengths=None, n_null=5, **loop_kwargs):
        """Vary the number of trials. Returns {'n', 'theory', 'empirical'}.

        Every repetition draws a subset of trials and a set of half splits, and
        the closed forms are evaluated on the same ones as the estimators.
        match_theory_trial keeps trial 0 in every subset and holds it out, so
        that LOOC, TTMC and NC_single always describe the same trial.

        With run_loop=True the recording is cut into runs (`run_lengths`, two
        halves by default) and, as on recorded data, every estimator and closed
        form is computed on each run separately and averaged over the runs,
        while LOOP holds out each run in turn. The null LOOP is computed on the
        first `n_null` repetitions of every n (key "LOOP_null").
        """
        n_values = list(n_values)
        keys = ALL_ESTIMATORS + (("LOOP", "LOOP_null") if run_loop else ())
        empirical = {k: np.zeros(len(n_values)) for k in keys}
        theory = {k: np.zeros(len(n_values)) for k in ALL_ESTIMATORS}
        windows = run_slices(self.T, run_lengths) if run_loop else [None]
        rng = np.random.default_rng()

        for i, n in enumerate(n_values):
            for rep in range(n_rep):
                if match_theory_trial:
                    others = np.random.choice(np.arange(1, self.n_trials),
                                              size=n - 1, replace=False)
                    indices = np.insert(others, 0, 0)
                else:
                    indices = np.random.choice(self.n_trials, size=n, replace=False)
                half_splits = random_half_splits(n)
                for window in windows:
                    estimates = self.estimate(trial_indices=indices, index=0,
                                              time_slice=window, half_splits=half_splits)
                    for k in ALL_ESTIMATORS:
                        empirical[k][i] += estimates[k] / (n_rep * len(windows))
                    if with_theory:
                        theory_values = self.theory(trial_indices=indices, index=0,
                                                    time_slice=window,
                                                    half_splits=half_splits)
                        for k in ALL_ESTIMATORS:
                            theory[k][i] += theory_values[k] / (n_rep * len(windows))
                if run_loop:
                    empirical["LOOP"][i] += self.loop(indices, 0, run_lengths,
                                                      **loop_kwargs) / n_rep
                    if rep < n_null:
                        empirical["LOOP_null"][i] += self.loop(
                            indices, 0, run_lengths, null=True, rng=rng,
                            **loop_kwargs) / min(n_null, n_rep)

        return {"n": np.array(n_values), "theory": theory if with_theory else None,
                "empirical": empirical}
