"""Recorded datasets: loading, cleaning and the estimator sweeps."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..estimators.empirical.estimators import (allen_noise_ceiling,
                                               get_unique_subsets,
                                               leave_one_out_correlation,
                                               mean_to_one_correlation,
                                               random_half_splits,
                                               split_half_correlation)
from ..estimators.empirical.loop import (DEFAULT_ALPHA_FRACTIONS, circular_shift,
                                         loop_estimate, run_folds)
from .loaders import DATASETS
from .preprocessing import clean


def trial_grid(n_trials):
    """Trial counts swept by `sweep_n`: every second one, every eighth from 20 trials."""
    step = 2 if n_trials < 20 else 8
    return list(range(2, n_trials + 1, step))


class EmpiricalDataset:
    """A recorded dataset, shape (n_trials, n_units, n_samples).

    `data` is the cleaned recording. The raw one is kept for LOOP, which cleans
    every run and every validation segment on its own.
    """

    def __init__(self, data, name, raw=None, clean_kwargs=None, loop=True):
        self.data = data
        self.name = name
        self.raw = raw
        self.clean_kwargs = dict(clean_kwargs or {})
        self.loop = loop

    @classmethod
    def load(cls, name, roi=None, clean_kwargs=None, **loader_kwargs):
        """Load a dataset by name and apply the standard cleaning.

        name : one of the keys of `DATASETS`.
        roi  : required for Sherlock, unused otherwise.
        """
        spec = DATASETS[name]
        if spec["rois"] is not None:
            if roi is None:
                raise ValueError(f"{name} requires an roi, one of {spec['rois']}")
            raw = spec["loader"](roi, **loader_kwargs)
        else:
            raw = spec["loader"](**loader_kwargs)

        kwargs = dict(spec["clean"])
        kwargs.update(clean_kwargs or {})
        data = clean(raw.reshape(-1, raw.shape[-1]), **kwargs).reshape(raw.shape)

        label = name if roi is None else f"{name}_{roi}"
        return cls(data, label, raw=raw, clean_kwargs=kwargs, loop=spec["loop"])

    @property
    def n_trials(self):
        return self.data.shape[0]

    @property
    def n_units(self):
        return self.data.shape[1]

    @property
    def n_samples(self):
        return self.data.shape[-1]

    @property
    def run_lengths(self):
        return self.clean_kwargs.get("run_lengths")

    def preprocess(self, x):
        """The standard cleaning applied to one stretch of recording, (..., T)."""
        kwargs = dict(self.clean_kwargs, run_lengths=None)
        return clean(x.reshape(-1, x.shape[-1]), **kwargs).reshape(x.shape)

    def sweep_n(self, n_bootstraps=10, N=None, run_loop=False, n_null=5, n_segments=4,
                alpha_fractions=DEFAULT_ALPHA_FRACTIONS, device="cuda", seed=0):
        """Every estimator as a function of the number of trials.

        The single-trial estimators are computed for each held-out trial in turn
        over `n_bootstraps` distinct subsets of the others; the mean estimators
        are computed over `n_bootstraps` distinct subsets of all trials.

        Without LOOP every estimator uses the whole cleaned recording. With
        run_loop=True each run is held out in turn: LOOP is fitted on the other
        runs and scored on it, and every other estimator is computed on each run
        separately, the results being averaged over the runs, so that all of
        them see the same data. LOOP is also computed on null data, every trial
        circularly shifted within each run, for the first `n_null` held-out
        trials of every n (key "LOOP_null"); a leak-free LOOP gives zero there.

        Returns a dict of arrays and the list of trial counts. The single-trial
        estimators have shape (len(n_list), n_trials, n_bootstraps, n_units) and
        the mean estimators (len(n_list), 1, n_bootstraps, n_units).
        """
        if run_loop and not self.loop:
            raise ValueError(f"LOOP is not run on {self.name}; see DATASETS")
        N = self.n_trials if N is None else N
        n_list = trial_grid(N)
        rng = np.random.default_rng(seed)

        if run_loop:
            runs, segments = run_folds(self.raw, self.run_lengths, self.preprocess,
                                       n_segments)
            views = runs
        else:
            views = [self.data]

        def on_views(estimator, idx, **kwargs):
            return np.mean([estimator(view[idx], **kwargs) for view in views], axis=0)

        def loop_on(view_runs, view_segments, idx):
            return loop_estimate([run[idx] for run in view_runs],
                                 [[s[idx] for s in run] for run in view_segments],
                                 index=0, alpha_fractions=alpha_fractions, device=device)

        single_names = ["LOOC", "TTMC"] + (["LOOP", "LOOP_null"] if run_loop else [])
        results_single = {
            name: np.full((len(n_list), self.n_trials, n_bootstraps, self.n_units),
                          np.nan)
            for name in single_names}

        for j, n in enumerate(tqdm(n_list, desc=f"{self.name} single")):
            for trial_idx in range(self.n_trials):
                others = np.delete(np.arange(self.n_trials), trial_idx)
                subsets = get_unique_subsets(others, n - 1, n_bootstraps=n_bootstraps)
                for i, indices in enumerate(subsets):
                    idx = np.insert(indices, 0, trial_idx)
                    results_single["LOOC"][j, trial_idx, i] = \
                        on_views(leave_one_out_correlation, idx, index=0)
                    results_single["TTMC"][j, trial_idx, i] = \
                        on_views(mean_to_one_correlation, idx, index=0)
                    if not run_loop:
                        continue

                    results_single["LOOP"][j, trial_idx, i] = loop_on(runs, segments, idx)
                    if i == 0 and trial_idx < n_null:
                        shifted = circular_shift(self.raw[idx], self.run_lengths, rng)
                        null_runs, null_segments = run_folds(
                            shifted, self.run_lengths, self.preprocess, n_segments)
                        results_single["LOOP_null"][j, trial_idx, i] = loop_on(
                            null_runs, null_segments, np.arange(len(idx)))

        results_mean = {name: np.full((len(n_list), 1, n_bootstraps, self.n_units),
                                      np.nan)
                        for name in ("SH", "SHSB", "ATV")}

        for j, n in enumerate(tqdm(n_list, desc=f"{self.name} mean")):
            subsets = get_unique_subsets(np.arange(self.n_trials), n,
                                         n_bootstraps=n_bootstraps)
            for i, indices in enumerate(subsets):
                idx = np.asarray(indices)
                splits = random_half_splits(n)
                results_mean["SH"][j, 0, i] = on_views(split_half_correlation, idx,
                                                       splits=splits)
                results_mean["SHSB"][j, 0, i] = on_views(split_half_correlation, idx,
                                                         correction=True, splits=splits)
                results_mean["ATV"][j, 0, i] = on_views(allen_noise_ceiling, idx)

        results = dict(results_single)
        results.update(results_mean)
        return results, n_list

    def sweep_T(self, T_values, n_rep=30):
        """Every estimator as a function of the number of samples.

        For each length in `T_values`, `n_rep` contiguous windows are drawn
        uniformly at random from the recording. LOOC and TTMC are averaged over
        every trial held out in turn, so that no single trial drives the curve.

        All trials are used at every T; this measures the finite-T bias alone,
        with the number of trials held fixed.

        Returns a dict of (len(T_values), n_rep) arrays.
        """
        T_values = [int(T) for T in T_values]
        keys = ("LOOC", "TTMC", "SH", "SHSB", "ATV")
        results = {key: np.zeros((len(T_values), n_rep)) for key in keys}

        for i, T in enumerate(tqdm(T_values, desc=f"{self.name} window length")):
            for j in range(n_rep):
                start = np.random.randint(0, self.n_samples - T)
                window = self.data[..., start:start + T]

                results["LOOC"][i, j] = np.mean(
                    [leave_one_out_correlation(window, index=k).mean()
                     for k in range(self.n_trials)])
                results["TTMC"][i, j] = np.mean(
                    [mean_to_one_correlation(window, index=k).mean()
                     for k in range(self.n_trials)])
                results["SH"][i, j] = split_half_correlation(window).mean()
                results["SHSB"][i, j] = split_half_correlation(
                    window, correction=True).mean()
                results["ATV"][i, j] = allen_noise_ceiling(window).mean()

        return results