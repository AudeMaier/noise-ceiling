"""Recorded datasets: loading, cleaning, unit selection and the estimator sweep."""

from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from ..estimators.empirical.estimators import (allen_noise_ceiling,
                                               get_unique_subsets,
                                               leave_one_out_correlation,
                                               mean_to_one_correlation,
                                               split_half_correlation)
from ..estimators.empirical.loop import loop_estimate, make_split
from ..selection import run_aic_procedure
from .loaders import DATASETS
from .preprocessing import clean

DEFAULT_LOOP_ALPHAS = np.logspace(-5, 15, 10)


class EmpiricalDataset:
    """A recorded dataset, shape (n_trials, n_units, n_samples)."""

    def __init__(self, data, name, chunked=False):
        self.data = data
        self.name = name
        self.chunked = chunked

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
            data = spec["loader"](roi, **loader_kwargs)
        else:
            data = spec["loader"](**loader_kwargs)

        kwargs = dict(spec["clean"])
        kwargs.update(clean_kwargs or {})
        shape = data.shape
        data = clean(data.reshape(-1, shape[-1]), **kwargs).reshape(shape)

        label = name if roi is None else f"{name}_{roi}"
        return cls(data, label, chunked=spec["chunked"])

    @property
    def n_trials(self):
        return self.data.shape[0]

    @property
    def n_units(self):
        return self.data.shape[1]

    @property
    def n_samples(self):
        return self.data.shape[-1]

    def select_signal_units(self, threshold, n_bootstrap=50, n_samples=100,
                            cache=None, recompute=False):
        """Keep only the units the AIC procedure identifies as carrying signal.

        cache     : path to an .npy file holding the Delta AIC distribution. It
                    is written after every computation, and reused unless
                    `recompute`, since the procedure is by far the slowest part
                    of the pipeline.
        recompute : redo the selection even when the cache exists. The procedure
                    is stochastic, so this generally yields a different set of
                    units.

        Returns the mask of kept units.
        """
        if cache is not None and os.path.exists(cache) and not recompute:
            aic_dist = np.load(cache)
        else:
            _, aic_dist = run_aic_procedure(
                self.data, n_list=range(2, self.n_trials, 2),
                n_bootstrap=n_bootstrap, n_samples=n_samples,
                aic_threshold=threshold)
            if cache is not None:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.save(cache, aic_dist)

        mask = aic_dist.mean(axis=1) > threshold
        print(f"{self.name}: keeping {mask.sum()} of {len(mask)} units")
        self.data = self.data[:, mask]
        return mask

    def sweep_n(self, n_bootstraps=10, N=None, run_loop=False, match_windows=True,
                test_fraction=0.25, n_chunk_test=5, test_buffer=50,
                n_validations=10, device="cpu", alphas=DEFAULT_LOOP_ALPHAS):
        """Every estimator as a function of the number of trials.

        The single-trial estimators are computed for each held-out trial in turn
        over `n_bootstraps` distinct subsets of the others; the mean estimators
        are computed over `n_bootstraps` distinct subsets of all trials.

        With run_loop=True one train/test split is drawn per subset and shared by
        LOOP and, if match_windows, the single-trial correlation estimators, so
        that all of them are evaluated on the same held-out samples.

        Returns a dict of arrays and the list of trial counts. The single-trial
        estimators have shape (len(n_list), n_trials, n_bootstraps, n_units) and
        the mean estimators (len(n_list), 1, n_bootstraps, n_units).
        """
        data = self.data
        N = self.n_trials if N is None else N
        if N < 20:
            n_list = list(range(2, N + 1, 2))
        else:
            n_list = list(range(2, N+1, 8))

        single_names = ["LOOC", "TTMC"] + (["LOOP"] if run_loop else [])
        results_single = {
            name: np.full((len(n_list), self.n_trials, n_bootstraps, self.n_units),
                          np.nan)
            for name in single_names}

        for j, n in enumerate(tqdm(n_list, desc=f"{self.name} single")):
            for trial_idx in range(self.n_trials):
                others = np.delete(np.arange(self.n_trials), trial_idx)
                subsets = get_unique_subsets(others, n - 1, n_bootstraps=n_bootstraps)
                for i, indices in enumerate(subsets):
                    subset = data[np.insert(indices, 0, trial_idx)]

                    if not run_loop:
                        results_single["LOOC"][j, trial_idx, i] = \
                            leave_one_out_correlation(subset, index=0)
                        results_single["TTMC"][j, trial_idx, i] = \
                            mean_to_one_correlation(subset, index=0)
                        continue

                    split = make_split(self.n_samples, chunked=self.chunked,
                                       n_chunks=n_chunk_test,
                                       test_fraction=test_fraction,
                                       buffer_size=test_buffer)
                    evaluated = subset[..., split[1]] if match_windows else subset

                    results_single["LOOC"][j, trial_idx, i] = \
                        leave_one_out_correlation(evaluated, index=0)
                    results_single["TTMC"][j, trial_idx, i] = \
                        mean_to_one_correlation(evaluated, index=0)
                    results_single["LOOP"][j, trial_idx, i] = loop_estimate(
                        subset, split, alphas, index=0, device=device,
                        n_validations=n_validations, val_chunk=self.chunked)

        mean_estimators = {
            "SH": lambda idx: split_half_correlation(data[idx]),
            "SHSB": lambda idx: split_half_correlation(data[idx], correction=True),
            "SNR": lambda idx: allen_noise_ceiling(data[idx]),
        }
        results_mean = {name: np.full((len(n_list), 1, n_bootstraps, self.n_units),
                                      np.nan)
                        for name in mean_estimators}

        for j, n in enumerate(tqdm(n_list, desc=f"{self.name} mean")):
            subsets = get_unique_subsets(np.arange(self.n_trials), n,
                                         n_bootstraps=n_bootstraps)
            for i, indices in enumerate(subsets):
                for name, estimator in mean_estimators.items():
                    results_mean[name][j, 0, i] = estimator(indices)

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
        keys = ("LOOC", "TTMC", "SH", "SHSB", "SNR")
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
                results["SNR"][i, j] = allen_noise_ceiling(window).mean()
 
        return results