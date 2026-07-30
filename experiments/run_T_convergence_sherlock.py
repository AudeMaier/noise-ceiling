"""Finite-T trend of the estimators on recorded data.

Every estimator is measured on random windows of increasing length, with all
trials kept. The mean over windows is fitted with a - b T^-p, so that the
asymptote and the rate of approach can be read off separately.

    python experiments/run_T_convergence_sherlock.py
    python experiments/run_T_convergence_sherlock.py --roi V1 --roi A1 --n-rep 50

Results are cached per ROI; delete the pickle to recompute.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from noise_ceiling.analysis import fit_saturating
from noise_ceiling.datasets.empirical import EmpiricalDataset
from noise_ceiling.io import ensure_dir, load_pickle, save_pickle
from noise_ceiling.plotting import plot_finite_T

ESTIMATORS = ["SH", "SHSB", "SNR", "LOOC", "TTMC"]
T_VALUES = list(range(10, 100))
FIGURE = "figures/finite_T_sherlock.svg"


def measure(roi, n_rep, run_lengths, select_units, recompute):
    cache = f"results/finite_T_sherlock_{roi}.pkl"
    if os.path.exists(cache) and not recompute:
        print(f"loaded {cache}")
        return load_pickle(cache)["results"]

    clean_kwargs = None if run_lengths else {"run_lengths": None}
    dataset = EmpiricalDataset.load("sherlock", roi=roi, clean_kwargs=clean_kwargs)
    print(f"{dataset.name}: {dataset.n_trials} trials, {dataset.n_units} units, "
          f"{dataset.n_samples} samples")

    if select_units:
        dataset.select_signal_units(threshold=2,
                                    cache=f"results/AIC_dist/aic_dist_sherlock_{roi}.npy")

    results = dataset.sweep_T(T_VALUES, n_rep=n_rep)
    save_pickle(cache, {"T": T_VALUES, "n_rep": n_rep, "results": results,
                        "run_lengths": run_lengths, "select_units": select_units})
    return results


def main(rois, n_rep, recompute, free_exponent):
    panels, fits = {}, {}

    for roi in rois:
        results = measure(roi, n_rep=n_rep, recompute=recompute, run_lengths=False, select_units=False)
        panels[roi] = results
        fits[roi] = {}

        print(f"\n{roi}:  fit  NC(T) = a - b T^-p")
        header = f"{'estimator':<11}{'a':>9}{'b':>12}"
        header += f"{'p':>9}{'95% CI on p':>20}" if free_exponent else ""
        print(header)
        print("-" * len(header))

        for key in ESTIMATORS:
            popt, se = fit_saturating(T_VALUES, np.nanmean(results[key], axis=1),
                                      free_exponent=free_exponent)
            fits[roi][key] = popt
            if popt is None:
                print(f"{key:<11}{'fit failed':>27}")
                continue
            line = f"{key:<11}{popt[0]:9.3f}{popt[1]:12.4g}"
            if free_exponent:
                low, high = popt[2] - 1.96 * se[2], popt[2] + 1.96 * se[2]
                line += f"{popt[2]:9.3f}   [{low:7.3f}, {high:7.3f}]"
            print(line)

    fig = plot_finite_T(panels, T_VALUES, ESTIMATORS, fits,
                        free_exponent=free_exponent)
    ensure_dir(FIGURE)
    fig.savefig(FIGURE, bbox_inches="tight")
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", action="append", dest="rois",
                        help="repeatable; defaults to V1 and IT")
    parser.add_argument("--n-rep", type=int, default=300,
                        help="random windows per length")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--fixed-exponent", action="store_true",
                        help="fit a - b/T instead of estimating the exponent")
    args = parser.parse_args()

    main(args.rois or ["V1", "IT"], args.n_rep, args.recompute, not args.fixed_exponent)