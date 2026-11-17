"""Finite-T trend of the estimators on recorded data.

Every estimator is measured on random windows of increasing length, with all
trials kept. The mean over windows is fitted with a - b T^-p, so that the
asymptote and the rate of approach can be read off separately.

    python experiments/run_T_convergence_sherlock.py
    python experiments/run_T_convergence_sherlock.py --roi V1 --roi A1 --n-rep 50

Results are cached per ROI and reused when computed with the same number of
windows and the same cleaning; --recompute forces a new run.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from noise_ceiling.analysis import fit_saturating
from noise_ceiling.datasets.empirical import EmpiricalDataset
from noise_ceiling.io import ensure_dir, load_pickle, rename_legacy, save_pickle
from noise_ceiling.plotting import plot_finite_T

ESTIMATORS = ["SHSB", "ATV", "LOOC", "TTMC"]
T_VALUES = list(range(10, 100))
FIGURE = "figures/figures_paper/finite_T_sherlock.pdf"


def measure(roi, n_rep, run_lengths, recompute):
    cache = f"results/finite_T_sherlock_{roi}.pkl"
    if os.path.exists(cache) and not recompute:
        blob = load_pickle(cache)
        if (blob["n_rep"] == n_rep and blob["run_lengths"] == run_lengths
                and list(blob["T"]) == T_VALUES):
            print(f"loaded {cache}")
            return rename_legacy(blob["results"])
        print(f"{cache} was computed with other settings; recomputing")

    clean_kwargs = None if run_lengths else {"run_lengths": None}
    dataset = EmpiricalDataset.load("sherlock", roi=roi, clean_kwargs=clean_kwargs)
    print(f"{dataset.name}: {dataset.n_trials} trials, {dataset.n_units} units, "
          f"{dataset.n_samples} samples")

    results = dataset.sweep_T(T_VALUES, n_rep=n_rep)
    save_pickle(cache, {"T": T_VALUES, "n_rep": n_rep, "results": results,
                        "run_lengths": run_lengths})
    return results


def main(rois, n_rep, recompute, free_exponent, T_max, run_lengths, sh=False):
    estimators = (["SH"] if sh else []) + ESTIMATORS
    panels, fits, errors = {}, {}, {}
    shown = [i for i, T in enumerate(T_VALUES) if T <= T_max]
    T_shown = [T_VALUES[i] for i in shown]

    for roi in rois:
        results = measure(roi, n_rep=n_rep, recompute=recompute,
                          run_lengths=run_lengths)
        fits[roi], errors[roi] = {}, {}

        print(f"\n{roi}:  fit  NC(T) = a - b T^-p")
        header = f"{'estimator':<11}{'a':>9}{'b':>12}"
        header += f"{'p':>9}{'95% CI on p':>20}" if free_exponent else ""
        print(header)
        print("-" * len(header))

        for key in estimators:
            popt, se = fit_saturating(T_shown, np.nanmean(results[key], axis=1)[shown],
                                      free_exponent=free_exponent)
            fits[roi][key], errors[roi][key] = popt, se
            if popt is None:
                print(f"{key:<11}{'fit failed':>27}")
                continue
            line = f"{key:<11}{popt[0]:9.3f}{popt[1]:12.4g}"
            if free_exponent:
                low, high = popt[2] - 1.96 * se[2], popt[2] + 1.96 * se[2]
                line += f"{popt[2]:9.3f}   [{low:7.3f}, {high:7.3f}]"
            print(line)
        panels[roi] = {key: values[shown] for key, values in results.items()}

    fig = plot_finite_T(panels, T_shown, estimators, fits,
                        free_exponent=free_exponent, errors=errors,
                        legend_fontsize=10, xlabel="Stimulus length $T$")
    ensure_dir(FIGURE)
    fig.savefig(FIGURE, bbox_inches="tight")
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", action="append", dest="rois",
                        help="repeatable; defaults to V1 and IT")
    parser.add_argument("--n-rep", type=int, default=100,
                        help="random windows per length")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--fixed-exponent", action="store_true",
                        help="fit a - b/T instead of estimating the exponent")
    parser.add_argument("--t-max", type=int, default=100,
                        help="largest window length fitted and plotted")
    parser.add_argument("--single-run", action="store_true",
                        help="clean the two runs as one, keeping the step at "
                             "the run boundary")
    parser.add_argument("--sh", action="store_true", help="also fit and plot SH")
    args = parser.parse_args()

    main(args.rois or ["V1", "IT"], args.n_rep, args.recompute,
         not args.fixed_exponent, args.t_max, run_lengths=not args.single_run,
         sh=args.sh)