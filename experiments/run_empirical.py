"""Noise ceiling estimators against the number of trials, on recorded data.

Everything that differs between the datasets lives in noise_ceiling.datasets.
loaders.DATASETS: paths, cleaning, AIC threshold, and whether test chunks have
to be purged.

    python experiments/run_empirical.py sherlock          # every ROI
    python experiments/run_empirical.py sherlock --roi V1
    python experiments/run_empirical.py things
    python experiments/run_empirical.py eeg
"""

from __future__ import annotations

import argparse

import matplotlib
import torch

matplotlib.use("Agg")

from noise_ceiling.datasets.empirical import EmpiricalDataset
from noise_ceiling.datasets.loaders import DATASETS
from noise_ceiling.io import ensure_dir, save_results
from noise_ceiling.plotting import plot_sweep


def run(name, roi=None, run_loop=False, recompute_aic=False, device=None):
    spec = DATASETS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmpiricalDataset.load(name, roi=roi)
    print(f"{dataset.name}: {dataset.n_trials} trials, {dataset.n_units} units, "
          f"{dataset.n_samples} samples")

    if spec["aic"] is not None:
        dataset.select_signal_units(
            threshold=spec["aic"]["threshold"],
            n_bootstrap=spec["aic"]["n_bootstrap"],
            n_samples=spec["aic"]["n_samples"],
            cache=spec["aic_cache"].format(roi=roi),
            recompute=recompute_aic)

    results, n_list = dataset.sweep_n(n_bootstraps=spec["n_bootstraps"],
                                      run_loop=run_loop, device=device)

    result_path = spec["results"].format(roi=roi)
    save_results(result_path, results)

    figure_path = f"figures/intermediate_figures/sweep_{dataset.name}.pdf"
    ensure_dir(figure_path)
    plot_sweep(results, n_list, title=dataset.name).savefig(figure_path)
    print(f"wrote {result_path} and {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--roi", default=None,
                        help="Sherlock only; every ROI is run if omitted")
    parser.add_argument("--loop", action="store_true",
                        help="also compute the LOOP estimator (slow, needs a GPU)")
    parser.add_argument("--recompute-aic", action="store_true",
                        help="redo the unit selection instead of reusing the cache")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rois = DATASETS[args.dataset]["rois"]
    if rois is None:
        run(args.dataset, run_loop=args.loop, recompute_aic=args.recompute_aic,
            device=args.device)
    else:
        for roi in ([args.roi] if args.roi else rois):
            run(args.dataset, roi=roi, run_loop=args.loop,
                recompute_aic=args.recompute_aic, device=args.device)
