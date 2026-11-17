"""Noise ceiling estimators against the number of trials, on recorded data.

Everything that differs between the datasets lives in noise_ceiling.datasets.
loaders.DATASETS: paths, cleaning, and whether LOOP is run.

    python experiments/run_empirical.py sherlock          # every ROI
    python experiments/run_empirical.py sherlock --roi V1
    python experiments/run_empirical.py things
    python experiments/run_empirical.py eeg
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from noise_ceiling.datasets.empirical import EmpiricalDataset
from noise_ceiling.datasets.loaders import DATASETS
from noise_ceiling.io import ensure_dir, save_results
from noise_ceiling.plotting import plot_sweep


def run(name, roi=None, run_loop=False, device=None):
    spec = DATASETS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmpiricalDataset.load(name, roi=roi)
    print(f"{dataset.name}: {dataset.n_trials} trials, {dataset.n_units} units, "
          f"{dataset.n_samples} samples")
    if run_loop and not spec["loop"]:
        print(f"LOOP is not run on {name}; computing the other estimators only")
        run_loop = False

    results, n_list = dataset.sweep_n(n_bootstraps=spec["n_bootstraps"],
                                      run_loop=run_loop, device=device)

    # the null LOOP is a check of the split, shown on the sweep figure only
    result_path = spec["results"].format(roi=roi)
    saved = {k: v for k, v in results.items() if k != "LOOP_null"}
    save_results(result_path, {**saved, "n_list": np.array(n_list)})

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
                        help="also compute LOOP, holding out each run in turn; every "
                             "estimator is then computed per run (slow, needs a GPU)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rois = DATASETS[args.dataset]["rois"]
    if rois is None:
        run(args.dataset, run_loop=args.loop, device=args.device)
    else:
        for roi in ([args.roi] if args.roi else rois):
            run(args.dataset, roi=roi, run_loop=args.loop, device=args.device)
