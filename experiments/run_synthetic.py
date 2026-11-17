"""Noise ceiling estimators against the number of trials, on synthetic data.

Writes results/synthetic_results_{name}.pkl, read by make_figures.py, and a
diagnostic figure comparing the empirical sweep against the closed forms.

    python experiments/run_synthetic.py configs/synthetic/within_subject.yaml
    python experiments/run_synthetic.py            # all configs
"""

from __future__ import annotations

import argparse
import glob

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

from noise_ceiling.config import load_config
from noise_ceiling.datasets.empirical import trial_grid
from noise_ceiling.datasets.synthetic import (ALL_ESTIMATORS, MEAN_ESTIMATORS,
                                              SINGLE_ESTIMATORS, SyntheticDataset)
from noise_ceiling.io import ensure_dir, save_pickle
from noise_ceiling.plotting import plot_theory_vs_empirical

DEFAULT_CONFIGS = sorted(set(glob.glob("configs/synthetic/*.yaml"))
                         - set(glob.glob("configs/synthetic/finite_T.yaml")))


def run(config_path):
    config = load_config(config_path)
    name, params, sweep = config["name"], config["model"], config["sweep"]
    run_loop = config["estimators"]["run_loop"]
    keys = ALL_ESTIMATORS + (("LOOP", "LOOP_null") if run_loop else ())
    n_values = trial_grid(params["n"])
    runs = []
    for _ in tqdm(range(sweep["n_iter"]), desc=name):
        dataset = SyntheticDataset(**params)
        runs.append(dataset.sweep_n(n_values, n_rep=sweep["n_rep"],
                                    match_theory_trial=True,
                                    run_loop=run_loop,
                                    run_lengths=config["estimators"].get("run_lengths")))

    empirical = {k: np.array([r["empirical"][k] for r in runs]) for k in keys}
    theory = {k: np.array([r["theory"][k] for r in runs]) for k in ALL_ESTIMATORS}

    figure_path = f"figures/intermediate_figures/synthetic_results_{name}.pdf"
    ensure_dir(figure_path)
    fig = plot_theory_vs_empirical(
        empirical, theory, n_values, MEAN_ESTIMATORS, SINGLE_ESTIMATORS,
        n_repetitions=sweep["n_iter"],
        title=f"{name}   (dashed = theory, solid = empirical, 95% CI)", loop=run_loop)
    fig.savefig(figure_path)

    # the null LOOP is a check of the split, shown on the diagnostic figure only
    empirical.pop("LOOP_null", None)
    result_path = f"results/synthetic_results_{name}.pkl"
    save_pickle(result_path, {"n_list": n_values, "empirical": empirical,
                              "theory": theory, "params": params})
    print(f"wrote {figure_path} and {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    for path in parser.parse_args().configs:
        run(path)
