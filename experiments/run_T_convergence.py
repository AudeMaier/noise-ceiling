"""Finite-T bias of the noise ceiling estimators.

Bias is measured against the estimate at the full recording length rather than
against the closed forms, which are accurate only to order 1/n in absolute
level; that offset would otherwise dominate the measured bias beyond a few
hundred samples and invert its sign. Anchoring cancels any constant offset and
tests the claim under review, that the bias decays as 1/T.

Since measured(T) = c (T^-p - T_ref^-p), the exponent is estimated with
fit_power rather than a log-log slope, which would be biased steep.

    python experiments/run_T_convergence.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_ceiling.analysis import bias, fit_power
from noise_ceiling.config import load_config
from noise_ceiling.datasets.synthetic import ALL_ESTIMATORS, SyntheticDataset
from noise_ceiling.io import ensure_dir, load_pickle, save_pickle
from noise_ceiling.style import SWEEP_COLORS

CACHE = "data/finite_T_bias/finite_T_bias_tmax.pkl"
FIGURE = "figures/finite_T_bias.png"


def simulate(params, sweep, T_values):
    runs = []
    for _ in tqdm(range(sweep["n_outer"]), desc="datasets"):
        dataset = SyntheticDataset(**params)
        result = dataset.sweep_T(T_values, n_rep=sweep["n_rep"],
                                 realized=sweep["realized"], anchor="tmax",
                                 T_ref=params["T"])
        runs.append(bias(result))
    return runs


def main(config_path):
    config = load_config(config_path)
    params, sweep = config["model"], config["sweep"]
    T_values = np.logspace(np.log10(sweep["T_min"]), np.log10(sweep["T_max"]),
                           sweep["n_T"])
    T_ref = params["T"]

    if os.path.exists(CACHE):
        blob = load_pickle(CACHE)
        runs, T_ref = blob["runs"], blob["T_ref"]
        print(f"loaded {len(runs)} runs from {CACHE}")
    else:
        runs = simulate(params, sweep, T_values)
        save_pickle(CACHE, {"runs": runs, "T": T_values, "T_ref": T_ref,
                            "params": params})

    measured = {k: np.array([r[k] for r in runs]) for k in ALL_ESTIMATORS}

    n_fitted = sweep["n_T_fitted"]
    print(f"\nfit  measured(T) = c (T^-p - T_ref^-p),  T_ref = {T_ref}")
    print(f"mean +- SE of p over {len(runs)} datasets")
    print(f"{'estimator':<11}{'p':>9}{'SE':>8}{'95% CI':>18}{'c':>10}")
    print("-" * 56)

    fits = {}
    for k in ALL_ESTIMATORS:
        per_run = [fit_power(T_values[:n_fitted], row[:n_fitted], T_ref)
                   for row in measured[k]]
        p = np.array([x[1] for x in per_run])
        c = np.array([x[0] for x in per_run])
        p, c = p[np.isfinite(p)], c[np.isfinite(c)]
        mean, se = p.mean(), p.std(ddof=1) / np.sqrt(len(p))
        fits[k] = (mean, se, c.mean())
        print(f"{k:<11}{mean:9.3f}{se:8.3f}   "
              f"[{mean - 1.96 * se:6.3f}, {mean + 1.96 * se:6.3f}]{c.mean():10.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in ALL_ESTIMATORS:
        mean = measured[k].mean(0)
        se = measured[k].std(0, ddof=1) / np.sqrt(measured[k].shape[0])
        p, _, c = fits[k]
        ax.errorbar(T_values, mean, yerr=1.96 * se, color=SWEEP_COLORS[k],
                    capsize=2, lw=1.4, label=f"{k}  (p = {p:.2f})")
        ax.plot(T_values, c * (T_values ** -p - float(T_ref) ** -p), "--",
                color="gray", lw=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples $T$")
    ax.set_ylabel(r"Bias  $\mathbb{E}[\widehat{NC}(T_{\mathrm{ref}})]"
                  r"-\mathbb{E}[\widehat{NC}(T)]$")
    ax.set_title("Finite-$T$ bias (positive: every estimator underestimates)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(FIGURE)
    fig.savefig(FIGURE, dpi=150)
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/synthetic/finite_T.yaml")
    main(parser.parse_args().config)
