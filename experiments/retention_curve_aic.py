from __future__ import annotations

import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_ceiling.datasets import generation
from noise_ceiling.estimators.empirical.estimators import (split_half_correlation,
                                                           true_noise_ceiling_mean)
from noise_ceiling.io import ensure_dir
from noise_ceiling.selection import run_aic_procedure

# noise levels pooled to populate the whole range of true noise ceilings
SIGMAS = {"things": [0.4, 0.8, 1.2, 1.6, 2.0, 2.5], "sherlock": [0.4, 0.8, 1.2, 1.6, 2.0, 2.5]}
THRESHOLDS = [0, 2, 4, 6, 8, 10]
BINS = np.arange(0, 1.01, 0.05)
AIC = dict(n_bootstrap=50, n_samples=100)
SETTINGS = {
    # THINGS fMRI1: 12 repetitions of 100 discrete stimuli, within-subject
    "things": dict(X=10, Y=10, K=300, crop=[100, 100], T=100, n=12, width=1.0,
                   sigma=0.5, shared_fraction=0.2, sigma_origin=1.0,
                   within_subject=True, gaussian_signals=True),
    # Sherlock: 16 subjects, continuous stimulus, HRF-convolved
    "sherlock": dict(X=10, Y=10, K=300, crop=[100, 100], T=2000, n=16, width=1.0,
                     sigma=0.8, shared_fraction=0.2, sigma_origin=2,
                     within_subject=False, gaussian_signals=False),
}


def realisation(params, aic_config, rng_seed=None):
    """True ceiling and AIC score for every unit of one realisation."""
    if rng_seed is not None:
        np.random.seed(rng_seed)

    T, n = params["T"], params["n"]
    grids = generation.generate_voxel_grids(**params)
    flat = grids["voxel_grids"].reshape(n, -1, T)
    flat_noiseless = grids["noiseless_grids"].reshape(n, -1, T)

    truth = true_noise_ceiling_mean(flat, flat_noiseless)
    _, aic_dist = run_aic_procedure(flat, n_list=range(2, n, 2),
                                    n_bootstrap=aic_config["n_bootstrap"],
                                    n_samples=aic_config["n_samples"],
                                    aic_threshold=0)
    shsb = split_half_correlation(flat, correction=True)
    sh = split_half_correlation(flat, correction=False)
    return truth, aic_dist.mean(axis=1), shsb, sh


def collect(setting, sigmas, reps, seed, aic_config):
    params = SETTINGS[setting]
    truths, scores, ranks = [], [], []

    for i, sigma in enumerate(sigmas):
        for r in range(reps):
            truth, score, shsb, sh = realisation(
                {**params, "sigma": sigma}, aic_config,
                rng_seed=None if seed is None else seed + 100 * i + r)
            truths.append(truth)
            scores.append(score)
            ranks.append(np.lexsort((sh, shsb)))

    return truths, scores, ranks


def retention(truths, scores, ranks, threshold, bins):
    """Share of units kept in each bin of true noise ceiling, for both rules."""
    print(bins)
    kept_aic, all_truth = [], [], []
    for truth, score, order in zip(truths, scores, ranks):
        keep = score > threshold
        kept_aic.append(keep)
        all_truth.append(truth)

    truth = np.concatenate(all_truth)
    kept_aic = np.concatenate(kept_aic)

    index = np.digitize(truth, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        in_bin = index == b
        count = int(in_bin.sum())
        if count < 10:
            continue
        aic_share = float(kept_aic[in_bin].mean())
        rows.append({
            "threshold": threshold,
            "nc_low": float(bins[b]), "nc_high": float(bins[b + 1]),
            "nc_mid": float((bins[b] + bins[b + 1]) / 2),
            "count": count,
            "aic": aic_share,
            "aic_se": float(np.sqrt(aic_share * (1 - aic_share) / count))
        })
    return rows


def half_retention(rows, key="aic"):
    """True ceiling at which retention crosses one half, by linear interpolation."""
    points = [(r["nc_mid"], r[key]) for r in rows]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < 0.5 <= y1:
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return np.nan


def plot(all_rows, setting):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#4A93D8", "#4CC275", "#EB8642", "#C55995", "#630303", "#535353"]

    for color, (threshold, rows) in zip(colors, sorted(all_rows.items())):
        x = [r["nc_mid"] for r in rows]
        y = [r["aic"] for r in rows]
        e = [2 * r["aic_se"] for r in rows]
        ax.errorbar(x, y, yerr=e, color=color, linewidth=2, capsize=2,
                    label=f"AIC threshold {threshold:g}")

    ax.axhline(0.5, color="lightgray", linestyle=":", zorder=0)
    ax.set_xlabel("True noise ceiling of the unit")
    ax.set_ylabel("Probability of being kept")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Retention of the AIC selection, {setting}\n", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=sorted(SETTINGS), default="things")
    parser.add_argument("--reps", type=int, default=10,
                        help="realisations per noise level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    truths, scores, ranks = collect(args.setting, SIGMAS[args.setting],
                                    args.reps, args.seed, AIC)
    pooled = np.concatenate(truths)
    print(f"\n{args.setting}: {len(pooled)} units pooled over "
          f"sigma in {SIGMAS[args.setting]}, {args.reps} realisations each")
    print(f"true noise ceiling spans {pooled.min():.2f} to {pooled.max():.2f}, "
          f"mean {pooled.mean():.2f}")

    all_rows = {}
    for threshold in THRESHOLDS:
        rows = retention(truths, scores, ranks, threshold, BINS)
        all_rows[threshold] = rows
        print(f"\nAIC threshold {threshold:g}   "
              f"(50% retention at true ceiling {half_retention(rows):.3f}, "
              )
        print(f"{'true NC':>14}{'units':>7}{'kept AIC':>10}")
        for r in rows:
            print(f"  {r['nc_low']:.2f} - {r['nc_high']:.2f}{r['count']:>10}"
                  f"{r['aic']:>10.2f}")

    out = args.out or f"results/retention_{args.setting}.csv"
    ensure_dir(out)
    flat_rows = [r for rows in all_rows.values() for r in rows]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    figure = f"figures/retention_{args.setting}.svg"
    ensure_dir(figure)
    plot(all_rows, args.setting).savefig(figure, bbox_inches="tight")
    print(f"\nwrote {out} and {figure}")