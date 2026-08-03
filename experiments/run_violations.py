from __future__ import annotations

import argparse
import copy

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

from noise_ceiling.config import load_config
from noise_ceiling.datasets.synthetic import (ALL_ESTIMATORS, MEAN_ESTIMATORS,
                                              SINGLE_ESTIMATORS)
from noise_ceiling.datasets.violations import (ViolatedDataset, Violations,
                                               prepare_real_noise)
from noise_ceiling.io import ensure_dir, save_pickle
import matplotlib.pyplot as plt

from noise_ceiling.style import LABELS, PAPER_COLORS

PRESETS = {
    "hrf": ("hrf_subject_sd", "hrf_voxel_sd", "hrf_voxel_smooth",
            "hrf_dispersion_ratio"),
    "real_noise": ("real_noise",),
    "oscillation": ("oscillation_amplitude", "oscillation_freq_hz",
                    "oscillation_phase_jitter", "highpass_cutoff_s"),
    "highpass": ("highpass_cutoff_s",),
    "corr": ("shared_individual_corr",),
    "sigma_trial": ("sigma_trial_sd",),
}


def build_violations(config, enabled):
    """Take the `violations` block of the config, keep only the enabled keys."""
    configured = config.get("violations", {}) or {}
    keep = set()
    for name in enabled:
        keep.update(PRESETS[name])
    return Violations.from_dict({k: v for k, v in configured.items() if k in keep})


def load_real_noise(config, params, roi, rng):
    """Recorded data, time-shifted per trial, as the noise term."""
    from noise_ceiling.datasets.empirical import EmpiricalDataset

    source = config.get("real_noise", {})
    clean_kwargs = dict(source.get("clean_kwargs", {"standardize": False}))
    dataset = EmpiricalDataset.load(source.get("dataset", "sherlock"),
                                    roi=roi or source.get("roi"),
                                    clean_kwargs=clean_kwargs)
    noise, diagnostics = prepare_real_noise(
        dataset.data, n=params["n"], X=params["X"], Y=params["Y"], T=params["T"],
        rng=rng, scaling=source.get("scaling", "global"), return_diagnostics=True)
    print(f"  recorded noise from {dataset.name}: "
          f"inter-trial r = {diagnostics['isc_before']:+.4f} before shifting, "
          f"{diagnostics['isc_after']:+.4f} after; "
          f"noise level spread across trials = {diagnostics['trial_sd_spread']:.3f}, "
          f"across units = {diagnostics['unit_sd_spread']:.3f}")
    if abs(diagnostics["isc_after"]) > 0.02:
        print("  WARNING: residual inter-trial correlation after shifting; the "
              "noise still carries shared signal and v* is no longer exact.")
    if diagnostics["trial_sd_spread"] < 0.01:
        print("  WARNING: the noise level is identical in every trial, so the "
              "homoscedasticity assumption is not violated. Load with "
              "clean_kwargs={'standardize': False}.")
    return noise, diagnostics


def run_one(params, violations, sweep, run_loop, chunked, label):
    """Estimator sweep over n, averaged over dataset regenerations."""
    n_values = list(range(2, params["n"] + 1, 2))
    keys = ALL_ESTIMATORS + (("LOOP",) if run_loop else ())

    runs = []
    for _ in tqdm(range(sweep["n_iter"]), desc=label, leave=False):
        dataset = ViolatedDataset(violations=violations, **params)
        runs.append(dataset.sweep_n(n_values, n_rep=sweep["n_rep"],
                                    match_theory_trial=True,
                                    run_loop=run_loop, chunked=chunked))

    return n_values, {k: np.array([r["empirical"][k] for r in runs]) for k in keys}


def plot(results, n_values, title, path):
    conditions = list(results)
    fig, axes = plt.subplots(2, len(conditions), figsize=(4.2 * len(conditions), 7),
                             squeeze=False, sharex=True, sharey=True)

    for col, condition in enumerate(conditions):
        data = results[condition]
        for row, group in enumerate((MEAN_ESTIMATORS, SINGLE_ESTIMATORS)):
            ax = axes[row][col]
            for key in group:
                values = data[key]
                mean = values.mean(0)
                err = 2 * values.std(0) / np.sqrt(max(values.shape[0], 1))
                style = dict(color=PAPER_COLORS.get(key))
                if key.startswith("NC_"):
                    style.update(color="black", linestyle="--", linewidth=2)
                ax.errorbar(n_values, mean, yerr=err, label=LABELS.get(key, key),
                            capsize=2, **style)
            if row == 0:
                ax.set_title(condition, fontsize=10)
            if col == 0:
                ax.set_ylabel("noise ceiling")
            ax.set_xlabel("number of trials (n)")
    axes[0][0].legend(fontsize=7)
    axes[1][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    ensure_dir(path)
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    for name in PRESETS:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true",
                            help=f"enable the {name} violation")
    parser.add_argument("--all", action="store_true", help="enable every violation")
    parser.add_argument("--baseline", action="store_true",
                        help="also run with no violation, as a reference column")
    parser.add_argument("--roi", default=None, help="ROI for the recorded noise")
    parser.add_argument("--sweep", default=None,
                        help="violation field to sweep, e.g. hrf_subject_sd")
    parser.add_argument("--sweep-values", nargs="*", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="suffix for the output files")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config = load_config(args.config)
    params, sweep = config["model"], config["sweep"]
    run_loop = config.get("estimators", {}).get("run_loop", False)
    chunked = config.get("estimators", {}).get("chunked", False)

    enabled = [name for name in PRESETS if getattr(args, name)]
    if args.all:
        enabled = [k for k in PRESETS if k != "highpass"]
    violations = build_violations(config, enabled)

    if "real_noise" in enabled:
        noise, _ = load_real_noise(config, params, args.roi, rng)
        violations.real_noise = noise

    conditions = {}
    if args.baseline or not enabled:
        conditions["baseline"] = Violations()
    if enabled:
        if args.sweep:
            values = args.sweep_values or [0.0, 1.0, 2.0]
            for value in values:
                variant = copy.copy(violations)
                setattr(variant, args.sweep, value)
                conditions[f"{args.sweep}={value:g}"] = variant
        else:
            conditions["+".join(enabled)] = violations

    results = {}
    n_values = None
    for label, variant in conditions.items():
        print(f"[{label}] violations: {variant.describe()}")
        n_values, results[label] = run_one(params, variant, sweep, run_loop,
                                           chunked, label)

    tag = args.tag or ("_".join(enabled) if enabled else "baseline")
    name = f"{config['name']}_{tag}"

    figure_path = f"figures/intermediate_figures/violations_{name}.pdf"
    plot(results, n_values, f"{config['name']} - {tag}   (dashed black = true NC)",
         figure_path)

    result_path = f"results/violations_{name}.pkl"
    save_pickle(result_path, {"n_list": n_values, "empirical": results,
                              "params": params, "enabled": enabled,
                              "violations": {k: vars(v) for k, v in conditions.items()}})
    print(f"wrote {figure_path} and {result_path}")


if __name__ == "__main__":
    main()
