"""Paper figures, from the results written by the run scripts.

    python experiments/make_figures.py

Figure 2 : THINGS_fMRI1 against the within-subject synthetic regime.
Figure 3 : Sherlock V1 and IT against the matching synthetic regimes.
Figure 4 : every Sherlock ROI on its own.
EEG: THINGS_EEG2 on its own.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from noise_ceiling.io import ensure_dir, load_pickle, load_results
from noise_ceiling.plotting import plot_empirical, plot_synthetic_vs_empirical

MEAN_ESTIMATORS = ["NC_mean", "SH", "SHSB", "SNR"]
SINGLE_ESTIMATORS = ["NC_single", "LOOC", "TTMC"]

N_SYNTHETIC = 10      # dataset regenerations behind the synthetic error bars
N_EMPIRICAL = 10      # subsets behind the recorded error bars

# (figure name, synthetic regime, recorded results, trial counts)
COMPARISONS = [
    ("figure_2_THINGS", "within_subject", "results/THINGS_results.npy",
     [2, 4, 6, 8, 10, 12]),
    ("figure_3_V1", "sensory_areas", "results/sherlock_results_V1.npy",
     [2, 4, 6, 8, 10, 12, 14, 16]),
    ("figure_3_IT", "cognitive_areas", "results/sherlock_results_IT.npy",
     [2, 4, 6, 8, 10, 12, 14, 16]),
]

ROIS = ["V1", "A1", "IT", "amygdala", "hippocampus"]
ROI_TRIALS = [2, 4, 6, 8, 10, 12, 14, 16]
EEG_TRIALS = [2, 10, 18, 26, 34, 42, 50, 58, 66, 74]


def save(fig, path):
    ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    for figure, regime, empirical_path, n_list in COMPARISONS:
        synthetic = load_pickle(f"results/synthetic_results_{regime}.pkl")
        empirical = load_results(empirical_path)

        for suffix, estimators in (("mean", MEAN_ESTIMATORS),
                                   ("single", SINGLE_ESTIMATORS)):
            fig = plot_synthetic_vs_empirical(
                synthetic["empirical"], synthetic["theory"], empirical, n_list,
                estimators, n_synthetic=N_SYNTHETIC, n_empirical=N_EMPIRICAL)
            save(fig, f"figures/{figure}_{suffix}.svg")

    for roi in ROIS:
        results = load_results(f"results/sherlock_results_{roi}.npy")
        fig = plot_empirical(results, ROI_TRIALS, MEAN_ESTIMATORS,
                             SINGLE_ESTIMATORS, n_empirical=N_EMPIRICAL, title=roi)
        save(fig, f"figures/figure_4_{roi}.svg")

    results = load_results("results/eeg_results.npy")
    fig = plot_empirical(results, EEG_TRIALS, MEAN_ESTIMATORS,
                         SINGLE_ESTIMATORS, n_empirical=N_EMPIRICAL, title="EEG")
    save(fig, "figures/EEG.svg")


if __name__ == "__main__":
    main()
