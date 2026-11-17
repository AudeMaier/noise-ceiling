"""Paper figures, from the results written by the run scripts.

    python experiments/make_figures.py
    python experiments/make_figures.py --recompute

Every figure is written to figures/figures_paper/:

Figure 2 : figure_2.pdf, one row per dataset (THINGS fMRI1, THINGS EEG2,
           Sherlock V1 and IT), each against its matching synthetic regime: the
           trial-averaged pair of panels on the left, the single-trial pair on
           the right, the rows grouped by regime. Every pair is also written on
           its own, figure_2_<row>_{mean,single}.svg.
Figure 3 : figure_3_<roi>.svg, every Sherlock ROI on its own.
EEG      : EEG.svg, THINGS EEG2 on its own.

The .svg figures are written twice: with their legend and as <name>_no_legend.svg.

The figures need only the mean of every curve at each n, while the recorded
results keep every unit, subset and held-out trial (over a gigabyte for EEG).
The means are therefore saved once to CURVES and reused, so that changing the
style takes seconds. They are recomputed when a result file is newer than
CURVES, or with --recompute.
"""

from __future__ import annotations
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from noise_ceiling.datasets.empirical import trial_grid
from noise_ceiling.datasets.loaders import DATASETS
from noise_ceiling.io import ensure_dir, load_pickle, load_results, rename_legacy, save_pickle
from noise_ceiling.plotting import plot_empirical, plot_synthetic_vs_empirical
from noise_ceiling.style import FONT_LEGEND, FONT_TITLE, PANEL_GAP, PANEL_SIZE

FIGURE_DIR = "figures/figures_paper"
CURVES = "results/figure_curves.pkl"

MEAN_ESTIMATORS = ["NC_mean", "SHSB", "ATV"]
SINGLE_ESTIMATORS = ["NC_single", "LOOC", "TTMC"]
SINGLE_ESTIMATORS_SHERLOCK = ["NC_single", "LOOC", "TTMC", "LOOP"]

# Figure 2, top to bottom: (row label, file name, synthetic regime, recorded dataset, ROI)
ROWS = [
    ("THINGS fMRI1", "THINGS", "within_subject", "things", None),
    ("THINGS EEG2", "EEG", "eeg", "eeg", None),
    ("Sherlock V1", "V1", "sensory_areas", "sherlock", "V1"),
    ("Sherlock IT", "IT", "cognitive_areas", "sherlock", "IT"),
]
# (label, first row, last row) of each bracket on the right of Figure 2
REGIMES = [("Homogeneous regime", 0, 1), ("Heterogeneous regime", 2, 3)]

# layout of Figure 2, in inches
ROW_GAP = 0.45      # between rows, room for the tick labels
PAIR_GAP = 0.9      # between the two pairs, room for the y label
BRACKET_X = 0.3     # from the right edge of the panels
BRACKET_TICK = 0.08


def _synthetic_path(regime):
    return f"results/synthetic_results_{regime}.pkl"


def _recorded():
    """(name, roi) of every recorded result the figures use."""
    return ([("sherlock", roi) for roi in DATASETS["sherlock"]["rois"]]
            + [("things", None), ("eeg", None)])


def _key(name, roi):
    return name if roi is None else f"{name}_{roi}"


def _load_recorded(name, roi):
    """Mean of every recorded curve over all but the first axis, and the trial counts.

    Older result files do not store the counts; they are then rebuilt from the
    number of trials with the grid `sweep_n` uses.
    """
    results = load_results(DATASETS[name]["results"].format(roi=roi))
    n_list = results.pop("n_list", None)
    if n_list is None:
        n_list = trial_grid(max(v.shape[1] for v in results.values()))
    curves = {key: np.nanmean(values, axis=tuple(range(1, values.ndim)))
              for key, values in results.items()}
    return {"curves": curves, "n_list": list(n_list)}


def load_curves(recompute=False):
    """Synthetic results and mean recorded curves, from CURVES when it is up to date."""
    regimes = sorted({regime for _, _, regime, _, _ in ROWS})
    sources = ([_synthetic_path(regime) for regime in regimes]
               + [DATASETS[name]["results"].format(roi=roi) for name, roi in _recorded()])
    if (not recompute and os.path.exists(CURVES)
            and os.path.getmtime(CURVES) >= max(map(os.path.getmtime, sources))):
        print(f"loaded {CURVES}")
        return load_pickle(CURVES)

    curves = {"synthetic": {}, "recorded": {}}
    for regime in regimes:
        synthetic = load_pickle(_synthetic_path(regime))
        curves["synthetic"][regime] = {"empirical": rename_legacy(synthetic["empirical"]),
                                       "theory": rename_legacy(synthetic["theory"]),
                                       "n_list": list(synthetic["n_list"])}
    for name, roi in _recorded():
        curves["recorded"][_key(name, roi)] = _load_recorded(name, roi)
        print(f"read {_key(name, roi)}")
    save_pickle(CURVES, curves)
    print(f"wrote {CURVES}")
    return curves


def _save(fig, name):
    path = f"{FIGURE_DIR}/{name}"
    ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def save_both(make_figure, name):
    """Write the figure with and without its legend; make_figure(legend) builds it."""
    for legend, suffix in ((True, ""), (False, "_no_legend")):
        _save(make_figure(legend), f"{name}{suffix}.svg")


def _pair(curves, regime, name, roi, estimators, loop, **kwargs):
    """Arguments of plot_synthetic_vs_empirical for one dataset and its synthetic regime."""
    synthetic = curves["synthetic"][regime]
    recorded = curves["recorded"][_key(name, roi)]
    return dict(synthetic=synthetic["empirical"], theory=synthetic["theory"],
                empirical=recorded["curves"], n_list=synthetic["n_list"],
                estimators=estimators, n_list_empirical=recorded["n_list"], loop=loop,
                **kwargs)


def figure_2(curves, mean_estimators, loop):
    width, height = PANEL_SIZE
    pair_width = 2 * width + PANEL_GAP
    fig_width = 2 * pair_width + PAIR_GAP
    fig_height = len(ROWS) * height + (len(ROWS) - 1) * ROW_GAP
    fig = plt.figure(figsize=(fig_width, fig_height))

    def bottom(row):
        return fig_height - (row + 1) * height - row * ROW_GAP

    def panel(row, pair, k):
        left = pair * (pair_width + PAIR_GAP) + k * (width + PANEL_GAP)
        return fig.add_axes([left / fig_width, bottom(row) / fig_height,
                             width / fig_width, height / fig_height])

    for row, (label, _, regime, name, roi) in enumerate(ROWS):
        labels = {}
        if row > 0:
            labels["titles"] = ("", "")
        if row < len(ROWS) - 1:
            labels["xlabel"] = ""

        for pair, estimators in enumerate((mean_estimators, SINGLE_ESTIMATORS)):
            plot_synthetic_vs_empirical(**_pair(
                curves, regime, name, roi, estimators, loop=loop and pair == 1,
                legend=False, axes=(panel(row, pair, 0), panel(row, pair, 1)), **labels))

        fig.text(0, (bottom(row) + height / 2) / fig_height, label, rotation=90,
                 ha="center", va="center", fontsize=FONT_TITLE + 1, fontweight="bold",
                 transform=offset_copy(fig.transFigure, fig=fig, x=-68, units="points"))

    # one legend per pair, taken from the last row, which has every curve (LOOP included)
    last_row = fig.axes[-4:]
    for pair in range(2):
        handles, names = last_row[2 * pair].get_legend_handles_labels()
        center = (pair * (pair_width + PAIR_GAP) + pair_width / 2) / fig_width
        fig.legend(handles, names, ncol=2, fontsize=FONT_LEGEND, loc="lower center",
                   bbox_to_anchor=(center, 1),
                   bbox_transform=offset_copy(fig.transFigure, fig=fig,
                                              y=FONT_TITLE + 10, units="points"))

    # Positions in inches, converted to figure fractions: saving with
    # bbox_inches="tight" moves the figure origin, which inch transforms do not follow.
    x = fig_width + BRACKET_X
    for label, first, last in REGIMES:
        top = bottom(first) + height + (0.35 if first == 0 else 0.05)
        low = bottom(last) - (0.55 if last == len(ROWS) - 1 else 0.3)
        xs = [(x - BRACKET_TICK) / fig_width, x / fig_width, x / fig_width,
              (x - BRACKET_TICK) / fig_width]
        ys = [top / fig_height, top / fig_height, low / fig_height, low / fig_height]
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color="k", linewidth=1.2))
        fig.text((x + 0.25) / fig_width, (top + low) / 2 / fig_height, label, rotation=90,
                 ha="center", va="center", fontsize=FONT_TITLE + 1)
    return fig


def main(loop=True, sh=False, recompute=False):
    mean_estimators = MEAN_ESTIMATORS[:1] + (["SH"] if sh else []) + MEAN_ESTIMATORS[1:]
    curves = load_curves(recompute)

    # Figure 2, as one figure and as separate pairs of panels
    _save(figure_2(curves, mean_estimators, loop), "figure_2.pdf")
    for _, file_name, regime, name, roi in ROWS:
        for suffix, estimators in (("mean", mean_estimators), ("single", SINGLE_ESTIMATORS)):
            save_both(lambda legend: plot_synthetic_vs_empirical(**_pair(
                curves, regime, name, roi, estimators,
                loop=loop and suffix == "single", legend=legend)),
                f"figure_2_{file_name}_{suffix}")

    # Figure 3, and THINGS EEG2 on its own
    for roi in DATASETS["sherlock"]["rois"]:
        recorded = curves["recorded"][_key("sherlock", roi)]
        save_both(lambda legend: plot_empirical(
            recorded["curves"], recorded["n_list"], mean_estimators,
            SINGLE_ESTIMATORS_SHERLOCK, title=roi, legend=legend), f"figure_3_{roi}")
    recorded = curves["recorded"]["eeg"]
    save_both(lambda legend: plot_empirical(
        recorded["curves"], recorded["n_list"], mean_estimators, SINGLE_ESTIMATORS,
        title="EEG", legend=legend), "EEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="leave LOOP out of the single-trial panels of Figure 2")
    parser.add_argument("--sh", action="store_true",
                        help="include SH in the trial-averaged panels")
    parser.add_argument("--recompute", action="store_true",
                        help="reread the result files instead of the saved curves")
    args = parser.parse_args()
    main(args.loop, args.sh, args.recompute)
