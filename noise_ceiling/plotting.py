"""Figures.

`plot_synthetic_vs_empirical` and `plot_empirical` produce the paper figures;
`plot_theory_vs_empirical` and `plot_sweep` are the diagnostics the run scripts
write while a sweep is in progress.

Estimators are given as short keys; any key missing from a result dict is
skipped, which is how the recorded datasets drop the two true noise ceilings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy

from .style import (ELINEWIDTH, FONT_LABEL, FONT_LEGEND, FONT_TICKS, FONT_TITLE,
                    FONT_XLABEL, LABELS, LINEWIDTH, LINEWIDTH_THEORY, LINEWIDTHS,
                    PANEL_GAP, PANEL_SIZE, PAPER_COLORS, SWEEP_COLORS)


def _mean_over_all_but_first(values):
    """Mean over every axis except the first."""
    return np.nanmean(values, axis=tuple(range(1, values.ndim)))


def _mean_and_ci(values):
    """Mean over the first axis (repetitions) and the half-width of its asymptotic
    95% interval, 1.96 standard errors."""
    count = np.sum(~np.isnan(values), axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(count)
    return np.nanmean(values, axis=0), 1.96 * se


def _panel_row(n_panels):
    """A figure holding a row of panels of exactly PANEL_SIZE inches.

    Tick labels, axis labels, titles and legends are drawn outside the panels;
    saving with bbox_inches="tight" grows the file to include them.
    """
    width, height = PANEL_SIZE
    fig_width = n_panels * width + (n_panels - 1) * PANEL_GAP
    fig = plt.figure(figsize=(fig_width, height))
    axes = [fig.add_axes([i * (width + PANEL_GAP) / fig_width, 0, width / fig_width, 1])
            for i in range(n_panels)]
    return fig, axes


def _above_panels(fig, points):
    """Figure coordinates shifted up by `points`, to place text above the panel titles."""
    return offset_copy(fig.transFigure, fig=fig, y=points, units="points")


def _style_panel(ax, title, ylim, xlabel):
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(labelsize=FONT_TICKS)
    ax.set_xlabel(xlabel, fontsize=FONT_XLABEL)
    ax.set_title(title, fontsize=FONT_TITLE)


def plot_synthetic_vs_empirical(synthetic, theory, empirical, n_list, estimators,
                                n_list_empirical=None, ylim=(0, 0.9),
                                titles=("Synthetic Data", "Brain Data"),
                                xlabel="Number of trials (n)",
                                ylabel="Noise Ceiling Value", legend_ncol=2, loop=False,
                                legend=True, axes=None):
    """Synthetic and recorded estimates side by side, closed forms dotted.

    synthetic, theory : (n_repetitions, len(n_list)), one entry per estimator.
                        The synthetic error bars are asymptotic 95% intervals
                        over the repetitions (generated datasets).
    empirical         : (len(n_list_empirical), ...), averaged over every
                        remaining axis.
    n_list_empirical  : trial counts of the recorded sweep, n_list if None.
    axes              : (synthetic, recorded) axes to draw into, for a figure
                        holding several pairs; a new figure is made if None.
    """
    n_list_empirical = n_list if n_list_empirical is None else n_list_empirical
    if axes is None:
        fig, (ax_syn, ax_emp) = _panel_row(2)
    else:
        (ax_syn, ax_emp), fig = axes, axes[0].figure

    synthetic_keys = list(estimators) + (["LOOP"] if loop and "LOOP" in synthetic else [])
    for key in synthetic_keys:
        mean, ci = _mean_and_ci(synthetic[key])
        ax_syn.errorbar(n_list, mean, yerr=ci, color=PAPER_COLORS[key], label=LABELS[key],
                        linewidth=LINEWIDTHS.get(key, LINEWIDTH), elinewidth=ELINEWIDTH,
                        zorder=1)
        if key in theory:
            ax_syn.plot(n_list, np.nanmean(theory[key], axis=0), color="k",
                        linestyle="dotted", linewidth=LINEWIDTH_THEORY, zorder=2)

    empirical_keys = [k for k in estimators if k in empirical]
    if loop and "LOOP" in empirical:
        empirical_keys.append("LOOP")
    for key in empirical_keys:
        ax_emp.plot(n_list_empirical, _mean_over_all_but_first(empirical[key]),
                    label=LABELS[key], color=PAPER_COLORS[key],
                    linewidth=LINEWIDTHS.get(key, LINEWIDTH))

    for ax, title in ((ax_syn, titles[0]), (ax_emp, titles[1])):
        _style_panel(ax, title, ylim, xlabel)

    ax_syn.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax_emp.set_yticklabels([])
    if legend:
        handles, labels = ax_syn.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=legend_ncol, fontsize=FONT_LEGEND,
                   loc="lower center", bbox_to_anchor=(0.5, 1),
                   bbox_transform=_above_panels(fig, FONT_TITLE + 10))
    return fig


def plot_empirical(results, n_list, mean_estimators, single_estimators,
                   ylim=(0, 0.9), title=None,
                   xlabel="Number of trials (n)", ylabel="Noise Ceiling Value",
                   legend=True):
    """Mean and single-trial estimators of one recorded dataset, side by side."""
    fig, (ax_mean, ax_single) = _panel_row(2)
    panels = ((ax_mean, mean_estimators, r"$NC^{mean}$"),
              (ax_single, single_estimators, r"$NC^{single}$"))

    for ax, estimators, panel_title in panels:
        for key in estimators:
            if key not in results:
                continue
            ax.plot(n_list, _mean_over_all_but_first(results[key]),
                    label=LABELS[key], color=PAPER_COLORS[key],
                    linewidth=LINEWIDTHS.get(key, LINEWIDTH))
        _style_panel(ax, panel_title, ylim, xlabel)

    ax_mean.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax_single.set_yticklabels([])
    if legend:
        handles, labels = [sum(pair, []) for pair in zip(
            ax_mean.get_legend_handles_labels(), ax_single.get_legend_handles_labels())]
        fig.legend(handles, labels, fontsize=FONT_LEGEND, loc="center left",
                   bbox_to_anchor=(1, 0.5),
                   bbox_transform=offset_copy(fig.transFigure, fig=fig, x=8, units="points"))

    if title is not None:
        fig.text(0.5, 1, title, ha="center", va="bottom", fontsize=FONT_TITLE + 3,
                 transform=_above_panels(fig, FONT_TITLE + 10))
    return fig


def plot_finite_T(panels, T_values, estimators, fits, free_exponent=True,
                  xlabel="Number of samples (T)", ylabel="Noise Ceiling Value",
                  errors=None, legend_fontsize=13):
    """Estimates against window length, one panel per dataset.

    panels : ordered mapping from panel title to a dict of (len(T), n_rep) arrays.
    fits   : same keys as `panels`, mapping estimator to the parameters returned
             by `fit_saturating`, or None where the fit failed.
    errors : optional, like `fits` but with the standard errors returned by
             `fit_saturating`; the legend then gives the 95% interval of p.

    Points are the mean over windows; the dashed line is the fitted
    a - b T^-p, and the legend gives the exponent.
    """
    from .analysis import saturating

    titles = list(panels)
    fig, axes = plt.subplots(1, len(titles), figsize=(5 * len(titles), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(min(T_values), max(T_values), 200)

    for ax, title in zip(axes, titles):
        for key in estimators:
            if key not in panels[title]:
                continue

            popt = fits[title].get(key)
            if popt is None:
                ax.plot([], [], color=PAPER_COLORS[key], label=f"{LABELS[key]}")
                continue
            exponent = popt[2] if free_exponent else 1.0
            asymptote = popt[0]
            label = f"{LABELS[key]}   $p = {exponent:.2f}"
            if free_exponent and errors is not None and errors[title].get(key) is not None:
                label += rf" \pm {1.96 * errors[title][key][2]:.2f}"
            ax.plot(T_values, np.nanmean(panels[title][key], axis=1), "x",
                    color=PAPER_COLORS[key], markersize=6, label=label + "$")
            ax.plot(grid, saturating(grid, *popt), "k--",
                    linewidth=1.6)
            ax.axhline(asymptote, linestyle='--', color='darkgray', linewidth=1.6)

        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=13)
        #put legend outside of the plot
        ax.legend(fontsize=legend_fontsize, loc="upper left", bbox_to_anchor=(1, 1))

    axes[0].set_ylabel(ylabel, fontsize=12)
    fig.tight_layout()
    return fig


def plot_theory_vs_empirical(empirical, theory, n_values, mean_estimators,
                             single_estimators, n_repetitions, title=None, loop=False):
    """Diagnostic: empirical sweep against the closed forms, with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    panels = ((axes[0], mean_estimators, r"$NC^{\mathrm{mean}}$ estimators"),
              (axes[1], single_estimators, r"$NC^{\mathrm{single}}$ estimators"))

    for ax, estimators, panel_title in panels:
        for key in estimators:
            mean = np.nanmean(empirical[key], axis=0)
            se = np.nanstd(empirical[key], axis=0, ddof=1) / np.sqrt(n_repetitions)
            ax.errorbar(n_values, mean, yerr=2 * se, label=key,
                        color=SWEEP_COLORS[key], capsize=2, lw=1.4)
            ax.plot(n_values, np.nanmean(theory[key], axis=0), "k--", lw=0.9, zorder=0)
        if loop and ax == axes[1]:
            mean = np.nanmean(empirical["LOOP"], axis=0)
            se = np.nanstd(empirical["LOOP"], axis=0, ddof=1) / np.sqrt(n_repetitions)
            ax.errorbar(n_values, mean, yerr=2 * se, label="LOOP", color=SWEEP_COLORS["LOOP"], capsize=2, lw=1.4)
            if "LOOP_null" in empirical:
                ax.plot(n_values, np.nanmean(empirical["LOOP_null"], axis=0), "--",
                        color=SWEEP_COLORS["LOOP_null"], lw=1.4, label=LABELS["LOOP_null"])
        ax.set_xlabel("Number of trials")
        ax.set_ylabel("Noise ceiling estimate")
        ax.set_title(panel_title)
        ax.legend(fontsize=8)

    if title is not None:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_sweep(results, n_list, title=None):
    """Diagnostic: every estimator of a recorded sweep on one pair of axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, values in results.items():
        ax.plot(n_list, _mean_over_all_but_first(values), label=LABELS.get(key, key),
                color=SWEEP_COLORS.get(key),
                linestyle="--" if key.endswith("_null") else "-")
    ax.set_xlabel("Number of trials (n)")
    ax.set_ylabel("Noise ceiling estimate")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return fig
