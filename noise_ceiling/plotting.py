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
from matplotlib.lines import Line2D

from .style import (ELINEWIDTH, LABELS, LINEWIDTH, LINEWIDTH_THEORY, PAPER_COLORS,
                    SWEEP_COLORS)


def _over_all_but_first(values):
    """Mean and standard deviation over every axis except the first."""
    axis = tuple(range(1, values.ndim))
    return np.nanmean(values, axis=axis), np.nanstd(values, axis=axis)


def plot_synthetic_vs_empirical(synthetic, theory, empirical, n_list, estimators,
                                n_synthetic=10, n_empirical=50, ylim=(0, 0.9),
                                titles=("Synthetic Data", "fMRI Data"),
                                xlabel="Number of trials (n)",
                                ylabel="Noise Ceiling Value", legend_ncol=2, loop=False):
    """Synthetic and recorded estimates side by side, closed forms dotted.

    synthetic, theory : (n_repetitions, len(n_list)), one entry per estimator.
    empirical         : (len(n_list), ...), averaged over every remaining axis.
    n_synthetic, n_empirical : denominators of the standard errors.
    """
    fig, (ax_syn, ax_emp) = plt.subplots(1, 2, figsize=(5.5, 3))

    for key in estimators:
        ax_syn.errorbar(n_list, np.nanmean(synthetic[key], axis=0),
                        yerr=2 * np.nanstd(synthetic[key], axis=0) / np.sqrt(n_synthetic),
                        label=LABELS[key], color=PAPER_COLORS[key],
                        linewidth=LINEWIDTH, elinewidth=ELINEWIDTH, zorder=1)
        ax_syn.plot(n_list, np.nanmean(theory[key], axis=0), color="k",
                    linestyle="dotted", linewidth=LINEWIDTH_THEORY, zorder=2)
    if loop and "LOOP" in synthetic:
        ax_syn.errorbar(n_list, np.nanmean(synthetic["LOOP"], axis=0),
                        yerr=2 * np.nanstd(synthetic["LOOP"], axis=0) / np.sqrt(n_synthetic),
                        label=LABELS["LOOP"], color=PAPER_COLORS["LOOP"],
                        linewidth=LINEWIDTH, elinewidth=ELINEWIDTH, zorder=1)
    for key in estimators:
        if key not in empirical:
            continue
        mean, std = _over_all_but_first(empirical[key])
        ax_emp.errorbar(n_list, mean, yerr=2 * std / np.sqrt(n_empirical),
                        label=LABELS[key], color=PAPER_COLORS[key],
                        linewidth=LINEWIDTH, elinewidth=ELINEWIDTH)
    if loop and "LOOP" in empirical:
        mean, std = _over_all_but_first(empirical["LOOP"])
        ax_emp.errorbar(n_list, mean, yerr=2 * std / np.sqrt(n_empirical),
                        label=LABELS["LOOP"], color=PAPER_COLORS["LOOP"],
                        linewidth=LINEWIDTH, elinewidth=ELINEWIDTH)

    for ax, title in zip((ax_syn, ax_emp), titles):
        ax.set_ylim(*ylim)
        ax.set_xticks(n_list)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=13)

    ax_syn.set_ylabel(ylabel, fontsize=12)
    ax_emp.set_yticklabels([])
    ax_syn.legend(ncol=legend_ncol, fontsize=12, bbox_to_anchor=(0.5, 1.5),
                  loc="upper center")
    return fig


def plot_empirical(results, n_list, mean_estimators, single_estimators,
                   n_empirical=50, ylim=(0, 0.9), title=None,
                   xlabel="Number of trials (n)", ylabel="Noise Ceiling Value",
                   legend=True):
    """Mean and single-trial estimators of one recorded dataset, side by side."""
    fig, (ax_mean, ax_single) = plt.subplots(1, 2, figsize=(5, 3))

    for ax, estimators in ((ax_mean, mean_estimators), (ax_single, single_estimators)):
        for key in estimators:
            if key not in results:
                continue
            mean, std = _over_all_but_first(results[key])
            ax.errorbar(n_list, mean, yerr=2 * std / np.sqrt(n_empirical),
                        label=LABELS[key], color=PAPER_COLORS[key],
                        linewidth=LINEWIDTH, elinewidth=ELINEWIDTH)
        ax.set_ylim(*ylim)
        ax.set_xticks(n_list)
        ax.set_xlabel(xlabel)
        if legend:
            ax.legend(fontsize=10)

    ax_mean.set_title(r"$NC^{mean}$ estimators", fontsize=13)
    ax_single.set_title(r"$NC^{single}$ estimators", fontsize=13)
    ax_mean.set_ylabel(ylabel, fontsize=12)
    ax_single.set_yticklabels([])

    if title is not None:
        fig.suptitle(title, fontsize=17, y=1.1)
    fig.subplots_adjust(wspace=0.05)
    return fig


def plot_finite_T(panels, T_values, estimators, fits, free_exponent=True,
                  xlabel="Number of samples (T)", ylabel="Noise Ceiling Value"):
    """Estimates against window length, one panel per dataset.

    panels : ordered mapping from panel title to a dict of (len(T), n_rep) arrays.
    fits   : same keys as `panels`, mapping estimator to the parameters returned
             by `fit_saturating`, or None where the fit failed.

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
            ax.plot(T_values, np.nanmean(panels[title][key], axis=1), "x",
                    color=PAPER_COLORS[key], markersize=4)

            popt = fits[title].get(key)
            if popt is None:
                ax.plot([], [], color=PAPER_COLORS[key], label=f"{LABELS[key]}")
                continue
            exponent = popt[2] if free_exponent else 1.0
            asymptote = popt[0]
            ax.plot(grid, saturating(grid, *popt), "k--",
                    linewidth=1.6,
                    label=f"{LABELS[key]}   $p$ = {exponent:.2f}")
            ax.axhline(asymptote, linestyle='--', color='darkgray', linewidth=1.6)

        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(ylabel, fontsize=12)
    fig.tight_layout()
    return fig


def plot_theory_vs_empirical(empirical, theory, n_values, mean_estimators,
                             single_estimators, n_repetitions, title=None, loop=False):
    """Diagnostic: empirical sweep against the closed forms, with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    panels = ((axes[0], mean_estimators, r"$NC^{\mathrm{mean}}$ estimators"),
              (axes[1], single_estimators, r"$NC^{\mathrm{single}}$ estimators"))
    print(empirical.keys())

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
        mean, _ = _over_all_but_first(values)
        ax.plot(n_list, mean, label=LABELS.get(key, key),
                color=SWEEP_COLORS.get(key))
    ax.set_xlabel("Number of trials (n)")
    ax.set_ylabel("Noise ceiling estimate")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return fig

def plot_selection_comparison(curves, n_values, estimators=("NC_mean", "SHSB"),
                              colors=None, linestyles=("--", "-"),
                              xlabel="Number of trials (n)",
                              ylabel="Noise Ceiling Value", title=None):
    """One curve per (unit selection, estimator) pair.
 
    curves : mapping from selection name to {estimator: array}, each array of
             shape (n_repetitions, len(n_values)) or (len(n_values),).
 
    Colour identifies the selection, line style the estimator, so that the gap
    between an estimator and the true ceiling can be read within each colour.
    """
    default_colors = ["#7A7A7A", "#4A93D8", "#EB8642", "#4CC275", "#C55995"]
    colors = colors or {name: default_colors[i % len(default_colors)]
                        for i, name in enumerate(curves)}
 
    fig, ax = plt.subplots(figsize=(6.5, 4))
 
    for name, per_estimator in curves.items():
        for key, linestyle in zip(estimators, linestyles):
            values = np.atleast_2d(per_estimator[key])
            mean = np.nanmean(values, axis=0)
            ax.plot(n_values, mean, linestyle=linestyle, color=colors[name],
                    linewidth=2, label=None)
            if values.shape[0] > 1:
                se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                ax.fill_between(n_values, mean - 2 * se, mean + 2 * se,
                                color=colors[name], alpha=0.15, linewidth=0)
 
    selection_handles = [Line2D([], [], color=colors[name], linewidth=2, label=name)
                         for name in curves]
    estimator_handles = [Line2D([], [], color="k", linestyle=style,
                                linewidth=2, label=LABELS.get(key, key))
                         for key, style in zip(estimators, linestyles)]
 
    first = ax.legend(handles=selection_handles, fontsize=9, loc="lower right",
                      title="Units kept", title_fontsize=9)
    ax.add_artist(first)
    ax.legend(handles=estimator_handles, fontsize=9, loc="upper left")
 
    ax.set_xticks(n_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig