"""Figure style, shared by every plot.

Estimators are referred to by short keys everywhere in the code; the display
names live here and nowhere else.
"""

from __future__ import annotations

LINEWIDTH = 3.5
# SHSB is drawn under ATV and wider, so that a rim of it shows where they overlap.
LINEWIDTHS = {"SHSB": 4.7, "ATV": 2.8}
LINEWIDTH_THEORY = 2.5
ELINEWIDTH = 2

# Paper figures (make_figures.py): every panel is PANEL_SIZE inches (width,
# height), side by side PANEL_GAP inches apart, whatever the figure.
PANEL_SIZE = (1.8, 1.7)
PANEL_GAP = 0.3
FONT_TICKS = 12
FONT_LABEL = 13
FONT_XLABEL = 12
FONT_TITLE = 14
FONT_LEGEND = 12

# Checked for protanopia and deuteranopia against each other and the grey of
# the true noise ceilings, which is light so that it differs from every colour
# in lightness as well as hue. Curves sharing a panel are the ones that must
# stay apart: (NC_mean, SH, SHSB, ATV) and (NC_single, LOOC, TTMC, LOOP).
PAPER_COLORS = {
    "NC_mean": "#b4b4b4",
    "SH": "#4b3f9e",
    "SHSB": "#2a78d6",
    "ATV": "#eb6834",
    "NC_single": "#b4b4b4",
    "LOOC": "#1b9e9e",
    "TTMC": "#eda100",
    "LOOP": "#a8325e",
}

LABELS = {
    "NC_mean": r"${NC}^{mean}$",
    "SH": r"$\widehat{NC}^{SH}$",
    "SHSB": r"$\widehat{NC}^{SHSB}$",
    "ATV": r"$\widehat{NC}^{ATV}$",
    "NC_single": r"${NC}^{single}$",
    "LOOC": r"$\widehat{NC}^{LOOC}$",
    "TTMC": r"$\widehat{NC}^{TTMC}$",
    "LOOP": r"$\widehat{NC}^{LOOP}$",
    "LOOP_null": r"$\widehat{NC}^{LOOP}$ null",
}

# used by the diagnostic figures the run scripts produce
SWEEP_COLORS = {
    "ATV": "tab:orange",
    "SHSB": "tab:green",
    "NC_mean": "darkgray",
    "SH": "tab:blue",
    "LOOC": "tab:red",
    "TTMC": "tab:purple",
    "NC_single": "tab:brown",
    "LOOP": "tab:pink",
    "LOOP_null": "tab:pink",
}

# keys used by result files written before the estimators were given their
# current short names
LEGACY_NAMES = {
    "True Noise Ceiling (Mean)": "NC_mean",
    "True Noise Ceiling (Single)": "NC_single",
    "Split-Half Correlation": "SH",
    "Spearman-Brown Noise Ceiling": "SHSB",
    "Allen Noise Ceiling": "ATV",
    "SNR": "ATV",
    "Leave-One-Out Correlation": "LOOC",
    "Mean-to-One Correlation": "TTMC",
    "Leave-One-Out Prediction": "LOOP",
}
