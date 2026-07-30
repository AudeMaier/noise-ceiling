"""Figure style, shared by every plot.

Estimators are referred to by short keys everywhere in the code; the display
names live here and nowhere else.
"""

from __future__ import annotations

LINEWIDTH = 3
LINEWIDTH_THEORY = 2.5
ELINEWIDTH = 2

PAPER_COLORS = {
    "NC_mean": "grey",
    "SH": "#4A93D8",
    "SHSB": "#4CC275",
    "SNR": "#C55995",
    "NC_single": "grey",
    "LOOC": "#EB8642",
    "TTMC": "#F35C5C",
    "LOOP": "#8E6FBF",
}

LABELS = {
    "NC_mean": r"${NC}^{mean}$",
    "SH": r"$\widehat{NC}^{SH}$",
    "SHSB": r"$\widehat{NC}^{SHSB}$",
    "SNR": r"$\widehat{NC}^{SNR}$",
    "NC_single": r"${NC}^{single}$",
    "LOOC": r"$\widehat{NC}^{LOOC}$",
    "TTMC": r"$\widehat{NC}^{TTMC}$",
    "LOOP": r"$\widehat{NC}^{LOOP}$",
}

# used by the diagnostic figures the run scripts produce
SWEEP_COLORS = {
    "SNR": "tab:orange",
    "SHSB": "tab:green",
    "NC_mean": "darkgray",
    "SH": "tab:blue",
    "LOOC": "tab:red",
    "TTMC": "tab:purple",
    "NC_single": "tab:brown",
    "LOOP": "tab:pink",
}

# keys used by result files written before the estimators were given short names
LEGACY_NAMES = {
    "True Noise Ceiling (Mean)": "NC_mean",
    "True Noise Ceiling (Single)": "NC_single",
    "Split-Half Correlation": "SH",
    "Spearman-Brown Noise Ceiling": "SHSB",
    "Allen Noise Ceiling": "SNR",
    "Leave-One-Out Correlation": "LOOC",
    "Mean-to-One Correlation": "TTMC",
    "Leave-One-Out Prediction": "LOOP",
}
