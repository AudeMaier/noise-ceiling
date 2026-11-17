"""Fit of an estimate against the number of samples T, NC(T) = a - b T^-p."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def saturating(T, a, b, p=1.0):
    """NC(T) = a - b T^-p, the estimate approaching its asymptote from below."""
    return a - b * np.asarray(T, dtype=float) ** -p


def fit_saturating(T, values, free_exponent=True):
    """Fit `saturating` to a measured curve.

    free_exponent=False fixes p = 1, i.e. assumes the 1/T decay rather than
    measuring it.

    Returns the parameters and their standard errors, both ordered
    (a, b, p) or (a, b).
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < (4 if free_exponent else 3):
        return None, None

    amplitude = max(y[finite][-1] - y[finite][0], 1e-3) * T[finite][0]
    if free_exponent:
        model, p0 = saturating, (y[finite][-1], amplitude, 1.0)
    else:
        def model(t, a, b):
            return saturating(t, a, b, 1.0)
        p0 = (y[finite][-1], amplitude)

    try:
        popt, pcov = curve_fit(model, T[finite], y[finite], p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        return None, None
    return popt, np.sqrt(np.maximum(np.diag(pcov), 0))
