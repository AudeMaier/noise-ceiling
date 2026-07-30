"""Bias of the estimators as a function of the number of samples."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .datasets.synthetic import ALL_ESTIMATORS


def bias(result):
    """Bias of each estimator in a `sweep_T` result. Positive means underestimation.

    Uses the estimate at T_ref as the reference when the sweep was anchored on
    it, and the closed forms otherwise.
    """
    reference = result.get("reference")
    if reference is not None:
        return {k: reference[k] - result["empirical"][k] for k in ALL_ESTIMATORS}
    return {k: result["theory"][k] - result["empirical"][k] for k in ALL_ESTIMATORS}


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


def fit_power(T, measured, T_ref, p0=(1.0, 1.0)):
    """Fit measured(T) = c (T^-p - T_ref^-p) and return (c, p, standard error of p).

    A log-log slope would be biased steep for an anchored curve, because the
    T_ref term flattens it at large T.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(measured, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return np.nan, np.nan, np.nan

    def model(t, c, p):
        return c * (t ** -p - float(T_ref) ** -p)

    try:
        (c, p), cov = curve_fit(model, T[finite], y[finite], p0=p0, maxfev=40000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    return float(c), float(p), float(np.sqrt(np.maximum(cov[1, 1], 0)))
