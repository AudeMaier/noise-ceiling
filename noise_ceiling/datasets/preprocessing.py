"""Trend removal and standardisation before noise ceiling estimation.

The low-frequency basis is projected out run by run and each unit is then
z-scored. For a continuous stimulus this is the usual high-pass filter, with
`cutoff_s` in seconds and `tr` the sampling period. For a discrete stimulus the
last axis indexes stimuli rather than time; the same projection then removes
smooth trends across the stimulus ordering, and `cutoff_s` only sets how many
basis functions are removed.
"""

from __future__ import annotations

import numpy as np


def n_cosines(T, tr, cutoff_s):
    """Number of DCT regressors for a given cutoff, following the SPM convention."""
    return max(1, int(np.floor(2.0 * T * tr / float(cutoff_s))))


def dct_basis(T, tr=None, cutoff_s=None, K=None, add_constant=True):
    """Orthonormal low-frequency DCT basis, shape (T, K + 1).

    Give either `K` directly or both `tr` and `cutoff_s`. The columns are
    orthonormal, so the high-pass projector is I - Q Q^T.
    """
    if K is None:
        if tr is None or cutoff_s is None:
            raise ValueError("give either K, or both tr and cutoff_s")
        K = n_cosines(T, tr, cutoff_s)

    t = np.arange(T)
    columns = [np.ones(T)] if add_constant else []
    columns += [np.cos(np.pi * k * (2 * t + 1) / (2 * T)) for k in range(1, K + 1)]
    Q, _ = np.linalg.qr(np.stack(columns, axis=1))
    return Q


def project_out(x, Q):
    """Remove the column space of Q from the last axis of x, shape (..., T)."""
    x = np.asarray(x, dtype=float)
    return x - (x @ Q) @ Q.T


def clean(x, tr=1.5, run_lengths=None, cutoff_s=128.0, standardize=True,
          confounds=None):
    """Project out confounds and low-frequency trends, then z-score.

    x           : (units, T).
    run_lengths : lengths of the concatenated runs, summing to T. Setting this
                  removes the step at each run boundary. Defaults to one run.
    confounds   : (T, p) nuisance regressors, or None.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    T = x.shape[-1]
    if run_lengths is None:
        run_lengths = [T]
    if sum(run_lengths) != T:
        raise ValueError(f"run_lengths sum to {sum(run_lengths)}, expected {T}")

    out = np.empty_like(x)
    start = 0
    for run_length in run_lengths:
        run = slice(start, start + run_length)
        segment = x[..., run]

        if confounds is not None:
            C = np.asarray(confounds, float)[run]
            Q_confounds, _ = np.linalg.qr(C - C.mean(0))
            segment = project_out(segment, Q_confounds)

        segment = project_out(segment, dct_basis(run_length, tr=tr, cutoff_s=cutoff_s))

        if standardize:
            sd = segment.std(-1, keepdims=True)
            segment = segment / np.where(sd > 0, sd, 1.0)

        out[..., run] = segment
        start += run_length

    return out
