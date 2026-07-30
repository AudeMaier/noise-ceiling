"""Tensor helpers for the ridge solver."""

from __future__ import annotations

import numpy as np
import torch


def z_score(x, dim=0, eps=1e-8):
    """Z-score along `dim`, guarding against zero variance."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


def mult_diag(d, mtx, left=True):
    """Multiply `mtx` by the diagonal matrix with diagonal `d`."""
    if left:
        return (d.unsqueeze(1) * mtx.T).T
    return d.unsqueeze(0) * mtx


def svd_wrapper(X, singcutoff=1e-10):
    """Thin SVD with singular values below `singcutoff` dropped.

    Falls back to the NumPy SVD when the torch driver fails to converge.
    """
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        U_np, S_np, Vh_np = np.linalg.svd(X.cpu().numpy(), full_matrices=False)
        U = torch.from_numpy(U_np).to(X.device)
        S = torch.from_numpy(S_np).to(X.device)
        Vh = torch.from_numpy(Vh_np).to(X.device)

    n_good = torch.sum(S > singcutoff).item()
    return U[:, :n_good], S[:n_good], Vh[:n_good]
