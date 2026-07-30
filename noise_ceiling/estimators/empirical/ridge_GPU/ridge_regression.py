"""Ridge regression on the GPU.

Torch port of the ridge utilities from
https://github.com/HuthLab/encoding-model-scaling-laws
"""

from __future__ import annotations

import torch

from .ridge_utils import mult_diag, svd_wrapper, z_score


def ridge_corr_torch(Rstim, Pstim, Rresp, Presp, alphas,
                     singcutoff=1e-30, use_corr=True, normalpha=False):
    """Score a grid of ridge parameters on a held-out set.

    Rstim, Pstim : train and test predictors, (n_samples, n_features).
    Rresp, Presp : train and test targets, (n_samples, n_targets).
    alphas       : ridge parameters to score.
    use_corr     : score with the correlation rather than the signed sqrt(R^2).

    Returns the score for every (alpha, target), shape (n_alphas, n_targets).
    """
    U, S, Vh = svd_wrapper(Rstim, singcutoff=singcutoff)
    nalphas = [a * S[0].item() for a in alphas] if normalpha else alphas

    UR = torch.matmul(U.T, Rresp)
    PVh = torch.matmul(Pstim, Vh.T)
    zPresp = z_score(Presp, dim=0)
    Prespvar = Presp.var(dim=0)

    scores = []
    for na in nalphas:
        D = S / (S ** 2 + na ** 2)
        pred = torch.matmul(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            score = (zPresp * z_score(pred, dim=0)).mean(dim=0)
        else:
            Rsq = 1 - (Presp - pred).var(dim=0) / Prespvar
            score = torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)

        scores.append(torch.nan_to_num(score))

    return torch.stack(scores)


def ridge_corr_pred_torch(Rstim, Pstim, Rresp, Presp, valphas,
                          singcutoff=1e-30, use_corr=True, normalpha=True):
    """Fit with one ridge parameter per target and score on the test set.

    valphas : ridge parameter for each target, shape (n_targets,).

    Returns the test score per target, shape (n_targets,).
    """
    U, S, Vh = svd_wrapper(Rstim, singcutoff=singcutoff)
    nalphas = valphas * S[0].item() if normalpha else valphas

    UR = torch.matmul(U.T, Rresp)
    PVh = torch.matmul(Pstim, Vh.T)
    zPresp = z_score(Presp, dim=0)
    Prespvar = Presp.var(dim=0)

    corr = torch.zeros(Rresp.shape[1], device=Rstim.device).to(float)
    for ua in torch.unique(nalphas):
        targets = torch.nonzero(nalphas == ua).squeeze()
        if targets.ndim == 0:
            targets = targets.unsqueeze(0)

        D = S / (S ** 2 + ua ** 2)
        pred = torch.matmul(torch.matmul(PVh, torch.diag(D)), UR[:, targets])

        if use_corr:
            corr[targets] = (zPresp[:, targets] * z_score(pred, dim=0)).mean(dim=0)
        else:
            Rsq = 1 - (Presp[:, targets] - pred).var(dim=0) / Prespvar[targets]
            corr[targets] = torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)

    return corr
