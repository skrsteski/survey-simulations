"""
Wrapper around ppi_py.ppi_mean_ci that also returns variance and lambda.
"""

from __future__ import annotations
import numpy as np
from ppi_py import ppi_mean_ci as _ppi_mean_ci_upstream
from ppi_py.ppi import _calc_lam_glm
from ppi_py.utils import construct_weight_vector, reshape_to_2d


def ppi_mean_ci_with_var(Y, Yhat, Yhat_unlabeled, alpha=0.05, lam=None):
    """
    Compute PPI confidence interval, variance, and power-tuning lambda.

    :param Y: labeled ground-truth responses
    :param Yhat: labeled model predictions
    :param Yhat_unlabeled: unlabeled model predictions
    :param alpha: significance level
    :param lam: power-tuning parameter (None = data-driven optimal)
    :return: ((lo, hi), ppi_var, lam_used)
    """
    Y = np.asarray(Y, dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    Yhat_unlabeled = np.asarray(Yhat_unlabeled, dtype=float)
    n, N = len(Y), len(Yhat_unlabeled)

    Y2 = reshape_to_2d(Y)
    Yhat2 = reshape_to_2d(Yhat)
    Yhat_u2 = reshape_to_2d(Yhat_unlabeled)
    w = construct_weight_vector(n, None, vectorized=True)
    w_u = construct_weight_vector(N, None, vectorized=True)

    if lam is None:
        base = (w_u * Yhat_u2).mean(0) + (w * (Y2 - Yhat2)).mean(0)
        grads = w * (Y2 - base)
        grads_hat = w * (Yhat2 - base)
        grads_hat_u = w_u * (Yhat_u2 - base)
        lam_used = float(_calc_lam_glm(
            grads, grads_hat, grads_hat_u, np.eye(1),
            coord=None, clip=True, optim_mode="overall",
        ))
    else:
        lam_used = float(lam)

    lo, hi = _ppi_mean_ci_upstream(Y, Yhat, Yhat_unlabeled, alpha=alpha, lam=lam_used)
    ci = (float(np.asarray(lo).squeeze()), float(np.asarray(hi).squeeze()))

    imputed_std = (w_u * (lam_used * Yhat_u2)).std(0) / np.sqrt(N)
    rectifier_std = (w * (Y2 - lam_used * Yhat2)).std(0) / np.sqrt(n)
    ppi_var = float((imputed_std ** 2 + rectifier_std ** 2)[0])

    return ci, ppi_var, lam_used
