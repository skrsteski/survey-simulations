"""
Bootstrap evaluation metrics for PPI survey simulation.

Computes bias, ESS gain, and coverage by repeatedly drawing random
labeled subsets from the full test set and evaluating PPI (or LLM-only)
estimates against the known population parameter theta*.

Bias:     E[|theta_hat - theta*| / theta*] x 100
ESS gain: (Var_human / Var_method - 1) x 100
Coverage: fraction of trials where the CI contains theta*
"""

from __future__ import annotations
import numpy as np
from ppi_py import ppi_mean_pointestimate
from .ppi import ppi_mean_ci_with_var

RNG = np.random.default_rng(42)
def reset_rng(seed):
    global RNG
    RNG = np.random.default_rng(seed)


# -- helpers ----------------------------------------------------------------

def _split(y, yhat, labeled_idx):
    """
    Split into labeled (y_l, yhat_l) and unlabeled (yhat_u), dropping NaN.
    """
    mask = np.zeros(len(y), dtype=bool)
    mask[labeled_idx] = True
    y_l, yh_l = y[mask], yhat[mask]
    yh_u = yhat[~mask]
    valid = ~(np.isnan(y_l) | np.isnan(yh_l))
    return y_l[valid], yh_l[valid], yh_u[~np.isnan(yh_u)]


def _ess_gain(human_var, method_var):
    """
    ESS gain (%). Positive = method more efficient than human-only.
    """
    if method_var <= 0:
        return np.nan
    return ((human_var / method_var) - 1.0) * 100.0


# -- single-split estimators -----------------------------------------------

def analytical_ci_width(data, alpha=0.05):
    """
    Normal-approximation CI width at level (1 - alpha).
    """
    valid = data[~np.isnan(data)]
    if len(valid) <= 1:
        return np.nan
    se = np.std(valid, ddof=1) / np.sqrt(len(valid))
    return 2 * 1.96 * se


def analytical_coverage(data, theta_true, alpha=0.05):
    """
    1.0 if the normal CI covers theta_true, 0.0 otherwise.
    """
    valid = data[~np.isnan(data)]
    if len(valid) <= 1:
        return np.nan
    theta_hat = np.mean(valid)
    se = np.std(valid, ddof=1) / np.sqrt(len(valid))
    margin = 1.96 * se
    return 1.0 if (theta_hat - margin) <= theta_true <= (theta_hat + margin) else 0.0


def ess_gain_analytical(y_true, yhat, labeled_idx, method="ppi", lam=None):
    """
    ESS gain for a single labeled/unlabeled split.

    :param method: "ppi" or "llm"
    :param lam: PPI lambda (None = optimal)
    """
    y_l, yh_l, yh_u = _split(y_true, yhat, labeled_idx)
    if len(y_l) <= 1 or len(yh_u) == 0:
        return np.nan

    human_var = float(np.var(y_l, ddof=1) / len(y_l))
    if method == "ppi":
        _, ppi_var, _ = ppi_mean_ci_with_var(y_l, yh_l, yh_u, lam=lam)
        method_var = ppi_var
    else:
        method_var = float(np.var(yh_u, ddof=1) / len(yh_u))

    return _ess_gain(human_var, method_var)


def ppi_once(y_true, yhat, labeled_idx, lam=None):
    """
    PPI point estimate and CI for a single split.

    :return: (theta_ppi, ci_width, ci_lo, ci_hi, lam_used)
    """
    y_l, yh_l, yh_u = _split(y_true, yhat, labeled_idx)
    if len(y_l) <= 1 or len(yh_u) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    (lo, hi), _, lam_used = ppi_mean_ci_with_var(y_l, yh_l, yh_u, alpha=0.05, lam=lam)
    theta_ppi = float(np.asarray(
        ppi_mean_pointestimate(y_l, yh_l, yh_u, lam=lam_used)
    ).squeeze())
    return theta_ppi, hi - lo, lo, hi, lam_used


# -- bootstrap evaluation --------------------------------------------------

def trials_metrics(y_true, yhat, trial_n, n_trials, theta_true, mode="ppi", lam=None):
    """
    Bootstrap-averaged bias, ESS, and coverage over random labeled splits.

    :param y_true: ground-truth array (N,)
    :param yhat: prediction array (N,)
    :param trial_n: labeled subset size per trial
    :param n_trials: number of bootstrap repetitions
    :param theta_true: true population parameter
    :param mode: "ppi" or "llm"
    :param lam: PPI lambda (None = optimal, 1.0 = standard PPI)
    :return: dict {bias, ess, coverage, lam, theta_hat, ci_width, n_trials_used}
    """
    empty = {"bias": np.nan, "ess": np.nan, "coverage": np.nan,
             "lam": np.nan, "theta_hat": np.nan, "ci_width": np.nan, "n_trials_used": 0}

    if len(y_true) <= trial_n:
        return empty

    biases, esses, covs, lams, thetas, ci_widths = [], [], [], [], [], []

    for _ in range(n_trials):
        idx = RNG.choice(len(y_true), size=trial_n, replace=False)
        y_l, yh_l, yh_u = _split(y_true, yhat, idx)
        if len(y_l) <= 1 or len(yh_u) == 0:
            continue

        if mode == "ppi":
            (lo, hi), ppi_var, lam_used = ppi_mean_ci_with_var(y_l, yh_l, yh_u, alpha=0.05, lam=lam)
            theta_hat = float(np.asarray(
                ppi_mean_pointestimate(y_l, yh_l, yh_u, lam=lam_used)
            ).squeeze())
            method_var = ppi_var
            covs.append(1.0 if lo <= theta_true <= hi else 0.0)
            lams.append(lam_used)
            ci_widths.append(hi - lo)
        elif mode == "llm":
            theta_hat = float(np.mean(yh_u))
            method_var = float(np.var(yh_u, ddof=1) / len(yh_u))
            ci_widths.append(analytical_ci_width(yh_u))
        else:
            raise ValueError(f"mode must be 'ppi' or 'llm', got {mode!r}")

        biases.append(abs(theta_hat - theta_true) / (abs(theta_true) + 1e-10) * 100)
        thetas.append(theta_hat)
        human_var = float(np.var(y_l, ddof=1) / len(y_l))
        esses.append(_ess_gain(human_var, method_var))

    if not biases:
        return empty

    return {
        "bias": float(np.mean(biases)),
        "ess": float(np.nanmean(esses)),
        "coverage": float(np.mean(covs)) if covs else np.nan,
        "lam": float(np.mean(lams)) if lams else np.nan,
        "theta_hat": float(np.mean(thetas)),
        "ci_width": float(np.nanmean(ci_widths)) if ci_widths else np.nan,
        "n_trials_used": len(biases),
    }
