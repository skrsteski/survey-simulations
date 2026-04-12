"""
Evaluation loop for PPI-rectified survey simulation.

Runs bootstrap-averaged metrics (bias, ESS, coverage) for each model CSV
in three modes: unrectified LLM, PPI with optimal lambda, and PPI with
lambda=1. Outputs a single results.json per dataset.
"""

from __future__ import annotations
import fnmatch
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .metrics import trials_metrics
from .utils import extract_method, find_model_csvs


def eval_file(prepare_fn, human_df, pred_path, trial_ns, n_trials=5):
    """
    Evaluate one model CSV in all three modes (llm, rec_lam_opt, rec_lam_1).

    :param prepare_fn: callable (human_df, pred_path) -> (y, yhat, theta_true)
    :param human_df: human ground-truth dataframe
    :param pred_path: path to a model predictions CSV
    :param trial_ns: labeled subset sizes; first element used for headline metrics
    :param n_trials: bootstrap repetitions per trial_n
    :return: nested dict with model metadata and per-mode results
    """
    y, yhat, theta_true = prepare_fn(human_df, pred_path)

    model_name = f"{Path(pred_path).parent.name}:{Path(pred_path).stem}"
    is_baseline = "baseline" in pred_path
    base_trial_n = trial_ns[0]

    result = {
        "model": model_name,
        "method": extract_method(model_name),
        "n_responses": len(y),
        "theta_true": theta_true,
    }

    modes = [("llm", "llm", None), ("rec_lam_opt", "ppi", None), ("rec_lam_1", "ppi", 1.0)]
    for key, mode, lam in modes:
        if is_baseline and mode == "ppi":
            result[key] = None
            continue

        block = {}
        ess_by_n, cov_by_n = {}, {}

        for trial_n in trial_ns:
            bt = trials_metrics(y, yhat, trial_n, n_trials, theta_true, mode=mode, lam=lam)
            ess_by_n[str(trial_n)] = bt["ess"]
            cov_by_n[str(trial_n)] = bt["coverage"]

            if trial_n == base_trial_n:
                block["theta_hat"] = bt["theta_hat"]
                block["bias"] = bt["bias"]
                block["ci_width"] = bt["ci_width"]
                if mode == "ppi":
                    block["lam"] = bt["lam"]

        block["ess"] = ess_by_n
        block["coverage"] = cov_by_n
        result[key] = block

    return result


def run_evaluation(
    dataset_name, prepare_fn, human_df,
    base_dir="outputs/simulated_data", trial_ns=None,
    n_trials=5, seed=0, only=None, results_dir=None,
):
    """
    Evaluate all model CSVs for one dataset and write results.json.

    :param dataset_name: e.g. "nhanes", "atp1", "atp2"
    :param prepare_fn: callable (human_df, pred_path) -> (y, yhat, theta_true)
    :param human_df: pre-loaded human ground-truth dataframe
    :param base_dir: directory containing model_name/method.csv files
    :param trial_ns: labeled subset sizes (default [50, 100, 150, 200])
    :param n_trials: bootstrap repetitions
    :param seed: RNG seed recorded in output metadata
    :param only: optional fnmatch patterns to filter CSVs
    :param results_dir: output directory (default outputs/results/)
    """
    if trial_ns is None:
        trial_ns = [50, 100, 150, 200]
    if results_dir is None:
        results_dir = Path("outputs/results")

    model_csvs = find_model_csvs(base_dir)
    if only:
        model_csvs = [p for p in model_csvs if any(fnmatch.fnmatch(p, pat) for pat in only)]

    results = []
    for p in tqdm(model_csvs, desc=f"Evaluating {dataset_name}"):
        results.append(eval_file(prepare_fn, human_df, p, trial_ns, n_trials=n_trials))

    payload = {
        "dataset": dataset_name,
        "config": {"n_trials": n_trials, "trial_ns": trial_ns, "seed": seed},
        "results": _clean(results),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")


def _clean(obj):
    if isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj
