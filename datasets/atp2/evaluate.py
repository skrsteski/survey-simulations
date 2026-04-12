import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synrec.evaluate import run_evaluation
from synrec.metrics import reset_rng

DATASET_NAME = "atp2"
DATA_PATH = Path("data/human_test.csv")

RESPONSE_MAP = {
    "A": 4,       # Excellent
    "B": 3,       # Good
    "C": 2,       # Only fair
    "D": 1,       # Poor
    "E": np.nan,  # Not sure
    "F": np.nan,  # Refused/Web blank
}

CHOICE_MAP = {
    "Poor": "D",
    "Only fair": "C",
    "Good": "B",
    "Excellent": "A",
    "Not sure": "E",
    "Refused/Web blank": "F",
}


def load_human():
    df = pd.read_csv(DATA_PATH)
    df["answer"] = df["correct_answer"].map(CHOICE_MAP)
    return df[["id", "answer"]].dropna()


def _to_numeric(responses):
    if isinstance(responses, pd.Series):
        return responses.map(RESPONSE_MAP).values
    return np.array([RESPONSE_MAP.get(r, np.nan) for r in responses])


def prepare(human_df, pred_path):
    pred_df = pd.read_csv(pred_path)
    if "id" not in pred_df.columns or "answer" not in pred_df.columns:
        raise ValueError(f"{pred_path}: missing 'id' or 'answer'")
    
    pred_df = pred_df[["id", "answer"]].rename(columns={"answer": "answer_pred"})
    df = human_df.merge(pred_df, on="id", how="inner").dropna()
    y = np.array(_to_numeric(df["answer"]))
    yhat = np.array(_to_numeric(df["answer_pred"]))
    theta_true = float(np.nanmean(y))
    valid = ~(np.isnan(y) | np.isnan(yhat))
    return y[valid], yhat[valid], theta_true


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="outputs/simulated_data")
    parser.add_argument("--trial_ns", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reset_rng(args.seed)
    run_evaluation(
        DATASET_NAME, prepare, load_human(),
        base_dir=args.base_dir, trial_ns=args.trial_ns,
        n_trials=args.n_trials, seed=args.seed, only=args.only,
    )
