import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synrec.evaluate import run_evaluation
from synrec.metrics import reset_rng

DATASET_NAME = "nhanes"
DATA_PATH = Path("data/human_test.csv")


def load_human():
    df = pd.read_csv(DATA_PATH)
    df = df[["id", "energy", "ingr_descr_eng"]].dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    return df.groupby("id")["energy"].sum().reset_index()


def prepare(human_df, pred_path):
    pred_df = pd.read_csv(pred_path)
    if "id" not in pred_df.columns or "energy" not in pred_df.columns:
        raise ValueError(f"{pred_path}: missing 'id' or 'energy'")
    pred_df = pred_df.rename(columns={"energy": "energy_pred"})
    pred_df["energy_pred"] = pd.to_numeric(pred_df["energy_pred"], errors="coerce")
    pred_df["id"] = pred_df["id"].astype(int)
    pred_df = pred_df.groupby("id")["energy_pred"].sum().reset_index()

    df = human_df.merge(pred_df, on="id", how="inner").dropna(subset=["energy", "energy_pred"])
    y = np.array(df["energy"].values, dtype=float)
    yhat = np.array(df["energy_pred"].values, dtype=float)
    return y, yhat, float(np.mean(y))


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
