"""
Reproduce the main table of the paper. Usage: python reproduce.py
"""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = {"nhanes": "NHANES", "atp1": "ATP Q1", "atp2": "ATP Q2"}
METHODS = {
    "ft": "Domain-FT",
    "subpop": "SubPOP-FT",
    "concept_guided": "Persona-guided",
    "silicon_sampling": "Demo-only",
}
BASELINES = {"random", "prev_day"}
TRIAL_NS = ["100", "50", "150", "200"]


def run_and_collect():
    rows = []
    for dataset in DATASETS:
        ds_dir = ROOT / "datasets" / dataset
        cmd = [sys.executable, "evaluate.py", "--trial_ns", *TRIAL_NS, "--n_trials", "100"]
        print(f"\n=== {dataset} ===")
        subprocess.run(cmd, cwd=ds_dir, check=True)
        payload = json.loads((ds_dir / "outputs/results/results.json").read_text())
        
        for r in payload["results"]:
            method = r["method"]
            llm = r.get("llm")
            if method in METHODS and llm:
                rows.append({
                    "dataset": DATASETS[dataset],
                    "block": "Synthesize only",
                    "method": METHODS[method] + " | None",
                    "bias": llm["bias"],
                    "ess": llm["ess"].get("100"),
                })
            elif method in BASELINES and llm:
                rows.append({
                    "dataset": DATASETS[dataset],
                    "block": "Baseline",
                    "method": "Baseline",
                    "bias": llm["bias"],
                    "ess": llm["ess"].get("100"),
                })

            for key, label in [("rec_lam_opt", "Rec_λ_opt"), ("rec_lam_1", "Rec_λ=1")]:
                ppi = r.get(key)
                if method not in METHODS or not ppi:
                    continue
                rows.append({
                    "dataset": DATASETS[dataset],
                    "block": label,
                    "method": f"{METHODS[method]} | {label}",
                    "bias": ppi["bias"],
                    "ess": ppi["ess"].get("100"),
                })

    return pd.DataFrame(rows)


def print_table(df):
    print("\n===== MAIN TABLE - method averages across models =====")
    block_order = ["Baseline", "Synthesize only", "Rec_λ=1", "Rec_λ_opt"]
    for metric, header in [("bias", "Bias (%) ↓"), ("ess", "ESS Gain (%) ↑")]:
        print(f"\n{header}")
        rows_out = []
        for block in block_order:
            sub = df[df["block"] == block]
            if sub.empty:
                continue
            pv = sub.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
            pv = pv.reindex(columns=list(DATASETS.values()))
            pv["Avg"] = pv.mean(axis=1)
            pv["_block"] = block
            rows_out.append(pv)
        out = pd.concat(rows_out)
        print(out.drop(columns="_block").round(2).to_string(na_rep="—"))


if __name__ == "__main__":
    print_table(run_and_collect())
