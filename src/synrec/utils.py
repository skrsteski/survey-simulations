from __future__ import annotations
import glob
import os
from pathlib import Path

METHODS_ORDER = [
    "concept_guided", "ft", "subpop",
    "silicon_sampling", "random", "prev_day",
]


def extract_method(model_name: str) -> str:
    for m in METHODS_ORDER:
        if f":{m}" in model_name:
            return m
    return "other"


def find_model_csvs(base_dir: str) -> list[str]:
    return [
        p for p in sorted(glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True))
        if "_old" not in Path(p).parts
    ]
