from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

from synrec.methods.simulate_gpt import get_gpt_responses
from synrec.methods.simulate_hf import get_hf_responses
from synrec.utils import parse_multi_choice_response, get_multi_choice_info

### ----------------
# CONSTANTS
### ----------------
load_dotenv()
client = openai.OpenAI()


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file {path} did not load as a dict.")
        return cfg


def init_logging(model_name: str):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_name = f"log_{stamp}_{model_name[:3]}.log"
    logs_path = Path("outputs/logs")
    logs_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_path / log_name,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )
    logging.info(f"Logging initialized: {log_name}")


### ----------------
# RUN SIMULATION
### ----------------
async def run_simulation(method: str, out_path: Path | None, cfg: dict):
    test_dataset = pd.read_csv("data/atp/human_test.csv")

    init_logging(method)
    logging.info(f"Loaded dataset with {len(test_dataset)} entries")

    prompts = test_dataset["prompt"]
    persona_descriptions = test_dataset["persona_description"]
    prompts = [eval(prompt) if isinstance(prompt, str) else prompt for prompt in prompts]
    if cfg.get("concept_guided", False):
        for i, prompt in enumerate(prompts):
            pass
            prompt[0]["content"] += (
                f"\n\nAdditional profile information:\n{persona_descriptions[i]}"
            )

    # get response
    if cfg.get("use_hf", False):
        responses = get_hf_responses(prompts, cfg)
    else:
        responses = get_gpt_responses(prompts, cfg)

    # dump responses
    for i, response in enumerate(responses):
        logging.info(f"Response {i}: {response}")

    choice = test_dataset["choices"].tolist()[0]
    index2ans, all_choices = get_multi_choice_info(choice)
    parsed_answers = []
    for i, response in enumerate(responses):
        parsed_answers.append(parse_multi_choice_response(response, all_choices, index2ans))

    out_csv = out_path
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    test_dataset = test_dataset.copy()
    test_dataset["answer"] = parsed_answers
    # write all columns (original + parsed_answer) to CSV
    test_dataset.to_csv(out_csv, index=False)
    logging.info(f"Successfully simulated data => {out_csv}")
