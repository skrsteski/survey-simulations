from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import openai
import yaml
from datasets import load_dataset
from dotenv import load_dotenv

from synrec.methods.simulate_gpt import get_gpt_responses
from synrec.methods.simulate_hf import get_hf_responses
from synrec.retrieval import FuzzyMatchSearch, NutritionSearch
from synrec.utils import parse_food_entries, save_ground_cache, scaled_macros, ground_many_with_cache

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
        filemode="w"
    )
    logging.info(f"Logging initialized: {log_name}")

def get_searcher(cfg):
    retrieval_type = cfg.get("retrieval_type", "nutrition")
    if retrieval_type == "nutrition":
        return NutritionSearch(
            embeddings_path="data/fndds/embeddings.pt",
            items_path="data/fndds/items.json"
        )
    elif retrieval_type == "fuzzy":
        return FuzzyMatchSearch()
    else:
        raise ValueError(f"Unknown retrieval_type: {retrieval_type}")

def safe_divide(numerator, denominator, default=1.0):
    if numerator is None or denominator is None or denominator == 0:
        return default
    return numerator / denominator

### ----------------
# RUN SIMULATION
### ----------------
async def run_simulation(method: str, out_path: Path | None, cfg: dict):
    test_dataset = load_dataset("StefanKrsteski/nhanes2015_food_consumption_survey", split="test")

    # DEBUG MODE
    if cfg.get("debug", False):
        test_dataset = test_dataset.select(range(100))

    init_logging(method)
    logging.info(f"Loaded dataset with {len(test_dataset)} entries")

    prompts = test_dataset["prompt"]
    persona_descriptions = test_dataset["persona_description"]
    one_shot_examples = test_dataset["one_shot_prompt"]
    
    prompts = [eval(prompt) if isinstance(prompt, str) else prompt for prompt in prompts]
    # if cfg["concept_guided"] is true then add the "persona_description" column just at the end of "system" 
    if cfg.get("concept_guided", False):
        for i, prompt in enumerate(prompts):
            prompt[0]["content"] += f"\n\nAdditional profile information:\n{persona_descriptions[i]}"
    elif cfg.get("one_shot", False): # same as above just different phrasing 
        for i, prompt in enumerate(prompts):
            prompt[0]["content"] += f"\n\Example of what you ate in a day in the past:\n{one_shot_examples[i]}"

    demographics = [
        {
            "id": rec["id"],
            "recall_n": rec["recall_n"],
            "sex": rec["sex"],
            "age": rec["age"],
            "wgt": rec["wgt"],
            "hgt": rec["hgt"],
            "race": rec["race"],
            "hh_income": rec["hh_income"],
            "marital_status": rec["marital_status"],
            "hh_size": rec["hh_size"],
            "special_diet": rec["special_diet"],
            "preg_lact": rec["preg_lact"],
            "pa_cat": rec["pa_cat"],
            "smok": rec["smok"],
        }
        for rec in test_dataset
    ]

    if cfg.get("use_hf", False):
        responses = get_hf_responses(prompts, cfg)
    else:
        responses = get_gpt_responses(prompts, cfg)

    searcher = get_searcher(cfg)
    out_csv = out_path or Path(f"outputs/results/foodlines_{method}_batch.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "food_idx", "food_name", "input_desc", "matched_desc", "grams_eaten",
            "energy", "protein", "fat", "carb", "sugar", "fiber",
            "sex", "age", "wgt", "hgt", "race", "hh_income", "marital_status",
            "hh_size", "special_diet", "preg_lact"
        ])
        all_rows = await process_responses_with_search(
            responses,
            test_dataset["id"],
            cfg,
            searcher,
            demographics
        )
        writer.writerows(all_rows)
        save_ground_cache()
        logging.info(f"Successfully simulated data => {out_csv}")


async def process_responses_with_search(responses: list[str], pids: list[int], cfg, searcher, demographics):
    all_foods = []
    # print(responses)
    # exit()
    for response, pid, demo in zip(responses, pids, demographics):
        foods = parse_food_entries(response)
        for idx, entry in enumerate(foods, 1):
            all_foods.append((entry, pid, idx, demo))

    # first-stage search
    queries = [entry.name for entry, _, _, _ in all_foods]
    first_pass_results = searcher.search_many(queries, top_k=cfg["retrieval"]["first_stage_top_k"])
    # grounding
    grounding_inputs = [
        (f"{entry.name} - {entry.description} - reported grams: {entry.grams}",
         [c["description"] for c in first_pass])
        for (entry, _, _, _), first_pass in zip(all_foods, first_pass_results)
    ]
    texts, candidate_lists = zip(*grounding_inputs)
    groundings = await ground_many_with_cache(list(texts), list(candidate_lists), cfg)
    # collect grounded matches and prepare second-stage queries
    second_stage_queries = []
    second_stage_meta = []
    for (entry, pid, food_idx, demo), grounded in zip(all_foods, groundings):
        if not grounded or not grounded.components:
            continue
        for comp in grounded.components:
            second_stage_queries.append(comp.match)
            second_stage_meta.append((entry, pid, food_idx, demo, grounded, comp))

    # second-stage search
    second_results = searcher.search_many(second_stage_queries, top_k=cfg["retrieval"]["second_stage_top_k"])
    # final rows
    output_rows = []
    for (entry, pid, food_idx, demo, grounded, comp), nutrient_matches in zip(second_stage_meta, second_results):
        if not nutrient_matches:
            continue
        nutrient = nutrient_matches[0]
        total_component_grams = sum(c.grams for c in grounded.components)
        scaling_factor = safe_divide(entry.grams, total_component_grams, default=1.0)
        adjusted_grams = comp.grams * scaling_factor
        macros = scaled_macros(nutrient["nutrients"], adjusted_grams)

        output_rows.append([
            pid, food_idx, entry.name, entry.description, comp.match, adjusted_grams,
            macros["energy"], macros["protein"], macros["fat"],
            macros["carb"], macros["sugar"], macros["fiber"],
            demo.get("sex"), demo.get("age"), demo.get("wgt"),
            demo.get("hgt"), demo.get("race"), demo.get("hh_income"),
            demo.get("marital_status"), demo.get("hh_size"),
            demo.get("special_diet"), demo.get("preg_lact")
        ])

    return output_rows