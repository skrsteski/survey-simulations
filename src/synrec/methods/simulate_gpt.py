import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _run_one_batch(prompts: List[List[dict]], cfg: dict) -> List[str]:
    model = cfg["openai_model"]
    tmp_dir = Path(tempfile.mkdtemp(prefix="gpt_batch_"))
    input_path = tmp_dir / "batch_input.jsonl"
    with open(input_path, "w") as f:
        for idx, msgs in enumerate(prompts):
            line = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": msgs},
            }
            f.write(json.dumps(line) + "\n")

    with open(input_path, "rb") as in_file:
        file_obj = client.files.create(file=in_file, purpose="batch")
    file_id = file_obj.id
    logging.info(f"Uploaded batch, file_id={file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    logging.info(f"Created batch {batch_id} with {len(prompts)} requests.")

    while True:
        b = client.batches.retrieve(batch_id)
        logging.info(b)
        if b.status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(5)
    if b.status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended with status {b.status}")

    # Build a list with the same order as prompts; fill with "" for failed requests
    responses = [""] * len(prompts)

    # Successful outputs
    if b.output_file_id:
        content = client.files.content(b.output_file_id).text
        for ln in content.strip().splitlines():
            obj = json.loads(ln)
            cid = int(obj["custom_id"])
            body = obj["response"]["body"]
            responses[cid] = body["choices"][0]["message"]["content"]

    return responses


def get_gpt_responses(prompts: List[List[dict]], cfg: dict) -> List[str]:
    if not prompts:
        return []
    return _run_one_batch(prompts, cfg)
