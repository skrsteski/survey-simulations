import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel


def init_hf_chat(cfg: dict):
    base_model_name = cfg["hf_model"]
    adapter_path = cfg.get("adapter", None)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_args = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "attn_implementation": "flash_attention_2",
    }
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_args)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,
        batch_size=cfg.get("batch_size", 64),
    )


def get_hf_responses(prompts: List, cfg: dict) -> List[str]:
    hf_chat = init_hf_chat(cfg)
    gen_kwargs = {
        "temperature": cfg.get("temperature", 0.7),
        "max_new_tokens": cfg.get("max_tokens", 1024),
        "pad_token_id": hf_chat.tokenizer.pad_token_id,
    }

    formatted_prompts = []
    for prompt in prompts:
        formatted = hf_chat.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    outputs = hf_chat(formatted_prompts, **gen_kwargs)
    return [output[0]["generated_text"] for output in outputs]
