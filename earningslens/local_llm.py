from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Any

import torch

from earningslens import config

LOGGER = logging.getLogger(__name__)
_GENERATION_LOCK = Lock()


def _device_config() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_LLM_SOURCE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def _get_model():
    from transformers import AutoModelForCausalLM

    device, dtype = _device_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_LLM_SOURCE,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    LOGGER.info("Loaded local LLM %s on %s", config.LOCAL_LLM_SOURCE, device)
    return model


def warmup_local_llm() -> None:
    _get_tokenizer()
    _get_model()


def _build_prompt(prompt: str) -> str:
    tokenizer = _get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": "You are a precise financial analysis assistant. Follow formatting instructions exactly. Return only the requested content.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"


def _max_input_tokens(tokenizer: Any) -> int:
    model_max_length = int(getattr(tokenizer, "model_max_length", 2048) or 2048)
    if model_max_length <= 0 or model_max_length > 32768:
        return 4096
    return max(512, min(model_max_length, 4096))


def _prepare_inputs(prompt_text: str, tokenizer: Any, device: str) -> dict[str, torch.Tensor]:
    max_input_tokens = _max_input_tokens(tokenizer)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    total_tokens = int(input_ids.shape[1])
    if total_tokens > max_input_tokens:
        head_tokens = max_input_tokens * 3 // 5
        tail_tokens = max_input_tokens - head_tokens
        LOGGER.warning(
            "Prompt exceeded model input budget (%s > %s tokens); truncating with head/tail preservation.",
            total_tokens,
            max_input_tokens,
        )
        input_ids = torch.cat([input_ids[:, :head_tokens], input_ids[:, -tail_tokens:]], dim=1)
        attention_mask = torch.cat([attention_mask[:, :head_tokens], attention_mask[:, -tail_tokens:]], dim=1)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def generate_text(prompt: str, max_new_tokens: int = 300) -> str:
    with _GENERATION_LOCK:
        tokenizer = _get_tokenizer()
        model = _get_model()
        device, _ = _device_config()

        prompt_text = _build_prompt(prompt)
        inputs = _prepare_inputs(prompt_text, tokenizer, device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
