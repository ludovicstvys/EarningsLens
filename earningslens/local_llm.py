from __future__ import annotations

import logging
from functools import lru_cache

import torch

from earningslens import config

LOGGER = logging.getLogger(__name__)


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


def generate_text(prompt: str, max_new_tokens: int = 300) -> str:
    tokenizer = _get_tokenizer()
    model = _get_model()
    device, _ = _device_config()

    prompt_text = _build_prompt(prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}

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
