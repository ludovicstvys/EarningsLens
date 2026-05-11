from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_CACHE_DIR = Path(os.getenv("LOCAL_MODEL_CACHE_DIR", PROJECT_ROOT / ".models")).expanduser()

LOCAL_LLM_MODEL_ID = os.getenv("MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
SENTENCE_EMBEDDER_MODEL_ID = os.getenv("SENTENCE_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _preferred_model_source(env_var: str, local_subdir: str, model_id: str) -> str:
    explicit = os.getenv(env_var)
    if explicit:
        return explicit
    local_path = LOCAL_MODEL_CACHE_DIR / local_subdir
    if local_path.exists():
        return str(local_path)
    return model_id


LOCAL_LLM_SOURCE = _preferred_model_source("LOCAL_LLM_PATH", "local-llm", LOCAL_LLM_MODEL_ID)
FINBERT_SOURCE = _preferred_model_source("LOCAL_FINBERT_PATH", "finbert", FINBERT_MODEL_ID)
SENTENCE_EMBEDDER_SOURCE = _preferred_model_source(
    "LOCAL_SENTENCE_EMBEDDER_PATH",
    "sentence-embedder",
    SENTENCE_EMBEDDER_MODEL_ID,
)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
HEDGING_FLAG_THRESHOLD = 25.0
SENTIMENT_FLAG_THRESHOLD = -0.15
TOPIC_SIMILARITY_THRESHOLD = 0.75
EVASION_FLAG_THRESHOLD = 4
MAX_QA_PAIRS = 6
MIN_QA_PAIR_CONFIDENCE = 0.55
ANALYSIS_TIMEOUT_SECONDS = 45
