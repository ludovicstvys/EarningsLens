from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz

from earningslens import config, prompts
from earningslens.local_llm import generate_text
from earningslens.models import TopicDrift, Transcript
from earningslens.vocab import RISK_KEYWORDS

WORD_RE = re.compile(r"\S+")
RISK_TERMS = [keyword.lower() for keywords in RISK_KEYWORDS.values() for keyword in keywords]


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.SENTENCE_EMBEDDER_SOURCE)


def warmup_topic_models() -> None:
    _get_embedder()


def _chunk_text(text: str, max_words: int = 320) -> list[str]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _mean_embedding(text: str) -> np.ndarray:
    chunks = _chunk_text(text)
    if not chunks:
        return np.zeros(384, dtype=float)
    embeddings = _get_embedder().encode(chunks)
    array = np.array(embeddings, dtype=float)
    return array.mean(axis=0)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Local model response did not contain JSON.")
    return json.loads(text[start:end + 1])


def _clean_themes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(theme).strip() for theme in value if str(theme).strip()]


def _extract_themes_pair(current_prepared_text: str, prior_prepared_text: str) -> tuple[list[str], list[str]]:
    response = generate_text(
        prompts.THEME_COMPARISON_PROMPT.format(
            current_prepared_text=current_prepared_text,
            prior_prepared_text=prior_prepared_text,
        ),
        max_new_tokens=320,
    )
    payload = _extract_json(response)
    return _clean_themes(payload.get("current_themes")), _clean_themes(payload.get("prior_themes"))


def _theme_in_list(candidate: str, themes: list[str]) -> bool:
    candidate_lower = candidate.lower()
    candidate_words = set(candidate_lower.split())
    for theme in themes:
        theme_lower = theme.lower()
        theme_words = set(theme_lower.split())
        if fuzz.ratio(candidate_lower, theme_lower) >= 80:
            return True
        if candidate_lower in theme_lower or theme_lower in candidate_lower:
            return True
        if candidate_words.issubset(theme_words) or theme_words.issubset(candidate_words):
            return True
    return False


def _set_difference(source: list[str], comparator: list[str]) -> list[str]:
    return [theme for theme in source if not _theme_in_list(theme, comparator)]


def analyze_topics(current: Transcript, prior: Transcript) -> TopicDrift:
    current_emb = _mean_embedding(current.prepared_text)
    prior_emb = _mean_embedding(prior.prepared_text)
    semantic_similarity = _cosine_similarity(current_emb, prior_emb)

    current_themes, prior_themes = _extract_themes_pair(current.prepared_text, prior.prepared_text)
    new_themes = _set_difference(current_themes, prior_themes)
    dropped_themes = _set_difference(prior_themes, current_themes)

    flagged = (
        semantic_similarity < config.TOPIC_SIMILARITY_THRESHOLD
        or len(new_themes) >= 3
        or any(any(term in theme.lower() for term in RISK_TERMS) for theme in new_themes)
    )

    return TopicDrift(
        current_themes=current_themes,
        prior_themes=prior_themes,
        new_themes=new_themes,
        dropped_themes=dropped_themes,
        semantic_similarity=semantic_similarity,
        flagged=flagged,
    )
