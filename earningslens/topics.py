from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz

from earningslens import config, prompts
from earningslens.local_llm import generate_text
from earningslens.model_memory import clear_model_memory
from earningslens.models import TopicDrift, Transcript
from earningslens.vocab import RISK_KEYWORDS

WORD_RE = re.compile(r"\S+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-&]{1,}")
RISK_TERMS = [keyword.lower() for keywords in RISK_KEYWORDS.values() for keyword in keywords]
LOGGER = logging.getLogger(__name__)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "this", "to", "was", "we",
    "were", "with", "will", "would", "you", "your",
}


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.SENTENCE_EMBEDDER_SOURCE)


def warmup_topic_models() -> None:
    _get_embedder()


def release_topic_models() -> None:
    _get_embedder.cache_clear()
    clear_model_memory()


def _chunk_text(text: str, max_words: int = 320) -> list[str]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _truncate_for_prompt(text: str, max_words: int = 900) -> str:
    words = WORD_RE.findall(text)
    if len(words) <= max_words:
        return text
    head_words = max_words * 2 // 3
    tail_words = max_words - head_words
    return " ".join(words[:head_words] + ["..."] + words[-tail_words:])


def _mean_embedding(text: str) -> np.ndarray:
    chunks = _chunk_text(text)
    if not chunks:
        embedding_dim = _get_embedder().get_sentence_embedding_dimension()
        return np.zeros(embedding_dim, dtype=float)
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


def _fallback_themes(text: str, limit: int = 6) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(tokens) < 2:
        return []
    counts: Counter[str] = Counter()
    for ngram_size in range(2, 5):
        for idx in range(len(tokens) - ngram_size + 1):
            phrase_tokens = tokens[idx:idx + ngram_size]
            if phrase_tokens[0] in STOPWORDS or phrase_tokens[-1] in STOPWORDS:
                continue
            if all(token in STOPWORDS for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            counts[phrase] += 1
    themes: list[str] = []
    for phrase, _count in counts.most_common(limit * 4):
        if _theme_in_list(phrase, themes):
            continue
        themes.append(phrase)
        if len(themes) >= limit:
            break
    return themes


def _parse_theme_payload(prompt: str) -> dict | None:
    retry_suffix = (
        "\n\nIMPORTANT: Respond with a single valid JSON object only. "
        'Do not include commentary, markdown, or code fences.'
    )
    last_error: Exception | None = None
    for attempt in range(2):
        response = generate_text(prompt if attempt == 0 else prompt + retry_suffix, max_new_tokens=320)
        try:
            return _extract_json(response)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Theme extraction parse failed on attempt %s: %s. Raw response: %r", attempt + 1, exc, response[:400])
    if last_error:
        LOGGER.warning("Falling back to deterministic theme extraction after parse failures: %s", last_error)
    return None


def _extract_themes_pair(current_prepared_text: str, prior_prepared_text: str) -> tuple[list[str], list[str], str, list[str]]:
    prompt = prompts.THEME_COMPARISON_PROMPT.format(
        current_prepared_text=_truncate_for_prompt(current_prepared_text),
        prior_prepared_text=_truncate_for_prompt(prior_prepared_text),
    )
    payload = _parse_theme_payload(prompt) or {}
    notes: list[str] = []
    source = "llm_json"
    current_themes = _clean_themes(payload.get("current_themes"))
    prior_themes = _clean_themes(payload.get("prior_themes"))
    if not current_themes:
        current_themes = _fallback_themes(current_prepared_text)
        notes.append("Current-quarter themes came from extractive fallback, not model JSON.")
        source = "fallback_extract"
    if not prior_themes:
        prior_themes = _fallback_themes(prior_prepared_text)
        notes.append("Prior-quarter themes came from extractive fallback, not model JSON.")
        source = "fallback_extract"
    return current_themes, prior_themes, source, notes


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
    if config.LOW_MEMORY_MODE:
        release_topic_models()

    current_themes, prior_themes, source, notes = _extract_themes_pair(current.prepared_text, prior.prepared_text)
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
        source=source,
        notes=notes,
    )
