from __future__ import annotations

import logging
import re
from functools import lru_cache

from earningslens import config
from earningslens.model_memory import clear_model_memory
from earningslens.models import SentimentAnalysis, Transcript

LOGGER = logging.getLogger(__name__)
WORD_RE = re.compile(r"\S+")


@lru_cache(maxsize=1)
def _get_finbert():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=config.FINBERT_SOURCE, return_all_scores=True)


def warmup_sentiment_model() -> None:
    _get_finbert()


def release_sentiment_model() -> None:
    _get_finbert.cache_clear()
    clear_model_memory()


def _analysis_section_text(transcript: Transcript, section: str) -> str:
    return " ".join(turn.text for turn in transcript.turns if turn.section == section and turn.role != "operator")


def _chunk_text(text: str, max_words: int = 320) -> list[str]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _score(text: str) -> float:
    chunks = _chunk_text(text)
    if not chunks:
        return 0.0

    finbert = _get_finbert()
    raw_results = finbert(chunks, batch_size=min(8, len(chunks)))
    if isinstance(raw_results, dict):
        results = [[raw_results]]
    elif isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
        results = [raw_results]
    else:
        results = raw_results
    scores: list[float] = []
    for result in results:
        by_label = {item["label"].lower(): float(item["score"]) for item in result}
        scores.append(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))
    return sum(scores) / len(scores)


def analyze_sentiment(transcript: Transcript) -> SentimentAnalysis:
    prepared_text = _analysis_section_text(transcript, "prepared")
    prepared_score = _score(prepared_text)

    qa_text = _analysis_section_text(transcript, "qa")
    if not qa_text.strip():
        raise ValueError(f"Transcript {transcript.company} {transcript.quarter} has no parsed Q&A text.")

    qa_score = _score(qa_text)
    gap = qa_score - prepared_score
    return SentimentAnalysis(
        prepared_score=prepared_score,
        qa_score=qa_score,
        gap=gap,
        flagged=gap <= config.SENTIMENT_FLAG_THRESHOLD,
        source="finbert",
    )
