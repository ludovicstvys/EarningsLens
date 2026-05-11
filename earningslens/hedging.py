from __future__ import annotations

import re
from collections import Counter

from earningslens import config
from earningslens.models import HedgingAnalysis, Transcript
from earningslens.vocab import HEDGE_WORDS

WORD_RE = re.compile(r"\b\w+\b")
HEDGE_PATTERNS = {word: re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in HEDGE_WORDS}


def _analysis_text(transcript: Transcript) -> str:
    return " ".join(turn.text for turn in transcript.turns if turn.role != "operator")


def _count_hedges(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word, pattern in HEDGE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            counts[word] = len(matches)
    return counts


def analyze_hedging(current: Transcript, prior: Transcript) -> HedgingAnalysis:
    current_text = _analysis_text(current)
    prior_text = _analysis_text(prior)
    current_counts = _count_hedges(current_text)
    prior_counts = _count_hedges(prior_text)

    current_words = max(len(WORD_RE.findall(current_text)), 1)
    prior_words = max(len(WORD_RE.findall(prior_text)), 1)
    current_density = sum(current_counts.values()) / current_words * 1000
    prior_density = sum(prior_counts.values()) / prior_words * 1000
    delta_pct = ((current_density - prior_density) / prior_density * 100) if prior_density > 0 else 0.0

    return HedgingAnalysis(
        current_density=current_density,
        prior_density=prior_density,
        delta_pct=delta_pct,
        flagged=delta_pct >= config.HEDGING_FLAG_THRESHOLD,
        top_hedges=current_counts.most_common(5),
    )
