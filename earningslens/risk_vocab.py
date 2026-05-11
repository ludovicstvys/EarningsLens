from __future__ import annotations

import re

from earningslens.models import RiskVocabItem, Transcript
from earningslens.vocab import RISK_KEYWORDS

RISK_PATTERNS = {
    category: {keyword: re.compile(rf"\b{re.escape(keyword.lower())}\b", re.IGNORECASE) for keyword in keywords}
    for category, keywords in RISK_KEYWORDS.items()
}


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "fx": "foreign exchange",
        "write down": "writedown",
        "write-down": "writedown",
        "one time": "one-time",
        "supply-chain": "supply chain",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _count_keyword(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def analyze_risk_vocab(current: Transcript, prior: Transcript) -> list[RiskVocabItem]:
    current_text = _normalize_text(current.full_text)
    prior_text = _normalize_text(prior.full_text)
    items: list[RiskVocabItem] = []
    for category, patterns in RISK_PATTERNS.items():
        for keyword, pattern in patterns.items():
            current_count = _count_keyword(current_text, pattern)
            prior_count = _count_keyword(prior_text, pattern)
            delta = current_count - prior_count
            if current_count == 0 and prior_count == 0:
                continue
            if prior_count == 0 and current_count > 0:
                delta_pct = 999.0
            elif prior_count == 0:
                delta_pct = 0.0
            else:
                delta_pct = delta / prior_count * 100
            items.append(
                RiskVocabItem(
                    keyword=keyword,
                    category=category,
                    current_count=current_count,
                    prior_count=prior_count,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )
    return sorted(items, key=lambda item: abs(item.delta), reverse=True)
