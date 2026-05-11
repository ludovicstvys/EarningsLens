from __future__ import annotations

import re

from earningslens.models import RiskVocabItem, Transcript
from earningslens.vocab import RISK_KEYWORDS

RISK_PATTERNS = {
    category: {keyword: re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE) for keyword in keywords}
    for category, keywords in RISK_KEYWORDS.items()
}


def _count_keyword(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def analyze_risk_vocab(current: Transcript, prior: Transcript) -> list[RiskVocabItem]:
    items: list[RiskVocabItem] = []
    for category, patterns in RISK_PATTERNS.items():
        for keyword, pattern in patterns.items():
            current_count = _count_keyword(current.full_text, pattern)
            prior_count = _count_keyword(prior.full_text, pattern)
            delta = current_count - prior_count
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
