from __future__ import annotations

import json

from earningslens import prompts
from earningslens.local_llm import generate_text
from earningslens.models import EarningsBrief, EvasionScore, HedgingAnalysis, RiskVocabItem, SentimentAnalysis, TopicDrift


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Local model response did not contain JSON.")
    return json.loads(text[start:end + 1])


def _format_risk_movers(risk_vocab: list[RiskVocabItem], top_n: int = 8) -> str:
    lines = []
    for item in risk_vocab[:top_n]:
        delta_pct = "NEW" if item.delta_pct == 999.0 else f"{item.delta_pct:+.1f}%"
        lines.append(
            f"- {item.keyword} ({item.category}): prior={item.prior_count}, current={item.current_count}, delta={item.delta:+d}, delta_pct={delta_pct}"
        )
    return "\n".join(lines) or "- None"


def _format_evasions(evasions: list[EvasionScore], top_n: int = 3) -> str:
    flagged = [item for item in evasions if item.flagged][:top_n]
    if not flagged:
        return "- None"
    return "\n".join(
        f"- {item.question_speaker} -> {item.answer_speaker}: responsiveness {item.responsiveness}/10 because {item.reasoning}"
        for item in flagged
    )


def synthesize(
    company: str,
    quarter: str,
    prior_quarter: str,
    sentiment: SentimentAnalysis,
    hedging: HedgingAnalysis,
    topics: TopicDrift,
    risk_vocab: list[RiskVocabItem],
    evasions: list[EvasionScore],
) -> EarningsBrief:
    flag_count = sum(
        [
            sentiment.flagged,
            hedging.flagged,
            topics.flagged,
            any(item.flagged for item in evasions),
        ]
    )
    overall_signal = "green" if flag_count == 0 else "amber" if flag_count == 1 else "red"

    response = generate_text(
        prompts.SYNTHESIS_PROMPT.format(
            company=company,
            quarter=quarter,
            prior_quarter=prior_quarter,
            sentiment_prepared=sentiment.prepared_score,
            sentiment_qa=sentiment.qa_score,
            sentiment_gap=sentiment.gap,
            sentiment_flagged=sentiment.flagged,
            hedging_prior=hedging.prior_density,
            hedging_current=hedging.current_density,
            hedging_delta=hedging.delta_pct,
            hedging_flagged=hedging.flagged,
            top_hedges=hedging.top_hedges,
            topic_similarity=topics.semantic_similarity,
            new_themes=topics.new_themes,
            dropped_themes=topics.dropped_themes,
            topics_flagged=topics.flagged,
            risk_movers=_format_risk_movers(risk_vocab),
            n_evasions=len([item for item in evasions if item.flagged]),
            evasion_summary=_format_evasions(evasions),
            signal=overall_signal,
        ),
        max_new_tokens=260,
    )
    payload = _extract_json(response)

    return EarningsBrief(
        company=company,
        quarter=quarter,
        prior_quarter=prior_quarter,
        sentiment=sentiment,
        hedging=hedging,
        topics=topics,
        risk_vocab=risk_vocab,
        evasions=evasions,
        overall_signal=overall_signal,  # type: ignore[arg-type]
        flag_count=flag_count,
        executive_summary=str(payload.get("executive_summary", "")).strip(),
        bullet_points=[str(item).strip() for item in payload.get("bullet_points", []) if str(item).strip()],
    )
