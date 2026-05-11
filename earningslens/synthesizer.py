from __future__ import annotations

import json
import logging

from earningslens import prompts
from earningslens.local_llm import generate_text
from earningslens.models import EarningsBrief, EvasionScore, HedgingAnalysis, RiskVocabItem, SentimentAnalysis, TopicDrift

LOGGER = logging.getLogger(__name__)


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


def _format_delta_pct(value: float) -> str:
    return "NEW" if value == 999.0 else f"{value:+.1f}%"


def _truncate_list(items: list[str], max_items: int = 6) -> list[str]:
    return items[:max_items]


def _build_fallback_summary(
    company: str,
    quarter: str,
    overall_signal: str,
    sentiment: SentimentAnalysis,
    hedging: HedgingAnalysis,
    topics: TopicDrift,
    evasions: list[EvasionScore],
) -> str:
    signal_label = overall_signal.upper()
    flagged_evasions = sum(1 for item in evasions if item.flagged)
    highlights = []
    if sentiment.flagged:
        highlights.append(f"Q&A sentiment trailed prepared remarks by {sentiment.gap:+.2f}")
    if hedging.flagged:
        if hedging.delta_pct == 999.0:
            highlights.append("hedging density rose from zero in the prior quarter")
        else:
            highlights.append(f"hedging density moved {_format_delta_pct(hedging.delta_pct)} vs prior quarter")
    if topics.flagged:
        highlights.append(f"theme similarity fell to {topics.semantic_similarity:.2f}")
    if flagged_evasions:
        highlights.append(f"{flagged_evasions} Q&A exchanges looked evasive")
    if not highlights:
        highlights.append("the main tracked dimensions stayed stable versus the prior quarter")
    return f"{signal_label} on {company}'s {quarter} call: " + "; ".join(highlights[:3]) + "."


def _build_fallback_bullets(
    sentiment: SentimentAnalysis,
    hedging: HedgingAnalysis,
    topics: TopicDrift,
    risk_vocab: list[RiskVocabItem],
    evasions: list[EvasionScore],
) -> list[str]:
    bullets = [
        f"Prepared sentiment {sentiment.prepared_score:+.2f}; Q&A sentiment {sentiment.qa_score:+.2f}.",
        f"Hedging density {hedging.current_density:.1f} per 1000 words vs {hedging.prior_density:.1f} prior ({_format_delta_pct(hedging.delta_pct)}).",
        f"Topic similarity scored {topics.semantic_similarity:.2f} vs the prior quarter.",
    ]
    if risk_vocab:
        top_risk = risk_vocab[0]
        bullets.append(
            f"Top risk-vocab mover: {top_risk.keyword} {top_risk.delta:+d} to {top_risk.current_count} mentions."
        )
    flagged_evasions = [item for item in evasions if item.flagged]
    if flagged_evasions:
        worst = min(flagged_evasions, key=lambda item: item.responsiveness)
        bullets.append(
            f"Lowest Q&A responsiveness was {worst.responsiveness}/10 from {worst.answer_speaker}."
        )
    return bullets[:5]


def _parse_synthesis_payload(prompt: str) -> dict | None:
    last_error: Exception | None = None
    retry_suffix = (
        "\n\nIMPORTANT: Respond with a single valid JSON object only. "
        'Do not include commentary, markdown, or code fences. If unsure, still emit the JSON schema exactly.'
    )
    for attempt in range(2):
        response = generate_text(prompt if attempt == 0 else prompt + retry_suffix, max_new_tokens=260)
        try:
            return _extract_json(response)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Synthesis JSON parse failed on attempt %s: %s. Raw response: %r", attempt + 1, exc, response[:400])
    if last_error:
        LOGGER.warning("Falling back to deterministic synthesis after parse failures: %s", last_error)
    return None


def synthesize(
    company: str,
    quarter: str,
    prior_quarter: str,
    sentiment: SentimentAnalysis,
    hedging: HedgingAnalysis,
    topics: TopicDrift,
    risk_vocab: list[RiskVocabItem],
    evasions: list[EvasionScore],
    analysis_provenance: dict[str, str] | None = None,
    analysis_notes: dict[str, list[str]] | None = None,
    parser_overview: dict[str, object] | None = None,
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

    prompt = prompts.SYNTHESIS_PROMPT.format(
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
        new_themes=_truncate_list(topics.new_themes),
        dropped_themes=_truncate_list(topics.dropped_themes),
        topics_flagged=topics.flagged,
        risk_movers=_format_risk_movers(risk_vocab),
        n_evasions=len([item for item in evasions if item.flagged]),
        evasion_summary=_format_evasions(evasions),
        signal=overall_signal,
    )
    payload = _parse_synthesis_payload(prompt) or {}
    executive_summary = str(payload.get("executive_summary", "")).strip()
    bullet_points = [str(item).strip() for item in payload.get("bullet_points", []) if str(item).strip()]
    synthesis_source = "llm_json"
    synthesis_notes = list((analysis_notes or {}).get("synthesis", []))
    if not executive_summary:
        executive_summary = _build_fallback_summary(
            company=company,
            quarter=quarter,
            overall_signal=overall_signal,
            sentiment=sentiment,
            hedging=hedging,
            topics=topics,
            evasions=evasions,
        )
        synthesis_source = "fallback_summary"
        synthesis_notes.append("Executive summary came from deterministic synthesis fallback.")
    if not bullet_points:
        bullet_points = _build_fallback_bullets(
            sentiment=sentiment,
            hedging=hedging,
            topics=topics,
            risk_vocab=risk_vocab,
            evasions=evasions,
        )
        synthesis_source = "fallback_summary"
        synthesis_notes.append("Bullet points came from deterministic synthesis fallback.")

    final_provenance = dict(analysis_provenance or {})
    final_notes = dict(analysis_notes or {})
    final_provenance["synthesis"] = synthesis_source
    final_notes["synthesis"] = synthesis_notes

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
        executive_summary=executive_summary,
        bullet_points=bullet_points,
        analysis_provenance=final_provenance,
        analysis_notes=final_notes,
        parser_overview=parser_overview or {},
    )
