from __future__ import annotations

import json
import logging
import re
from statistics import mean

from earningslens import config, prompts
from earningslens.local_llm import generate_text
from earningslens.models import EvasionAnswerTurn, EvasionScore, SpeakerTurn, Transcript

LOGGER = logging.getLogger(__name__)
WORD_RE = re.compile(r"\b[a-z0-9]+\b")
QUANT_HINT_RE = re.compile(r"\b(how much|how many|quantify|percentage|percent|basis points|bps|number|amount|magnitude)\b", re.IGNORECASE)
MULTIPART_HINT_RE = re.compile(r"\b(and|as well as|whether)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for match in re.finditer(r"{", text):
        try:
            payload, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    if not candidates:
        raise ValueError("Local model response did not contain valid JSON.")

    for key in ("scores", "responsiveness", "themes", "current_themes"):
        keyed_candidates = [item for item in candidates if key in item]
        if keyed_candidates:
            return keyed_candidates[-1]
    return candidates[-1]


def _truncate_for_prompt(text: str, max_words: int = 140) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    head_words = max_words * 2 // 3
    tail_words = max_words - head_words
    return " ".join(words[:head_words] + ["..."] + words[-tail_words:])


def _token_set(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _looks_like_echo(reasoning: str, answer: str) -> bool:
    reasoning_tokens = _token_set(reasoning)
    answer_tokens = _token_set(answer)
    if not reasoning_tokens or not answer_tokens:
        return False
    overlap = len(reasoning_tokens & answer_tokens) / max(1, len(reasoning_tokens))
    return overlap >= 0.8 or reasoning.strip().lower() in answer.strip().lower()


def _fallback_reasoning(question: str, answer: str, responsiveness: int | None = None) -> str:
    notes: list[str] = []
    lowered_question = question.lower()
    lowered_answer = answer.lower()
    if QUANT_HINT_RE.search(lowered_question) and not NUMBER_RE.search(answer):
        notes.append("The answer did not quantify the issue despite an explicit request for numbers.")
    if MULTIPART_HINT_RE.search(lowered_question) and "segment" in lowered_question and "segment" not in lowered_answer:
        notes.append("The answer did not clearly address every part of the multi-part question.")
    if not notes and responsiveness is not None and responsiveness >= 8:
        notes.append("The answer addressed the main question with relevant demand, capacity, monetization, and efficiency details.")
    if not notes and responsiveness is not None and responsiveness >= 6:
        notes.append("The answer addressed the topic directionally but did not fully resolve every part of the question.")
    if not notes:
        notes.append("The answer was only partially responsive and relied on generalities rather than a complete direct answer.")
    return " ".join(notes)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "about",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "there",
        "these",
        "think",
        "this",
        "to",
        "we",
        "what",
        "where",
        "which",
        "who",
        "with",
        "you",
    }
    return {token for token in WORD_RE.findall(text.lower()) if len(token) > 2 and token not in stopwords}


def _content_overlap(question: str, answer: str) -> float:
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(answer)
    return len(question_tokens & answer_tokens) / max(1, len(question_tokens))


def _looks_like_missing_question_claim(reasoning: str) -> bool:
    lowered = reasoning.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not asked",
            "was not asked",
            "wasn't asked",
            "question was not",
            "question wasn't",
            "not able to provide a specific answer",
            "unable to provide a specific answer",
        )
    )


def _looks_like_speaker_names(reasoning: str, answer_speaker: str) -> bool:
    normalized_reasoning = re.sub(r"\s+", " ", reasoning).strip().lower()
    normalized_speaker = re.sub(r"\s+", " ", answer_speaker).strip().lower()
    if not normalized_reasoning:
        return True
    if normalized_speaker and normalized_reasoning == normalized_speaker:
        return True

    reasoning_tokens = WORD_RE.findall(reasoning)
    if len(reasoning_tokens) < 3 and not any(char in reasoning for char in ".;:"):
        return True

    speaker_tokens = set(WORD_RE.findall(answer_speaker.lower()))
    if reasoning_tokens and speaker_tokens:
        overlap = sum(1 for token in reasoning_tokens if token.lower() in speaker_tokens) / len(reasoning_tokens)
        if overlap >= 0.8:
            return True

    return False


def _looks_like_invalid_reasoning(reasoning: str) -> bool:
    normalized = re.sub(r"\s+", " ", reasoning).strip().lower()
    if not normalized:
        return True
    if "<" in reasoning or ">" in reasoning:
        return True
    if normalized in {"reasoning", "one sentence", "fallback score from single-pair pass"}:
        return True
    if re.fullmatch(r"(responsiveness|score)\s*[:=]?\s*\d{1,2}(?:/10)?\.?", normalized):
        return True
    if "write a" in normalized and "bullet" in normalized:
        return True
    return False


def _heuristic_score_single_pair(question: str, answer: str) -> tuple[int, str]:
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(answer)
    answer_word_count = len(WORD_RE.findall(answer))
    overlap = _content_overlap(question, answer)

    if not answer_tokens or answer_word_count < 12:
        return 3, "The answer was too brief to materially address the analyst's question."

    lowered_answer = answer.lower()
    if re.search(r"\b(no comment|don't comment|do not comment|not going to comment|cannot comment)\b", lowered_answer):
        return 3, "The answer declined to address the question directly."

    if QUANT_HINT_RE.search(question) and not NUMBER_RE.search(answer):
        return 6, "The answer addressed the topic directionally but did not provide the requested figures."

    if overlap >= 0.2:
        return 7, "The answer addressed several terms from the question but stayed broad rather than fully specific."
    if overlap >= 0.1:
        return 5, "The answer partially touched the question while leaning on generalities."
    return 4, "The answer mostly discussed adjacent themes rather than the specific question asked."


def _adjust_score(question: str, answer: str, answer_speaker: str, responsiveness: int, reasoning: str) -> tuple[int, str, list[str]]:
    notes: list[str] = []
    adjusted = responsiveness
    final_reasoning = reasoning.strip()
    lowered_question = question.lower()
    lowered_answer = answer.lower()

    if _looks_like_echo(final_reasoning, answer):
        final_reasoning = _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning echoed the answer; replaced with fallback explanation.")

    if _looks_like_invalid_reasoning(final_reasoning):
        final_reasoning = _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning was invalid or placeholder-like; replaced with fallback explanation.")

    if _looks_like_speaker_names(final_reasoning, answer_speaker):
        final_reasoning = _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning contained only speaker names; replaced with fallback explanation.")

    if adjusted <= 3 and _looks_like_missing_question_claim(final_reasoning) and _content_overlap(question, answer) >= 0.05:
        adjusted, final_reasoning = _heuristic_score_single_pair(question, answer)
        notes.append("Model claimed the question was not asked despite content overlap; replaced with deterministic score.")

    if QUANT_HINT_RE.search(lowered_question) and not NUMBER_RE.search(answer):
        adjusted = min(adjusted, 7)
        notes.append("Capped score because a quantitative question was answered without concrete figures.")

    if (" and " in lowered_question or "whether" in lowered_question) and "segment" in lowered_question and "segment" not in lowered_answer:
        adjusted = min(adjusted, 7)
        notes.append("Capped score because a multi-part question left the segment-specific part unanswered.")

    if adjusted == 10 and ("varied" in lowered_answer or "generally" in lowered_answer or "some" in lowered_answer):
        adjusted = 8
        notes.append("Reduced perfect score because the answer was qualified rather than fully specific.")

    return max(0, min(10, adjusted)), final_reasoning, notes


def _extract_pairs(transcript: Transcript) -> list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]]:
    qa_turns = [turn for turn in transcript.turns if turn.section == "qa" and turn.role != "operator"]
    pairs: list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]] = []
    idx = 0
    while idx < len(qa_turns):
        turn = qa_turns[idx]
        if turn.role != "analyst":
            idx += 1
            continue

        question = turn
        idx += 1
        answer_turns: list[SpeakerTurn] = []
        while idx < len(qa_turns):
            candidate = qa_turns[idx]
            if candidate.role == "analyst":
                break
            if candidate.role == "executive":
                answer_turns.append(candidate)
            idx += 1

        if answer_turns:
            notes: list[str] = []
            confidence_values = [question.role_confidence, *[item.role_confidence for item in answer_turns]]
            pair_confidence = mean(confidence_values)
            if question.speaker == "Unknown":
                notes.append("Question speaker was unknown.")
                pair_confidence = min(pair_confidence, 0.25)
            if any(item.speaker == "Unknown" for item in answer_turns):
                notes.append("At least one answer speaker was unknown.")
                pair_confidence = min(pair_confidence, 0.4)
            if question.role_confidence < config.MIN_QA_PAIR_CONFIDENCE:
                notes.append("Question role confidence was low.")
            if mean(item.role_confidence for item in answer_turns) < config.MIN_QA_PAIR_CONFIDENCE:
                notes.append("Answer role confidence was low.")
            pairs.append((question, answer_turns, pair_confidence, notes))
    return pairs


def _build_evasion_score(
    question: SpeakerTurn,
    answer_turns: list[SpeakerTurn],
    responsiveness: int,
    reasoning: str,
    pair_confidence: float = 1.0,
    low_confidence: bool = False,
    source: str = "model",
    notes: list[str] | None = None,
) -> EvasionScore:
    answer = " ".join(turn.text for turn in answer_turns)
    answer_speaker = ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns))
    adjusted_responsiveness, adjusted_reasoning, adjustment_notes = _adjust_score(
        question.text,
        answer,
        answer_speaker,
        responsiveness,
        reasoning,
    )
    return EvasionScore(
        question=question.text,
        question_speaker=question.speaker,
        answer=answer,
        answer_speaker=answer_speaker,
        answer_turns=[EvasionAnswerTurn(speaker=turn.speaker, text=turn.text) for turn in answer_turns],
        responsiveness=adjusted_responsiveness,
        reasoning=adjusted_reasoning,
        flagged=(adjusted_responsiveness <= config.EVASION_FLAG_THRESHOLD) and not low_confidence,
        pair_confidence=pair_confidence,
        low_confidence=low_confidence,
        source=source,
        notes=[*(notes or []), *adjustment_notes],
    )


def _score_single_pair(question: SpeakerTurn, answer_turns: list[SpeakerTurn]) -> tuple[int, str]:
    answer = " ".join(turn.text for turn in answer_turns)
    answer_speaker = ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns))
    prompt = prompts.EVASION_SCORING_PROMPT.format(
        q_speaker=question.speaker,
        question=_truncate_for_prompt(question.text),
        a_speaker=answer_speaker,
        answer=_truncate_for_prompt(answer, 180),
    )
    retry_suffix = (
        "\n\nIMPORTANT: Respond with a single valid JSON object only. "
        'Do not include commentary, markdown, or code fences.'
    )
    last_error: Exception | None = None
    for attempt in range(2):
        response = generate_text(prompt if attempt == 0 else prompt + retry_suffix, max_new_tokens=120)
        try:
            payload = _extract_json(response)
            responsiveness = int(payload.get("responsiveness", 0))
            reasoning = str(payload.get("reasoning", "")).strip() or "Fallback score from single-pair pass."
            return max(0, min(10, responsiveness)), reasoning
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            LOGGER.warning("Single-pair evasion parse failed on attempt %s: %s. Raw response: %r", attempt + 1, exc, response[:300])
    LOGGER.warning("Using deterministic fallback for single-pair evasion scoring: %s", last_error)
    return _heuristic_score_single_pair(question.text, answer)


def _score_pairs(pairs: list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]]) -> list[EvasionScore]:
    prompt_items = []
    scored_pairs: list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]] = []
    skipped_pairs: list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]] = []
    for question, answer_turns, pair_confidence, notes in pairs:
        if pair_confidence < config.MIN_QA_PAIR_CONFIDENCE:
            skipped_pairs.append((question, answer_turns, pair_confidence, notes))
            continue
        pair_id = len(scored_pairs)
        scored_pairs.append((question, answer_turns, pair_confidence, notes))
        prompt_items.append(
            {
                "pair_id": pair_id,
                "question_speaker": question.speaker,
                "question": _truncate_for_prompt(question.text),
                "answer_speaker": ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns)),
                "answer": _truncate_for_prompt(" ".join(turn.text for turn in answer_turns), 180),
            }
        )

    if not scored_pairs:
        return [
            _build_evasion_score(
                question,
                answer_turns,
                responsiveness=10,
                reasoning="Skipped scoring because speaker parsing confidence was too low.",
                pair_confidence=pair_confidence,
                low_confidence=True,
                source="skipped_low_confidence",
                notes=notes,
            )
            for question, answer_turns, pair_confidence, notes in skipped_pairs
        ]

    prompt = prompts.EVASION_BATCH_SCORING_PROMPT.format(qa_pairs_json=json.dumps(prompt_items, ensure_ascii=True))
    retry_suffix = (
        "\n\nIMPORTANT: Respond with a single valid JSON object only. "
        'Do not include commentary, markdown, or code fences.'
    )
    payload = None
    last_error: Exception | None = None
    for attempt in range(2):
        response = generate_text(
            prompt if attempt == 0 else prompt + retry_suffix,
            max_new_tokens=min(360, 80 + len(prompt_items) * 45),
        )
        try:
            payload = _extract_json(response)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Batch evasion parse failed on attempt %s: %s. Raw response: %r", attempt + 1, exc, response[:400])
    if payload is None:
        LOGGER.warning("Falling back to single-pair evasion scoring for all pairs after batch parse failures: %s", last_error)
        return [
            _build_evasion_score(
                question,
                answer_turns,
                *_score_single_pair(question, answer_turns),
                pair_confidence=pair_confidence,
                source="single_pair_fallback",
                notes=notes,
            )
            for question, answer_turns, pair_confidence, notes in scored_pairs
        ] + [
            _build_evasion_score(
                question,
                answer_turns,
                responsiveness=10,
                reasoning="Skipped scoring because speaker parsing confidence was too low.",
                pair_confidence=pair_confidence,
                low_confidence=True,
                source="skipped_low_confidence",
                notes=notes,
            )
            for question, answer_turns, pair_confidence, notes in skipped_pairs
        ]

    raw_scores = payload.get("scores", [])
    score_by_pair_id: dict[int, tuple[int, str]] = {}
    if isinstance(raw_scores, list):
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            try:
                pair_id = int(item.get("pair_id"))
            except (TypeError, ValueError):
                continue
            try:
                responsiveness = int(item.get("responsiveness", 0))
            except (TypeError, ValueError):
                responsiveness = 0
            reasoning = str(item.get("reasoning", "")).strip()
            score_by_pair_id[pair_id] = (max(0, min(10, responsiveness)), reasoning)

    missing_pair_ids = [pair_id for pair_id in range(len(scored_pairs)) if pair_id not in score_by_pair_id]
    if missing_pair_ids:
        LOGGER.warning("Batch evasion scorer omitted pair_ids %s; rescoring individually.", missing_pair_ids)
        for pair_id in missing_pair_ids:
            question, answer_turns, _pair_confidence, _notes = scored_pairs[pair_id]
            score_by_pair_id[pair_id] = _score_single_pair(question, answer_turns)

    results: list[EvasionScore] = []
    for pair_id, (question, answer_turns, pair_confidence, notes) in enumerate(scored_pairs):
        responsiveness, reasoning = score_by_pair_id.get(pair_id, (0, "Unable to score this pair."))
        source = "batch_llm" if pair_id not in missing_pair_ids else "single_pair_fallback"
        results.append(
            _build_evasion_score(
                question,
                answer_turns,
                responsiveness,
                reasoning,
                pair_confidence=pair_confidence,
                source=source,
                notes=notes,
            )
        )
    results.extend(
        _build_evasion_score(
            question,
            answer_turns,
            responsiveness=10,
            reasoning="Skipped scoring because speaker parsing confidence was too low.",
            pair_confidence=pair_confidence,
            low_confidence=True,
            source="skipped_low_confidence",
            notes=notes,
        )
        for question, answer_turns, pair_confidence, notes in skipped_pairs
    )
    return results


def analyze_evasion(transcript: Transcript, progress_callback=None) -> list[EvasionScore]:
    pairs = _extract_pairs(transcript)
    if len(pairs) > config.MAX_QA_PAIRS:
        pairs = sorted(pairs, key=lambda pair: len(pair[0].text.split()), reverse=True)[: config.MAX_QA_PAIRS]

    if not pairs:
        return []
    if progress_callback:
        progress_callback(0.2, f"Scoring {len(pairs)} Q&A pairs")
    results = _score_pairs(pairs)
    if progress_callback:
        progress_callback(1.0, f"Scored {len(pairs)} Q&A pairs")
    return results
