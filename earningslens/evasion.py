from __future__ import annotations

import logging
import re
from statistics import mean

from earningslens import config, prompts
from earningslens.local_llm import generate_text_batch
from earningslens.model_memory import clear_model_memory
from earningslens.models import EvasionAnswerTurn, EvasionCoverageItem, EvasionScore, SpeakerTurn, Transcript

LOGGER = logging.getLogger(__name__)
WORD_RE = re.compile(r"\b[a-z0-9]+\b")
QUANT_HINT_RE = re.compile(r"\b(how much|how many|quantify|percentage|percent|basis points|bps|number|amount|magnitude)\b", re.IGNORECASE)
MULTIPART_HINT_RE = re.compile(r"\b(and|as well as|whether)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
QUESTION_START_RE = re.compile(r"\b(can you|could you|would you|how|what|where|when|why|whether|is there|are there|do you|does|did)\b", re.IGNORECASE)
FILLER_PREFIX_RE = re.compile(r"^(and|also|then|secondly|second|first|finally|just|maybe|on that|relatedly)\s*,?\s+", re.IGNORECASE)
META_PREAMBLE_RE = re.compile(
    r"^(?:a |one |my |last |another |separate |follow[- ]?up |quick |different |next |final )?"
    r"(?:quick |brief |short |separate |different |follow[- ]?up |last |final )?"
    r"(?:question|one|thing|topic|note|point|item|ask)s?\b[^.?!]{0,40}[.,:!]\s+",
    re.IGNORECASE,
)
ANSWER_FILLER_PREFIX_RE = re.compile(
    r"^(?:(?:yeah|yes|yep|sure|right|ok|okay|absolutely|certainly|of course|"
    r"great question|good question|thanks(?:\s+for\s+(?:the|that)\s+question)?|"
    r"thank you(?:\s+for\s+(?:the|that)\s+question)?|well)[\s,.!]+)+",
    re.IGNORECASE,
)
SCORE_LINE_RE = re.compile(r"^\s*SCORE\s*[:=]?\s*(\d{1,2})(?:\s*/\s*10)?\s*$", re.IGNORECASE | re.MULTILINE)
WHY_LINE_RE = re.compile(
    r"(?:WHY|REASON|REASONING|RATIONALE|EXPLANATION)\s*[:=]?\s*(.+?)(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
CoverageTuple = tuple[str, str, str]
ScoreTuple = tuple[int, str, list[CoverageTuple]]


def _truncate_for_prompt(text: str, max_words: int = 140) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    head_words = max_words * 2 // 3
    tail_words = max_words - head_words
    return " ".join(words[:head_words] + ["..."] + words[-tail_words:])


def _clean_question_part(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    for _ in range(2):
        stripped = META_PREAMBLE_RE.sub("", cleaned, count=1).strip()
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = cleaned.strip(" .?;:,")
    cleaned = FILLER_PREFIX_RE.sub("", cleaned).strip(" .?;:,")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def _strip_answer_filler(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    stripped = ANSWER_FILLER_PREFIX_RE.sub("", cleaned).strip()
    return stripped or cleaned


def _split_question_parts(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return []

    candidates: list[str] = []
    for sentence in re.split(r"\?\s+|;\s+", text):
        cleaned = _clean_question_part(sentence)
        if cleaned:
            candidates.append(cleaned)

    if len(candidates) == 1:
        base = candidates[0]
        should_split = (
            len(WORD_RE.findall(base)) >= 14
            and (
                re.search(r"\b(and whether|and how|and what|and where|and when|and why|as well as whether|as well as how)\b", base, re.IGNORECASE)
                or (QUANT_HINT_RE.search(base) and re.search(r"\b(whether|by segment|by geography|by product|by region)\b", base, re.IGNORECASE))
            )
        )
        if should_split:
            candidates = [_clean_question_part(part) for part in re.split(r"\b(?:and|as well as)\b(?=\s+(?:whether|how|what|where|when|why|by segment|by geography|by product|by region))", base, flags=re.IGNORECASE)]
            candidates = [part for part in candidates if part]

    if len(candidates) == 1:
        base = candidates[0]
        starts = list(QUESTION_START_RE.finditer(base))
        if len(starts) > 1 and len(WORD_RE.findall(base)) >= 18:
            split_candidates: list[str] = []
            for idx, match in enumerate(starts):
                end = starts[idx + 1].start() if idx + 1 < len(starts) else len(base)
                part = _clean_question_part(base[match.start() : end])
                if part:
                    split_candidates.append(part)
            if len(split_candidates) > 1:
                candidates = split_candidates

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if len(WORD_RE.findall(candidate)) < 3:
            continue
        key = re.sub(r"\W+", " ", candidate.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return (deduped or [_clean_question_part(text)])[:4]


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
        notes.append("The answer addressed the main question with relevant operational detail.")
    if not notes and responsiveness is not None and responsiveness >= 6:
        notes.append("The answer addressed the topic directionally but did not fully resolve every part of the question.")
    if not notes:
        notes.append("The answer was only partially responsive and relied on generalities rather than a complete direct answer.")
    return " ".join(notes)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "about", "all", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
        "do", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "our", "that", "the",
        "their", "there", "these", "think", "this", "to", "we", "what", "where", "which", "who",
        "with", "you",
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
    if normalized in {"reasoning", "one sentence", "fallback score from heuristic pass"}:
        return True
    if re.fullmatch(r"(responsiveness|score)\s*[:=]?\s*\d{1,2}(?:/10)?\.?", normalized):
        return True
    if "write a" in normalized and "bullet" in normalized:
        return True
    return False


def _looks_like_vague_reasoning(reasoning: str) -> bool:
    normalized = re.sub(r"\s+", " ", reasoning).strip().lower()
    vague_phrases = (
        "addressed the topic but did not provide all requested detail",
        "addressed the topic but lacked some detail",
        "addressed the topic but stayed broad",
    )
    return any(phrase in normalized for phrase in vague_phrases) and len(_content_tokens(reasoning)) <= 8


def _coverage_reasoning(coverage: list[CoverageTuple]) -> str:
    missed = [part for part, status, _r in coverage if status == "missed"]
    partial = [part for part, status, _r in coverage if status == "partial"]
    answered = [part for part, status, _r in coverage if status == "answered"]
    if missed:
        return f"The answer did not address: {missed[0]}."
    if partial:
        return f"The answer only partially addressed: {partial[0]}."
    if answered:
        return "The answer addressed the identified question parts with relevant detail."
    return "The answer was scored without structured coverage detail."


def _derive_score_from_coverage(coverage: list[CoverageTuple], model_score: int | None = None) -> int:
    if not coverage:
        return max(0, min(10, int(model_score or 0)))
    values = {"answered": 1.0, "partial": 0.5, "missed": 0.0}
    ratio = sum(values.get(status, 0.5) for _p, status, _r in coverage) / len(coverage)
    missed_count = sum(1 for _p, status, _r in coverage if status == "missed")
    partial_count = sum(1 for _p, status, _r in coverage if status == "partial")
    if ratio == 1:
        return 9 if model_score is None else max(8, min(10, model_score))
    if missed_count == 0:
        return 7 if partial_count else 8
    if ratio >= 0.5:
        return 6
    if ratio >= 0.25:
        return 4
    return 3


def _heuristic_coverage(question_parts: list[str], answer: str) -> list[CoverageTuple]:
    coverage: list[CoverageTuple] = []
    for part in question_parts:
        overlap = _content_overlap(part, answer)
        if QUANT_HINT_RE.search(part) and not NUMBER_RE.search(answer):
            coverage.append((part, "partial" if overlap >= 0.1 else "missed", "The answer did not provide the requested figure or range."))
        elif "segment" in part.lower() and "segment" not in answer.lower():
            coverage.append((part, "partial" if overlap >= 0.1 else "missed", "The answer did not give segment-specific detail."))
        elif overlap >= 0.25:
            coverage.append((part, "answered", "The answer discussed this part with overlapping specifics."))
        elif overlap >= 0.1:
            coverage.append((part, "partial", "The answer touched this part but stayed broad."))
        else:
            coverage.append((part, "missed", "The answer did not materially address this part."))
    return coverage


def _heuristic_score_single_pair(question: str, answer: str) -> tuple[int, str]:
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


def _adjust_score(
    question: str,
    answer: str,
    answer_speaker: str,
    responsiveness: int,
    reasoning: str,
    coverage: list[CoverageTuple],
) -> tuple[int, str, list[str]]:
    notes: list[str] = []
    adjusted = responsiveness
    final_reasoning = reasoning.strip()
    lowered_question = question.lower()
    lowered_answer = answer.lower()

    if coverage:
        missed_count = sum(1 for _p, status, _r in coverage if status == "missed")
        partial_count = sum(1 for _p, status, _r in coverage if status == "partial")
        if missed_count and adjusted >= 8:
            adjusted = 6 if missed_count == 1 and partial_count == 0 else 4
            notes.append("Capped score because coverage analysis found missed question parts.")
        elif missed_count == 0 and partial_count == 0 and adjusted < 7:
            adjusted = 7

    if _looks_like_vague_reasoning(final_reasoning):
        final_reasoning = _coverage_reasoning(coverage) if coverage else _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning was too generic; replaced with coverage-based explanation.")

    if _looks_like_echo(final_reasoning, answer):
        final_reasoning = _coverage_reasoning(coverage) if coverage else _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning echoed the answer; replaced with fallback explanation.")

    if _looks_like_invalid_reasoning(final_reasoning):
        final_reasoning = _coverage_reasoning(coverage) if coverage else _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning was invalid or placeholder-like; replaced with fallback explanation.")

    if _looks_like_speaker_names(final_reasoning, answer_speaker):
        final_reasoning = _coverage_reasoning(coverage) if coverage else _fallback_reasoning(question, answer, adjusted)
        notes.append("Model reasoning contained only speaker names; replaced with fallback explanation.")

    if adjusted <= 3 and _looks_like_missing_question_claim(final_reasoning) and _content_overlap(question, answer) >= 0.05:
        adjusted, final_reasoning = _heuristic_score_single_pair(question, answer)
        notes.append("Model claimed the question was not asked despite content overlap; replaced with deterministic score.")

    if QUANT_HINT_RE.search(lowered_question) and not NUMBER_RE.search(answer):
        if adjusted > 7:
            notes.append("Capped score because a quantitative question was answered without concrete figures.")
        adjusted = min(adjusted, 7)

    if (" and " in lowered_question or "whether" in lowered_question) and "segment" in lowered_question and "segment" not in lowered_answer:
        if adjusted > 7:
            notes.append("Capped score because a multi-part question left the segment-specific part unanswered.")
        adjusted = min(adjusted, 7)

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


def _parse_score_response(response: str) -> tuple[int, str] | None:
    if not response:
        return None
    score_match = SCORE_LINE_RE.search(response)
    if not score_match:
        return None
    score = max(0, min(10, int(score_match.group(1))))
    why_match = WHY_LINE_RE.search(response)
    if why_match:
        why = re.sub(r"\s+", " ", why_match.group(1)).strip(" .;:,")
    else:
        why = ""
    return score, why


def _build_evasion_score(
    question: SpeakerTurn,
    answer_turns: list[SpeakerTurn],
    responsiveness: int,
    reasoning: str,
    coverage: list[CoverageTuple] | None = None,
    pair_confidence: float = 1.0,
    low_confidence: bool = False,
    source: str = "model",
    notes: list[str] | None = None,
) -> EvasionScore:
    answer = " ".join(turn.text for turn in answer_turns)
    scoring_answer = _strip_answer_filler(answer)
    answer_speaker = ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns))
    question_parts = _split_question_parts(question.text)
    coverage_items = coverage if coverage else _heuristic_coverage(question_parts, scoring_answer)
    adjusted_responsiveness, adjusted_reasoning, adjustment_notes = _adjust_score(
        question.text,
        scoring_answer,
        answer_speaker,
        responsiveness,
        reasoning,
        coverage_items,
    )
    return EvasionScore(
        question=question.text,
        question_speaker=question.speaker,
        answer=answer,
        answer_speaker=answer_speaker,
        answer_turns=[EvasionAnswerTurn(speaker=turn.speaker, text=turn.text) for turn in answer_turns],
        question_parts=question_parts,
        coverage=[EvasionCoverageItem(part=part, status=status, reasoning=item_reasoning) for part, status, item_reasoning in coverage_items],
        responsiveness=adjusted_responsiveness,
        reasoning=adjusted_reasoning,
        flagged=(adjusted_responsiveness <= config.EVASION_FLAG_THRESHOLD) and not low_confidence,
        pair_confidence=pair_confidence,
        low_confidence=low_confidence,
        source=source,
        notes=[*(notes or []), *adjustment_notes],
    )


RETRY_SUFFIX = (
    "\n\nRespond with ONLY two lines: a SCORE line with an integer 0-10 and a WHY line with one sentence of evidence. "
    "Do not copy the example or any placeholder text."
)


def _build_evasion_prompt(question: SpeakerTurn, answer_turns: list[SpeakerTurn]) -> tuple[str, str, list[str]]:
    answer = _strip_answer_filler(" ".join(turn.text for turn in answer_turns))
    answer_speaker = ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns))
    prompt = prompts.EVASION_SCORING_PROMPT.format(
        q_speaker=question.speaker,
        question=_truncate_for_prompt(question.text),
        a_speaker=answer_speaker,
        answer=_truncate_for_prompt(answer, 180),
    )
    return prompt, answer, _split_question_parts(question.text)


def _resolve_llm_response(response: str, answer: str, question_parts: list[str]) -> ScoreTuple | None:
    parsed = _parse_score_response(response)
    if parsed is None:
        return None
    score, why = parsed
    coverage = _heuristic_coverage(question_parts, answer)
    if not why:
        why = _coverage_reasoning(coverage)
    return score, why, coverage


def _heuristic_score_tuple(question_text: str, answer: str, question_parts: list[str]) -> ScoreTuple:
    coverage = _heuristic_coverage(question_parts, answer)
    score, reasoning = _heuristic_score_single_pair(question_text, answer)
    return score, reasoning, coverage


def _run_batch(prompts_list: list[str]) -> list[str]:
    try:
        return generate_text_batch(prompts_list, max_new_tokens=64)
    except Exception as exc:
        LOGGER.debug("Evasion batch generation failed: %s", exc)
        return [""] * len(prompts_list)


def _score_pairs(
    pairs: list[tuple[SpeakerTurn, list[SpeakerTurn], float, list[str]]],
    progress_callback=None,
) -> list[EvasionScore]:
    results: list[EvasionScore | None] = [None] * len(pairs)
    total = len(pairs)

    llm_indices: list[int] = []
    llm_prompts: list[str] = []
    llm_answers: list[str] = []
    llm_question_parts: list[list[str]] = []
    skipped = 0

    for idx, (question, answer_turns, pair_confidence, notes) in enumerate(pairs):
        if pair_confidence < config.MIN_QA_PAIR_CONFIDENCE:
            results[idx] = _build_evasion_score(
                question,
                answer_turns,
                responsiveness=10,
                reasoning="Skipped scoring because speaker parsing confidence was too low.",
                pair_confidence=pair_confidence,
                low_confidence=True,
                source="skipped_low_confidence",
                notes=notes,
            )
            skipped += 1
        else:
            prompt, answer, question_parts = _build_evasion_prompt(question, answer_turns)
            llm_indices.append(idx)
            llm_prompts.append(prompt)
            llm_answers.append(answer)
            llm_question_parts.append(question_parts)

    batch_size = max(1, config.EVASION_BATCH_SIZE)
    completed = skipped
    llm_outcomes: dict[int, tuple[ScoreTuple, str]] = {}

    for batch_start in range(0, len(llm_indices), batch_size):
        batch_local = list(range(batch_start, min(batch_start + batch_size, len(llm_indices))))
        responses = _run_batch([llm_prompts[i] for i in batch_local])

        retry_locals: list[int] = []
        for offset, local_idx in enumerate(batch_local):
            tup = _resolve_llm_response(responses[offset], llm_answers[local_idx], llm_question_parts[local_idx])
            if tup is not None:
                llm_outcomes[llm_indices[local_idx]] = (tup, "model")
            else:
                LOGGER.debug("Evasion response missing SCORE line: %r", responses[offset][:200])
                retry_locals.append(local_idx)

        if retry_locals:
            retry_responses = _run_batch([llm_prompts[i] + RETRY_SUFFIX for i in retry_locals])
            for retry_pos, local_idx in enumerate(retry_locals):
                pair_idx = llm_indices[local_idx]
                tup = _resolve_llm_response(retry_responses[retry_pos], llm_answers[local_idx], llm_question_parts[local_idx])
                if tup is not None:
                    llm_outcomes[pair_idx] = (tup, "model")
                else:
                    LOGGER.info("Using heuristic fallback for evasion scoring after batch retry failure: %r", retry_responses[retry_pos][:200])
                    question_text = pairs[pair_idx][0].text
                    fallback = _heuristic_score_tuple(question_text, llm_answers[local_idx], llm_question_parts[local_idx])
                    llm_outcomes[pair_idx] = (fallback, "heuristic_fallback")

        completed += len(batch_local)
        if progress_callback:
            fraction = 0.2 + 0.75 * (completed / max(1, total))
            progress_callback(min(0.95, fraction), f"Scored {completed} of {total} Q&A pairs")

    for local_idx, pair_idx in enumerate(llm_indices):
        question, answer_turns, pair_confidence, notes = pairs[pair_idx]
        (responsiveness, reasoning, coverage), source = llm_outcomes[pair_idx]
        results[pair_idx] = _build_evasion_score(
            question,
            answer_turns,
            responsiveness,
            reasoning,
            coverage=coverage,
            pair_confidence=pair_confidence,
            source=source,
            notes=notes,
        )

    return [r for r in results if r is not None]


def analyze_evasion(transcript: Transcript, progress_callback=None) -> list[EvasionScore]:
    pairs = _extract_pairs(transcript)

    if not pairs:
        return []
    if progress_callback:
        progress_callback(0.2, f"Scoring {len(pairs)} Q&A pairs")
    try:
        results = _score_pairs(pairs, progress_callback=progress_callback)
    finally:
        clear_model_memory()
    if progress_callback:
        progress_callback(1.0, f"Scored {len(pairs)} Q&A pairs")
    return results
