from __future__ import annotations

import json

from earningslens import config, prompts
from earningslens.local_llm import generate_text
from earningslens.models import EvasionScore, SpeakerTurn, Transcript


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Local model response did not contain JSON.")
    return json.loads(text[start:end + 1])


def _extract_pairs(transcript: Transcript) -> list[tuple[SpeakerTurn, list[SpeakerTurn]]]:
    qa_turns = [turn for turn in transcript.turns if turn.section == "qa" and turn.role != "operator"]
    pairs: list[tuple[SpeakerTurn, list[SpeakerTurn]]] = []
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
            pairs.append((question, answer_turns))
    return pairs


def _build_evasion_score(question: SpeakerTurn, answer_turns: list[SpeakerTurn], responsiveness: int, reasoning: str) -> EvasionScore:
    answer = " ".join(turn.text for turn in answer_turns)
    answer_speaker = ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns))
    return EvasionScore(
        question=question.text,
        question_speaker=question.speaker,
        answer=answer,
        answer_speaker=answer_speaker,
        responsiveness=responsiveness,
        reasoning=reasoning,
        flagged=responsiveness <= config.EVASION_FLAG_THRESHOLD,
    )


def _score_pairs(pairs: list[tuple[SpeakerTurn, list[SpeakerTurn]]]) -> list[EvasionScore]:
    prompt_items = []
    for pair_id, (question, answer_turns) in enumerate(pairs):
        prompt_items.append(
            {
                "pair_id": pair_id,
                "question_speaker": question.speaker,
                "question": question.text,
                "answer_speaker": ", ".join(dict.fromkeys(turn.speaker for turn in answer_turns)),
                "answer": " ".join(turn.text for turn in answer_turns),
            }
        )

    response = generate_text(
        prompts.EVASION_BATCH_SCORING_PROMPT.format(
            qa_pairs_json=json.dumps(prompt_items, ensure_ascii=True),
        ),
        max_new_tokens=min(700, 120 + len(prompt_items) * 80),
    )
    payload = _extract_json(response)
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

    results: list[EvasionScore] = []
    for pair_id, (question, answer_turns) in enumerate(pairs):
        responsiveness, reasoning = score_by_pair_id.get(pair_id, (0, "Model did not return a score for this pair."))
        results.append(_build_evasion_score(question, answer_turns, responsiveness, reasoning))
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
