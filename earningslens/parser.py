from __future__ import annotations

import logging
import re

from earningslens.models import SpeakerTurn, Transcript

LOGGER = logging.getLogger(__name__)

QA_BOUNDARY_PATTERNS = [
    re.compile(r"^\s*(questions?\s*(and|&)\s*answers?|q\s*&\s*a|q&a)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*operator[:\s].*we will now (begin|open) the question", re.IGNORECASE),
]
COLON_ONLY_SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z.\-\s']{1,60}):\s*$")
INLINE_COLON_SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z.\-\s']{1,60}):\s*(.+)$")
DASH_SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z.\-\s']{1,60})\s*[—–-]\s*([^:]{1,80})\s*:\s*(.*)$")
TITLE_HINT_RE = re.compile(r"\b(ceo|cfo|chief|president|coo|chairman|chairwoman)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b")


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    filtered: list[str] = []
    for line in lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            filtered.append("")
            continue
        if re.fullmatch(r"page \d+ of \d+", stripped, re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        filtered.append(stripped)
    return "\n".join(filtered).strip()


def _find_qa_boundary(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        for pattern in QA_BOUNDARY_PATTERNS:
            if pattern.match(line):
                return idx
    return None


def _clean_speaker(raw_speaker: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in re.split(r"\s*[—–-]\s*", raw_speaker, maxsplit=1)]
    speaker = parts[0]
    title = parts[1] if len(parts) > 1 else None
    return speaker, title


def _detect_turn_start(line: str) -> tuple[str, str | None, str | None] | None:
    colon_match = COLON_ONLY_SPEAKER_RE.match(line)
    if colon_match:
        speaker, title = _clean_speaker(colon_match.group(1))
        return speaker, title, None

    inline_colon_match = INLINE_COLON_SPEAKER_RE.match(line)
    if inline_colon_match:
        speaker, title = _clean_speaker(inline_colon_match.group(1))
        return speaker, title, inline_colon_match.group(2).strip()

    dash_match = DASH_SPEAKER_RE.match(line)
    if dash_match:
        speaker, title, remainder = dash_match.groups()
        clean_speaker, title = _clean_speaker(f"{speaker} - {title}")
        return clean_speaker, title, remainder.strip()

    return None


def _infer_role(speaker: str, title: str | None, prepared_execs: set[str], qa_only_speakers: set[str]) -> str:
    if speaker.lower() == "operator":
        return "operator"
    if speaker in prepared_execs:
        return "executive"
    if title and TITLE_HINT_RE.search(title):
        return "executive"
    if speaker in qa_only_speakers:
        return "analyst"
    return "unknown"


def parse_transcript(text: str, company: str, quarter: str, ticker: str | None = None) -> Transcript:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("Transcript is empty.")
    if len(WORD_RE.findall(normalized)) < 200:
        raise ValueError("Transcript too short for meaningful analysis.")

    lines = normalized.split("\n")
    qa_boundary_idx = _find_qa_boundary(lines)

    detected_turns: list[dict[str, object]] = []
    current_turn: dict[str, object] | None = None

    for line_idx, line in enumerate(lines):
        if not line:
            continue
        if qa_boundary_idx is not None and line_idx == qa_boundary_idx:
            if current_turn:
                detected_turns.append(current_turn)
                current_turn = None
            continue

        speaker_start = _detect_turn_start(line)
        if speaker_start:
            speaker, title, remainder = speaker_start
            if current_turn:
                detected_turns.append(current_turn)
            current_turn = {
                "speaker": speaker,
                "title": title,
                "line_idx": line_idx,
                "text_parts": [remainder] if remainder else [],
            }
            continue

        if current_turn is None:
            current_turn = {
                "speaker": "Unknown",
                "title": None,
                "line_idx": line_idx,
                "text_parts": [line],
            }
        else:
            current_turn["text_parts"].append(line)

    if current_turn:
        detected_turns.append(current_turn)

    if not detected_turns:
        detected_turns = [{
            "speaker": "Unknown",
            "title": None,
            "line_idx": 0,
            "text_parts": [normalized],
        }]
    elif len(detected_turns) > 1 and detected_turns[0]["speaker"] == "Unknown":
        detected_turns = detected_turns[1:]

    prepared_execs = {
        turn["speaker"]
        for turn in detected_turns
        if (qa_boundary_idx is None or int(turn["line_idx"]) < qa_boundary_idx)
        and turn.get("title")
        and TITLE_HINT_RE.search(str(turn["title"]))
    }
    prepared_speakers = {
        turn["speaker"]
        for turn in detected_turns
        if qa_boundary_idx is None or int(turn["line_idx"]) < qa_boundary_idx
    }
    qa_speakers = {
        turn["speaker"]
        for turn in detected_turns
        if qa_boundary_idx is not None and int(turn["line_idx"]) > qa_boundary_idx
    }
    qa_only_speakers = qa_speakers - prepared_speakers - {"Operator"}

    turns: list[SpeakerTurn] = []
    for idx, turn in enumerate(detected_turns):
        line_idx = int(turn["line_idx"])
        section = "qa" if qa_boundary_idx is not None and line_idx > qa_boundary_idx else "prepared"
        speaker = str(turn["speaker"])
        title = str(turn["title"]) if turn.get("title") else None
        role = _infer_role(speaker, title, prepared_execs, qa_only_speakers)
        text_value = " ".join(part for part in turn["text_parts"] if part).strip()
        turns.append(
            SpeakerTurn(
                idx=idx,
                speaker=speaker,
                role=role,  # type: ignore[arg-type]
                section=section,  # type: ignore[arg-type]
                text=text_value,
            )
        )

    if qa_boundary_idx is None:
        LOGGER.warning("No Q&A boundary detected; treating transcript as prepared remarks only.")

    return Transcript(company=company, ticker=ticker, quarter=quarter, turns=turns, raw_text=normalized)
