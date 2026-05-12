from pathlib import Path

from earningslens.evasion import _extract_json, analyze_evasion
from earningslens.models import SpeakerTurn, Transcript
from earningslens.parser import parse_transcript


def test_evasion_falls_back_to_single_pair_when_batch_parse_fails(monkeypatch):
    responses = iter(
        [
            "not json",
            "still not json",
            '{"responsiveness": 6, "reasoning": "Answered the question with limited specificity."}',
        ]
    )

    def fake_generate_text(prompt: str, max_new_tokens: int = 300) -> str:
        return next(responses)

    monkeypatch.setattr("earningslens.evasion.generate_text", fake_generate_text)

    transcript = Transcript(
        company="ACME",
        quarter="Q2 2025",
        turns=[
            SpeakerTurn(idx=0, speaker="Analyst One", role="analyst", role_confidence=0.9, section="qa", text="Can you quantify demand trends?"),
            SpeakerTurn(idx=1, speaker="CEO", role="executive", role_confidence=0.95, section="qa", text="Demand was stable, though we did not give exact figures."),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness == 6
    assert "limited specificity" in result[0].reasoning


def test_evasion_extracts_real_json_after_echoed_prompt_example():
    response = """
    Respond with ONLY this JSON:
    {"responsiveness": <int 0-10>, "reasoning": "<one sentence>"}

    {"responsiveness": 7, "reasoning": "Addressed the topic but stayed broad."}
    """

    assert _extract_json(response) == {
        "responsiveness": 7,
        "reasoning": "Addressed the topic but stayed broad.",
    }


def test_evasion_uses_heuristic_instead_of_zero_when_single_pair_parse_fails(monkeypatch):
    monkeypatch.setattr("earningslens.evasion.generate_text", lambda prompt, max_new_tokens=300: "not json")

    transcript = Transcript(
        company="ACME",
        quarter="Q2 2025",
        turns=[
            SpeakerTurn(
                idx=0,
                speaker="Analyst One",
                role="analyst",
                role_confidence=0.9,
                section="qa",
                text="How do you evaluate customer concentration risk and the ability to fulfill large AI contracts?",
            ),
            SpeakerTurn(
                idx=1,
                speaker="CEO",
                role="executive",
                role_confidence=0.95,
                section="qa",
                text=(
                    "We evaluate customer concentration risk through a broad portfolio and flexible capacity. "
                    "The AI contracts are delivered over time, which gives us lead time to manage supply."
                ),
            ),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness > 0
    assert "Unable to parse" not in result[0].reasoning


def test_evasion_replaces_echoed_reasoning_and_caps_partial_quant_answers(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text",
        lambda prompt, max_new_tokens=300: (
            '{"scores": [{"pair_id": 0, "responsiveness": 10, '
            '"reasoning": "We did see some elongation, particularly in larger deals, but the exact pattern varied by account, by geography, and by internal customer process."}]}'
        ),
    )

    transcript = Transcript(
        company="ACME",
        quarter="Q2 2025",
        turns=[
            SpeakerTurn(
                idx=0,
                speaker="Mark Chen",
                role="analyst",
                role_confidence=0.95,
                section="qa",
                text="Can you quantify how much sales cycles elongated in the quarter and whether the softness was concentrated in a specific customer segment?",
            ),
            SpeakerTurn(
                idx=1,
                speaker="Jane Smith",
                role="executive",
                role_confidence=0.95,
                section="qa",
                text="We did see some elongation, particularly in larger deals, but the exact pattern varied by account, by geography, and by internal customer process. I would say generally that customers are being more thoughtful.",
            ),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness <= 8
    assert "did not quantify" in result[0].reasoning.lower()
    assert any("echoed the answer" in note.lower() for note in result[0].notes)


def test_evasion_replaces_false_question_not_asked_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text",
        lambda prompt, max_new_tokens=300: (
            '{"scores": [{"pair_id": 0, "responsiveness": 0, '
            '"reasoning": "We are not able to provide a specific answer to your question, as it was not asked."}]}'
        ),
    )

    transcript = Transcript(
        company="MSFT",
        quarter="Q1 2026",
        turns=[
            SpeakerTurn(
                idx=0,
                speaker="Brent Thill",
                role="analyst",
                role_confidence=0.93,
                section="qa",
                text=(
                    "Amy, on the bookings blowout. Can you give us a sense of what you're seeing "
                    "in that 51% RPO and 110-plus percent bookings growth that gives you confidence "
                    "about the breadth and extent of these deals globally?"
                ),
            ),
            SpeakerTurn(
                idx=1,
                speaker="Amy Hood",
                role="executive",
                role_confidence=0.93,
                section="qa",
                text=(
                    "RPO is nearly $400 billion and spans multiple products and customers of all sizes. "
                    "The weighted average duration is around 2 years, and contracts are accumulating "
                    "across many customers who plan to use them fairly soon."
                ),
            ),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness > 3
    assert "not asked" not in result[0].reasoning.lower()
    assert any("question was not asked" in note.lower() for note in result[0].notes)


def test_evasion_replaces_speaker_names_used_as_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text",
        lambda prompt, max_new_tokens=300: (
            '{"scores": [{"pair_id": 0, "responsiveness": 8, "reasoning": "Amy Hood, Satya Nadella"}]}'
        ),
    )

    transcript = Transcript(
        company="MSFT",
        quarter="Q1 2026",
        turns=[
            SpeakerTurn(
                idx=0,
                speaker="Mark Moerdler",
                role="analyst",
                role_confidence=0.93,
                section="qa",
                text=(
                    "How confident are you that software can monetize the global AI investments, "
                    "and what factors are you monitoring to avoid overbuilding?"
                ),
            ),
            SpeakerTurn(
                idx=1,
                speaker="Amy Hood",
                role="executive",
                role_confidence=0.93,
                section="qa",
                text=(
                    "We monitor booked RPO, asset lives, capacity constraints, and product usage. "
                    "Short-lived GPUs and CPUs align with contract durations, while demand remains above supply."
                ),
            ),
            SpeakerTurn(
                idx=2,
                speaker="Satya Nadella",
                role="executive",
                role_confidence=0.93,
                section="qa",
                text=(
                    "We also focus on token factory efficiency, fleet fungibility, and high-value Copilot, "
                    "security, health, and consumer monetization opportunities."
                ),
            ),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].reasoning != "Amy Hood, Satya Nadella"
    assert [turn.speaker for turn in result[0].answer_turns] == ["Amy Hood", "Satya Nadella"]
    assert "speaker names" in " ".join(result[0].notes).lower()


def test_evasion_replaces_placeholder_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text",
        lambda prompt, max_new_tokens=300: '{"scores": [{"pair_id": 0, "responsiveness": 8, "reasoning": "Score: 8/10"}]}',
    )

    transcript = Transcript(
        company="ACME",
        quarter="Q2 2025",
        turns=[
            SpeakerTurn(
                idx=0,
                speaker="Analyst One",
                role="analyst",
                role_confidence=0.9,
                section="qa",
                text="How are you thinking about capacity planning for AI demand?",
            ),
            SpeakerTurn(
                idx=1,
                speaker="CEO",
                role="executive",
                role_confidence=0.95,
                section="qa",
                text=(
                    "We are planning capacity against contracted demand, utilization trends, "
                    "and the pace at which new AI products are consuming infrastructure."
                ),
            ),
        ],
        raw_text="Q&A transcript",
    )

    result = analyze_evasion(transcript)

    assert result[0].reasoning != "Score: 8/10"
    assert "placeholder-like" in " ".join(result[0].notes).lower()


def test_evasion_preserves_msft_multi_speaker_answer_turns(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text",
        lambda prompt, max_new_tokens=300: (
            '{"scores": ['
            '{"pair_id": 0, "responsiveness": 8, "reasoning": "Addressed demand and monetization with relevant operational details."},'
            '{"pair_id": 1, "responsiveness": 8, "reasoning": "Addressed demand and monetization with relevant operational details."},'
            '{"pair_id": 2, "responsiveness": 8, "reasoning": "Addressed demand and monetization with relevant operational details."},'
            '{"pair_id": 3, "responsiveness": 8, "reasoning": "Addressed demand and monetization with relevant operational details."},'
            '{"pair_id": 4, "responsiveness": 8, "reasoning": "Addressed demand and monetization with relevant operational details."},'
            '{"pair_id": 5, "responsiveness": 8, "reasoning": "Amy Hood, Satya Nadella"}'
            "]} "
        ),
    )

    transcript = parse_transcript(
        Path("data/samples/MSFT_Q1_2026.txt").read_text(),
        company="Microsoft",
        quarter="Q1 2026",
    )

    results = analyze_evasion(transcript)
    mark_result = next(item for item in results if item.question_speaker == "Mark Moerdler")

    assert mark_result.answer_speaker == "Amy Hood, Satya Nadella"
    assert [turn.speaker for turn in mark_result.answer_turns] == ["Amy Hood", "Satya Nadella"]
    assert mark_result.reasoning != "Amy Hood, Satya Nadella"
