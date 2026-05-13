from pathlib import Path

from earningslens import prompts
from earningslens.evasion import _parse_score_response, _split_question_parts, analyze_evasion
from earningslens.models import SpeakerTurn, Transcript
from earningslens.parser import parse_transcript


def _make_transcript(turns: list[SpeakerTurn]) -> Transcript:
    return Transcript(
        company="ACME",
        quarter="Q2 2025",
        turns=turns,
        raw_text="Q&A transcript",
    )


def test_split_question_parts_keeps_single_intent_question_together():
    question = "How are you thinking about capacity planning for AI demand?"

    assert _split_question_parts(question) == ["How are you thinking about capacity planning for AI demand"]


def test_split_question_parts_separates_multi_part_question():
    question = "How confident are you that software can monetize AI investments, and what factors are you monitoring to avoid overbuilding?"

    assert _split_question_parts(question) == [
        "How confident are you that software can monetize AI investments",
        "What factors are you monitoring to avoid overbuilding",
    ]


def test_split_question_parts_preserves_quant_and_segment_requirements():
    question = "Can you quantify how much sales cycles elongated and whether the softness was concentrated in a specific customer segment?"

    assert _split_question_parts(question) == [
        "Can you quantify how much sales cycles elongated",
        "Whether the softness was concentrated in a specific customer segment",
    ]


def test_parse_score_response_extracts_score_and_why():
    response = "SCORE: 8\nWHY: The answer addressed demand with specific operational detail."

    assert _parse_score_response(response) == (
        8,
        "The answer addressed demand with specific operational detail",
    )


def test_parse_score_response_clamps_out_of_range_scores():
    response = "SCORE: 99\nWHY: The answer addressed everything."

    score, _why = _parse_score_response(response)
    assert score == 10


def test_parse_score_response_returns_none_without_score_line():
    assert _parse_score_response("This is some commentary without a score line.") is None


def test_parse_score_response_ignores_format_ranges():
    assert _parse_score_response("SCORE: 0-10 integer\nWHY: one evidence sentence") is None


def test_parse_score_response_tolerates_extra_chatter_before_score():
    response = "Sure, here is my assessment.\nSCORE: 6\nWHY: The answer hedged on the requested specifics."

    assert _parse_score_response(response) == (6, "The answer hedged on the requested specifics")


def test_evasion_prompt_does_not_include_copyable_domain_example():
    assert "EXAMPLE" not in prompts.EVASION_SCORING_PROMPT
    assert "specific number of days or percentage" not in prompts.EVASION_SCORING_PROMPT


def test_evasion_uses_model_score_when_response_is_parseable(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["SCORE: 6\nWHY: The answer touched demand but did not provide exact figures."] * len(prompts),
    )

    transcript = _make_transcript(
        [
            SpeakerTurn(idx=0, speaker="Analyst One", role="analyst", role_confidence=0.9, section="qa", text="Can you quantify demand trends?"),
            SpeakerTurn(idx=1, speaker="CEO", role="executive", role_confidence=0.95, section="qa", text="Demand was stable across the quarter, though we did not give exact figures."),
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].source == "model"
    assert result[0].responsiveness == 6
    assert "exact figures" in result[0].reasoning


def test_evasion_falls_back_to_heuristic_when_response_unparseable(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["I cannot do this task right now."] * len(prompts),
    )

    transcript = _make_transcript(
        [
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
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].source == "heuristic_fallback"
    assert result[0].responsiveness > 0


def test_evasion_caps_quant_question_without_numbers(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["SCORE: 10\nWHY: The answer covered demand thoroughly."] * len(prompts),
    )

    transcript = _make_transcript(
        [
            SpeakerTurn(idx=0, speaker="Analyst One", role="analyst", role_confidence=0.9, section="qa", text="Can you quantify demand trends this quarter?"),
            SpeakerTurn(idx=1, speaker="CEO", role="executive", role_confidence=0.95, section="qa", text="Demand was solid and we feel confident about the overall trajectory."),
        ]
    )

    result = analyze_evasion(transcript)

    assert result[0].responsiveness <= 7
    assert any("quantitative" in note.lower() for note in result[0].notes)


def test_evasion_replaces_speaker_names_used_as_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["SCORE: 8\nWHY: Amy Hood, Satya Nadella"] * len(prompts),
    )

    transcript = _make_transcript(
        [
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
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].reasoning != "Amy Hood, Satya Nadella"
    assert [turn.speaker for turn in result[0].answer_turns] == ["Amy Hood", "Satya Nadella"]
    assert "speaker names" in " ".join(result[0].notes).lower()


def test_evasion_replaces_placeholder_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["SCORE: 8\nWHY: Score: 8/10"] * len(prompts),
    )

    transcript = _make_transcript(
        [
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
        ]
    )

    result = analyze_evasion(transcript)

    assert result[0].reasoning != "Score: 8/10"
    assert "placeholder-like" in " ".join(result[0].notes).lower()


def test_evasion_replaces_echoed_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: [
            "SCORE: 10\n"
            "WHY: We did see some elongation, particularly in larger deals, but the exact pattern varied by account, by geography, and by internal customer process."
        ] * len(prompts),
    )

    transcript = _make_transcript(
        [
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
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness <= 8
    assert any("echoed the answer" in note.lower() for note in result[0].notes)


def test_evasion_replaces_false_question_not_asked_reasoning(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: [
            "SCORE: 0\nWHY: We are not able to provide a specific answer to your question, as it was not asked."
        ] * len(prompts),
    )

    transcript = _make_transcript(
        [
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
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].responsiveness > 3
    assert "not asked" not in result[0].reasoning.lower()
    assert any("question was not asked" in note.lower() for note in result[0].notes)


def test_evasion_scores_each_pair_with_separate_llm_call(monkeypatch):
    responses = iter(
        [
            "SCORE: 8\nWHY: Specific demand actions described.",
            "SCORE: 5\nWHY: General margin direction but not segment specifics.",
            "SCORE: 6\nWHY: Renewal behavior addressed only directionally.",
        ]
    )
    prompts_seen = 0

    def fake_generate_text_batch(prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        nonlocal prompts_seen
        prompts_seen += len(prompts)
        return [next(responses) for _ in prompts]

    monkeypatch.setattr("earningslens.evasion.generate_text_batch", fake_generate_text_batch)

    transcript = _make_transcript(
        [
            SpeakerTurn(idx=0, speaker="Analyst One", role="analyst", role_confidence=0.9, section="qa", text="Can you quantify demand trends?"),
            SpeakerTurn(idx=1, speaker="CEO", role="executive", role_confidence=0.95, section="qa", text="Demand grew 8% and backlog remained strong."),
            SpeakerTurn(idx=2, speaker="Analyst Two", role="analyst", role_confidence=0.9, section="qa", text="How are margins changing by segment?"),
            SpeakerTurn(idx=3, speaker="CFO", role="executive", role_confidence=0.95, section="qa", text="Margins improved overall, but mix varied across the portfolio."),
            SpeakerTurn(idx=4, speaker="Analyst Three", role="analyst", role_confidence=0.9, section="qa", text="What is changing in enterprise renewal behavior?"),
            SpeakerTurn(idx=5, speaker="COO", role="executive", role_confidence=0.95, section="qa", text="Renewals are moving through normal channels, and we are focused on customer success."),
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 3
    assert prompts_seen == 3
    assert [item.source for item in result] == ["model"] * 3


def test_evasion_preserves_msft_multi_speaker_answer_turns(monkeypatch):
    monkeypatch.setattr(
        "earningslens.evasion.generate_text_batch",
        lambda prompts, max_new_tokens=64: ["SCORE: 8\nWHY: Addressed demand and monetization with relevant operational details."] * len(prompts),
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
    assert len(mark_result.reasoning) > 0


def test_evasion_skips_low_confidence_pairs(monkeypatch):
    called = []

    def fake_generate_text_batch(prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        called.extend(prompts)
        return ["SCORE: 8\nWHY: Should not be called."] * len(prompts)

    monkeypatch.setattr("earningslens.evasion.generate_text_batch", fake_generate_text_batch)

    transcript = _make_transcript(
        [
            SpeakerTurn(idx=0, speaker="Unknown", role="analyst", role_confidence=0.2, section="qa", text="Can you quantify demand trends?"),
            SpeakerTurn(idx=1, speaker="Unknown", role="executive", role_confidence=0.2, section="qa", text="Demand grew significantly across the quarter."),
        ]
    )

    result = analyze_evasion(transcript)

    assert len(result) == 1
    assert result[0].low_confidence is True
    assert result[0].source == "skipped_low_confidence"
    assert called == []
