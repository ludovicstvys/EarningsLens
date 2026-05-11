from earningslens.evasion import analyze_evasion
from earningslens.models import SpeakerTurn, Transcript


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
