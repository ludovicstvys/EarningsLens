from pathlib import Path

from earningslens.parser import parse_transcript


def test_parser_marks_roles_and_confidence_for_typical_earnings_call_format():
    transcript_text = """
Jane Doe - Chief Executive Officer:
Welcome everyone. We delivered strong growth this quarter. Margin expansion also improved.

John Smith - Chief Financial Officer:
Revenue and free cash flow both improved meaningfully. Guidance remains unchanged.

Questions and Answers

Operator:
Our first question comes from Alex Brown with Morgan Stanley.

Alex Brown - Morgan Stanley:
Can you discuss enterprise demand and pricing?

Jane Doe - Chief Executive Officer:
Enterprise demand remained solid and pricing was stable across geographies.
""".strip()

    # Pad the transcript so it passes the minimum-word threshold.
    transcript_text = transcript_text + "\n" + ("Growth remained stable. " * 120)

    parsed = parse_transcript(transcript_text, company="ACME", quarter="Q2 2025")

    assert parsed.qa_boundary_detected is True
    assert parsed.parse_confidence > 0.6
    assert any(turn.role == "analyst" for turn in parsed.turns if turn.section == "qa")
    assert any(turn.role == "executive" for turn in parsed.turns if turn.section == "qa")


def test_parser_detects_inline_ir_qa_transition_and_dash_colon_turns():
    transcript_text = """
Operator - Operator: Greetings, and welcome to the call.

Jamie Lee - Chief Executive Officer: We delivered growth across products. Customers renewed at healthy rates.

Pat Rivera - Chief Financial Officer: Revenue increased and margins expanded. With that, I will turn it back to investor relations.

Morgan Chen - Vice President of Investor Relations: Thanks, Pat. We'll now move over to Q&A. Operator, can you please repeat your instructions?

Operator - Operator: Our first question comes from Alex Brown with Morgan Stanley.

Alex Brown - Morgan Stanley: Can you discuss enterprise demand and pricing?

Jamie Lee - Chief Executive Officer: Enterprise demand remained solid and pricing was stable across geographies.

Morgan Chen - Vice President of Investor Relations: That wraps up the Q&A portion of today's call.
""".strip()

    transcript_text = transcript_text + "\n" + ("Growth remained stable. " * 120)

    parsed = parse_transcript(transcript_text, company="ACME", quarter="Q2 2025")

    assert parsed.qa_boundary_detected is True
    assert parsed.qa_text
    assert any(turn.speaker == "Alex Brown" and turn.role == "analyst" for turn in parsed.turns)
    assert any(turn.speaker == "Jamie Lee" and turn.role == "executive" and turn.section == "qa" for turn in parsed.turns)
    assert any(turn.speaker == "Morgan Chen" and turn.role == "operator" for turn in parsed.turns)


def test_parser_detects_goldman_sample_qa_roster_transition():
    parsed = parse_transcript(
        Path("data/samples/GS_Q1_2026.txt").read_text(),
        company="Goldman Sachs",
        quarter="Q1 2026",
        ticker="GS",
    )

    assert parsed.qa_boundary_detected is True
    assert parsed.parse_confidence > 0.9
    assert any(turn.speaker == "Glenn Schorr" and turn.role == "analyst" for turn in parsed.turns if turn.section == "qa")
    assert any(turn.speaker == "Denis Coleman" and turn.role == "executive" for turn in parsed.turns if turn.section == "qa")
