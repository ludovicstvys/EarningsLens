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
