from earningslens.hedging import analyze_hedging
from earningslens.models import SpeakerTurn, Transcript


def _transcript(text: str, quarter: str) -> Transcript:
    return Transcript(
        company="ACME",
        quarter=quarter,
        turns=[
            SpeakerTurn(idx=0, speaker="CEO", role="executive", role_confidence=0.95, section="prepared", text=text),
        ],
        raw_text=text,
    )


def test_hedging_flags_new_density_when_prior_is_zero():
    current = _transcript("We may face uncertainty and could see pressure.", "Q2 2025")
    prior = _transcript("Revenue was strong and margins expanded.", "Q1 2025")

    result = analyze_hedging(current, prior)

    assert result.prior_density == 0.0
    assert result.current_density > 0.0
    assert result.delta_pct == 999.0
    assert result.flagged is True
