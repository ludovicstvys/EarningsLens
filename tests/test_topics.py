import numpy as np

from earningslens.models import SpeakerTurn, Transcript
from earningslens.topics import analyze_topics


def _transcript(text: str, quarter: str) -> Transcript:
    return Transcript(
        company="ACME",
        quarter=quarter,
        turns=[
            SpeakerTurn(idx=0, speaker="CEO", role="executive", role_confidence=0.95, section="prepared", text=text),
        ],
        raw_text=text,
    )


def test_topics_fallbacks_when_model_returns_non_json(monkeypatch):
    monkeypatch.setattr("earningslens.topics.generate_text", lambda prompt, max_new_tokens=320: "not json")
    monkeypatch.setattr("earningslens.topics._mean_embedding", lambda text: np.array([1.0, 0.0]))

    current = _transcript("Cloud expansion and pricing discipline drove cloud expansion and pricing discipline.", "Q2 2025")
    prior = _transcript("Supply chain recovery and margin improvement supported supply chain recovery.", "Q1 2025")

    result = analyze_topics(current, prior)

    assert result.current_themes
    assert result.prior_themes
    assert isinstance(result.semantic_similarity, float)
