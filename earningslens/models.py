from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Section = Literal["prepared", "qa"]
Role = Literal["operator", "executive", "analyst", "unknown"]
Signal = Literal["green", "amber", "red"]


class SpeakerTurn(BaseModel):
    idx: int
    speaker: str
    role: Role
    section: Section
    text: str


class Transcript(BaseModel):
    company: str
    ticker: Optional[str] = None
    quarter: str
    turns: list[SpeakerTurn]
    raw_text: str

    @property
    def prepared_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.section == "prepared")

    @property
    def qa_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.section == "qa")

    @property
    def full_text(self) -> str:
        return " ".join(t.text for t in self.turns)


class SentimentAnalysis(BaseModel):
    prepared_score: float
    qa_score: float
    gap: float
    flagged: bool


class HedgingAnalysis(BaseModel):
    current_density: float
    prior_density: float
    delta_pct: float
    flagged: bool
    top_hedges: list[tuple[str, int]]


class TopicDrift(BaseModel):
    current_themes: list[str]
    prior_themes: list[str]
    new_themes: list[str]
    dropped_themes: list[str]
    semantic_similarity: float
    flagged: bool


class RiskVocabItem(BaseModel):
    keyword: str
    category: str
    current_count: int
    prior_count: int
    delta: int
    delta_pct: float


class EvasionScore(BaseModel):
    question: str
    question_speaker: str
    answer: str
    answer_speaker: str
    responsiveness: int
    reasoning: str
    flagged: bool


class EarningsBrief(BaseModel):
    company: str
    quarter: str
    prior_quarter: str
    sentiment: SentimentAnalysis
    hedging: HedgingAnalysis
    topics: TopicDrift
    risk_vocab: list[RiskVocabItem]
    evasions: list[EvasionScore]
    overall_signal: Signal
    flag_count: int
    executive_summary: str
    bullet_points: list[str]
