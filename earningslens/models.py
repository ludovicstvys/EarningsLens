from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Section = Literal["prepared", "qa"]
Role = Literal["operator", "executive", "analyst", "unknown"]
Signal = Literal["green", "amber", "red"]


class SpeakerTurn(BaseModel):
    idx: int
    speaker: str
    title: Optional[str] = None
    role: Role
    role_confidence: float = 0.0
    section: Section
    text: str
    parse_notes: list[str] = Field(default_factory=list)


class Transcript(BaseModel):
    company: str
    ticker: Optional[str] = None
    quarter: str
    turns: list[SpeakerTurn]
    raw_text: str
    qa_boundary_detected: bool = False
    parse_confidence: float = 0.0
    parser_notes: list[str] = Field(default_factory=list)

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
    source: str = "model"
    notes: list[str] = Field(default_factory=list)


class HedgingAnalysis(BaseModel):
    current_density: float
    prior_density: float
    delta_pct: float
    flagged: bool
    top_hedges: list[tuple[str, int]]
    source: str = "lexicon"
    notes: list[str] = Field(default_factory=list)


class TopicDrift(BaseModel):
    current_themes: list[str]
    prior_themes: list[str]
    new_themes: list[str]
    dropped_themes: list[str]
    semantic_similarity: float
    flagged: bool
    source: str = "model"
    notes: list[str] = Field(default_factory=list)


class RiskVocabItem(BaseModel):
    keyword: str
    category: str
    current_count: int
    prior_count: int
    delta: int
    delta_pct: float


class EvasionAnswerTurn(BaseModel):
    speaker: str
    text: str


class EvasionCoverageItem(BaseModel):
    part: str
    status: Literal["answered", "partial", "missed"]
    reasoning: str


class EvasionScore(BaseModel):
    question: str
    question_speaker: str
    answer: str
    answer_speaker: str
    answer_turns: list[EvasionAnswerTurn] = Field(default_factory=list)
    question_parts: list[str] = Field(default_factory=list)
    coverage: list[EvasionCoverageItem] = Field(default_factory=list)
    responsiveness: int
    reasoning: str
    flagged: bool
    pair_confidence: float = 1.0
    low_confidence: bool = False
    source: str = "model"
    notes: list[str] = Field(default_factory=list)


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
    analysis_provenance: dict[str, str] = Field(default_factory=dict)
    analysis_notes: dict[str, list[str]] = Field(default_factory=dict)
    parser_overview: dict[str, object] = Field(default_factory=dict)
