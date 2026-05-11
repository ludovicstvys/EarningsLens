# EarningsLens — Build Spec for Claude Code

> Hand this entire document to Claude Code. Decisions are pre-made; implement as written. Where you must deviate, leave a `# DEVIATION:` comment explaining why.

---

## 1. Project Overview

**Name:** EarningsLens
**One-liner:** AI that reads earnings calls like a senior equity analyst — in 90 seconds, not 6 hours.
**Audience:** Junior buy-side / sell-side analysts, corporate strategy teams, finance students, retail investors.

**Core value loop (the demo):**
1. User picks (or uploads) **two consecutive earnings call transcripts** from the same company — the latest quarter and the prior quarter.
2. App parses both, runs five analyses in parallel, generates a 1-page analyst brief.
3. User sees a dashboard: overall signal (🟢🟡🔴), executive summary, four diagnostic tabs (Sentiment / Hedging / Topics / Risk Vocab), and a Q&A "evasion" inspector.

**The five analyses (each must be visibly demonstrated in the UI):**

| # | Analysis | Technique |
|---|---|---|
| 1 | **Sentiment gap** between prepared remarks and Q&A | FinBERT (transformer classifier) |
| 2 | **Hedging creep** quarter-over-quarter | Lexicon match + normalized frequency |
| 3 | **Topic drift** — themes added/dropped vs prior quarter | Sentence embeddings + LLM theme extraction |
| 4 | **Risk vocabulary tracking** — finance-specific keyword deltas | Curated lexicon + count delta |
| 5 | **Evasion detection** in Q&A — does the answer address the question? | LLM-as-judge with rubric |

Final overall signal aggregates these five.

**Non-goals (do NOT build):**
- Live ticker integration / market data
- User accounts / auth
- Database persistence (session state only)
- Stock-price prediction or trading recommendation (and the UI must say so — see §6 disclaimer)
- Multi-company portfolio view
- Audio transcription (text in, text out — v2 problem)

---

## 2. Tech Stack (locked)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Stable, broad library support |
| UI | Streamlit ≥ 1.32 | Single file, fastest demo path |
| Transcript parsing | Plain regex + state machine | No model needed |
| Sentiment | HuggingFace `ProsusAI/finbert` via `transformers` pipeline | Finance-domain tuned, runs on CPU |
| Embeddings (topic drift) | `sentence-transformers` with `all-MiniLM-L6-v2` | Lightweight, CPU-friendly |
| Theme extraction & evasion scoring & final synthesis | Anthropic Claude API (`anthropic` SDK) | Best structured output |
| Env management | `python-dotenv` | Standard |
| Testing | `pytest` | Standard |
| PDF brief export | `reportlab` | Standard, no system deps |

For the latest Claude API model identifiers refer to https://docs.claude.com/en/docs/about-claude/models — do NOT hardcode a model name from memory; read it from `MODEL` in `.env`.

---

## 3. Architecture & Data Flow

```
two transcripts (.txt) — current quarter, prior quarter
        │
        ▼
   parser.py  ──►  Transcript × 2   (with structured SpeakerTurn list)
        │
        ├──► sentiment.py  ──►  SentimentAnalysis    (current only)
        │
        ├──► hedging.py    ──►  HedgingAnalysis      (current vs prior)
        │
        ├──► topics.py     ──►  TopicDrift           (current vs prior)
        │
        ├──► risk_vocab.py ──►  list[RiskVocab]      (current vs prior)
        │
        └──► evasion.py    ──►  list[EvasionScore]   (current Q&A only)
                                      │
                                      ▼
                            synthesizer.py
                                      │
                                      ▼
                              EarningsBrief (signal + executive summary)
                                      │
                                      ▼
                                  app.py (Streamlit dashboard)
```

All inter-module data is passed as Pydantic models (`earningslens/models.py`). No globals. The five analyzers can run in parallel — see §5.8.

---

## 4. File Structure

```
earningslens/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                       # Streamlit entry point
├── earningslens/
│   ├── __init__.py
│   ├── models.py                # Pydantic data models
│   ├── parser.py                # Transcript → SpeakerTurns
│   ├── sentiment.py             # FinBERT on prepared vs Q&A
│   ├── hedging.py               # Hedge lexicon match + density
│   ├── topics.py                # Embedding similarity + LLM themes
│   ├── risk_vocab.py            # Risk-keyword delta counting
│   ├── evasion.py               # LLM-as-judge on Q&A pairs
│   ├── synthesizer.py           # Final brief assembly + LLM summary
│   ├── vocab.py                 # Hedge words + risk keywords (constants)
│   ├── prompts.py               # All LLM prompt templates
│   └── config.py                # env loading, constants
├── data/
│   └── samples/
│       ├── ACME_Q2_2025.txt     # Clean baseline
│       ├── ACME_Q3_2025.txt     # Stressed (will trigger 🔴)
│       ├── BRAVO_Q1_2025.txt    # Healthy baseline
│       └── BRAVO_Q2_2025.txt    # Continued healthy (will be 🟢)
├── scripts/
│   └── make_samples.py          # Generates the four synthetic transcripts
├── tests/
│   ├── fixtures/
│   │   └── mini_transcript.txt
│   ├── test_parser.py
│   ├── test_hedging.py
│   ├── test_risk_vocab.py
│   └── test_topics.py
└── docs/
    ├── user_guide.md
    └── architecture.md
```

---

## 5. Module Specifications

### 5.1 `earningslens/models.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

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
    quarter: str                       # e.g. "Q3 2025"
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
    prepared_score: float              # -1.0 (neg) to +1.0 (pos)
    qa_score: float
    gap: float                         # qa_score - prepared_score
    flagged: bool                      # True if gap <= -0.15

class HedgingAnalysis(BaseModel):
    current_density: float             # hedges per 1000 words
    prior_density: float
    delta_pct: float                   # (current - prior) / prior * 100
    flagged: bool                      # True if delta_pct >= 25
    top_hedges: list[tuple[str, int]]  # (word, count) top 5 in current

class TopicDrift(BaseModel):
    current_themes: list[str]          # 5–8 short noun phrases
    prior_themes: list[str]
    new_themes: list[str]              # in current, not prior
    dropped_themes: list[str]          # in prior, not current
    semantic_similarity: float         # 0-1, cosine of mean embeddings
    flagged: bool                      # True if similarity < 0.75 OR new_themes contains risk language

class RiskVocabItem(BaseModel):
    keyword: str
    category: str                      # macro / supply / demand / etc
    current_count: int
    prior_count: int
    delta: int
    delta_pct: float                   # 0 if prior_count was 0 and current_count > 0 → 999.0 sentinel

class EvasionScore(BaseModel):
    question: str
    question_speaker: str
    answer: str
    answer_speaker: str
    responsiveness: int                # 0-10
    reasoning: str
    flagged: bool                      # True if responsiveness <= 4

class EarningsBrief(BaseModel):
    company: str
    quarter: str
    prior_quarter: str
    sentiment: SentimentAnalysis
    hedging: HedgingAnalysis
    topics: TopicDrift
    risk_vocab: list[RiskVocabItem]    # full list, UI filters
    evasions: list[EvasionScore]       # full list, UI shows flagged
    overall_signal: Signal
    flag_count: int
    executive_summary: str             # 3-sentence LLM synthesis
    bullet_points: list[str]           # 3–5 bullets for the brief
```

### 5.2 `earningslens/parser.py`

**Public function:** `parse_transcript(text: str, company: str, quarter: str, ticker: str | None = None) -> Transcript`

Algorithm (state machine):
1. Normalize whitespace, strip page headers/footers.
2. Detect the Q&A boundary using regex (case-insensitive, in order; first match wins):
   - `r"^\s*(questions?\s*(and|&)\s*answers?|q\s*&\s*a|q&a)\b.*$"`
   - `r"^\s*operator[:\s].*we will now (begin|open) the question"`
   - If none found → treat entire transcript as `prepared`. (Still useful for sentiment baseline; log warning.)
3. Split into speaker turns. A turn boundary is a line matching `r"^([A-Z][A-Za-z.\-\s']{1,60}):\s*$"` OR `r"^([A-Z][A-Za-z.\-\s']{1,60})\s*[—–-]\s*"`. Text until next boundary belongs to that speaker.
4. Classify speaker role using a heuristic:
   - "Operator" → `operator`
   - Names appearing in prepared section with titles like "CEO", "CFO", "Chief", "President" → `executive`
   - Names appearing only after Q&A boundary, never in prepared section → `analyst`
   - Speaker recognized as executive in prepared and also speaks in Q&A → `executive` in both
   - Else → `unknown`
5. Tag each turn's `section` based on whether it falls before/after the Q&A boundary.
6. Drop operator turns from downstream sentiment/hedging analysis (but keep them in the `turns` list).

Edge cases:
- Single-speaker monologue (no detectable turns) → entire text becomes one turn, role `unknown`.
- Empty input → raise `ValueError("Transcript is empty.")`.
- Less than 200 words → raise `ValueError("Transcript too short for meaningful analysis.")`.

### 5.3 `earningslens/sentiment.py`

**Public function:** `analyze_sentiment(transcript: Transcript) -> SentimentAnalysis`

Implementation:
- Load FinBERT pipeline once at module level (cache).
- `_score(text)` chunks text into ≤ 400-token slices (FinBERT has 512 max), runs pipeline on each chunk, averages the per-chunk scores.
- Per chunk: `score = P(positive) - P(negative)`, range [-1, +1].
- Compute on `transcript.prepared_text` and `transcript.qa_text` separately (excluding operator turns).
- `gap = qa_score - prepared_score`.
- `flagged = gap <= -0.15` (Q&A meaningfully more negative than prepared).

```python
from transformers import pipeline
_finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
```

If `qa_text` is empty → set `qa_score = prepared_score`, `flagged = False`, log warning.

### 5.4 `earningslens/hedging.py`

**Public function:** `analyze_hedging(current: Transcript, prior: Transcript) -> HedgingAnalysis`

Implementation:
- Use `vocab.HEDGE_WORDS` (see §7).
- For each transcript: count case-insensitive word-boundary matches in `prepared_text + qa_text` (exclude operator turns). Normalize: `density = total_hedge_count / total_word_count * 1000`.
- `delta_pct = (current_density - prior_density) / prior_density * 100` (if `prior_density > 0`, else 0).
- `flagged = delta_pct >= 25`.
- `top_hedges` = top 5 hedge words in current by count.

### 5.5 `earningslens/risk_vocab.py`

**Public function:** `analyze_risk_vocab(current: Transcript, prior: Transcript) -> list[RiskVocabItem]`

Implementation:
- Use `vocab.RISK_KEYWORDS` (see §7) — dict of `category → [keywords]`.
- For each keyword, count case-insensitive word-boundary matches across full text of each transcript.
- Build `RiskVocabItem` per keyword.
- Sort returned list by `abs(delta)` descending so UI surfaces the biggest movers first.
- Sentinel: if `prior_count == 0` and `current_count > 0`, set `delta_pct = 999.0` (UI renders this as "NEW").

### 5.6 `earningslens/topics.py`

**Public function:** `analyze_topics(current: Transcript, prior: Transcript) -> TopicDrift`

Two-part algorithm:

**Part A — Semantic similarity (no LLM):**
- Embed `prepared_text` of each transcript using `all-MiniLM-L6-v2`. (Chunk at 400 tokens, average embeddings.)
- `semantic_similarity = cosine_similarity(current_emb, prior_emb)`.

**Part B — Theme extraction (LLM):**
- For each transcript, call Claude with `prompts.THEME_EXTRACTION_PROMPT` on the **prepared remarks only** (Q&A is too noisy for theme extraction).
- Expect JSON: `{"themes": ["theme 1", "theme 2", ...]}` — 5 to 8 themes, each a short noun phrase ≤ 6 words.
- `new_themes = current_themes \ prior_themes` using fuzzy match (RapidFuzz ratio ≥ 80 = "same theme").
- `dropped_themes = prior_themes \ current_themes`, same fuzzy match.
- `flagged = semantic_similarity < 0.75 OR len(new_themes) >= 3 OR any new theme matches a RISK_KEYWORD`.

### 5.7 `earningslens/evasion.py`

**Public function:** `analyze_evasion(transcript: Transcript) -> list[EvasionScore]`

Implementation:
1. Extract Q&A pairs from `transcript.turns`:
   - Walk turns in Q&A section in order.
   - A "pair" = consecutive `analyst` turn → `executive` turn (the next executive turn after an analyst speaks). Multiple consecutive executive turns answering one question are concatenated.
   - Skip operator turns.
2. Cap at 12 pairs (longest-question priority if more) to bound LLM cost.
3. For each pair, call Claude with `prompts.EVASION_SCORING_PROMPT`. Expect JSON `{"responsiveness": 0-10, "reasoning": "..."}`.
4. `flagged = responsiveness <= 4`.
5. Return full list. UI filters to `flagged` for the main view.

Concurrency: use `concurrent.futures.ThreadPoolExecutor(max_workers=5)`. Show a Streamlit progress bar.

### 5.8 `earningslens/synthesizer.py`

**Public function:** `synthesize(company, quarter, prior_quarter, sentiment, hedging, topics, risk_vocab, evasions) -> EarningsBrief`

Steps:
1. Compute `flag_count = sum([sentiment.flagged, hedging.flagged, topics.flagged, any(e.flagged for e in evasions)])`. (Risk vocab is informational only — does not contribute to flag count, but the synthesizer prompt sees the top movers.)
2. Map: 0 flags → `green`, 1 flag → `amber`, 2+ flags → `red`.
3. Build a structured "analyst signal package" dict and pass it to Claude with `prompts.SYNTHESIS_PROMPT`. Expect JSON: `{"executive_summary": "...", "bullet_points": ["...", "..."]}`.
4. Assemble and return `EarningsBrief`.

Run order in `app.py` — use `ThreadPoolExecutor` to run sentiment / hedging / topics / risk_vocab / evasion **in parallel**, then synthesize once all complete.

### 5.9 `earningslens/config.py`

```python
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-sonnet-4-5-20250929")  # set in .env
HEDGING_FLAG_THRESHOLD = 25.0      # %
SENTIMENT_FLAG_THRESHOLD = -0.15
TOPIC_SIMILARITY_THRESHOLD = 0.75
EVASION_FLAG_THRESHOLD = 4
MAX_QA_PAIRS = 12
```

---

## 6. Streamlit UI Spec (`app.py`)

**Layout (top to bottom):**

1. **Header:** "🔍 EarningsLens" + tagline "Read earnings calls like a senior analyst."
2. **Disclaimer banner (always visible at top, dismissible):** "EarningsLens is an analytical tool, not investment advice. Outputs are AI-generated and may contain errors. Do your own diligence."
3. **Sidebar:**
   - Mode toggle: "Use sample" / "Upload my own".
   - If sample: dropdown of paired samples (ACME Q2→Q3 2025, BRAVO Q1→Q2 2025).
   - If upload: two file uploaders (current quarter, prior quarter) + text inputs for company name and quarter labels.
   - "Analyze" primary button.
4. **Main pane (after analysis):**
   - **Top hero row:**
     - Big circle badge with overall signal (🟢/🟡/🔴) and label "GREEN — Quarter looks clean" / "AMBER — Watch list" / "RED — Multiple stress signals".
     - To its right: 3-sentence executive summary in a quoted block.
     - Below: 3–5 bullet points.
   - **Four-tab dashboard** (`st.tabs`):
     - **Tab 1 — Sentiment:** Two horizontal bars (prepared score, Q&A score) on the -1 to +1 axis. Gap value below with flag icon. Caption explains what a negative gap means.
     - **Tab 2 — Hedging:** Side-by-side bar chart of densities (current vs prior). Big delta percentage. Word cloud or table of top 5 hedges.
     - **Tab 3 — Topics:** Semantic similarity gauge (0–1). Two columns: "New themes" (red chips if flagged risk), "Dropped themes" (grey chips). Full theme list expandable.
     - **Tab 4 — Risk vocab:** Sortable table: keyword | category | prior | current | Δ | Δ%. Highlight rows where `delta_pct >= 50` or `delta_pct == 999`.
   - **"Q&A Evasion Inspector" section** (collapsible, expanded by default if any flagged):
     - List of flagged Q&A pairs, each as a card: analyst question (bold) → executive answer (italic blockquote) → "Responsiveness: X/10" badge → reasoning.
     - Toggle to "show all Q&A pairs, not just flagged."
   - **Bottom action row:**
     - "Download analyst brief (PDF)" button — generates 1-page PDF via `reportlab` (executive summary, bullet points, all four diagnostic numbers, top 3 evasions).
     - "Re-run" button.

**State management:** Use `st.session_state` for the parsed `EarningsBrief` so re-renders don't re-run analyses.

**Loading UX:** Progress bar with stages: "Parsing transcripts" → "Running sentiment / hedging / topics / risk vocab / evasion in parallel" → "Synthesizing brief".

---

## 7. Vocabularies (`earningslens/vocab.py`)

Implement exactly:

```python
HEDGE_WORDS = [
    # Modal verbs of uncertainty
    "may", "might", "could", "would", "should",
    # Epistemic stance
    "believe", "think", "expect", "anticipate", "hope", "hopefully",
    "assume", "suppose", "estimate", "project",
    # Vague quantifiers
    "some", "somewhat", "several", "various", "a few", "a number of",
    # Temporal hedges
    "eventually", "in due course", "going forward", "over time",
    "in the coming quarters", "in the near term", "in the medium term",
    # Adverbial hedges
    "potentially", "possibly", "perhaps", "maybe", "arguably",
    "generally", "broadly", "roughly", "approximately", "around",
    # Risk-adjacent nominal hedges
    "uncertainty", "uncertainties", "challenges", "headwinds",
    "softness", "pressure", "choppy", "volatility",
]

RISK_KEYWORDS = {
    "macro":        ["inflation", "recession", "rate hike", "interest rates", "FX",
                     "currency", "geopolitical", "macro"],
    "supply":       ["supply chain", "shortage", "constraint", "constraints",
                     "disruption", "logistics", "backlog"],
    "demand":       ["churn", "retention", "attrition", "slowdown", "soft demand",
                     "weakness", "weakening", "deferral"],
    "competitive":  ["competitive pressure", "pricing pressure", "share loss",
                     "competition", "discounting"],
    "operational":  ["restructure", "restructuring", "impairment", "write-down",
                     "writedown", "charge", "one-time", "non-recurring", "layoffs"],
    "regulatory":   ["regulatory", "regulation", "compliance", "investigation",
                     "lawsuit", "litigation", "subpoena"],
    "liquidity":    ["covenant", "refinance", "default", "liquidity", "leverage",
                     "debt", "amortization", "going concern"],
}
```

Matching is case-insensitive, word-boundary aware. Multi-word phrases match as exact substrings (case-insensitive).

---

## 8. LLM Prompts (`earningslens/prompts.py`)

```python
THEME_EXTRACTION_PROMPT = """You are an equity analyst. Extract the 5 to 8 main themes management discussed in this earnings call's prepared remarks.

A "theme" is a short noun phrase, max 6 words. Examples: "AI product roadmap", "China revenue softness", "margin expansion via pricing", "cloud migration progress".

Do NOT include filler topics like "introductions" or "thanks to the team". Focus on substantive business topics.

PREPARED REMARKS:
\"\"\"
{prepared_text}
\"\"\"

Respond with ONLY this JSON — no markdown, no preamble:
{{"themes": ["theme one", "theme two", "..."]}}
"""

EVASION_SCORING_PROMPT = """You are scoring whether an executive's answer actually addressed an analyst's question on an earnings call.

ANALYST QUESTION (asked by {q_speaker}):
\"\"\"
{question}
\"\"\"

EXECUTIVE ANSWER (given by {a_speaker}):
\"\"\"
{answer}
\"\"\"

Score the answer's responsiveness on a 0–10 scale:
- 10 = Direct, specific, fully answers what was asked.
- 7–9 = Answers the question but with some hedging or partial information.
- 4–6 = Pivots to adjacent topics or gives generalities instead of specifics.
- 0–3 = Clearly evades — refuses to answer, changes subject, or gives non-answer corporate boilerplate.

Penalize: declining to quantify when a quantitative answer was asked; pivoting to unrelated good news; "we don't comment on..." when context suggests they could.
Do not penalize: legitimate "we'll address in next quarter's guidance" if the question genuinely asked about future guidance.

Respond with ONLY this JSON — no markdown, no preamble:
{{"responsiveness": <int 0-10>, "reasoning": "<one sentence, ≤ 25 words>"}}
"""

SYNTHESIS_PROMPT = """You are a senior equity analyst writing the top of a 1-page brief on {company}'s {quarter} earnings call vs {prior_quarter}.

Here is the data your team's analysts produced:

SENTIMENT GAP (Q&A vs prepared remarks):
  Prepared score: {sentiment_prepared:+.2f}
  Q&A score:      {sentiment_qa:+.2f}
  Gap:            {sentiment_gap:+.2f}    Flagged: {sentiment_flagged}

HEDGING DENSITY (hedges per 1000 words):
  Prior quarter:  {hedging_prior:.1f}
  This quarter:   {hedging_current:.1f}
  Change:         {hedging_delta:+.1f}%   Flagged: {hedging_flagged}
  Top hedges this quarter: {top_hedges}

TOPIC DRIFT:
  Semantic similarity to prior quarter: {topic_similarity:.2f}
  New themes this quarter: {new_themes}
  Dropped themes: {dropped_themes}
  Flagged: {topics_flagged}

TOP RISK-VOCAB MOVERS (current vs prior count):
{risk_movers}

FLAGGED Q&A EVASIONS ({n_evasions}):
{evasion_summary}

OVERALL SIGNAL: {signal}

Write a brief that a portfolio manager could read in 30 seconds:

Respond with ONLY this JSON — no markdown, no preamble:
{{
  "executive_summary": "<3 sentences, ≤ 70 words total. Lead with the signal. Be concrete, name specifics from the data above.>",
  "bullet_points": ["<bullet 1, ≤ 20 words>", "<bullet 2>", "<bullet 3>", "<optional bullet 4>", "<optional bullet 5>"]
}}

Tone: dry, factual, no hype. If signal is green, say so plainly. Never give investment advice or price targets.
"""
```

---

## 9. `requirements.txt`

```
streamlit>=1.32
transformers>=4.40
torch>=2.2
sentence-transformers>=2.7
anthropic>=0.40
pydantic>=2.6
python-dotenv>=1.0
rapidfuzz>=3.6
reportlab>=4.1
numpy>=1.26
pytest>=8.0
```

---

## 10. `.env.example`

```
ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-sonnet-4-5-20250929
```

---

## 11. Setup & Run Commands

In `README.md`, document exactly these steps:

```bash
# 1. Clone & enter
git clone <repo> && cd earningslens

# 2. Virtual env
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY

# 5. Generate sample transcripts (one-time)
python scripts/make_samples.py

# 6. Run
streamlit run app.py
```

First run downloads FinBERT (~440 MB) and MiniLM (~80 MB). Note this in README.

---

## 12. Sample Transcripts (`scripts/make_samples.py`)

Critical for a reproducible demo. The script generates four synthetic earnings call transcripts using Claude API. Real transcripts have copyright + version-control issues; synthetic ones let you guarantee the demo's signal direction.

**Specs for the script:**

- Generate 4 transcripts as `.txt` files in `data/samples/`.
- Each transcript ≈ 2500–3500 words, with this exact structure:

```
[Company] [Quarter] Earnings Conference Call

Operator: Good morning, and welcome...

Jane Smith - CEO: Thank you, operator. [Prepared remarks ~800 words]

John Doe - CFO: Thank you, Jane. [Prepared remarks ~600 words]

Operator: We will now begin the question and answer session.

Mark Chen - Goldman Sachs: Hi, thanks for taking my question. [Question]
Jane Smith - CEO: [Answer]

[6–10 Q&A pairs total]

Operator: This concludes today's call.
```

- **ACME pair (must produce 🔴 RED signal in current quarter):**
  - `ACME_Q2_2025.txt`: SaaS company, healthy quarter. CEO confident, prepared remarks upbeat. Q&A direct. Themes: product launches, enterprise expansion, healthy net retention.
  - `ACME_Q3_2025.txt`: same company, deteriorating. Prepared remarks still cautiously upbeat but Q&A reveals stress. Should trigger: sentiment gap (Q&A more negative), hedging creep (+40%+), new themes around "macro headwinds" and "elongated sales cycles", at least 2 evasive Q&A answers.

- **BRAVO pair (must produce 🟢 GREEN signal):**
  - `BRAVO_Q1_2025.txt` and `BRAVO_Q2_2025.txt`: industrial company, steady performer. Prepared remarks and Q&A similarly toned in both. Hedging stable. Themes consistent. All Q&A answers direct.

The script should call Claude with a generation prompt that produces these properties, then write each transcript to disk. Generate sequentially (one API call per transcript). Each prompt must include explicit instructions about which flags should fire, so the demo is deterministic.

---

## 13. Tests (minimum viable)

- `test_parser.py`: feeds `tests/fixtures/mini_transcript.txt` (a 20-line fake transcript you create); asserts correct number of turns, correct Q&A boundary detection, correct role classification of operator/executive/analyst.
- `test_hedging.py`: synthetic 1000-word text with 30 known hedge words; asserts density = 30.0; second text with 50 hedges; asserts `delta_pct = 66.7` and `flagged = True`.
- `test_risk_vocab.py`: synthetic texts with known keyword counts; asserts per-keyword deltas and the `999.0` sentinel for new keywords.
- `test_topics.py`: mock the Claude client and the embedder; assert `new_themes` / `dropped_themes` set arithmetic and the fuzzy-match dedup (e.g., "AI roadmap" and "AI product roadmap" should be treated as same theme).

LLM-dependent code (`evasion.py`, `synthesizer.py`, the LLM part of `topics.py`) is NOT unit-tested end-to-end — mock the Anthropic client.

---

## 14. Acceptance Criteria (Claude Code: do not declare done until all pass)

1. `streamlit run app.py` launches without error on a fresh venv.
2. Selecting the ACME Q2→Q3 sample pair produces an `EarningsBrief` with `overall_signal == "red"` and `flag_count >= 2`.
3. Selecting the BRAVO Q1→Q2 sample pair produces an `EarningsBrief` with `overall_signal == "green"` and `flag_count == 0`.
4. Total analysis time for one sample pair is under 60 seconds on a laptop CPU (excluding first-run model downloads). Parallelism of the 5 analyzers must be in place.
5. All test files pass with `pytest`.
6. The disclaimer banner is visible on first load.
7. README contains setup steps and a screenshot of the dashboard showing a 🔴 result.
8. No API keys committed. `.env` is in `.gitignore`.

---

## 15. Documentation Deliverables (`docs/`)

- `user_guide.md`: 1 page. Screenshot of dashboard, explanation of the five signals, how to interpret each tab, big disclaimer.
- `architecture.md`: 1 page. Data-flow diagram from §3 + a paragraph per module explaining what AI technique it uses (this is the artifact that proves to graders you're using "AAA" techniques — emphasize: domain-tuned transformer, sentence embeddings, lexicon + statistics, LLM-as-judge, LLM synthesis).

---

## 16. Citations (include in README under "Credits")

- FinBERT: Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (2019). https://huggingface.co/ProsusAI/finbert
- Sentence-Transformers: Reimers & Gurevych, "Sentence-BERT" (2019). https://www.sbert.net
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- LLM-as-judge methodology: Zheng et al., "Judging LLM-as-a-Judge" (2023). https://arxiv.org/abs/2306.05685
- Hedging lexicon adapted from: Hyland, "Hedging in Scientific Research Articles" (1998), with finance-specific additions.
- Anthropic Claude API: https://docs.claude.com
- Streamlit: https://streamlit.io

---

## 17. Build Order (recommended for Claude Code)

1. Scaffolding: directory tree, requirements.txt, .env.example, .gitignore, models.py, config.py, vocab.py.
2. `parser.py` + a hand-written `tests/fixtures/mini_transcript.txt` + `test_parser.py`. Confirm green before moving on.
3. `hedging.py` + `test_hedging.py`.
4. `risk_vocab.py` + `test_risk_vocab.py`.
5. `sentiment.py` — verify by running on a small string manually.
6. `prompts.py`.
7. `topics.py` + `test_topics.py` (mocked).
8. `evasion.py` — verify with one fake Q&A pair.
9. `synthesizer.py` — verify end-to-end with mocked or real LLM.
10. `scripts/make_samples.py` → generate the four sample transcripts. Inspect them by hand; regenerate if a transcript doesn't have the property it's supposed to have.
11. `app.py` — full Streamlit UI with parallel execution.
12. README, user_guide, architecture docs.
13. Run the full acceptance checklist (§14). Iterate on the synthetic transcript prompts in `make_samples.py` if the demo signals don't fire correctly.

Commit after each step.