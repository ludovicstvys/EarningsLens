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

THEME_COMPARISON_PROMPT = """You are an equity analyst. Extract the 5 to 8 main themes management discussed in each of these two earnings call prepared-remarks sections.

A "theme" is a short noun phrase, max 6 words. Examples: "AI product roadmap", "China revenue softness", "margin expansion via pricing", "cloud migration progress".

Do NOT include filler topics like "introductions" or "thanks to the team". Focus on substantive business topics.

CURRENT QUARTER PREPARED REMARKS:
\"\"\"
{current_prepared_text}
\"\"\"

PRIOR QUARTER PREPARED REMARKS:
\"\"\"
{prior_prepared_text}
\"\"\"

Respond with ONLY this JSON — no markdown, no preamble:
{{
  "current_themes": ["theme one", "theme two", "..."],
  "prior_themes": ["theme one", "theme two", "..."]
}}
"""

EVASION_SCORING_PROMPT = """You score how responsively an executive answered an analyst on an earnings call.
Output exactly two lines and nothing else:
SCORE: <integer 0-10>
WHY: <one sentence naming the specific answered or missing detail>

Scale: 9-10 fully and specifically answers; 7-8 answers with some hedging; 4-6 pivots to adjacent topics or generalities; 0-3 refuses or changes the subject.

Penalize: declining to quantify when the analyst asked for numbers; pivoting to unrelated good news; vague corporate boilerplate.
Do not penalize: legitimately deferring to a future quarter's guidance when the analyst asked about future guidance.

QUESTION ({q_speaker}): {question}
ANSWER ({a_speaker}): {answer}
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
  "executive_summary": "Write 1-3 factual sentences, 70 words maximum. Lead with the signal and cite specific metrics.",
  "bullet_points": ["Write a factual bullet under 20 words.", "Write a second factual bullet.", "Write a third factual bullet."]
}}

Tone: dry, factual, no hype. If signal is green, say so plainly. Never give investment advice or price targets.
Do not copy the JSON example text. Do not use angle brackets or placeholder labels.
"""
