from __future__ import annotations

from pathlib import Path

from earningslens.local_llm import generate_text

OUTPUT_DIR = Path("data/samples")

TRANSCRIPT_SPECS = [
    {
        "filename": "ACME_Q2_2025.txt",
        "company": "ACME",
        "quarter": "Q2 2025",
        "profile": "SaaS company, healthy quarter. CEO confident, prepared remarks upbeat. Q&A direct. Themes: product launches, enterprise expansion, healthy net retention.",
        "flags": "This is the clean baseline. Do not create evasive answers. Keep hedging low and stable. No major new risk language.",
    },
    {
        "filename": "ACME_Q3_2025.txt",
        "company": "ACME",
        "quarter": "Q3 2025",
        "profile": "Same SaaS company, deteriorating quarter. Prepared remarks still cautiously upbeat but Q&A reveals stress.",
        "flags": "Must trigger a RED sample: Q&A meaningfully more negative than prepared remarks, hedging density at least 40% higher than Q2, new themes around macro headwinds and elongated sales cycles, and at least two evasive executive answers.",
    },
    {
        "filename": "BRAVO_Q1_2025.txt",
        "company": "BRAVO",
        "quarter": "Q1 2025",
        "profile": "Industrial company, steady performer. Prepared remarks and Q&A similarly toned. Direct answers. Stable demand and operations.",
        "flags": "This is a GREEN sample baseline. No evasive answers. Themes should be stable and boring in a good way.",
    },
    {
        "filename": "BRAVO_Q2_2025.txt",
        "company": "BRAVO",
        "quarter": "Q2 2025",
        "profile": "Same industrial company, continued healthy execution. Prepared remarks and Q&A similarly toned in both quarters. Direct answers.",
        "flags": "Must remain GREEN: hedging stable, themes consistent, no new risk-heavy themes, and direct Q&A answers.",
    },
]


def build_prompt(company: str, quarter: str, profile: str, flags: str) -> str:
    return f"""Write a synthetic earnings call transcript for {company} {quarter}.

Requirements:
- Output plain text only.
- Length 2500 to 3500 words.
- Use this exact structure:
  [Company] [Quarter] Earnings Conference Call

  Operator: Good morning, and welcome...
  Jane Smith - CEO: [Prepared remarks about 800 words]
  John Doe - CFO: [Prepared remarks about 600 words]
  Operator: We will now begin the question and answer session.
  Mark Chen - Goldman Sachs: [Question]
  Jane Smith - CEO: [Answer]
  ...continue until there are 6 to 10 Q&A pairs total...
  Operator: This concludes today's call.
- Keep speaker labels exactly in transcript style.
- Make the text realistic but synthetic, with no copyrighted source text.

Company/quarter profile:
{profile}

Flag behavior to enforce:
{flags}
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in TRANSCRIPT_SPECS:
        text = generate_text(
            build_prompt(**{k: spec[k] for k in ("company", "quarter", "profile", "flags")}),
            max_new_tokens=3000,
        )
        (OUTPUT_DIR / spec["filename"]).write_text(text)
        print(f"Wrote {OUTPUT_DIR / spec['filename']}")


if __name__ == "__main__":
    main()
