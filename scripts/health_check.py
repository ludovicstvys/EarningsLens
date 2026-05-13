from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from earningslens.evasion import analyze_evasion
from earningslens.hedging import analyze_hedging
from earningslens.local_llm import warmup_local_llm
from earningslens.parser import parse_transcript
from earningslens.risk_vocab import analyze_risk_vocab
from earningslens.sentiment import analyze_sentiment, release_sentiment_model, warmup_sentiment_model
from earningslens.topics import analyze_topics, release_topic_models, warmup_topic_models


def _load_sample(name: str) -> str:
    return (PROJECT_ROOT / "data" / "samples" / name).read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local health check for EarningsLens.")
    parser.add_argument("--full", action="store_true", help="Warm models and run a full sample analysis.")
    args = parser.parse_args()

    current_text = _load_sample("MSFT_Q1_2026.txt")
    prior_text = _load_sample("MSFT_Q4_2025.txt")

    current = parse_transcript(current_text, company="Microsoft", quarter="Q1 2026", ticker="MSFT")
    prior = parse_transcript(prior_text, company="Microsoft", quarter="Q4 2025", ticker="MSFT")

    print("parser: ok")
    print(f"current parse confidence: {current.parse_confidence:.2f}")
    print(f"prior parse confidence: {prior.parse_confidence:.2f}")

    hedging = analyze_hedging(current, prior)
    risk_vocab = analyze_risk_vocab(current, prior)
    print("lexicon analyses: ok")
    print(f"hedging delta: {hedging.delta_pct}")
    print(f"risk movers: {len(risk_vocab)}")

    if not args.full:
        print("health check complete")
        return 0

    warmup_sentiment_model()
    sentiment = analyze_sentiment(current)
    release_sentiment_model()

    warmup_topic_models()
    topics = analyze_topics(current, prior)

    warmup_local_llm()
    evasions = analyze_evasion(current)
    release_topic_models()

    print("model analyses: ok")
    print(f"sentiment gap: {sentiment.gap:+.2f}")
    print(f"topic similarity: {topics.semantic_similarity:.2f}")
    print(f"evasion pairs: {len(evasions)}")
    print("health check complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
