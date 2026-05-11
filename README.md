# EarningsLens

AI that reads earnings calls like a senior equity analyst, using local Hugging Face models.

![Dashboard screenshot](docs/dashboard_red_demo.svg)

## What It Does

EarningsLens compares two consecutive earnings call transcripts from the same company, runs five analyses in parallel, and produces a concise analyst brief:

- Sentiment gap between prepared remarks and Q&A using FinBERT
- Hedging creep quarter over quarter using a finance-oriented lexicon
- Topic drift using MiniLM embeddings plus a local Hugging Face instruct model
- Risk vocabulary tracking using a curated keyword set
- Q&A evasion detection using a local Hugging Face instruct model as a judge

The output is analytical only. It is not investment advice.

Recent reliability improvements:
- parser confidence and low-confidence Q&A suppression
- cached transcript fetch / parse paths for faster reruns
- per-analysis timeout and fallback isolation
- diagnostics and provenance in the Streamlit UI
- regression tests plus a local health-check script

## Setup

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
# edit .env if you want to change the local Hugging Face model id
# add ALPHA_VANTAGE_API_KEY if you want direct ticker search

# 5. Generate sample transcripts (one-time)
python scripts/make_samples.py

# 6. Run
streamlit run app.py

# 7. Optional checks
./.venv/bin/pytest -q
./.venv/bin/python scripts/health_check.py
```

First run downloads FinBERT (~440 MB), MiniLM (~80 MB), and the local instruct model configured in `.env`.

## Input Modes

- `Search company`: search by exact ticker, select a match, load recent quarters, pick the two quarters to compare, and analyze them via Alpha Vantage
- `Use sample`: use the built-in synthetic demo pairs

## Diagnostics And Controls

- Sidebar controls let you tune thresholds, Q&A confidence filtering, and per-analysis timeout.
- The app exposes analysis provenance so you can tell whether an output came from the main model path or a fallback.
- Enable `Show diagnostics` in the sidebar to inspect parsed speaker roles and turn segmentation.

## Local Model Choice

Default local model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Reason:
- Hugging Face's model card describes the 1.7B SmolLM2 variant as lightweight enough to run on-device.
- On an 8 GB Mac, this is a more realistic default than 7B+ instruct models.
- It is small enough to be practical locally while still usable for short structured tasks like theme extraction, Q&A scoring, and synthesis.

## Repository Layout

- [app.py](/Users/ludovicsaint-yves/Desktop/Cours/Projet/EarningsLens/app.py)
- [earningslens/](/Users/ludovicsaint-yves/Desktop/Cours/Projet/EarningsLens/earningslens)
- [tests/](/Users/ludovicsaint-yves/Desktop/Cours/Projet/EarningsLens/tests)
- [docs/user_guide.md](/Users/ludovicsaint-yves/Desktop/Cours/Projet/EarningsLens/docs/user_guide.md)
- [docs/architecture.md](/Users/ludovicsaint-yves/Desktop/Cours/Projet/EarningsLens/docs/architecture.md)

## Credits

- FinBERT: Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (2019). https://huggingface.co/ProsusAI/finbert
- Sentence-Transformers: Reimers & Gurevych, "Sentence-BERT" (2019). https://www.sbert.net
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- LLM-as-judge methodology: Zheng et al., "Judging LLM-as-a-Judge" (2023). https://arxiv.org/abs/2306.05685
- Hedging lexicon adapted from: Hyland, "Hedging in Scientific Research Articles" (1998), with finance-specific additions.
- Hugging Face SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- Streamlit: https://streamlit.io
