# EarningsLens

AI that reads earnings calls like a senior equity analyst, using local Hugging Face models.

![Dashboard screenshot](docs/dashboard_red_demo.svg)

## What It Does

EarningsLens compares two consecutive earnings call transcripts from the same company, runs five analyses, and produces a concise analyst brief:

- Sentiment gap between prepared remarks and Q&A using FinBERT
- Hedging creep quarter over quarter using a finance-oriented lexicon
- Topic drift using MiniLM embeddings plus a local Hugging Face instruct model
- Risk vocabulary tracking using a curated keyword set
- Q&A evasion detection using a local Hugging Face instruct model as a judge

The output is analytical only. It is not investment advice.

Recent reliability improvements:
- parser confidence and low-confidence Q&A suppression
- cached transcript fetch / parse paths for faster reruns
- fallback isolation when an analysis fails
- diagnostics and provenance in the Streamlit UI
- regression tests plus a local health-check script

## One-click setup

If you do not want to touch a terminal:

- **macOS** — double-click `setup.command` in Finder.
- **Windows** — double-click `setup.bat` in File Explorer.

The installer verifies Python 3.11, creates `.venv`, installs every pinned dependency, copies `.env.example` to `.env`, and pre-downloads the three Hugging Face models (~1.5 GB). If Python 3.11 is missing it stops with a link to python.org — everything else is automated. After it finishes, double-click `launch.command` (macOS) or `launch.bat` (Windows) to start the app.

## Manual setup

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
`LOW_MEMORY_MODE=1` is enabled by default so the heaviest model-backed analyses run in phases instead of all at once. Set `LOW_MEMORY_MODE=0` if you prefer the faster parallel path on a machine with more RAM.

## Input Modes

- `Search company`: search by exact ticker, select a match, load recent quarters, pick the two quarters to compare, and analyze them via Alpha Vantage
- `Use sample`: use the built-in synthetic demo pairs

## Diagnostics And Controls

- Sidebar controls let you tune thresholds and Q&A confidence filtering.
- `.env` controls memory strategy: `LOW_MEMORY_MODE=1` lowers peak RAM; `LOW_MEMORY_MODE=0` favors speed.
- The app exposes analysis provenance so you can tell whether an output came from the main model path or a fallback.
- Enable `Show diagnostics` in the sidebar to inspect parsed speaker roles and turn segmentation.

## Local Model Choice

Default local model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Reason:
- Hugging Face's model card describes the 1.7B SmolLM2 variant as lightweight enough to run on-device.
- On an 8 GB Mac, this is a more realistic default than 7B+ instruct models.
- It is small enough to be practical locally while still usable for short structured tasks like theme extraction, Q&A scoring, and synthesis.

## Evaluation

We tuned thresholds on the four sample transcripts shipped in `data/samples/` (Microsoft FY25 Q4 → FY26 Q1, Goldman Sachs Q4 2025 → Q1 2026) plus a handful of additional public calls we did not commit to the repo. Each flag threshold was chosen to keep the GREEN/AMBER/RED triage useful rather than alarmist:

| Signal | Threshold | Rationale |
|---|---|---|
| Sentiment gap (Q&A − prepared) | `≤ -0.15` | FinBERT scores are bounded in `[-1, 1]`; a 0.15 drop is consistently visible to a human reader of the Q&A vs the script and below it the gap is within noise. |
| Hedging delta quarter-over-quarter | `≥ +25 %` | Lexicon-based densities move slowly across quarters; a quarter that suddenly raises hedge density by a quarter again is qualitatively different. |
| Topic semantic similarity | `< 0.75` | MiniLM cosine similarity between two prepared-remarks transcripts of the same company typically sits in the 0.80–0.95 band; below 0.75 the prepared narrative has materially shifted. |
| Q&A evasion responsiveness | `≤ 4 / 10` | The LLM-as-judge rubric uses an integer 0–10 scale; ≤ 4 corresponds to *"pivots to adjacent topics or generalities"*. |
| Q&A pair confidence floor | `≥ 0.55` | Pairs whose parser confidence is below 0.55 are *excluded* from flags entirely — we'd rather miss an evasion than produce a false flag built on a misattributed speaker. |

Validation we ran:

- **Regression tests**: 25 `pytest` cases covering the parser, the evasion rubric, the score-adjustment heuristics, the question-splitter, and the answer-filler stripper. CI-able with `./.venv/bin/pytest -q`.
- **Health check**: `python scripts/health_check.py` loads each model, runs one sample call end-to-end, and reports any failure.
- **Empirical sanity check on flags**: on the bundled samples the system produces ~2-3 flags per quarter, most of which a human reviewer agrees with after re-reading the underlying turn. There is no claim of statistical accuracy — the signals are framed as reading aids, not as ground truth.

The thresholds live in `earningslens/config.py` and can be overridden via environment variables when re-tuning is needed.

## Supporting documents

- [docs/EarningsLens_User_Guide.pdf](docs/EarningsLens_User_Guide.pdf) — polished end-user guide
- [docs/user_guide.md](docs/user_guide.md) — short Markdown overview
- [docs/architecture.md](docs/architecture.md) — pipeline architecture
- [docs/course-concepts.md](docs/course-concepts.md) — mapping to *Introduction to AI for Business* concepts
- [PROMPTS.md](PROMPTS.md) — log of how AI coding assistants were used during development
- [data/samples/SOURCES.md](data/samples/SOURCES.md) — attribution for the bundled transcripts

## Repository Layout

- [app.py](app.py)
- [earningslens/](earningslens/)
- [tests/](tests/)
- [scripts/](scripts/)
- [docs/](docs/)
- [data/samples/](data/samples/)

## Credits

- FinBERT: Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (2019). https://huggingface.co/ProsusAI/finbert
- Sentence-Transformers: Reimers & Gurevych, "Sentence-BERT" (2019). https://www.sbert.net
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- LLM-as-judge methodology: Zheng et al., "Judging LLM-as-a-Judge" (2023). https://arxiv.org/abs/2306.05685
- Hedging lexicon adapted from: Hyland, "Hedging in Scientific Research Articles" (1998), with finance-specific additions.
- Hugging Face SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- Streamlit: https://streamlit.io
