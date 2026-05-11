#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Create it first with:"
  echo "python3.11 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Missing dependencies. Install them with:"
  echo "pip install -r requirements.txt"
  exit 1
fi

if [ ! -d ".models/local-llm" ] || [ ! -d ".models/finbert" ] || [ ! -d ".models/sentence-embedder" ]; then
  echo "Local model cache is missing. Prefetch once with:"
  echo "python scripts/prefetch_models.py"
  echo "The app can still start, but it may download models on first use."
fi

exec streamlit run app.py
