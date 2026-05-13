#!/bin/zsh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=================================================="
echo "  EarningsLens — one-click setup (macOS)"
echo "=================================================="
echo ""

# 1. Find a Python interpreter
if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
  PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")
  if [ "$PY_MINOR" != "11" ]; then
    echo "Warning: found $(which python3) (Python 3.$PY_MINOR), but 3.11 is recommended."
    echo "Continuing anyway — install Python 3.11 from https://www.python.org/downloads/"
    echo "if anything fails."
    echo ""
  fi
else
  echo "Error: no Python 3 interpreter found on this Mac."
  echo ""
  echo "Install Python 3.11 from https://www.python.org/downloads/macos/"
  echo "Then double-click this file again."
  echo ""
  read -k 1 "?Press any key to close..."
  exit 1
fi

echo "Using $(which $PY) ($($PY --version))"
echo ""

# 2. Create the virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "Step 1/5  Creating virtual environment in .venv ..."
  $PY -m venv .venv
else
  echo "Step 1/5  Reusing existing .venv"
fi

source .venv/bin/activate

# 3. Upgrade pip
echo "Step 2/5  Upgrading pip ..."
python -m pip install --upgrade pip --quiet

# 4. Install pinned dependencies
echo "Step 3/5  Installing Python dependencies (this can take a few minutes) ..."
python -m pip install -r requirements.txt --quiet

# 5. Copy .env if missing
if [ ! -f ".env" ]; then
  echo "Step 4/5  Creating .env from .env.example ..."
  cp .env.example .env
else
  echo "Step 4/5  Reusing existing .env"
fi

# 6. Prefetch Hugging Face models
echo "Step 5/5  Downloading local models (~1.5 GB total) ..."
echo "          FinBERT ~440 MB, MiniLM ~80 MB, SmolLM2 ~1 GB"
echo "          This is one-time; future launches are instant."
python scripts/prefetch_models.py

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "  To start the app, double-click   launch.command"
echo "  Or from a terminal:               streamlit run app.py"
echo ""
read -k 1 "?Press any key to close..."
