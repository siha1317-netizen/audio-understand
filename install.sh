#!/usr/bin/env sh
# macOS / Linux: creates .venv next to this file and installs dependencies.
# Usage:  sh install.sh        (CPU build of PyTorch on Linux; macOS uses the default wheel)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname)" = "Linux" ]; then
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install torch
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch, transformers, librosa; print('ok', torch.__version__, transformers.__version__, librosa.__version__)"
echo "Done. Run:  .venv/bin/python scripts/analyze.py <file>"
