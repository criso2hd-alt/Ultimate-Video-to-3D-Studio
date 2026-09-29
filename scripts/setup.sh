#!/usr/bin/env bash
# macOS / Linux: create .venv with Python 3.12, install the app, fetch the depth model.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "Python 3.12 is required (brew install python@3.12)."; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/python -m pip install -e ".[test,build]"
.venv/bin/python scripts/get_model.py
echo "Done. Run ./scripts/run.sh"
