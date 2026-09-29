#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Not set up yet. Run ./scripts/setup.sh first."; exit 1; }
exec .venv/bin/python main.py "$@"
