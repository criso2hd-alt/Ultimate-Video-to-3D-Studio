#!/usr/bin/env bash
# Builds "Ultimate Video to 3D Studio.app" (unsigned). Sign and notarise separately for distribution.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run ./scripts/setup.sh first."; exit 1; }
[ -f ultimate_video_3d/assets/onnx/Depth-Anything-V2-Small-hf.fp16.onnx ] || .venv/bin/python scripts/get_model.py
.venv/bin/python -m pytest -q
.venv/bin/python -m PyInstaller --noconfirm --clean --distpath release --workpath build UltimateVideo3DStudio.spec
echo "Built release/Ultimate Video to 3D Studio.app"
