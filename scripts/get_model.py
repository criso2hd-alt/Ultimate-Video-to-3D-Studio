"""Fetch the Depth Anything V2 Small (Apache-2.0) ONNX model into the app's assets folder.

The model is ~50 MB, so it is not committed. It is hosted as a release asset of
this repository and checked against a SHA-256 before it is put in place. To make
it yourself instead: ``python scripts/export_onnx.py --fp16`` (needs torch).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

NAME = "Depth-Anything-V2-Small-hf.fp16.onnx"
URL = f"https://github.com/criso2hd-alt/Ultimate-Video-to-3D-Studio/releases/download/models-v1/{NAME}"
SHA256 = "8ead3c1a9c3d97adaf966fc9df51894455ccd2b89a7e1f65d13becc2c7a8f16b"
DEST = Path(__file__).resolve().parent.parent / "ultimate_video_3d" / "assets" / "onnx"


def main() -> int:
    target = DEST / NAME
    if target.is_file():
        print(f"Depth model already present: {target}")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    print(f"Downloading {URL}")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response, open(part, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                digest.update(chunk)
    except Exception as error:  # noqa: BLE001
        part.unlink(missing_ok=True)
        print(f"Could not download the model: {error}\n"
              "Generate it locally instead:  python scripts/export_onnx.py --fp16", file=sys.stderr)
        return 1
    if digest.hexdigest() != SHA256:
        part.unlink(missing_ok=True)
        print("The downloaded model failed its integrity check and was discarded.", file=sys.stderr)
        return 1
    part.replace(target)
    print(f"Saved {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
