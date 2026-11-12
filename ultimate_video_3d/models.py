"""Fetch the depth model from this project's own GitHub release when the app does not have it.

The release zip ships the model inside it, so a normal install never gets here. It matters when the
app is run from source, when the file was deleted, or when a future version needs a model an older
install lacks: the app fetches it itself - nobody is ever asked to go to GitHub and download a file
by hand. It comes from a release asset of the project's own repository (no third-party host), is
saved into the app's ``models`` folder, resumes if interrupted, and is checked against a SHA-256
before it is put in place, so a truncated or tampered download is discarded rather than loaded.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from . import bootstrap, paths

NAME = "Depth-Anything-V2-Small-hf.fp16.onnx"
#: The project's release that holds model files (tag ``models-v1``). Kept as its own release so the
#: "latest version" the update check reads is always an app release, never this one.
RELEASE = "https://github.com/criso2hd-alt/Ultimate-Video-to-3D-Studio/releases/download/models-v1"
URL = f"{RELEASE}/{NAME}"
SHA256 = "8ead3c1a9c3d97adaf966fc9df51894455ccd2b89a7e1f65d13becc2c7a8f16b"
APPROX_BYTES = 50_000_000

#: The optional High quality model, Depth Anything 3 Mono-Large (Apache-2.0), exported by scripts/export_da3.py.
#: fp32 for DirectML / CPU, fp16 for the NVIDIA TensorRT-RTX add-on (see depth.MODELS). Name -> (sha256, bytes).
EXTRA: dict[str, tuple[str, int]] = {
    "Depth-Anything-3-Mono-Large.onnx": ("5650a4f498ed44570c0bccf85a5cf280698dad86b2a95e2a1c8a126ec6a5aff5", 1_336_938_991),
    "Depth-Anything-3-Mono-Large.fp16.onnx": ("5c28525da835a0976b6e79b36a9f72eef8caceaa7b9ce7e1ec78299f905bbcef", 668_688_412),
}


def size_of(name: str) -> int:
    return EXTRA[name][1] if name in EXTRA else APPROX_BYTES


def target_path(name: str = NAME) -> Path:
    return paths.model_dir() / name


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def install(on_bytes: bootstrap.BytesProgress | None = None, on_text: bootstrap.TextProgress | None = None,
            name: str = NAME) -> Path:
    """Download and verify a depth model file. Returns its path; raises ``OSError`` with a plain message."""
    url, sha = (URL, SHA256) if name == NAME else (f"{RELEASE}/{name}", EXTRA[name][0])
    destination = target_path(name)
    part = destination.with_name(destination.name + ".part")
    if on_text:
        on_text("Downloading the depth model…")
    if shutil.disk_usage(destination.parent).free < size_of(name) * 1.1 + 200_000_000:
        raise OSError(f"There is not enough free disk space for the depth model ({size_of(name) / 1e9:.1f} GB) "
                      f"in {destination.parent}.")
    bootstrap._download(url, part, on_bytes)                     # resumes a partial file
    if on_text:
        on_text("Checking the download…")
    if _sha256_of(part) != sha:
        part.unlink(missing_ok=True)                             # never keep a file that failed its check
        raise OSError("The downloaded depth model did not pass its integrity check, so it was discarded. "
                      "Please try again.")
    part.replace(destination)
    return destination
