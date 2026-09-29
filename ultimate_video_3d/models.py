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
from pathlib import Path

from . import bootstrap, paths

NAME = "Depth-Anything-V2-Small-hf.fp16.onnx"
#: The project's release that holds model files (tag ``models-v1``). Kept as its own release so the
#: "latest version" the update check reads is always an app release, never this one.
URL = f"https://github.com/criso2hd-alt/Ultimate-Video-to-3D-Studio/releases/download/models-v1/{NAME}"
SHA256 = "8ead3c1a9c3d97adaf966fc9df51894455ccd2b89a7e1f65d13becc2c7a8f16b"
APPROX_BYTES = 50_000_000


def target_path() -> Path:
    return paths.model_dir() / NAME


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def install(on_bytes: bootstrap.BytesProgress | None = None, on_text: bootstrap.TextProgress | None = None) -> Path:
    """Download and verify the depth model. Returns its path; raises ``OSError`` with a plain message."""
    destination = target_path()
    part = destination.with_name(destination.name + ".part")
    if on_text:
        on_text("Downloading the depth model…")
    bootstrap._download(URL, part, on_bytes)                     # resumes a partial file
    if on_text:
        on_text("Checking the download…")
    if _sha256_of(part) != SHA256:
        part.unlink(missing_ok=True)                             # never keep a file that failed its check
        raise OSError("The downloaded depth model did not pass its integrity check, so it was discarded. "
                      "Please try again.")
    part.replace(destination)
    return destination
