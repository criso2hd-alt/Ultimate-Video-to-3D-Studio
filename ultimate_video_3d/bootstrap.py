"""Fetch the video component (PyAV + FFmpeg, ~35 MB) on first use instead of shipping it.

The download is a pinned wheel from PyPI: a zip that is downloaded, unpacked into
the user's data folder, and put on ``sys.path``. No pip, no dependency
resolution - a URL and an unzip, so the failure modes are "no network" and "disk
full" rather than anything dependency-shaped. A partial download resumes.

Works for Windows, macOS (Intel and Apple silicon) and Linux: the right wheel is
chosen from the running platform.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import ssl
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import paths

AV_VERSION = "18.1.0"
AV_APPROX_BYTES = 40_000_000

BytesProgress = Callable[[int, int], None]
TextProgress = Callable[[str], None]

NETWORK_HELP = (
    "This is almost always something between the app and the download rather "
    "than a real outage:\n"
    "  • Turn off any VPN — this is the most common cause.\n"
    "  • Switch your DNS to 1.1.1.1 (Cloudflare) or 8.8.8.8 (Google).\n"
    "  • Disable a proxy, or pause antivirus HTTPS/SSL scanning.\n"
    "Then start it again — the download resumes where it stopped."
)

_NETWORK_ERRORS = (URLError, TimeoutError, socket.timeout, ConnectionError, ssl.SSLError)

#: Progress crosses threads as queued signals, and each one makes the GUI thread
#: take the GIL. Per-chunk reporting would starve its event loop and the window
#: would go "Not Responding" mid-download, so updates are coalesced.
_PROGRESS_INTERVAL = 0.05


class _Throttle:
    def __init__(self, callback: BytesProgress | None) -> None:
        self._callback = callback
        self._last = 0.0

    def __call__(self, done: int, total: int, force: bool = False) -> None:
        if self._callback is None:
            return
        now = time.monotonic()
        if force or now - self._last >= _PROGRESS_INTERVAL:
            self._last = now
            self._callback(done, total)


def activate_av() -> None:
    """Put a downloaded PyAV on the import path, DLLs included."""
    target = paths.av_runtime_dir()
    if not target.is_dir():
        return
    entry = str(target)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    if hasattr(os, "add_dll_directory"):
        for libs in (target / "av.libs", target / "av"):
            if libs.is_dir():
                try:
                    os.add_dll_directory(str(libs))
                except OSError:
                    pass


def av_is_ready() -> bool:
    activate_av()
    try:
        return importlib.util.find_spec("av") is not None
    except (ImportError, ValueError):
        return False


def _platform_tags() -> tuple[str, ...]:
    """Substrings a wheel filename must contain to suit this machine."""
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return ("win_arm64",) if "arm" in machine else ("win_amd64",)
    if sys.platform == "darwin":
        return ("macosx", "arm64") if machine == "arm64" else ("macosx", "x86_64")
    return ("manylinux", "aarch64" if "aarch" in machine or "arm" in machine else "x86_64")


def _resolve_av_wheel() -> tuple[str, int]:
    """Ask PyPI for this platform's wheel URL and size. Resolved at download time
    because hashed file paths on files.pythonhosted.org are not worth hard-coding."""
    import json

    request = Request(f"https://pypi.org/pypi/av/{AV_VERSION}/json",
                      headers={"User-Agent": "ultimate-video-3d-studio"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    needed = _platform_tags()
    candidates = [
        entry for entry in data.get("urls", [])
        if entry.get("filename", "").endswith(".whl")
        and all(part in entry["filename"] for part in needed)
        and "t-win" not in entry["filename"]        # free-threaded builds: different ABI
    ]
    for match in (lambda n: tag in n, lambda n: "abi3" in n):
        for entry in candidates:
            if match(entry["filename"]):
                return entry["url"], int(entry.get("size") or AV_APPROX_BYTES)
    raise OSError(f"No compatible PyAV {AV_VERSION} wheel on PyPI for this system ({', '.join(needed)}).")


def _remote_size(url: str) -> int:
    request = Request(url, headers={"User-Agent": "ultimate-video-3d-studio"}, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length") or 0)
    except (*_NETWORK_ERRORS, ValueError):
        return 0


#: A connection that is reset or goes silent partway is retried, resuming from what was already saved.
#: Security software that inspects web traffic does this now and then, and one hiccup should not make
#: a person start the whole download again.
DOWNLOAD_ATTEMPTS = 3


def _download(url: str, destination: Path, on_bytes: BytesProgress | None) -> None:
    """Fetch ``url`` to ``destination``, resuming a partial file, and retrying a few times if the
    connection drops."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_once(url, destination, on_bytes)
            return
        except OSError as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            del error
            time.sleep(1.0 * attempt)                        # a short, growing pause, then carry on from the saved part


def _download_once(url: str, destination: Path, on_bytes: BytesProgress | None) -> None:
    expected = _remote_size(url)
    have = destination.stat().st_size if destination.exists() else 0
    if expected and have == expected:
        if on_bytes:
            on_bytes(expected, expected)
        return
    if have > expected > 0:
        destination.unlink()
        have = 0

    headers = {"User-Agent": "ultimate-video-3d-studio"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        # 30 s socket timeout: a connection that goes silent (the classic VPN/DNS
        # stall) raises instead of looking like a frozen app.
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            resuming = response.status == 206
            if not resuming:
                have = 0
            total = expected or (int(response.headers.get("Content-Length") or AV_APPROX_BYTES) + have)
            done = have
            report = _Throttle(on_bytes)
            with open(destination, "ab" if resuming else "wb") as handle:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    report(done, total)
            report(done, total, force=True)
    except _NETWORK_ERRORS as error:
        raise OSError(f"The download stalled after {have // 1_048_576} MB.\n\n{NETWORK_HELP}") from error
    if done < total * 0.99:
        raise OSError(f"The download stopped early ({done // 1_048_576} of {total // 1_048_576} MB).\n\n{NETWORK_HELP}")


def _os_path(path: Path) -> str:
    r"""A path safe to open on Windows however long: the ``\\?\`` prefix opts out of MAX_PATH."""
    if os.name != "nt":
        return str(path)
    full = os.path.abspath(str(path))
    if full.startswith("\\\\?\\"):
        return full
    if full.startswith("\\\\"):
        return "\\\\?\\UNC\\" + full[2:]
    return "\\\\?\\" + full


def _extract(archive: Path, target: Path, on_bytes: BytesProgress | None) -> None:
    report = _Throttle(on_bytes)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        total = sum(m.file_size for m in members) or 1
        done = 0
        for member in members:
            # zip names always use "/"; dropping "", "." and ".." stops a crafted
            # entry escaping the target directory.
            parts = [p for p in member.filename.split("/") if p not in ("", ".", "..")]
            if not parts:
                continue
            dest = target.joinpath(*parts)
            if member.is_dir():
                os.makedirs(_os_path(dest), exist_ok=True)
                continue
            os.makedirs(_os_path(dest.parent), exist_ok=True)
            with bundle.open(member) as source, open(_os_path(dest), "wb") as sink:
                while True:
                    chunk = source.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    sink.write(chunk)
                    done += len(chunk)
                    report(done, total)
        report(done, total, force=True)


def install_av(on_bytes: BytesProgress | None = None, on_text: TextProgress | None = None) -> None:
    """Download and unpack PyAV. Raises on failure.

    Staged through a sibling directory and moved into place at the end, so an
    interrupted run leaves nothing ``av_is_ready`` would mistake for an install.
    """
    target = paths.av_runtime_dir()
    staging = target.with_name(target.name + ".partial")
    archive = target.with_name(target.name + ".whl.part")
    if staging.is_dir():
        shutil.rmtree(_os_path(staging), ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if on_text:
            on_text("Finding the video component…")
        url, _size = _resolve_av_wheel()
        if on_text:
            on_text(f"Downloading video support (PyAV {AV_VERSION})…")
        _download(url, archive, on_bytes)
        if on_text:
            on_text("Unpacking…")
        _extract(archive, staging, on_bytes)
        if not (staging / "av" / "__init__.py").is_file():
            raise OSError("The downloaded video component is missing its av package.")
        if target.is_dir():
            shutil.rmtree(_os_path(target), ignore_errors=True)
        staging.rename(target)
    finally:
        shutil.rmtree(_os_path(staging), ignore_errors=True)
    archive.unlink(missing_ok=True)
    activate_av()
