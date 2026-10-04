"""A plain-text diary of what the VR side did (``vr.log`` in the data folder), for the times VR will not start.

Starting a headset session goes through the runtime, the driver and the headset, and when it hangs there is nothing on screen
to say where. Each step writes a timestamped line here (session requested, runtime connected, state changes, errors with their
traceback, stops, the watchdog); the file is trimmed when it grows past ``LIMIT`` and nothing here can raise.
"""

from __future__ import annotations

import threading
import time
import traceback

LIMIT = 256 * 1024
_LOCK = threading.Lock()


def _path():
    from .. import paths

    return paths.data_dir() / "vr.log"


def log(message: str) -> None:
    """Add a line (any thread)."""
    try:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
        with _LOCK:
            path = _path()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > LIMIT:
                text = path.read_text(encoding="utf8", errors="replace")
                path.write_text(text[len(text) // 2:], encoding="utf8")          # keep the newer half
            with path.open("a", encoding="utf8") as handle:
                handle.write(line)
    except Exception:  # noqa: BLE001 - a diary must never be the thing that breaks
        pass


def log_error(where: str, error: BaseException) -> None:
    """An exception with its traceback."""
    log(f"{where}: {type(error).__name__}: {error}\n" + "".join(traceback.format_exception(error)).rstrip())
