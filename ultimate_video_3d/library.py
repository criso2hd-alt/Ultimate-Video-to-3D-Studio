"""The Library: the 3D videos this app has made, plus any the person adds, with no Qt in here.

What is stored is a small list (``library.json`` in the data folder): where each video is, what 3D layout it
is in, and a few facts for the grid. The videos themselves are never copied, so the Library costs no disk
space. A converted video is added automatically when its conversion finishes; a video from somewhere else is
added by hand, and a layout is guessed for it (from the file name and shape) which the person can correct.

Removing an entry only forgets it. *Deleting* moves the file to the Recycle Bin (Trash on a Mac), so a slip
of the mouse is recoverable; it never deletes for good, and it does nothing if the bin is unavailable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import paths, stereo

CONVERTED, ADDED = "converted", "added"

#: Layout keys that hold two eyes in one frame, and how.
SIDE_BY_SIDE = {"sbs_half", "sbs_full"}
TOP_BOTTOM = {"tb_half", "tb_full"}
#: The choices offered for a video whose layout is not known. "" is a flat 2D video.
LAYOUT_CHOICES: tuple[tuple[str, str], ...] = (
    ("sbs_half", "Side-by-Side (Half)"), ("sbs_full", "Side-by-Side (Full)"),
    ("tb_half", "Top-and-Bottom (Half)"), ("tb_full", "Top-and-Bottom (Full)"),
    ("anaglyph", "Anaglyph (red / cyan)"), ("", "Flat 2D"),
)


@dataclass
class Entry:
    id: str
    path: str
    title: str
    kind: str = ADDED
    layout: str = ""
    source: str = ""             # the file a converted video came from
    created: float = 0.0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    size: int = 0

    @property
    def file(self) -> Path:
        return Path(self.path)

    @property
    def missing(self) -> bool:
        return not self.file.exists()

    @property
    def is_stereo(self) -> bool:
        """Two eyes in one picture that the player can split (so Anaglyph can be offered)."""
        return self.layout in SIDE_BY_SIDE or self.layout in TOP_BOTTOM

    @property
    def layout_label(self) -> str:
        return dict(LAYOUT_CHOICES).get(self.layout, "Flat 2D") if self.layout != "rgbd" else "Colour + Depth"

    @property
    def detail(self) -> str:
        m, s = divmod(int(round(self.duration)), 60)
        h, m = divmod(m, 60)
        length = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        return f"{self.layout_label}  ·  {length}  ·  {self.size / 1e6:.0f} MB" if self.duration else self.layout_label


def guess_layout(path: Path, width: int, height: int) -> str:
    """The 3D layout of a video from elsewhere: file-name tags first, then the shape of the picture."""
    name = Path(path).stem.lower()
    tokens = [
        (r"(^|[^a-z])(f|full)[-_ .]?(sbs|side)|fsbs", "sbs_full"),
        (r"(^|[^a-z])(f|full)[-_ .]?(tb|ou|top)|ftb", "tb_full"),
        (r"(^|[^a-z])(h|half)?[-_ .]?(sbs|side[-_ ]?by[-_ ]?side)([^a-z]|$)|hsbs", "sbs_half"),
        (r"(^|[^a-z])(h|half)?[-_ .]?(tb|ou|over[-_ ]?under|top[-_ ]?bottom)([^a-z]|$)|htb|hou", "tb_half"),
        (r"anaglyph|[-_ ]ana([^a-z]|$)", "anaglyph"),
    ]
    for pattern, layout in tokens:
        if re.search(pattern, name):
            return layout
    if height and width / height >= 3.3:
        return "sbs_full"            # 32:9 and wider: two 16:9 pictures side by side
    if width and height / width >= 0.85:
        return "tb_full"             # as tall as it is wide: two pictures stacked
    return ""


def split_eyes(frame: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """The left and right pictures of a two-in-one frame, each back at its true shape.

    A half layout squeezes each eye to half the frame (in width for side-by-side, in height for top-and-bottom),
    so it is stretched back; a full layout already holds each eye at full size.
    """
    import cv2

    h, w = frame.shape[:2]
    if layout in SIDE_BY_SIDE:
        left, right = frame[:, : w // 2], frame[:, w // 2: w // 2 * 2]
        if layout == "sbs_half":
            left, right = (cv2.resize(e, (w // 2 * 2, h), interpolation=cv2.INTER_LINEAR) for e in (left, right))
    elif layout in TOP_BOTTOM:
        left, right = frame[: h // 2], frame[h // 2: h // 2 * 2]
        if layout == "tb_half":
            left, right = (cv2.resize(e, (w, h // 2 * 2), interpolation=cv2.INTER_LINEAR) for e in (left, right))
    else:
        return frame, frame
    return left, right


def view_of(frame_u8: np.ndarray, layout: str, mode: str) -> np.ndarray:
    """A decoded uint8 frame as shown on a normal screen: untouched ("original") or red/cyan ("anaglyph")."""
    if mode != "anaglyph" or not (layout in SIDE_BY_SIDE or layout in TOP_BOTTOM):
        return frame_u8
    left, right = split_eyes(frame_u8, layout)
    out = stereo.anaglyph(left.astype(np.float32) * (1 / 255.0), right.astype(np.float32) * (1 / 255.0))
    return (out * 255.0 + 0.5).astype(np.uint8)


# --- the list -------------------------------------------------------------------------------------


class Library:
    def __init__(self, file: Path | None = None, thumbs: Path | None = None) -> None:
        self._file = file or paths.data_dir() / "library.json"
        self._thumbs = thumbs or paths.data_dir() / "library-thumbs"
        self.entries: list[Entry] = []
        self.load()

    # -- storage ---------------------------------------------------------------------------------

    def load(self) -> None:
        try:
            data = json.loads(self._file.read_text(encoding="utf8"))
            known = set(Entry.__dataclass_fields__)
            self.entries = [Entry(**{k: v for k, v in item.items() if k in known}) for item in data.get("entries", [])]
        except (OSError, ValueError, TypeError):
            self.entries = []

    def save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            temp = self._file.with_name(self._file.name + ".tmp")
            temp.write_text(json.dumps({"entries": [asdict(e) for e in self.entries]}, indent=1), encoding="utf8")
            temp.replace(self._file)                           # never a half-written list
        except OSError:
            pass

    # -- adding ----------------------------------------------------------------------------------

    def find(self, path: Path) -> Entry | None:
        key = _key(path)
        return next((e for e in self.entries if _key(e.file) == key), None)

    def get(self, entry_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def _facts(self, path: Path) -> tuple[float, int, int, int]:
        from . import video

        try:
            info = video.probe(path)
            return info.duration, info.width, info.height, Path(path).stat().st_size
        except Exception:  # noqa: BLE001 - an unreadable video is still listed, just without details
            try:
                return 0.0, 0, 0, Path(path).stat().st_size
            except OSError:
                return 0.0, 0, 0, 0

    def add_converted(self, path: Path, layout: str, source: str = "") -> Entry:
        """Record a finished conversion. A video converted again over the same file just refreshes its entry."""
        path = Path(path)
        duration, width, height, size = self._facts(path)
        existing = self.find(path)
        entry = existing or Entry(id=uuid.uuid4().hex[:12], path=str(path), title=path.stem)
        entry.kind, entry.layout, entry.source = CONVERTED, layout, source
        entry.created, entry.duration, entry.width, entry.height, entry.size = time.time(), duration, width, height, size
        if existing is None:
            self.entries.insert(0, entry)
        else:
            self._forget_thumb(entry)
        self.save()
        return entry

    def add_external(self, path: Path, layout: str | None = None) -> Entry | None:
        """Add a video the person brought. Returns None if it is already listed (that entry is moved to the top)."""
        path = Path(path)
        if not path.is_file():
            return None
        existing = self.find(path)
        if existing is not None:
            self.entries.remove(existing)
            self.entries.insert(0, existing)
            self.save()
            return None
        duration, width, height, size = self._facts(path)
        entry = Entry(
            id=uuid.uuid4().hex[:12], path=str(path), title=path.stem, kind=ADDED,
            layout=guess_layout(path, width, height) if layout is None else layout,
            created=time.time(), duration=duration, width=width, height=height, size=size,
        )
        self.entries.insert(0, entry)
        self.save()
        return entry

    # -- changing --------------------------------------------------------------------------------

    def set_layout(self, entry_id: str, layout: str) -> None:
        entry = self.get(entry_id)
        if entry is not None:
            entry.layout = layout
            self._forget_thumb(entry)                          # the thumbnail shows one eye, so it depends on this
            self.save()

    def remove(self, entry_id: str) -> None:
        """Forget an entry; the file is left alone."""
        entry = self.get(entry_id)
        if entry is not None:
            self.entries.remove(entry)
            self._forget_thumb(entry)
            self.save()

    def delete_file(self, entry_id: str, attempts: int = 8) -> bool:
        """Move the video to the Recycle Bin and forget it. False (and nothing changes) if that was not possible.

        ``attempts`` > 1 retries briefly if the file is still open; a caller on the UI thread passes 1 and
        retries itself, keeping its event loop running (Qt's audio backend lets go of a file only then)."""
        entry = self.get(entry_id)
        if entry is None:
            return False
        if entry.file.exists() and not _trash_with_retries(entry.file, attempts):
            return False
        self.remove(entry_id)
        return True

    # -- thumbnails ------------------------------------------------------------------------------

    def thumb_path(self, entry: Entry) -> Path:
        return self._thumbs / f"{entry.id}.jpg"

    def _forget_thumb(self, entry: Entry) -> None:
        try:
            self.thumb_path(entry).unlink(missing_ok=True)
        except OSError:
            pass

    def make_thumbnail(self, entry: Entry) -> Path | None:
        """A 320 x 180 picture of the video (one eye of a 3D one), made on first need. Slow: call off the UI thread."""
        import cv2

        from . import video

        target = self.thumb_path(entry)
        if target.exists():
            return target
        if entry.missing:
            return None
        try:
            info = video.probe(entry.file)
            frame = video.read_frame_at(entry.file, int(info.frames * 0.12) if info.frames > 50 else 0)
            if frame is None:
                return None
            if entry.layout in SIDE_BY_SIDE or entry.layout in TOP_BOTTOM:
                frame = split_eyes(frame, entry.layout)[0]
            h, w = frame.shape[:2]
            scale = max(320 / w, 180 / h)                      # fill the tile, centred, rather than letterbox it
            resized = cv2.resize(frame, (max(320, round(w * scale)), max(180, round(h * scale))), interpolation=cv2.INTER_AREA)
            y0, x0 = (resized.shape[0] - 180) // 2, (resized.shape[1] - 320) // 2
            tile = (np.clip(resized[y0:y0 + 180, x0:x0 + 320], 0, 1) * 255 + 0.5).astype(np.uint8)
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            return target
        except Exception:  # noqa: BLE001 - a missing thumbnail is not an error
            return None


def _trash_with_retries(path: Path, attempts: int = 8, pause: float = 0.25) -> bool:
    """Windows will not move a file that something still has open, and a player, an indexer or a virus
    scanner can hold one for a moment after it has been closed. Try again briefly before giving up."""
    for attempt in range(attempts):
        if send_to_trash(path):
            return True
        if not path.exists():
            return True
        if attempt + 1 < attempts:
            time.sleep(pause)
    return False


def _key(path: Path) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


# --- the Recycle Bin -------------------------------------------------------------------------------


def send_to_trash(path: Path) -> bool:
    """Move a file to the Recycle Bin / Trash. Returns whether it worked; never deletes for good."""
    path = Path(path)
    try:
        if sys.platform == "win32":
            return _trash_windows(path)
        if sys.platform == "darwin":
            script = f'tell application "Finder" to delete POSIX file "{path}"'
            return subprocess.run(["osascript", "-e", script], capture_output=True, timeout=30).returncode == 0
        return subprocess.run(["gio", "trash", str(path)], capture_output=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001 - no bin means no delete
        return False


def _trash_windows(path: Path) -> bool:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 3
    FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    op.pFrom = str(path.resolve()) + "\0"                       # a double-NUL-terminated list; ctypes adds the last
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return result == 0 and not op.fAnyOperationsAborted and not path.exists()
