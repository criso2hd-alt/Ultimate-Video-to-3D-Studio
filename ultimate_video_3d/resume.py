"""Pause and resume for long conversions.

A conversion is encoded in *segments*: each is a small self-contained video file, closed as it fills (about a
minute of picture) or the moment the job is paused. The list of finished segments is kept beside them in a
``state.json``. To resume, the pipeline skips the frames those segments already cover, encodes the rest as new
segments, and at the end joins them into the one video (a straight copy of the packets, so nothing is
re-encoded). A crash or a closed window costs at most the segment that was open.

A job is recognised by what would change its result: the source file (path, size, modified time), the frame
range, and every export, 3D and colour setting. Change any of them and it is a different job, so a finished
part is never mixed into a picture made with other settings.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .settings import AppSettings

#: Frames per segment when nothing interrupts: about a minute of a 24 fps film.
SEGMENT_FRAMES = 1440
#: Frames re-read before the resume point so the depth smoothing has settled by the time it is used.
PREROLL = 12
#: Unfinished jobs older than this are removed at start-up.
KEEP_DAYS = 14


@dataclass
class Segment:
    name: str
    frames: int


@dataclass
class ResumeState:
    folder: Path
    source: str = ""
    total: int = 0
    encoder: str = ""
    segments: list[Segment] = field(default_factory=list)
    saved: float = 0.0

    @property
    def frames_done(self) -> int:
        return sum(s.frames for s in self.segments)

    @property
    def percent(self) -> int:
        return round(100 * self.frames_done / self.total) if self.total else 0

    def files(self) -> list[Path]:
        return [self.folder / s.name for s in self.segments]

    def valid(self) -> bool:
        return bool(self.segments) and all(p.exists() and p.stat().st_size > 0 for p in self.files())

    def next_segment_path(self, suffix: str) -> Path:
        return self.folder / f"seg{len(self.segments):05d}{suffix}"

    def add(self, path: Path, frames: int) -> None:
        if frames <= 0:
            path.unlink(missing_ok=True)
            return
        self.segments.append(Segment(path.name, frames))
        self.save()

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.saved = time.time()
        data = {"source": self.source, "total": self.total, "saved": self.saved, "encoder": self.encoder,
                "segments": [dataclasses.asdict(s) for s in self.segments]}
        temp = self.folder / "state.json.tmp"
        temp.write_text(json.dumps(data), encoding="utf8")
        temp.replace(self.folder / "state.json")            # never a half-written state

    def clear(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)
        self.segments = []


def job_key(source: Path, settings: AppSettings, start: int, limit: int | None) -> str:
    """Identify a conversion by everything that would change its picture."""
    try:
        stat = Path(source).stat()
        identity = [str(Path(source).resolve()), stat.st_size, int(stat.st_mtime)]
    except OSError:
        identity = [str(source)]
    payload = json.dumps(
        [identity, start, limit, dataclasses.asdict(settings.export), dataclasses.asdict(settings.threed),
         dataclasses.asdict(settings.colour), settings.force_cpu_depth],
        sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf8")).hexdigest()[:20]


def root() -> Path:
    path = paths.scratch_dir() / "resume"
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_state(source: Path, settings: AppSettings, start: int, limit: int | None, total: int) -> ResumeState:
    """The state for this job: what an earlier run left, or an empty one."""
    folder = root() / job_key(source, settings, start, limit)
    state = ResumeState(folder, str(source), total)
    try:
        data = json.loads((folder / "state.json").read_text(encoding="utf8"))
        state.segments = [Segment(str(s["name"]), int(s["frames"])) for s in data.get("segments", [])]
        state.saved = float(data.get("saved", 0.0))
        state.encoder = str(data.get("encoder", ""))
    except (OSError, ValueError, KeyError, TypeError):
        state.segments = []
    if state.segments and not state.valid():
        state.clear()                                          # a missing piece: the rest is useless
    return state


def find(source: Path, settings: AppSettings, start: int, limit: int | None, total: int) -> ResumeState | None:
    """An unfinished run of exactly this job, or None."""
    state = open_state(source, settings, start, limit, total)
    return state if state.segments and state.frames_done < (total or 1 << 60) else None


def cleanup_stale(days: int = KEEP_DAYS) -> None:
    """Remove unfinished jobs nobody came back to."""
    cutoff = time.time() - days * 86400
    try:
        for folder in root().iterdir():
            marker = folder / "state.json"
            when = marker.stat().st_mtime if marker.exists() else folder.stat().st_mtime
            if folder.is_dir() and when < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
    except OSError:
        pass
