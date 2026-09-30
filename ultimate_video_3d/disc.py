"""Reading a DVD or Blu-ray from a drive, and copying one title to a video file on disk.

The app never converts straight from a disc: a drive is slow, seeks badly and cannot be scrubbed. The disc is
copied first, as fast as the drive can deliver it, to a Matroska file (a straight copy of the streams, nothing
re-encoded, so it is quick and loses nothing), and the normal pipeline then runs on that file.

What is read:

- **DVD**: ``VIDEO_TS/VTS_nn_m.VOB`` plus the title's ``VTS_nn_0.IFO`` table, which says which cells (runs of
  2048-byte sectors) make up each title. Reading the table matters: a disc often holds several episodes in
  one title set, and many carry decoy titles.
- **Blu-ray**: ``BDMV/PLAYLIST/*.mpls`` (which clips a playlist plays, in order) and ``BDMV/STREAM/*.m2ts``.

Only discs the operating system can read as ordinary files are handled. A disc whose contents are scrambled
comes out of FFmpeg as noise, and is reported as unreadable instead of producing a broken file.

Pure Python, no Qt. Windows-only calls sit behind ``sys.platform`` and degrade to "no drives found".
"""

from __future__ import annotations

import os
import re
import string
import struct
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SECTOR = 2048
#: Sample starts must land on a whole DVD sector and a whole Blu-ray 6144-byte unit (both, so 6144).
ALIGN = 6144
#: Shorter than this is a menu, a logo or a trailer, not something to convert.
MIN_TITLE_SECONDS = 60.0

DVD, BLURAY = "dvd", "bluray"

Progress = Callable[[int, int], None]         # bytes done, bytes total


class DiscError(Exception):
    """Something about the disc the person should be told, in plain words."""


class Stopped(Exception):
    """The copy was cancelled."""


@dataclass(frozen=True)
class Drive:
    root: Path
    label: str
    kind: str                                  # DVD, BLURAY, or "" when no video disc is recognised

    @property
    def text(self) -> str:
        what = {DVD: "DVD", BLURAY: "Blu-ray"}.get(self.kind, "no video found")
        return f"{self.label or 'Disc'} ({self.root}) - {what}"


@dataclass
class Title:
    number: int
    kind: str
    seconds: float
    nbytes: int
    #: Files read back to back (Blu-ray clips), or the VOB files whose sectors ``cells`` index (DVD).
    parts: list[Path]
    #: DVD only: inclusive (first, last) sector runs within the joined VOB files. Empty = read every byte.
    cells: list[tuple[int, int]] = field(default_factory=list)
    #: Set when "Title N" (N = rank by length, filled in by list_titles) is not the name to show.
    label: str = ""
    #: Filled in by the caller once is_scrambled() has looked (it reads from the disc, so it is not automatic).
    scrambled: bool | None = None

    @property
    def name(self) -> str:
        return self.label or f"Title {self.number}"

    @property
    def text(self) -> str:
        m, s = divmod(int(round(self.seconds)), 60)
        h, m = divmod(m, 60)
        size = self.nbytes / 1e9
        return f"{self.name}  -  {h}:{m:02d}:{s:02d}  -  {size:.1f} GB"


# --- finding the disc --------------------------------------------------------------------------------


def _child(root: Path, name: str) -> Path | None:
    """``root/name`` matched without regard to case (a Mac or Linux mount may lower-case it)."""
    try:
        for entry in root.iterdir():
            if entry.name.upper() == name:
                return entry
    except OSError:
        return None
    return None


def disc_root(path: Path) -> tuple[Path, str] | None:
    """Accept a drive, the disc's folder, or its VIDEO_TS / BDMV folder itself. Returns (root, kind)."""
    path = Path(path)
    if path.name.upper() == "VIDEO_TS" and path.is_dir():
        return path.parent, DVD
    if path.name.upper() == "BDMV" and path.is_dir():
        return path.parent, BLURAY
    if _child(path, "VIDEO_TS"):
        return path, DVD
    if _child(path, "BDMV"):
        return path, BLURAY
    return None


def find_drives() -> list[Drive]:
    """Optical drives that have a disc in them (removable volumes on a Mac or Linux)."""
    roots: list[tuple[Path, str]] = []
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            mask = kernel32.GetLogicalDrives()
            for i, letter in enumerate(string.ascii_uppercase):
                if not mask & (1 << i):
                    continue
                root = f"{letter}:\\"
                if kernel32.GetDriveTypeW(root) != 5:          # DRIVE_CDROM
                    continue
                label = ctypes.create_unicode_buffer(261)
                ok = kernel32.GetVolumeInformationW(root, label, 261, None, None, None, None, 0)
                if ok:                                          # fails while the drive is empty
                    roots.append((Path(root), label.value))
        except Exception:  # noqa: BLE001 - a odd driver must not break the button
            return []
    else:
        bases = [Path("/Volumes")] if sys.platform == "darwin" else [Path("/media"), Path("/run/media"), Path("/mnt")]
        for base in bases:
            try:
                for entry in base.iterdir():
                    # /media/<user>/<disc> on Linux is one level deeper
                    candidates = [entry] + ([p for p in entry.iterdir()] if sys.platform != "darwin" and entry.is_dir() else [])
                    for candidate in candidates:
                        if candidate.is_dir():
                            roots.append((candidate, candidate.name))
            except OSError:
                continue
    drives = []
    for root, label in roots:
        found = disc_root(root)
        if found or sys.platform == "win32":                    # an empty-looking Windows optical drive is still worth listing
            drives.append(Drive(root, label, found[1] if found else ""))
    return drives


# --- listing titles ----------------------------------------------------------------------------------


def _bcd(value: int) -> int:
    return (value >> 4) * 10 + (value & 15)


def _bcd_time(raw: bytes) -> float:
    """A DVD playback time: hours, minutes, seconds, then frames with the frame rate in the top two bits."""
    hours, minutes, seconds, frames = _bcd(raw[0]), _bcd(raw[1]), _bcd(raw[2]), raw[3]
    fps = {3: 30.0, 1: 25.0}.get(frames >> 6, 25.0)
    return hours * 3600 + minutes * 60 + seconds + _bcd(frames & 0x3F) / fps


def _vob_parts(video_ts: Path, vts: int) -> list[Path]:
    """VTS_nn_1.VOB, VTS_nn_2.VOB ... in order (_0 is the menu and is left out)."""
    found = {}
    pattern = re.compile(rf"VTS_{vts:02d}_(\d)\.VOB$", re.IGNORECASE)
    for entry in video_ts.iterdir():
        m = pattern.match(entry.name)
        if m and int(m.group(1)) >= 1:
            found[int(m.group(1))] = entry
    return [found[k] for k in sorted(found)]


def _dvd_pgcs(ifo: bytes) -> Iterator[tuple[int, float, list[tuple[int, int]]]]:
    """(program chain number, seconds, sector runs) for each chain in a VTS_nn_0.IFO."""
    if ifo[:12] != b"DVDVIDEO-VTS" or len(ifo) < 0xD0:
        return
    pgcit = struct.unpack(">I", ifo[0xCC:0xD0])[0] * SECTOR             # VTS_PGCITI
    if not pgcit or pgcit + 8 > len(ifo):
        return
    count = struct.unpack(">H", ifo[pgcit:pgcit + 2])[0]
    for n in range(count):
        entry = pgcit + 8 + n * 8
        if entry + 8 > len(ifo):
            return
        offset = struct.unpack(">I", ifo[entry + 4:entry + 8])[0]
        pgc = pgcit + offset
        if pgc + 0xEE > len(ifo):
            continue
        cells = ifo[pgc + 3]
        seconds = _bcd_time(ifo[pgc + 4:pgc + 8])
        playback = pgc + struct.unpack(">H", ifo[pgc + 0xE8:pgc + 0xEA])[0]
        runs: list[tuple[int, int]] = []
        for c in range(cells):
            cell = playback + c * 24
            if cell + 24 > len(ifo):
                break
            category = ifo[cell]
            block_mode, block_type = category >> 6, (category >> 4) & 3
            if block_type == 1 and block_mode >= 2:                      # angle 2, 3 ...: the first angle is enough
                continue
            first, last = struct.unpack(">II", ifo[cell + 8:cell + 12] + ifo[cell + 20:cell + 24])
            if last >= first:
                runs.append((first, last))
        if runs:
            yield n + 1, seconds, runs


def dvd_titles(root: Path) -> list[Title]:
    video_ts = _child(root, "VIDEO_TS")
    if video_ts is None:
        return []
    titles: list[Title] = []
    sets = sorted({int(m.group(1)) for e in video_ts.iterdir()
                   if (m := re.match(r"VTS_(\d\d)_[1-9]\.VOB$", e.name, re.IGNORECASE))})
    for vts in sets:
        parts = _vob_parts(video_ts, vts)
        ifo = next((e for e in video_ts.iterdir() if e.name.upper() == f"VTS_{vts:02d}_0.IFO"), None)
        table: list[tuple[int, float, list[tuple[int, int]]]] = []
        if ifo is not None:
            try:
                table = list(_dvd_pgcs(ifo.read_bytes()))
            except (OSError, struct.error):
                table = []
        total = sum(p.stat().st_size for p in parts)
        if not table:
            # No usable table: offer the whole title set as one title, its length estimated from the file.
            titles.append(Title(len(titles) + 1, DVD, 0.0, total, parts, [], f"Title set {vts}"))
            continue
        for pgc, seconds, runs in table:
            if seconds < MIN_TITLE_SECONDS:
                continue
            nbytes = sum((b - a + 1) * SECTOR for a, b in runs)
            titles.append(Title(len(titles) + 1, DVD, seconds, nbytes, parts, runs))
    return titles


def _mpls_items(data: bytes) -> list[tuple[str, float]]:
    """(clip name, seconds) for each play item in a Blu-ray playlist file."""
    if data[:4] != b"MPLS" or len(data) < 20:
        return []
    start = struct.unpack(">I", data[8:12])[0]
    if start + 10 > len(data):
        return []
    count = struct.unpack(">H", data[start + 6:start + 8])[0]
    pos, items = start + 10, []
    for _ in range(count):
        if pos + 22 > len(data):
            break
        length = struct.unpack(">H", data[pos:pos + 2])[0]
        clip = data[pos + 2:pos + 7].decode("ascii", "replace")
        t_in, t_out = struct.unpack(">II", data[pos + 14:pos + 22])
        items.append((clip, max(0, t_out - t_in) / 45000.0))
        pos += 2 + length
    return items


def bluray_titles(root: Path) -> list[Title]:
    bdmv = _child(root, "BDMV")
    playlists = _child(bdmv, "PLAYLIST") if bdmv else None
    streams = _child(bdmv, "STREAM") if bdmv else None
    if playlists is None or streams is None:
        return []
    clips = {e.name.upper(): e for e in streams.iterdir()}
    found: list[Title] = []
    seen: set[tuple[str, ...]] = set()
    for entry in sorted(playlists.iterdir()):
        if entry.suffix.upper() != ".MPLS":
            continue
        try:
            items = _mpls_items(entry.read_bytes())
        except OSError:
            continue
        names = tuple(f"{clip}.M2TS" for clip, _ in items)
        parts = [clips[n] for n in names if n in clips]
        seconds = sum(s for _, s in items)
        if len(parts) != len(names) or not parts or seconds < MIN_TITLE_SECONDS or names in seen:
            continue
        seen.add(names)
        found.append(Title(0, BLURAY, seconds, sum(p.stat().st_size for p in parts), parts, [],
                           f"Playlist {entry.stem}"))
    found.sort(key=lambda t: -t.seconds)
    for i, title in enumerate(found, 1):
        title.number = i
    return found


def list_titles(root: Path, kind: str) -> list[Title]:
    """The titles worth offering, longest first (the main feature is nearly always the first)."""
    try:
        titles = dvd_titles(root) if kind == DVD else bluray_titles(root)
    except OSError as error:
        raise DiscError(f"The disc could not be read: {error}") from error
    titles.sort(key=lambda t: -(t.seconds or t.nbytes / 4e6))
    for i, title in enumerate(titles, 1):
        title.number = i
    return titles


# --- the byte stream of one title -------------------------------------------------------------------


class TitleReader:
    """A read-only, forward-only file object over a title: whole files back to back, or DVD sector runs."""

    def __init__(self, title: Title, on_bytes: Callable[[int], None] | None = None, skip: int = 0) -> None:
        self._on_bytes = on_bytes
        self.done = 0
        self._files = iter(title.parts)
        self._runs: list[tuple[Path, int, int]] = []          # (file, start byte, end byte) to read, in order
        if title.cells:
            sizes = [p.stat().st_size for p in title.parts]
            starts = [sum(sizes[:i]) for i in range(len(sizes))]
            for first, last in title.cells:
                lo, hi = first * SECTOR, (last + 1) * SECTOR
                for path, size, base in zip(title.parts, sizes, starts):
                    a, b = max(lo, base), min(hi, base + size)
                    if a < b:
                        self._runs.append((path, a - base, b - base))
        else:
            self._runs = [(p, 0, p.stat().st_size) for p in title.parts]
        while skip > 0 and self._runs:                            # start part-way in (a preview sample)
            path, a, b = self._runs[0]
            if skip >= b - a:
                skip -= b - a
                self._runs.pop(0)
            else:
                self._runs[0] = (path, a + skip, b)
                skip = 0
        self._index = 0
        self._handle = None
        self._left = 0

    def _advance(self) -> bool:
        if self._handle:
            self._handle.close()
            self._handle = None
        if self._index >= len(self._runs):
            return False
        path, a, b = self._runs[self._index]
        self._index += 1
        self._handle = open(path, "rb")
        self._handle.seek(a)
        self._left = b - a
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = 1 << 20
        out = []
        while size > 0:
            if self._left <= 0 and not self._advance():
                break
            data = self._handle.read(min(size, self._left))
            if not data:
                self._left = 0
                continue
            self._left -= len(data)
            size -= len(data)
            out.append(data)
        chunk = b"".join(out)
        self.done += len(chunk)
        if self._on_bytes and chunk:
            self._on_bytes(self.done)
        return chunk

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


def is_scrambled(title: Title) -> bool:
    """Whether a title's data is scrambled (copy-protected) rather than plain video, judged from its first few MB.

    Plain Blu-ray clips are transport streams with a sync byte 0x47 at the start of every 192-byte packet;
    encryption turns nearly all of those into noise. On a DVD the sectors carry a "scrambled" flag in their
    PES header. Deciding this up front takes a moment; letting FFmpeg struggle with noise takes a minute.
    """
    reader = TitleReader(title)
    try:
        data = reader.read(4 << 20)
    finally:
        reader.close()
    if len(data) < 64 * 192:
        return False                                            # too little to judge: let the real copy decide
    if title.kind == BLURAY:
        packets = len(data) // 192
        good = sum(1 for i in range(packets) if data[i * 192 + 4] == 0x47)
        return good < packets * 0.9
    video_pes = bytes((0, 0, 1, 0xE0))
    sectors = len(data) // SECTOR
    flagged = sum(1 for i in range(sectors)
                  if data[i * SECTOR + 14:i * SECTOR + 18] == video_pes and data[i * SECTOR + 20] & 0x30)
    return flagged > 0


PROTECTED = ("This disc is copy-protected, and this version of the app cannot read it. "
             "Discs without protection, and folders already copied to your computer, work.")


# --- copying ----------------------------------------------------------------------------------------


class _Retimer:
    """Keeps timestamps counting up when the disc's own clock restarts (a new cell, a new clip).

    A restart is an *event* with one shift, shared by every stream so sound and picture stay in step. But the
    streams do not reach it together: in the file, the picture runs ahead of the sound by a fraction of a
    second, so the picture has crossed while the sound is still finishing the old cell. Each stream therefore
    keeps its own place in the list of events and only takes the next one when its own timestamps restart.
    The first stream to restart creates the event; the others join it if their restart is close to it in the
    disc's clock.
    """

    JUMP_BACK = 0.25           # seconds a timestamp may step backwards before it is a restart
    JUMP_FORWARD = 8.0
    SAME_EVENT = 2.0

    def __init__(self) -> None:
        self.end = 0.0                                         # where the output has reached, seconds
        self._events: list[tuple[float, float]] = [(0.0, 0.0)]  # (shift, raw time of the restart)
        self._last: dict[int, float] = {}                      # stream index -> last raw dts, seconds
        self._on: dict[int, int] = {}                          # stream index -> the event it is on

    def offset(self, index: int, raw: float) -> float:
        last, event = self._last.get(index), self._on.get(index, 0)
        if last is not None and (raw < last - self.JUMP_BACK or raw > last + self.JUMP_FORWARD):
            following = event + 1
            if following < len(self._events) and abs(raw - self._events[following][1]) < self.SAME_EVENT:
                event = following
            else:
                self._events.append((self.end - raw, raw))
                event = len(self._events) - 1
            self._on[index] = event
        self._last[index] = raw
        return self._events[event][0]

    def moved(self, out_seconds: float) -> None:
        self.end = max(self.end, out_seconds)


def _open_title(title: Title, reader):
    """Open a title's byte stream as a demuxer. Raises DiscError if it is not video FFmpeg can read."""
    import av

    fmt = "mpeg" if title.kind == DVD else "mpegts"
    try:
        source = av.open(reader, mode="r", format=fmt,
                         options={"probesize": "20000000", "analyzeduration": "20000000"})
    except Exception as error:  # noqa: BLE001
        raise DiscError(_unreadable(title)) from error
    if not source.streams.video:
        source.close()
        raise DiscError(_unreadable(title))
    return source


def _retimed(source, streams, should_stop):
    """Yield (packet, seconds_from_start) for the given streams with timestamps that only ever count up.

    Sequence-end (flush) packets, which have no timestamp, are passed through with ``None``: the decoder needs
    them to release its last frames. The seconds are measured from the first picture, so the output starts at 0.
    """
    retimer = _Retimer()
    zero = None
    held = []                                                   # sound that arrives before the first picture
    lead = source.streams.video[0].index

    def stamp(packet):
        base = packet.time_base or packet.stream.time_base
        raw = float(packet.dts * base)
        shift = retimer.offset(packet.stream.index, raw)
        ticks = int(round(shift / float(base)))
        packet.dts += ticks
        if packet.pts is not None:
            packet.pts += ticks
        retimer.moved(float(packet.dts * base) + float((packet.duration or 0) * base))
        return float(packet.dts * base)

    for packet in source.demux(*streams):
        if should_stop and should_stop():
            raise Stopped
        if packet.dts is None:
            yield packet, None
            continue
        seconds = stamp(packet)
        if zero is None:
            if packet.stream.index != lead:
                held.append((packet, seconds))
                continue
            if not packet.is_keyframe:                          # a sample can start mid-picture-group: undecodable
                continue
            zero = seconds
            for early, at in held:
                if at >= zero:                                  # sound from before the first picture is dropped
                    yield _rebased(early, zero), at - zero
            held.clear()
        elif seconds < zero and packet.stream.index != lead:
            continue
        yield _rebased(packet, zero), seconds - zero


def _rebased(packet, zero: float):
    """Shift a packet so the first picture sits at time 0."""
    base = float(packet.time_base or packet.stream.time_base)
    ticks = int(round(zero / base))
    packet.dts -= ticks
    if packet.pts is not None:
        packet.pts -= ticks
    return packet


def copy_title(
    title: Title,
    destination: Path,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
    start_fraction: float = 0.0,
    max_seconds: float | None = None,
) -> Path:
    """Copy one title to a Matroska file, streams untouched. Raises DiscError, or Stopped if cancelled.

    ``start_fraction`` and ``max_seconds`` take a short piece from part-way in (a preview sample) instead.
    """
    from . import video

    if not video.is_available():
        raise DiscError("The video component is not installed yet.")
    import av

    if is_scrambled(title):
        raise DiscError(PROTECTED)
    total = max(1, title.nbytes)
    sampling = max_seconds is not None
    if sampling:
        total = max(1, min(total, int(title.nbytes * max_seconds / max(1.0, title.seconds or 1e9)) + (8 << 20)))
    skip = int(title.nbytes * start_fraction) // ALIGN * ALIGN
    reader = TitleReader(title, (lambda done: progress(min(done, total), total)) if progress else None, skip)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _open_title(title, reader) as source:
            with av.open(str(partial), mode="w", format="matroska") as out:
                mapping = {}
                for stream in source.streams:
                    if stream.type not in ("video", "audio", "subtitle"):
                        continue
                    try:
                        mapping[stream.index] = out.add_stream_from_template(stream)
                    except Exception:  # noqa: BLE001 - a stream Matroska will not take is left behind
                        continue
                wanted = [s for s in source.streams if s.index in mapping]
                for packet, seconds in _retimed(source, wanted, should_stop):
                    if seconds is None or packet.stream.index not in mapping:
                        continue
                    if sampling and seconds > max_seconds:
                        break
                    packet.stream = mapping[packet.stream.index]
                    out.mux(packet)
    except (Stopped, DiscError):
        _remove(partial)
        raise
    except Exception as error:  # noqa: BLE001
        _remove(partial)
        raise DiscError(f"{_unreadable(title)} ({error})") from error
    finally:
        reader.close()
    os.replace(partial, destination)
    if progress:
        progress(total, total)
    return destination


class DiscStream:
    """A title converted straight from the disc, without keeping a copy of it on disk.

    Used in place of a file path by :func:`pipeline.convert_video`. The disc is read once, front to back:
    :meth:`frames` decodes the picture and, in the same pass, writes the (small) soundtrack to a temporary
    file that the pipeline muxes onto the result afterwards. Nothing here can seek, so it always converts the
    whole title.
    """

    def __init__(self, title: Title, label: str, scratch: Path) -> None:
        self.title = title
        self.stem = f"{safe_name(label)} - {safe_name(title.name)}"
        self._audio = scratch / f"{self.stem}.audio.mkv"
        self.audio_written = False
        self.info = self._probe()

    def _probe(self):
        from . import video

        reader = TitleReader(self.title)
        try:
            with _open_title(self.title, reader) as source:
                info = video.describe(source)
        finally:
            reader.close()
        if self.title.seconds:
            info.duration = self.title.seconds
            info.frames = max(1, round(self.title.seconds * info.fps))
        return info

    @property
    def audio_path(self) -> Path | None:
        return self._audio if self.audio_written else None

    def discard(self) -> None:
        self.audio_written = False
        _remove(self._audio)

    def frames(self, start: int = 0, limit: int | None = None, should_stop=None, on_bytes=None):
        """Yield (index, rgb) for every frame of the title; ``start`` and ``limit`` count from the top."""
        import av

        from . import video

        self.discard()
        total = max(1, self.title.nbytes)
        reader = TitleReader(self.title, (lambda done: on_bytes(min(done, total), total)) if on_bytes else None)
        out = None
        try:
            with _open_title(self.title, reader) as source:
                vstream = source.streams.video[0]
                vstream.thread_type = "AUTO"
                sar = vstream.sample_aspect_ratio or vstream.codec_context.sample_aspect_ratio
                high = video._bit_depth(vstream.codec_context.pix_fmt or "") > 8
                astream = source.streams.audio[0] if source.streams.audio else None
                a_out = None
                if astream is not None:
                    out = av.open(str(self._audio), mode="w", format="matroska")
                    try:
                        a_out = out.add_stream_from_template(astream)
                    except Exception:  # noqa: BLE001 - not storable: convert without sound
                        out.close()
                        out, a_out, astream = None, None, None
                wanted = [vstream] + ([astream] if astream is not None else [])
                index, emitted = 0, 0
                for packet, seconds in _retimed(source, wanted, should_stop):
                    if packet.stream is vstream:
                        for frame in packet.decode():
                            if index >= start:
                                if limit is not None and emitted >= limit:
                                    return
                                yield index, video.frame_to_rgb(frame, high, sar=sar)
                                emitted += 1
                            index += 1
                    elif a_out is not None and seconds is not None:
                        packet.stream = a_out
                        out.mux(packet)
        except Stopped:
            return
        finally:
            reader.close()
            if out is not None:
                try:
                    out.close()
                    self.audio_written = True
                except Exception:  # noqa: BLE001
                    self.audio_written = False


def _unreadable(title: Title) -> str:
    return (f"{title.name} could not be read as video. The disc may be scratched or dirty, or it may be "
            "protected in a way this version cannot read.")


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def safe_name(text: str) -> str:
    """A file name from a disc label: letters, digits, dashes, spaces."""
    cleaned = re.sub(r"[^\w\- ]+", "", text).strip().replace("_", " ")
    return cleaned or "Disc"
