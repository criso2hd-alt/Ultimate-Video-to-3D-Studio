"""Look-ahead buffer for the 3D preview: render seconds ahead, play from the cache.

The live preview renders each frame as its moment arrives and drops any it cannot
finish in time. That is right for scrubbing and for adjusting a slider on a paused
picture - you want the answer *now* - but it is exactly what makes real-time
playback stutter: every slow frame is a visible hitch.

Playing is different. Nobody needs a frame the instant it is made, only at the
moment the audio clock reaches it. So while a 3D preview plays, a background
pipeline renders **ahead** of the playhead (5 or 10 seconds) and stores each
finished frame; a separate display thread just picks the frame the clock is on.
A slow frame now costs nothing unless the whole buffer drains, and then playback
pauses with a "Buffering" note and resumes when there is enough ahead - like any
streaming player - instead of stuttering.

Finished frames are kept as JPEGs (about 130 KB at 720p, so ten seconds is ~40 MB
where raw pictures would be ~800 MB). Any change to a setting that alters the
picture clears the cache and refills it from the current position; the paused
still updates instantly through the live path in the meantime.

The pipeline is the live preview's own render step (``preview.render_frame``), so a
buffered frame is the same picture the live path would have shown.
"""

from __future__ import annotations

import hashlib
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import depth as depthmod
from . import grade, video
from .preview import PreviewParams, render_frame

#: Playback starts once this much is rendered ahead ...
START_SECONDS = 1.0
#: ... after a seek, a little more ...
SEEK_SECONDS = 1.5
#: ... and after an *underrun* a bigger cushion, so one slow moment does not turn into a
#: run of short pauses: better a single, slightly longer wait than several brief ones.
UNDERRUN_SECONDS = 2.0
#: The preview frame-rate ladder, best first. The buffered preview is rendered on one of
#: these grids at most (a clip slower than the grid is untouched, and the export always
#: renders every frame). Which rung is used depends on this machine - see ``Tuning``.
FPS_LADDER = (30.0, 24.0, 15.0)
#: Rendering throughput (frames/s, with the fast fill settings) a rung needs: about
#: 1.7x the playback rate for 30, 1.4x for 24 - enough headroom that one slow moment
#: does not drain the buffer.
CAPABILITY_FOR = {30.0: 52.0, 24.0: 34.0}
#: Below this the machine cannot play a 3D preview in real time at all.
TOO_SLOW_FPS = 9.0
#: Parallel render workers in the look-ahead pipeline.
RENDER_WORKERS = 2
#: While less than this fraction of the look-ahead window is filled, the depth model runs
#: on alternate frames and the previous depth is reused in between. The ONNX call is the
#: pipeline's bottleneck (~25 ms under load, and resolution does not change it - the
#: network is a fixed 518 px), so this roughly doubles the fill rate exactly when the
#: buffer needs to grow. Once it is healthy, every frame gets its own depth again. The
#: edge-refine step re-aligns reused depth to each frame's own picture.
REUSE_BELOW_FRACTION = 0.6
#: Frames kept behind the playhead, so a small step back is instant.
KEEP_BEHIND_SECONDS = 1.0
JPEG_QUALITY = 90
#: While the video is *paused* the renderer keeps going, this far ahead of the playhead, and
#: writes the pictures to a cache folder on disk - so pressing Play starts at once and the
#: buffer bar under the timeline keeps growing. (While playing, the Settings "buffer" applies.)
PAUSED_FILL_SECONDS = 30.0
#: The disk cache never grows past this (all looks together): the looks not on screen go first,
#: then the frames farthest from the playhead.
CACHE_LIMIT_BYTES = 1 << 30
#: ... and it never eats into the last of the disk: it stops growing when less than this is free.
LOW_DISK_RESERVE = 2 << 30
#: How many different looks (Output, Anaglyph, a colour tweak ...) keep their own cache at once.
MAX_LOOKS = 12


class Tuning:
    """What frame rate the buffered preview renders at, chosen for *this* machine.

    ``Auto`` measures how fast the pipeline actually produces frames during the first
    fill and picks the highest rung it can sustain with headroom: a fast PC gets 30 fps,
    a mid-range one 24, a slow one 15. If playback ever underruns anyway it steps down a
    rung (and finally asks for a smaller preview). It never steps back up on its own, so
    it cannot oscillate. A fixed choice (30 or 24) is honoured as it is.
    """

    def __init__(self) -> None:
        self.choice = 0                      # 0 = auto, else 24 / 30
        self.fps = 24.0                      # the grid in use
        self.capability: float | None = None
        self._stamps: list[float] = []
        self.measured = False

    def reset_measurement(self) -> None:
        self._stamps, self.measured = [], False

    def set_choice(self, choice: int) -> None:
        self.choice = int(choice)
        if self.choice:
            self.fps = float(self.choice)
        elif not self.measured:
            self.fps = 24.0

    def note_frame(self, now: float) -> float | None:
        """Record a finished frame. Returns the measured capability once, in auto mode."""
        if self.measured:
            return None
        self._stamps.append(now)
        if len(self._stamps) < 26:
            return None
        window = self._stamps[2:]                             # skip the pipeline's warm-up frames
        span = window[-1] - window[0]
        self.measured = True
        if span <= 0:
            return None
        self.capability = (len(window) - 1) / span
        return self.capability

    def apply_capability(self, capability: float) -> float:
        """Pick the rung this machine can sustain (auto mode only). Returns the chosen fps."""
        if not self.choice:
            for fps in FPS_LADDER[:-1]:
                if capability >= CAPABILITY_FOR[fps]:
                    self.fps = fps
                    break
            else:
                self.fps = FPS_LADDER[-1]
        return self.fps

    def step_down(self) -> bool:
        """Move one rung down. False if already at the bottom."""
        lower = [f for f in FPS_LADDER if f < self.fps]
        if not lower:
            return False
        self.fps = lower[0]
        return True


@dataclass
class CachedFrame:
    """One finished preview picture. In memory (``jpegs``) or, once the cache spills it, on disk (``files``)."""

    jpegs: list[bytes]
    badge: str
    wiggle: bool
    hole_pct: float
    scope: bytes = b""                                  # a small JPEG of the graded frame, for the scopes
    files: list[Path] = field(default_factory=list)     # set when stored on disk
    scope_file: Path | None = None
    size: int = 0                                       # bytes held (memory or disk)

    def load(self) -> tuple[list[bytes], bytes] | None:
        """(picture JPEGs, scope JPEG), or None if the files are gone (the cache was just cleared)."""
        if not self.files:
            return self.jpegs, self.scope
        try:
            scope = self.scope_file.read_bytes() if self.scope_file is not None else b""
            return [p.read_bytes() for p in self.files], scope
        except OSError:
            return None

    def delete_files(self) -> None:
        for path in (*self.files, self.scope_file):
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass


def _encode(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not compress a preview frame")
    return data.tobytes()


def _decode(data: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def cleanup_stale_caches(root: Path, keep: Path | None = None) -> None:
    """Delete cache folders left behind by runs that crashed (a folder is named after its process id)."""
    try:
        import psutil

        for folder in root.iterdir():
            if folder == keep or not folder.is_dir():
                continue
            if not folder.name.isdigit() or not psutil.pid_exists(int(folder.name)):
                shutil.rmtree(folder, ignore_errors=True)
    except Exception:  # noqa: BLE001 - housekeeping must never block start-up
        pass


class RenderCache:
    """Finished preview frames by frame index. Thread-safe.

    With a ``directory`` the pictures are written to disk (as JPEG files) and only their
    names are kept in memory, so a long stretch of pre-rendered video costs disk, not RAM.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._frames: dict[int, CachedFrame] = {}
        self._dir = directory
        self._bytes = 0
        #: Bumped by every clear(). A renderer remembers the number it started under and its
        #: pictures are refused once it is stale, so a run that was told to stop (its
        #: settings changed) cannot slip a last old-looking frame into the emptied cache.
        self.generation = 0
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def _spill(self, index: int, frame: CachedFrame, generation: int) -> CachedFrame:
        """Write the frame's pictures to disk and return the light, file-backed version."""
        files = []
        for n, data in enumerate(frame.jpegs):
            path = self._dir / f"{generation}_{index:07d}_{n}.jpg"
            path.write_bytes(data)
            files.append(path)
        scope_file = None
        if frame.scope:
            scope_file = self._dir / f"{generation}_{index:07d}_s.jpg"
            scope_file.write_bytes(frame.scope)
        size = sum(len(d) for d in frame.jpegs) + len(frame.scope)
        return CachedFrame([], frame.badge, frame.wiggle, frame.hole_pct, b"", files, scope_file, size)

    def put(self, index: int, frame: CachedFrame, generation: int | None = None) -> None:
        """Store a finished frame. ``generation`` is the clear-count the renderer started under."""
        gen = self.generation if generation is None else generation
        if gen != self.generation:
            return                                       # from before the last clear(): stale
        if self._dir is not None:
            try:
                frame = self._spill(index, frame, gen)
            except OSError:
                pass                                     # disk full or gone: keep this one in memory
        if not frame.size:
            frame.size = sum(len(d) for d in frame.jpegs) + len(frame.scope)
        with self._lock:
            stale = gen != self.generation               # cleared while it was being written
            if not stale:
                old = self._frames.get(index)
                self._frames[index] = frame
                self._bytes += frame.size - (old.size if old else 0)
        if stale:
            frame.delete_files()

    def get(self, index: int) -> CachedFrame | None:
        with self._lock:
            return self._frames.get(index)

    def has(self, index: int) -> bool:
        with self._lock:
            return index in self._frames

    def _remove(self, keys: list[int]) -> None:
        with self._lock:
            gone = [self._frames.pop(k) for k in keys if k in self._frames]
            self._bytes -= sum(f.size for f in gone)
        for frame in gone:                                # file deletion happens outside the lock
            frame.delete_files()

    def clear(self) -> None:
        """Empty the cache *now*; the files are deleted in the background (a second of stalling the
        window, when hundreds of them are cached, is not worth waiting for)."""
        with self._lock:
            old, self._frames, self._bytes = list(self._frames.values()), {}, 0
            self.generation += 1
        if len(old) > 16:
            threading.Thread(target=lambda: [f.delete_files() for f in old], daemon=True,
                             name="uv3d-cache-reaper").start()
        else:
            for frame in old:
                frame.delete_files()

    def destroy(self) -> None:
        """Clear, and remove the cache folder itself (the app is closing)."""
        self.clear()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def enforce_limit(self, centre: int, max_bytes: int) -> None:
        """Over the size cap: drop the frames farthest from ``centre`` (the playhead) first."""
        if self._bytes <= max_bytes:
            return
        with self._lock:
            keys = sorted(self._frames, key=lambda k: abs(k - centre), reverse=True)
        total, drop = self._bytes, []
        for k in keys:
            if total <= max_bytes * 0.9:
                break
            frame = self.get(k)
            if frame is not None:
                total -= frame.size
                drop.append(k)
        self._remove(drop)

    def ahead(self, index: int, limit: int, max_gap: int = 3) -> int:
        """How many frame slots from ``index`` are covered by rendered frames (capped at ``limit``).

        A gap of up to ``max_gap`` missing frames is tolerated - the renderer skips
        alternate frames while it catches up, and the display holds the previous
        picture across the gap - but a longer hole ends the run.
        """
        with self._lock:
            last = -1
            for n in range(limit):
                if (index + n) in self._frames:
                    last = n
                elif n - last > max_gap:
                    break
            return last + 1

    def evict_before(self, index: int) -> None:
        with self._lock:
            keys = [k for k in self._frames if k < index]
        self._remove(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class BufferedRenderer:
    """Decode -> depth -> render -> compress, on three threads, filling the cache ahead of the clock."""

    def __init__(self, path: Path, info: video.VideoInfo, cache: RenderCache, clock, proxy_of,
                 tuning: "Tuning | None" = None, on_capability=None, housekeeping=None) -> None:
        self._path, self._info, self._cache, self._clock = path, info, cache, clock
        self._housekeeping = housekeeping                # called now and then with the playhead's frame index
        self.tuning = tuning or Tuning()
        self._on_capability = on_capability
        self._proxy_of = proxy_of                       # (av frame, high) -> proxy rgb, shared with the live feeder
        self._engine: depthmod.DepthEngine | None = None
        self._grader = grade.GradeEngine()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lifecycle = threading.RLock()
        self.finished = False                           # decoded to the end of the clip
        self.error = ""
        self.lookahead_s = 5.0
        #: What this run was started with, so a caller can tell whether it is still the right one.
        self.params: PreviewParams | None = None
        self.origin_ms = 0.0
        self._gen = 0
        self._puts = 0

    def _index(self, ms: float) -> int:
        return int(round(ms * (self._info.fps or 24.0) / 1000.0))

    def request_stop(self) -> None:
        """Ask the run to end without waiting for it (the UI thread must not block on the join)."""
        self._stop.set()

    def covers(self, params: PreviewParams, at_ms: float, cache: RenderCache | None = None) -> bool:
        """Whether a run for ``params`` (writing into ``cache``) is already going or done and reaches
        ``at_ms`` - it was started there, or its pictures already cover it - so there is no need to
        restart it."""
        return ((self.alive or self.finished) and not self._stop.is_set() and self.params == params
                and (cache is None or cache is self._cache)
                and self._gen == self._cache.generation
                and (abs(self.origin_ms - at_ms) < 1.0 or self._cache.has(self._index(at_ms))))

    def start(self, params: PreviewParams, start_ms: float, cache: RenderCache | None = None) -> None:
        """(Re)start filling ``cache`` (default: the one it already had) from ``start_ms``."""
        with self._lifecycle:
            self._start(params, start_ms, cache)

    def _start(self, params: PreviewParams, start_ms: float, cache: RenderCache | None) -> None:
        self.stop()
        if cache is not None:
            self._cache = cache                          # only swapped once the old run has fully ended
        self._stop = threading.Event()
        self.finished, self.error = False, ""
        self.params, self.origin_ms = params, float(start_ms)
        self._gen = self._cache.generation
        stop = self._stop
        q_depth: queue.Queue = queue.Queue(maxsize=3)
        q_render: queue.Queue = queue.Queue(maxsize=3)
        self._threads = [
            threading.Thread(target=self._decode, args=(stop, start_ms, q_depth, self._cache), daemon=True,
                             name="uv3d-buf-decode"),
            threading.Thread(target=self._depth, args=(stop, params, q_depth, q_render), daemon=True, name="uv3d-buf-depth"),
        ] + [
            # Two render workers: refine, JPEG compression and colour work overlap
            # (only the Numba kernels themselves take turns, via kernels.LOCK), which
            # lifts the whole pipeline from just under real time to comfortably over it.
            threading.Thread(target=self._render, args=(stop, params, q_render, self._cache, self._gen), daemon=True,
                             name=f"uv3d-buf-render-{i}")
            for i in range(RENDER_WORKERS)
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        with self._lifecycle:
            self.finished = False                       # a stopped run is not a completed one
            self._stop.set()
            for t in self._threads:
                if t.ident is not None and t is not threading.current_thread():
                    t.join(timeout=3.0)
            self._threads = []

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @staticmethod
    def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _get(q: queue.Queue, stop: threading.Event):
        while not stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    # -- stages ------------------------------------------------------------

    def _decode(self, stop: threading.Event, start_ms: float, out: queue.Queue, cache: RenderCache) -> None:
        import av

        try:
            container = av.open(str(self._path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base) if stream.time_base else 1.0 / 1000
            high = self._info.bit_depth > 8
            half_frame = 500.0 / (self._info.fps or 24.0)
            next_take = -1.0
            container.seek(int(start_ms / 1000.0 / tb), stream=stream, backward=True)
            for frame in container.decode(stream):
                if stop.is_set():
                    break
                ms = float(frame.pts) * tb * 1000.0 if frame.pts is not None else 0.0
                if ms + half_frame < start_ms:
                    continue                                     # decoding up from the keyframe
                grid = self.tuning.fps                            # read each frame: it can change mid-play
                interval = 1000.0 / grid if (self._info.fps or 24.0) > grid + 0.5 else 0.0
                if interval:
                    # Sample the clip on the preview frame-rate grid.
                    if next_take < 0:
                        next_take = ms
                    if ms + half_frame < next_take:
                        continue
                    next_take += interval
                    while next_take <= ms - half_frame:
                        next_take += interval
                index = self._index(ms)
                # Stay within the look-ahead window: do not race ahead of the playhead. (Checked before
                # the "already rendered" test so that a fully cached video is not decoded end to end.)
                while not stop.is_set() and ms - self._clock.now() > self.lookahead_s * 1000.0:
                    time.sleep(0.05)
                if stop.is_set():
                    break
                if cache.has(index):
                    continue                                     # already rendered (kept from earlier)
                if not self._put(out, (index, ms, self._proxy_of(frame, high)), stop):
                    break
            else:
                self.finished = True                             # decoded to the end of the clip
            container.close()
        except Exception as error:  # noqa: BLE001 - reported through .error
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self._put(out, None, stop)

    def _depth(self, stop: threading.Event, p: PreviewParams, src: queue.Queue, out: queue.Queue) -> None:
        try:
            self._engine = depthmod.shared_engine(p.force_cpu)
            stab = depthmod.DepthStabilizer(float(np.clip(p.temporal / 100.0 * 0.95, 0.0, 0.95)))
            steady = None
            reuse_turn = False
            while True:
                item = self._get(src, stop)
                if item is None:
                    break
                index, ms, frame = item
                filled = (ms - self._clock.now()) / (self.lookahead_s * 1000.0)
                if steady is not None and filled < REUSE_BELOW_FRACTION:
                    reuse_turn = not reuse_turn               # every other frame while the buffer is filling
                else:
                    reuse_turn = False
                if not (reuse_turn and steady is not None):
                    h, w = frame.shape[:2]
                    scale = 640.0 / max(h, w)
                    small = frame if scale >= 1.0 else cv2.resize(frame, (round(w * scale), round(h * scale)),
                                                                  interpolation=cv2.INTER_AREA)
                    steady = stab(self._engine.infer(small), small)
                if not self._put(out, (index, frame, steady), stop):
                    break
        except Exception as error:  # noqa: BLE001
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self._put(out, None, stop)

    def _render(self, stop: threading.Event, p: PreviewParams, src: queue.Queue, cache: RenderCache,
                gen: int) -> None:
        grader = grade.GradeEngine()                            # holds tone-table state: one per worker
        try:
            while True:
                item = self._get(src, stop)
                if item is None:
                    self._put(src, None, stop)                  # let the other worker see the end too
                    break
                index, frame, steady = item
                result = render_frame(p, frame, steady, grader)
                scope = _encode(result.scope, 85) if result.scope is not None else b""
                cache.put(index, CachedFrame(
                    [_encode(img) for img in result.images], result.badge, result.wiggle,
                    float(result.stats.get("hole_pct", 0.0)), scope), gen)
                self._puts += 1
                if self._puts % 60 == 0 and self._housekeeping is not None:
                    self._housekeeping(self._index(self._clock.now()))
                capability = self.tuning.note_frame(time.perf_counter())
                if capability is not None and self._on_capability is not None:
                    self._on_capability(capability)
        except Exception as error:  # noqa: BLE001
            self.error = f"{type(error).__name__}: {error}"


class BufferedPlayback(QObject):
    """Owns the cache, the renderer and the display thread; plays frames from the cache on the clock."""

    image = Signal(object)                  # (images, badge, wiggle, hole_pct)
    position = Signal(int)                  # ms of the frame just shown
    buffering = Signal(bool, float)         # (waiting for the buffer, progress 0..1)
    failed = Signal(str)
    notice = Signal(str)                    # plain-language message about what was tuned
    too_slow = Signal(float)                # measured fps: this machine cannot play 3D in real time
    smaller_preview_needed = Signal()       # even the lowest frame rate underran

    def __init__(self, clock, audio_play, audio_pause, parent: QObject | None = None,
                 cache_dir: Path | None = None) -> None:
        super().__init__(parent)
        self._clock = clock
        self._audio_play, self._audio_pause = audio_play, audio_pause
        self._root = cache_dir
        if cache_dir is not None:
            cleanup_stale_caches(cache_dir.parent, keep=cache_dir)
        #: One cache per look (mode + every setting that changes the picture + proxy size), kept
        #: until the video changes - so switching Output -> Anaglyph -> Output finds the first
        #: one still there. Oldest-used first. ``self.cache`` is the one on screen.
        self._caches: dict[str, RenderCache] = {}
        self._caches_lock = threading.RLock()
        self.proxy_height = 0
        self.cache = RenderCache()                       # an empty stand-in until a 3D look is chosen
        self._renderer: BufferedRenderer | None = None
        self._info: video.VideoInfo | None = None
        self._params: PreviewParams | None = None
        self._active = False
        self._state = "idle"                # "idle" | "buffering" | "playing"
        self._quit = False
        self._last_shown = -1
        self._resume_s = START_SECONDS
        self._duration_ms = 0.0
        self._restart_timer: threading.Timer | None = None
        self.lookahead_s = 5.0
        self.tuning = Tuning()
        self._thread = threading.Thread(target=self._display, daemon=True, name="uv3d-buf-display")
        self._thread.start()

    # -- setup ---------------------------------------------------------------

    def _select(self, params: PreviewParams) -> RenderCache:
        """Make the cache for this look the current one (creating it if it is new). Nothing is deleted."""
        key = hashlib.sha1(f"{params!r}|{self.proxy_height}".encode()).hexdigest()[:12]
        with self._caches_lock:
            cache = self._caches.pop(key, None)
            if cache is None:
                cache = RenderCache(self._root / key if self._root is not None else None)
            self._caches[key] = cache                    # most recently used goes last
            self.cache = cache
            while len(self._caches) > MAX_LOOKS:
                oldest = next(iter(self._caches))
                self._caches.pop(oldest).destroy()
        return cache

    def _allowed_bytes(self, total: int) -> int:
        try:
            free = shutil.disk_usage(self._root or Path.cwd()).free
        except OSError:
            return CACHE_LIMIT_BYTES
        return min(CACHE_LIMIT_BYTES, total + max(0, free - LOW_DISK_RESERVE))

    def _housekeeping(self, centre: int) -> None:
        """Keep the cache folder within its size cap and off the last of the disk: drop the looks that
        are not on screen (oldest first), then the frames farthest from the playhead."""
        with self._caches_lock:
            total = sum(c.nbytes for c in self._caches.values())
            cap = self._allowed_bytes(total)
            if total <= cap:
                return
            for key in list(self._caches):
                if total <= cap * 0.9:
                    break
                if self._caches[key] is self.cache:
                    continue
                gone = self._caches.pop(key)
                total -= gone.nbytes
                gone.destroy()
            if total > cap:
                self.cache.enforce_limit(centre, max(0, self.cache.nbytes - int(total - cap * 0.9)))

    def open(self, path: Path, info: video.VideoInfo, proxy_of) -> None:
        """A new video: the only time caches are thrown away (they are of the old picture)."""
        self.halt()
        with self._caches_lock:
            for cache in self._caches.values():
                cache.destroy()
            self._caches.clear()
            self.cache = RenderCache()
        self._info = info
        self._duration_ms = info.duration * 1000.0
        self.tuning.reset_measurement()
        self._renderer = BufferedRenderer(path, info, self.cache, self._clock, proxy_of,
                                          self.tuning, self._capability_measured, self._housekeeping)
        self._renderer.lookahead_s = self.lookahead_s

    @property
    def state(self) -> str:
        return self._state

    @property
    def playing_from_cache(self) -> bool:
        return self._active and self._state == "playing"

    def _index(self, ms: float) -> int:
        return int(round(ms * ((self._info.fps if self._info else 0) or 24.0) / 1000.0))

    def _capability_measured(self, capability: float) -> None:
        """First fill finished: pick the frame rate this machine can sustain."""
        if capability < TOO_SLOW_FPS:
            self.too_slow.emit(capability)
            return
        before = self.tuning.fps
        fps = self.tuning.apply_capability(capability)
        if not self.tuning.choice:
            self.notice.emit(
                f"Preview set to {fps:g} fps automatically for this computer "
                f"(rendering {capability:.0f} fps).")
        del before

    def set_fps_choice(self, choice: int) -> None:
        self.tuning.set_choice(choice)

    def _window_s(self) -> float:
        """How far ahead the renderer may run: the buffer setting while playing, more while paused."""
        return self.lookahead_s if self._active else PAUSED_FILL_SECONDS

    def set_lookahead(self, seconds: float) -> None:
        self.lookahead_s = float(seconds)
        if self._renderer is not None:
            self._renderer.lookahead_s = self._window_s()

    # -- control ---------------------------------------------------------------

    def begin(self, params: PreviewParams, start_ms: float) -> None:
        """Start playing from ``start_ms``: fill the buffer, then run on the clock.

        If the video was paused here with the 3D preview on, the renderer has been working
        ahead the whole time and the cache already holds pictures: that run simply carries
        on, and playback starts as soon as a second of them is ready (usually at once).
        """
        if self._renderer is None:
            return
        self._params = params
        self._active = True
        self._cancel_timer()
        cache = self._select(params)
        self._enter_buffering(start_ms, START_SECONDS)
        renderer = self._renderer
        renderer.lookahead_s = self.lookahead_s
        if not renderer.covers(params, start_ms, cache):
            renderer.start(params, start_ms, cache)

    def settle(self) -> None:
        """Paused while playing from the cache: stop showing frames but keep rendering ahead."""
        self._active = False
        self._state = "idle"
        self._cancel_timer()
        if self._renderer is not None:
            self._renderer.lookahead_s = PAUSED_FILL_SECONDS

    def prefill(self, params: PreviewParams, at_ms: float) -> None:
        """Paused: render ahead of ``at_ms`` into the (disk) cache, and keep going. Safe to call repeatedly."""
        if self._renderer is None or self._active:
            return
        self._params = params
        self._cancel_timer()
        cache = self._select(params)
        renderer = self._renderer
        renderer.lookahead_s = PAUSED_FILL_SECONDS
        if not renderer.covers(params, at_ms, cache):
            renderer.start(params, at_ms, cache)

    def prefill_later(self, params: PreviewParams, at_ms: float) -> None:
        """Paused and the playhead moved (a scrub): restart the fill from the new spot once it settles."""
        if self._renderer is None or self._active:
            return
        self._params = params
        self._renderer.request_stop()
        self._start_timer(lambda: self.prefill(params, at_ms))

    def halt(self) -> None:
        """Stop playing and stop rendering (the cache is kept for a quick resume)."""
        self._active = False
        self._state = "idle"
        self._cancel_timer()
        if self._renderer is not None:
            self._renderer.stop()

    def seek(self, ms: float) -> None:
        if not self._active or self._renderer is None or self._params is None:
            return
        self._enter_buffering(ms, SEEK_SECONDS)
        self._renderer.start(self._params, ms, self.cache)

    def invalidate(self, params: PreviewParams) -> None:
        """The picture changed (a setting, another view, the proxy size): switch to that look's cache.

        Nothing is deleted - each look keeps what it has, so going back to a view that was already
        rendered finds it complete. The renderer then carries on filling the new look from here,
        starting a moment after the *last* change so dragging a slider does not restart the
        pipeline on every pixel of travel. Playing, it refills the buffer; paused, it fills the
        disk cache from the playhead.
        """
        self._params = params
        self._select(params)
        if self._renderer is None:
            return
        self._renderer.request_stop()                       # the old look's run winds down by itself
        self._cancel_timer()
        if self._active:
            self._enter_buffering(self._clock.now(), START_SECONDS)
            self._start_timer(self._restart_after_change)
        else:
            self._start_timer(self._refill_paused)

    def _start_timer(self, callback) -> None:
        self._cancel_timer()
        timer = threading.Timer(0.35, callback)
        timer.daemon = True
        self._restart_timer = timer
        timer.start()

    def _restart_after_change(self) -> None:
        if self._active and self._renderer is not None and self._params is not None:
            self._renderer.start(self._params, self._clock.now(), self._select(self._params))

    def _refill_paused(self) -> None:
        if not self._active and self._params is not None and self._params.needs_depth:
            self.prefill(self._params, self._clock.now())

    def _underrun_step_down(self) -> None:
        """Playback ran dry: ask for less work per second so it does not happen again."""
        if self.tuning.choice:
            return                                        # the user chose a fixed rate: honour it
        if self.tuning.step_down():
            self.notice.emit(f"Playback caught up with the buffer, so the preview is now {self.tuning.fps:g} fps.")
        else:
            self.smaller_preview_needed.emit()

    def _cancel_timer(self) -> None:
        timer, self._restart_timer = self._restart_timer, None
        if timer is not None:
            timer.cancel()

    def _enter_buffering(self, at_ms: float, resume_s: float) -> None:
        self._resume_s = resume_s
        self._audio_pause()
        self._clock.stop()
        self._clock.set(at_ms)
        self._last_shown = -1
        self._state = "buffering"

    def shutdown(self) -> None:
        self._quit = True
        self.halt()
        self._thread.join(timeout=2.0)
        with self._caches_lock:                              # the cache folder does not outlive the app
            for cache in self._caches.values():
                cache.destroy()
            self._caches.clear()
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)

    # -- display thread ----------------------------------------------------------

    def _display(self) -> None:
        while not self._quit:
            if not self._active or self._info is None:
                time.sleep(0.03)
                continue
            try:
                self._tick()
            except Exception as error:  # noqa: BLE001 - never let the display thread die silently
                self.failed.emit(f"{type(error).__name__}: {error}")
                time.sleep(0.2)

    def _tick(self) -> None:
        fps = self._info.fps or 24.0
        now = self._clock.now()
        index = self._index(now)
        last_index = self._index(self._duration_ms) if self._duration_ms else 10 ** 9
        renderer = self._renderer

        if renderer is not None and renderer.error:
            self.failed.emit(renderer.error)
            self.halt()
            return

        if self._state == "buffering":
            remaining = max(0, last_index - index)
            need = max(1, min(int(self._resume_s * fps), remaining + 1))
            ahead = self.cache.ahead(index, need)
            self.buffering.emit(True, min(1.0, ahead / need))
            if ahead >= need:
                self._state = "playing"
                self._clock.start(now)
                self._audio_play(now)
                self.buffering.emit(False, 1.0)
            else:
                time.sleep(0.04)
            return

        frame = None
        for back in (0, 1, 2, 3):
            frame = self.cache.get(index - back)
            if frame is not None:
                break
        if frame is None:
            at_end = index >= last_index or (
                renderer is not None and renderer.finished and not renderer.alive
                and self.cache.ahead(index, 1) == 0 and index > last_index - 3)
            if at_end:
                self.seek(0.0)                       # end of the clip: loop
            else:
                self._enter_buffering(now, UNDERRUN_SECONDS)   # underrun: pause and refill
                self._underrun_step_down()
                if renderer is not None and not renderer.alive:
                    renderer.start(self._params, now, self.cache)
            return

        if index != self._last_shown:
            loaded = frame.load()
            if loaded is None:
                time.sleep(0.01)                     # its files were just cleared: treat as not there yet
                return
            pictures, scope = loaded
            self._last_shown = index
            self.image.emit(([_decode(j) for j in pictures], frame.badge, frame.wiggle, frame.hole_pct,
                             _decode(scope) if scope else None))
            self.position.emit(int(now))
        wait = ((index + 1) * 1000.0 / fps - self._clock.now()) / 1000.0
        time.sleep(min(max(wait, 0.002), 0.02))
