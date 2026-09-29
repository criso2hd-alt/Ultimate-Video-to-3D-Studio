"""The live preview: a proxy of the video (720p at most) through the real 3D and colour maths.

The preview never touches the full-resolution frame. Each frame arrives already
shrunk to a proxy and goes through exactly the same grade, depth, refine and
stereo code the export uses - only smaller - so what is on screen is what the
export will produce, at a size the GPU keeps up with while the video plays.

Two worker threads, so the two slowest things overlap instead of adding up:

    depth thread   the neural network (GPU) and its temporal stabiliser
    render thread  grade, edge refine, stereo warp and the chosen view (CPU cores)

Each has a "newest job wins" mailbox: if frames arrive faster than a stage can
take them, the stale ones are dropped rather than queued, so the picture stays in
step with the audio instead of falling behind. While paused, moving a slider
re-runs only the render thread against the depth already computed for that frame.

The views exist to answer one question - "is the 3D actually working?" - from
different angles: anaglyph and the output layout for glasses and headsets;
wiggle, depth and parallax for checking without any glasses at all; holes to see
where the renderer had to invent picture; the 3D orbit to look at the geometry.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import depth as depthmod
from . import grade, stereo

#: Default proxy height. A 720p frame is a ninth the pixels of 4K, which is what
#: makes depth + warp + grade fit inside a frame time on a mid-range GPU.
PROXY_HEIGHT = 720
PROXY_CHOICES = (360, 480, 540, 720)

#: (key, label). "original" is the plain 2D picture - the effect switched off. Shown as the
#: permanent bar under the picture, in this order.
PREVIEW_MODES = (
    ("original", "2D"),
    ("output", "Output"),
    ("anaglyph", "Anaglyph"),
    ("sbs", "Side by side"),
    ("wiggle", "Wiggle"),
    ("depth", "Depth"),
    ("parallax", "Parallax"),
    ("holes", "Holes"),
    ("orbit", "3D orbit"),
)

#: What each view is for (the tooltips on the bar).
PREVIEW_MODE_TIPS = {
    "original": "The plain video, no 3D. It plays smoothest.",
    "output": "Exactly what the export will contain, in the layout chosen under 3D.",
    "anaglyph": "Red/cyan - see the depth with cheap paper glasses.",
    "sbs": "Left and right eye next to each other, full size.",
    "wiggle": "The two eyes alternate: depth you can judge without any glasses.",
    "depth": "The depth map. Warm = near, cool = far.",
    "parallax": "How far each pixel moves between the eyes. Brighter = more.",
    "holes": "Where the renderer had to invent picture behind an edge (pink).",
    "orbit": "The depth as geometry, slowly orbiting.",
}


@dataclass(frozen=True)
class PreviewParams:
    """Everything the preview needs that is not the picture. Immutable, so any thread may hold it."""

    mode: str = "original"
    layout: str = "sbs_half"
    stereo: stereo.StereoParams = field(default_factory=stereo.StereoParams)
    temporal: float = 60.0
    edge_aware: bool = True
    depth_blur: float = 0.0
    colour: grade.ColourSettings = field(default_factory=grade.ColourSettings)
    force_cpu: bool = False

    @property
    def needs_depth(self) -> bool:
        return self.mode != "original"


@dataclass
class PreviewJob:
    frame: np.ndarray                       # float32 RGB 0..1, already proxy-sized
    frame_id: int
    new_frame: bool                         # False = same picture, settings changed
    params: PreviewParams


@dataclass
class PreviewResult:
    images: list[np.ndarray]                # uint8 HxWx3; several = alternate (wiggle)
    badge: str
    wiggle: bool
    scope: np.ndarray | None
    depth_ms: float = 0.0
    total_ms: float = 0.0
    device: str = ""
    depth: np.ndarray | None = None         # refined depth, for the point-cloud view
    colour: np.ndarray | None = None
    stats: dict = field(default_factory=dict)


def _u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _heat(values: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """0..1 -> RGB uint8 through a perceptual colour map."""
    return cv2.cvtColor(cv2.applyColorMap(_u8(values), colormap), cv2.COLOR_BGR2RGB)


def _shrink(image: np.ndarray, longest: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = longest / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


class _Mailbox:
    """One slot, newest wins, with a wake-up event."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item = None
        self.event = threading.Event()

    def put(self, item, merge=None) -> None:
        with self._lock:
            if merge is not None and self._item is not None:
                item = merge(self._item, item)
            self._item = item
        self.event.set()

    def take(self):
        self.event.wait()
        self.event.clear()
        with self._lock:
            item, self._item = self._item, None
        return item


class PreviewEngine(QObject):
    """Owns the preview threads. Call :meth:`submit` from any thread; results arrive as a signal."""

    rendered = Signal(object)               # PreviewResult
    device_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs = _Mailbox()
        self._renders = _Mailbox()
        self._alive = True
        self._engine: depthmod.DepthEngine | None = None
        self._stab = depthmod.DepthStabilizer(0.6)
        self._grader = grade.GradeEngine()
        self._steady: np.ndarray | None = None
        self._steady_id = -1
        self._last_done = 0.0
        self._fps = 0.0
        self._threads = [
            threading.Thread(target=self._depth_loop, daemon=True, name="uv3d-preview-depth"),
            threading.Thread(target=self._render_loop, daemon=True, name="uv3d-preview-render"),
        ]
        for t in self._threads:
            t.start()

    # -- API ---------------------------------------------------------------

    @staticmethod
    def _merge(old: PreviewJob, new: PreviewJob) -> PreviewJob:
        # A settings change landing before its frame was processed must not turn
        # the pending job into "reuse the depth" - that frame has none yet.
        if old.new_frame and not new.new_frame and old.frame_id == new.frame_id:
            new.new_frame = True
        return new

    def submit(self, job: PreviewJob) -> None:
        """Replace whatever is waiting. Stale jobs are dropped by design."""
        self._jobs.put(job, self._merge)

    def shutdown(self) -> None:
        self._alive = False
        self._jobs.event.set()
        self._renders.event.set()
        for t in self._threads:
            t.join(timeout=3.0)

    # -- depth thread ------------------------------------------------------

    def _ensure_engine(self, force_cpu: bool) -> None:
        if self._engine is not None and (self._engine.device == "CPU") == force_cpu:
            return
        self._engine = depthmod.shared_engine(force_cpu)
        self.device_ready.emit(self._engine.device)
        self._steady, self._steady_id = None, -1

    def _depth_loop(self) -> None:
        while self._alive:
            job = self._jobs.take()
            if job is None or not self._alive:
                continue
            try:
                steady, depth_ms = None, 0.0
                p = job.params
                if p.needs_depth:
                    self._ensure_engine(p.force_cpu)
                    self._stab.smoothing = float(np.clip(p.temporal / 100.0 * 0.95, 0.0, 0.95))
                    if job.new_frame or self._steady is None or self._steady_id != job.frame_id:
                        t = time.perf_counter()
                        small = _shrink(job.frame, 640)
                        self._steady = self._stab(self._engine.infer(small), small)
                        self._steady_id = job.frame_id
                        depth_ms = (time.perf_counter() - t) * 1000
                    steady = self._steady
                self._renders.put((job, steady, depth_ms))
            except Exception as error:  # noqa: BLE001 - the preview must never crash the app
                self.failed.emit(f"{type(error).__name__}: {error}")

    # -- render thread -----------------------------------------------------

    def _render_loop(self) -> None:
        while self._alive:
            item = self._renders.take()
            if item is None or not self._alive:
                continue
            try:
                result = self._render(*item)
            except Exception as error:  # noqa: BLE001
                self.failed.emit(f"{type(error).__name__}: {error}")
                continue
            self.rendered.emit(result)

    def _render(self, job: PreviewJob, steady: np.ndarray | None, depth_ms: float) -> PreviewResult:
        began = time.perf_counter()
        result = render_frame(job.params, job.frame, steady, self._grader)
        result.depth_ms = depth_ms
        result.device = self._engine.device if self._engine else ""
        now = time.perf_counter()
        result.total_ms = (now - began) * 1000
        if steady is not None:
            if self._last_done:
                inst = 1.0 / max(now - self._last_done, 1e-3)
                self._fps = inst if not self._fps else 0.8 * self._fps + 0.2 * inst
            self._last_done = now
            result.stats["fps"] = self._fps
        return result


def render_frame(p: PreviewParams, frame: np.ndarray, steady: np.ndarray | None,
                 grader: grade.GradeEngine) -> PreviewResult:
    """One preview picture: grade, and (in a 3D view) refine depth, warp and lay out the view.

    Shared by the live preview and the buffered look-ahead renderer, so a buffered
    frame is pixel-for-pixel what the live path would have shown.
    """
    graded = grader.apply(frame, p.colour) if p.colour.active else frame
    scope = _u8(_shrink(graded, 240))
    if steady is None:
        return PreviewResult([stereo.pack_frame("copy", None, graded, None, 8)], "", False, scope)
    d = depthmod.refine(steady, frame, p.edge_aware, p.depth_blur / 100.0 * 6.0)
    pair = stereo.synthesize(graded, d, p.stereo)
    images, badge, wiggle = build_views(p, pair, d, graded)
    orbit = p.mode == "orbit"
    return PreviewResult(
        images, badge, wiggle, scope,
        depth=d if orbit else None, colour=_u8(graded) if orbit else None,
        stats={"hole_pct": float(pair.hole_left[::3, ::3].mean() * 100.0)},
    )


def build_views(p: PreviewParams, pair: stereo.Pair, d: np.ndarray, graded: np.ndarray):
    mode = p.mode
    # The same fused kernel the export uses: layout + quantise in one pass, no
    # full-frame float temporaries (the numpy equivalents cost ~25 ms at 720p).
    if mode == "anaglyph":
        return [stereo.pack_frame("anaglyph", pair, graded, d, 8)], "Anaglyph · red/cyan glasses", False
    if mode == "sbs":
        return [stereo.pack_frame("sbs_full", pair, graded, d, 8)], "Left | Right", False
    if mode == "wiggle":
        eye = lambda img: stereo.pack_frame("copy", None, img, None, 8)   # noqa: E731
        return [eye(pair.left), eye(pair.right)], "Wiggle · no glasses needed", True
    if mode == "depth":
        return [_heat(d)], "Depth · warm = near", False
    if mode == "parallax":
        diff = np.abs(pair.left - pair.right).mean(axis=-1)
        scale = max(float(np.percentile(diff, 99.5)), 1e-3)
        return [_heat(np.clip(diff / scale, 0, 1), cv2.COLORMAP_INFERNO)], "Parallax · brighter = more shift", False
    if mode == "holes":
        mask = cv2.dilate((pair.hole_left | pair.hole_right).astype(np.float32), np.ones((3, 3), np.uint8))
        tint = np.array([1.0, 0.1, 0.8], np.float32)
        out = graded * 0.55 * (1 - mask[..., None]) + tint * mask[..., None]
        return [_u8(out)], "Holes · filled from the background", False
    if mode == "orbit":
        return [stereo.pack_frame("copy", None, graded, None, 8)], "3D orbit · depth as geometry", False
    packed = stereo.pack_frame(p.layout, pair, graded, d, 8)
    return [packed], f"Output · {stereo.LAYOUTS_BY_KEY[p.layout].label}", False
