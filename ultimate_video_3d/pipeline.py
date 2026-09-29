"""The whole conversion: decode -> grade -> depth -> stereo -> encode -> mux audio.

Three stages run on their own threads so nothing waits on anything else it does
not have to: a reader decodes ahead, the worker does the depth + stereo maths
(DirectML and Numba both release the GIL), and a writer feeds the encoder. On a
machine with a hardware encoder the encoder is effectively free, so throughput
is set by depth + warp, not by the slowest of the three summed.

Grading happens *before* depth is estimated but the depth is taken from the
ungraded picture: a heavy grade should not change how far away things look.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import depth as depthmod
from . import encoders, grade, hdr, paths, stereo, sysinfo, video
from .settings import AppSettings

Progress = Callable[[str], None]

_DEPTH_EDGE = 640          # longest edge of the copy fed to the depth model


@dataclass
class VideoProgress:
    stage: str             # "starting" | "converting" | "muxing" | "done" | "done-silent"
    index: int = 0
    total: int = 0
    fps: float = 0.0
    preview: np.ndarray | None = None
    encoder: str = ""
    message: str = ""


def output_path_for(source: Path, settings: AppSettings, folder: Path | None = None) -> Path:
    family = encoders.FAMILIES_BY_KEY[settings.export.family]
    tag = ""
    if settings.threed.enabled:
        tag = stereo.LAYOUTS_BY_KEY[settings.threed.layout].tag
    else:
        tag = "_converted"
    return (folder or paths.output_dir()) / f"{source.stem}{tag}{family.suffix}"


def eye_size(info: video.VideoInfo, settings: AppSettings) -> tuple[int, int]:
    """Per-eye frame size after the optional height cap, both even."""
    width, height = info.width, info.height
    cap = settings.export.max_height
    if cap and height > cap:
        width = round(width * cap / height)
        height = cap
    return width - width % 2, height - height % 2


def frame_size_for(info: video.VideoInfo, settings: AppSettings) -> tuple[int, int]:
    width, height = eye_size(info, settings)
    if settings.threed.enabled:
        return stereo.output_size(settings.threed.layout, width, height)
    return width, height


class _Reader(threading.Thread):
    """Decodes ahead of the worker so decode time hides behind compute."""

    def __init__(self, source: Path, start: int, limit: int | None, out: queue.Queue) -> None:
        super().__init__(daemon=True, name="uv3d-reader")
        self._args = (source, start, limit)
        self._out = out
        self.stop = threading.Event()

    def run(self) -> None:
        try:
            for item in video.frames(*self._args, should_stop=self.stop.is_set):
                while not self.stop.is_set():
                    try:
                        self._out.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    break
            self._out.put(None)
        except Exception as error:  # noqa: BLE001 - handed to the worker to raise
            self._out.put(error)


class _Writer(threading.Thread):
    """Feeds the encoder from its own thread."""

    def __init__(self, writer: video.VideoWriter) -> None:
        super().__init__(daemon=True, name="uv3d-writer")
        self._writer = writer
        self.queue: queue.Queue = queue.Queue(maxsize=6)
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                if self.error is None:
                    self._writer.write(item)
        except Exception as error:  # noqa: BLE001 - surfaced by the worker
            self.error = error
            while True:  # keep draining so the producer never deadlocks on a full queue
                if self.queue.get() is None:
                    break

    def submit(self, frame: np.ndarray) -> None:
        if self.error:
            raise self.error
        self.queue.put(frame)


class _DepthStage(threading.Thread):
    """Prepares each frame and estimates its depth, ahead of the warp.

    Everything here is stateful across frames (the stabiliser), so it lives on
    one thread; the warp and encode stages pull finished ``(frame, depth)`` pairs.
    """

    def __init__(self, source: queue.Queue, out: queue.Queue, settings: AppSettings,
                 info: video.VideoInfo, src_transfer: str, signal: str,
                 eye: tuple[int, int], say: Progress) -> None:
        super().__init__(daemon=True, name="uv3d-depth")
        self._in, self.out = source, out
        self._settings, self._info = settings, info
        self._src_transfer, self._signal, self._eye = src_transfer, signal, eye
        self._say = say
        self.stop = threading.Event()

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - re-raised by the consumer
            self.out.put(error)

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.out.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        threed = self._settings.threed
        eye_w, eye_h = self._eye
        engine = stabilizer = None
        if threed.enabled:
            engine = depthmod.DepthEngine()
            self._say(f"Depth model: {engine.load(self._say, self._settings.force_cpu_depth)}")
            stabilizer = depthmod.DepthStabilizer(threed.temporal / 100.0 * 0.95)
        try:
            while not self.stop.is_set():
                item = self._in.get()
                if item is None or isinstance(item, Exception):
                    self._put(item)
                    return
                _, frame = item
                if self._signal == video.SDR and self._src_transfer != video.SDR:
                    frame = hdr.tonemap_to_sdr(frame, self._src_transfer)
                if frame.shape[1] != eye_w or frame.shape[0] != eye_h:
                    frame = cv2.resize(frame, (eye_w, eye_h), interpolation=cv2.INTER_AREA)
                d = None
                if engine is not None:
                    small = _shrink(frame, _DEPTH_EDGE)
                    if self._signal != video.SDR:
                        small = hdr.tonemap_to_sdr(small, self._signal)   # depth is judged on an ordinary picture
                    d = stabilizer(engine.infer(small), small)
                if not self._put((frame, d)):
                    return
        finally:
            if engine is not None:
                engine.session = None


def convert_video(
    source: Path,
    destination: Path,
    settings: AppSettings,
    start: int = 0,
    limit: int | None = None,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
    gpus: list[sysinfo.GpuInfo] | None = None,
) -> Iterator[VideoProgress]:
    """Convert ``source`` to ``destination``; yields progress, then a final done."""
    from . import support

    support.require_intact()                       # see support.py: a modified copy does not convert
    say = progress or (lambda _m: None)
    if not video.is_available():
        raise RuntimeError("Video support (PyAV) has not been downloaded yet.")

    info = video.probe(source)
    export, threed = settings.export, settings.threed
    family = encoders.FAMILIES_BY_KEY[export.family]
    layout = stereo.LAYOUTS_BY_KEY[threed.layout]
    total = info.frames
    if limit is not None:
        total = min(limit, total - start) if total else limit

    src_transfer = info.hdr
    keep_hdr = src_transfer != video.SDR and export.hdr_mode == "keep" and family.hdr
    if src_transfer != video.SDR and export.hdr_mode == "keep" and not family.hdr:
        say(f"{family.label} cannot carry HDR, so this will be tone-mapped to SDR.")
    signal = src_transfer if keep_hdr else video.SDR

    eye_w, eye_h = eye_size(info, settings)
    out_w, out_h = frame_size_for(info, settings)
    profile = family.profiles[min(export.profile, len(family.profiles) - 1)] if family.profiles else None
    ten_bit = keep_hdr or (profile is not None and "10" in (profile.pix_fmt or family.pix_fmt_10bit)
                           and family.key == "prores")
    gpus = gpus if gpus is not None else sysinfo.detect_gpus()
    report = encoders.ProbeReport(**settings.encoder_probe) if settings.encoder_probe else None
    forced = None if export.encoder == "auto" else export.encoder
    say("Choosing an encoder…")
    encoder = encoders.pick(family, gpus, (out_w, out_h), info.fps, ten_bit, report, forced)
    if family.max_width and out_w > family.max_width:
        say(f"Note: {out_w}px wide exceeds {family.label}'s {family.max_width}px limit; some players may refuse it.")

    scratch = paths.scratch_dir()
    video_only = scratch / f"{source.stem}.video_only{family.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    writer = video.VideoWriter(
        video_only, family, encoder, info.fps, (out_w, out_h), export.quality, profile,
        hdr=signal, stereo_mode=layout.stereo_mode if threed.enabled else "",
    )
    async_writer = _Writer(writer)
    async_writer.start()
    yield VideoProgress("starting", total=total, encoder=encoder,
                        message=f"Encoding with {encoders.encoder_label(encoder)}")

    grader = grade.GradeEngine()
    params = threed.params()
    bits = 16 if writer.ten_bit else 8

    frames_queue: queue.Queue = queue.Queue(maxsize=4)
    reader = _Reader(source, start, limit, frames_queue)
    depth_queue: queue.Queue = queue.Queue(maxsize=3)
    depth_stage = _DepthStage(frames_queue, depth_queue, settings, info, src_transfer,
                              signal, (eye_w, eye_h), say)
    reader.start()
    depth_stage.start()

    done = 0
    began = time.monotonic()
    stopped = False
    try:
        while True:
            item = depth_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if should_stop is not None and should_stop():
                stopped = True
                break
            frame, steady = item
            d = None
            if steady is not None:
                # Refinement snaps depth edges to the picture's. It runs here, on
                # the warp thread, so the depth thread only has the model to feed.
                d = depthmod.refine(steady, frame, threed.edge_aware, threed.depth_blur / 100.0 * 6.0)

            graded = grader.apply(frame, settings.colour, signal) if settings.colour.active else frame
            pair = stereo.synthesize(graded, d, params) if threed.enabled else None
            packed = stereo.pack_frame(layout.key if threed.enabled else "copy", pair, graded, d, bits)

            async_writer.submit(packed)
            done += 1
            elapsed = max(time.monotonic() - began, 1e-3)
            preview = None
            if done % 3 == 0:
                preview = _shrink(packed.astype(np.float32) / (255.0 if bits == 8 else 65535.0), 640)
            yield VideoProgress("converting", done, total, done / elapsed, preview, encoder)
    finally:
        reader.stop.set()
        depth_stage.stop.set()
        async_writer.queue.put(None)
        async_writer.join()
    if async_writer.error:
        raise async_writer.error
    writer.close()

    if stopped:
        video_only.unlink(missing_ok=True)
        yield VideoProgress("stopped", done, total)
        return

    yield VideoProgress("muxing", done, total, message="Adding audio…")
    trimming = start > 0 or limit is not None
    window_start = start / info.fps if trimming else 0.0
    window_len = (done / info.fps) if trimming else None
    had_audio = video.mux_audio(
        source, video_only, destination, window_start, window_len,
        audio=export.audio, suffix=family.suffix,
    )
    video_only.unlink(missing_ok=True)
    yield VideoProgress("done", done, total,
                        message="" if had_audio else "No audio track.")


def _shrink(image: np.ndarray, longest: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = longest / float(max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)
