"""The preview player: video decoded on its own thread, audio played by Qt.

Why not just let Qt play the video and hand over frames? Because every frame then
crosses the UI thread on its way to the preview: a 4K frame has to be copied out
of Qt, converted and shrunk *on the thread that is also drawing the window and
answering the mouse*. At 30 fps that is the whole frame budget, and the interface
stops responding. Here the UI thread never touches a video frame at all:

    feeder thread: decode -> scale to a proxy inside FFmpeg -> hand to the preview engine
    UI thread:     draws finished pictures, handles clicks

Audio still comes from QMediaPlayer (no video sink, so it plays sound only) and
its position is the master clock the picture is paced against - so sound and
picture stay in step, and a slow frame is *dropped* rather than delaying
everything behind it. A clip with no audio is paced by the system clock.

Seeking is frame-accurate when it needs to be (a paused still, a released
scrubber) and keyframe-only while the scrubber is being dragged, because decoding
forward from a keyframe on a long-GOP 4K clip is exactly the slow part.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal

from . import hdr, paths, video
from .buffered import BufferedPlayback
from .preview import PreviewParams

#: A frame this far behind the clock is dropped rather than shown late.
LATE_MS = 90.0


def proxy_from_frame(frame, info: video.VideoInfo, high: bool, height: int) -> np.ndarray:
    """A decoded frame as a float RGB proxy ``height`` tall, tone-mapped to SDR if the source is HDR.

    Shared by the live feeder and the buffered renderer so both show the same picture.
    """
    h = min(height, info.height)
    w = round(info.width * h / info.height)
    rgb = video.frame_to_rgb(frame, high, (max(2, w - w % 2), max(2, h - h % 2)))
    if info.hdr != video.SDR:
        rgb = hdr.tonemap_to_sdr(rgb, info.hdr)        # the display, and the depth model, are SDR
    return rgb


class Clock:
    """A thread-safe playback clock: the audio position when there is one, else the system clock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._base_ms = 0.0
        self._base_t = time.monotonic()
        self._running = False

    def start(self, ms: float) -> None:
        with self._lock:
            self._base_ms, self._base_t, self._running = float(ms), time.monotonic(), True

    def stop(self) -> None:
        with self._lock:
            self._base_ms = self._now_locked()
            self._running = False

    def set(self, ms: float) -> None:
        with self._lock:
            self._base_ms, self._base_t = float(ms), time.monotonic()

    def _now_locked(self) -> float:
        if not self._running:
            return self._base_ms
        return self._base_ms + (time.monotonic() - self._base_t) * 1000.0

    def now(self) -> float:
        with self._lock:
            return self._now_locked()


class FrameFeeder(threading.Thread):
    """Decodes ahead of the clock and delivers proxy-sized frames."""

    def __init__(self, path: Path, info: video.VideoInfo, clock: Clock,
                 deliver: Callable[[np.ndarray, int, float], None]) -> None:
        super().__init__(daemon=True, name="uv3d-feeder")
        self._path, self._info, self._clock, self._deliver = path, info, clock, deliver
        self._cv = threading.Condition()
        self._playing = False
        self._seek: tuple[float, bool] | None = None
        self._quit = False
        self.proxy_height = 720
        #: True while delivering the picture for a keyframe-only seek (a scrub in progress). Its
        #: time is the nearest keyframe - on a short clip, frame 0 - so it must not move the playhead.
        self.approximate = False
        self.error = ""
        #: Cumulative counters, for diagnosing playback: how many frames were shown,
        #: how many were dropped for being late, and how late the shown ones were.
        self.stats = {"shown": 0, "dropped": 0, "behind_ms": 0.0}

    # -- control (any thread) ----------------------------------------------

    def play(self) -> None:
        with self._cv:
            self._playing = True
            self._cv.notify()

    def pause(self) -> None:
        with self._cv:
            self._playing = False
            self._cv.notify()

    def seek(self, ms: float, precise: bool = True) -> None:
        """Latest request wins: a drag can ask far faster than decoding can answer."""
        with self._cv:
            self._seek = (max(0.0, float(ms)), precise)
            self._cv.notify()

    def refresh(self) -> None:
        """Re-decode the current frame (the proxy size changed)."""
        self.seek(self._clock.now(), True)

    def shutdown(self) -> None:
        with self._cv:
            self._quit = True
            self._cv.notify()

    # -- thread ------------------------------------------------------------

    def _to_proxy(self, frame, high: bool) -> np.ndarray:
        return proxy_from_frame(frame, self._info, high, self.proxy_height)

    def run(self) -> None:
        try:
            self._loop()
        except Exception as error:  # noqa: BLE001 - a bad file must not take the app down
            self.error = f"{type(error).__name__}: {error}"

    def _wait_for_work(self) -> tuple[float, bool] | None:
        with self._cv:
            while not self._quit and not self._playing and self._seek is None:
                self._cv.wait(0.5)
            request, self._seek = self._seek, None
            return request

    def _loop(self) -> None:
        import av

        container = av.open(str(self._path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        high = self._info.bit_depth > 8
        tb = float(stream.time_base) if stream.time_base else 1.0 / 1000
        fps = self._info.fps or 24.0
        frames = None
        index_of = lambda ms: int(round(ms / 1000.0 * fps))          # noqa: E731

        def frame_ms(frame) -> float:
            if frame.pts is None:
                return 0.0
            return float(frame.pts) * tb * 1000.0

        def start_at(ms: float, precise: bool):
            """Seek, and return an iterator plus the first frame to show."""
            container.seek(int(ms / 1000.0 / tb), stream=stream, backward=True)
            it = container.decode(stream)
            for frame in it:
                if not precise or frame_ms(frame) + 500.0 / fps >= ms:
                    return it, frame
            return it, None

        try:
            while not self._quit:
                request = self._wait_for_work()
                if self._quit:
                    break
                if request is not None:
                    ms, precise = request
                    frames, first = start_at(ms, precise)
                    if first is not None:
                        if precise and not self._playing:
                            self._clock.set(frame_ms(first))     # a keyframe-only picture must not move the clock
                        self.approximate = not precise
                        try:
                            self._deliver(self._to_proxy(first, high), index_of(frame_ms(first)), frame_ms(first))
                        finally:
                            self.approximate = False
                    continue
                if not self._playing:
                    continue
                if frames is None:
                    frames, first = start_at(self._clock.now(), False)
                    if first is None:
                        continue
                    self._deliver(self._to_proxy(first, high), index_of(frame_ms(first)), frame_ms(first))
                    continue
                try:
                    frame = next(frames)
                except StopIteration:
                    frames, _ = start_at(0.0, False)                # loop; the audio loops itself
                    continue
                ms = frame_ms(frame)
                now = self._clock.now()
                if ms < now - LATE_MS:
                    self.stats["dropped"] += 1
                    continue                                          # behind: drop it, catch up
                while ms > now + 6.0 and self._playing and self._seek is None and not self._quit:
                    time.sleep(min(0.02, (ms - now) / 1000.0))
                    now = self._clock.now()
                if not self._playing or self._seek is not None or self._quit:
                    continue
                self.stats["shown"] += 1
                self.stats["behind_ms"] += self._clock.now() - ms
                self._deliver(self._to_proxy(frame, high), index_of(ms), ms)
        finally:
            container.close()


class PreviewPlayer(QObject):
    """Play / pause / seek, with a picture callback and a position signal.

    Two ways to put a picture on screen, chosen automatically:

    * **live** - the feeder decodes on the clock and the preview engine renders the
      newest frame it can. Instant, so it drives paused stills, scrubbing and any
      plain-2D playback.
    * **buffered** - while a *3D* preview plays, a look-ahead pipeline renders
      several seconds ahead and the picture comes from that cache, so a slow frame
      cannot show as a stutter (see ``buffered.py``).
    """

    position_changed = Signal(int)          # milliseconds, from the delivered picture
    state_changed = Signal(bool)            # True = playing (or waiting to play)
    buffered_image = Signal(object)         # (images, badge, wiggle, hole_pct) from the cache
    buffering_changed = Signal(bool, float)  # (waiting for the buffer, progress 0..1)
    notice = Signal(str)                    # something was tuned automatically
    smaller_preview_needed = Signal()       # ask the window for a lower preview resolution
    failed = Signal(str)
    _audio_command = Signal(str, int)       # display thread -> UI thread: "play"/"pause", ms

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        self._audio = QMediaPlayer(self)
        self._output = QAudioOutput(self)
        self._audio.setAudioOutput(self._output)          # no video sink: this plays sound only
        self._audio.setLoops(QMediaPlayer.Loops.Infinite)
        self._audio.positionChanged.connect(self._audio_position)
        self._audio_command.connect(self._on_audio_command)
        self._clock = Clock()
        self._feeder: FrameFeeder | None = None
        self._info: video.VideoInfo | None = None
        self._has_audio = False
        self._playing = False
        self._mode_buffered = False
        self._path: Path | None = None
        self.proxy_height = 540
        #: Set by the window: called on the feeder thread with (proxy_rgb, frame_index, ms).
        self.on_frame: Callable[[np.ndarray, int, float], None] | None = None
        #: What the preview would show and how far ahead to render; set by the window.
        self.buffer_params: PreviewParams | None = None
        self.buffer_seconds = 5
        self._buffered = BufferedPlayback(
            self._clock,
            lambda ms: self._audio_command.emit("play", int(ms)),
            lambda: self._audio_command.emit("pause", 0),
            self,
            cache_dir=paths.scratch_dir() / "preview-cache" / str(os.getpid()),
        )
        self._buffered.image.connect(self.buffered_image)
        self._buffered.buffering.connect(self.buffering_changed)
        self._buffered.position.connect(self.position_changed)
        self._buffered.failed.connect(self.failed)
        self._buffered.notice.connect(self.notice)
        self._buffered.smaller_preview_needed.connect(self.smaller_preview_needed)
        self._buffered.too_slow.connect(self._on_too_slow)
        self._too_slow = False

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def buffer_playing(self) -> bool:
        """True while the picture on screen is coming from the look-ahead cache."""
        return self._mode_buffered and self._buffered.playing_from_cache

    def buffer_ahead_seconds(self) -> float:
        info = self._info
        if info is None:
            return 0.0
        fps = info.fps or 24.0
        index = int(round(self._clock.now() * fps / 1000.0))
        return self._buffered.cache.ahead(index, int(30 * fps)) / fps

    def buffered_frames(self) -> tuple[int, int, bool]:
        """(first, last, waiting): the frame range rendered ahead of the playhead, for the timeline bar."""
        info = self._info
        if info is None:
            return 0, 0, False
        fps = info.fps or 24.0
        index = int(round(self._clock.now() * fps / 1000.0))
        ahead = self._buffered.cache.ahead(index, int(30 * fps))
        return index, index + ahead, self._buffered.state == "buffering"

    def set_preview_fps(self, choice: int) -> None:
        """0 = choose automatically for this computer; otherwise 24 or 30."""
        self._buffered.set_fps_choice(choice)

    def _on_too_slow(self, fps: float) -> None:
        """Measured too slow to play 3D in real time: fall back to live pictures for this session."""
        if self._too_slow:
            return
        self._too_slow = True
        self.notice.emit(
            f"This computer renders the 3D preview at only ~{fps:.0f} fps, too slow to play smoothly. "
            "Pause or scrub to inspect frames - the export is unaffected and uses full quality.")
        if self._playing and self._mode_buffered:
            self._to_live()
        elif not self._playing:
            self._buffered.halt()                        # no point pre-rendering what cannot be played

    def _can_buffer(self) -> bool:
        """A 3D view that the look-ahead cache can render (whether or not it is playing right now)."""
        p = self.buffer_params
        return bool(self.buffer_seconds > 0 and p is not None and p.needs_depth and p.mode != "orbit"
                    and self._info is not None)

    def _use_buffer(self) -> bool:
        return bool(not self._too_slow and self._can_buffer())

    def buffer_visible(self) -> bool:
        """Whether the timeline's buffer bar has anything to say (a 3D view is on)."""
        return self._can_buffer()

    def _prefill(self) -> None:
        """Paused: start (or keep) rendering ahead of the playhead into the disk cache."""
        if self._feeder is None or self._playing:
            return
        if self._use_buffer():
            self._buffered.prefill(self.buffer_params, self._clock.now())
        else:
            self._buffered.halt()

    # -- audio, on the UI thread ---------------------------------------------

    def _on_audio_command(self, command: str, ms: int) -> None:
        if not self._has_audio:
            return
        if command == "play":
            self._audio.setPosition(ms)
            self._audio.play()
        else:
            self._audio.pause()

    def _audio_position(self, ms: int) -> None:
        # The audio device is the master clock while sound is playing.
        if self._has_audio and self._playing and not (self._mode_buffered and not self._buffered.playing_from_cache):
            self._clock.set(ms)

    # -- pictures ------------------------------------------------------------

    def _delivered(self, rgb: np.ndarray, index: int, ms: float) -> None:
        if self.on_frame is not None:
            self.on_frame(rgb, index, ms)
        feeder = self._feeder
        if feeder is not None and feeder.approximate:
            return          # a keyframe-only scrub picture: its time is the keyframe's, not where the user pointed
        self.position_changed.emit(int(ms))            # queued to the UI thread: one int

    def _proxy_of(self, frame, high: bool) -> np.ndarray:
        return proxy_from_frame(frame, self._info, high, self.proxy_height)

    # -- opening -------------------------------------------------------------

    def open(self, path: Path, info: video.VideoInfo) -> None:
        self.close()
        self._path, self._info, self._has_audio = path, info, info.has_audio
        self._playing = self._mode_buffered = False
        self._too_slow = False
        self._clock.stop()
        self._clock.set(0.0)
        self._audio.setSource(QUrl.fromLocalFile(str(path)))
        self._buffered.open(path, info, self._proxy_of)
        self._buffered.set_lookahead(self.buffer_seconds)
        feeder = FrameFeeder(path, info, self._clock, self._delivered)
        feeder.proxy_height = self.proxy_height
        self._feeder = feeder
        feeder.start()
        feeder.seek(0.0, True)                         # show the first frame; stay paused
        self.state_changed.emit(False)
        self._prefill()                                # a 3D view already chosen starts preparing at once

    def close(self) -> None:
        self._buffered.halt()
        if self._feeder is not None:
            self._feeder.shutdown()
            self._feeder = None
        self._audio.stop()
        self._playing = self._mode_buffered = False

    def set_proxy_height(self, height: int) -> None:
        self.proxy_height = int(height)
        if self._feeder is not None:
            self._feeder.proxy_height = self.proxy_height
            self._feeder.refresh()
        self._buffered.proxy_height = self.proxy_height       # part of a look's identity: other sizes keep their own cache
        if self.buffer_params is not None and self._use_buffer() and (self._mode_buffered or not self._playing):
            self._buffered.invalidate(self.buffer_params)

    def set_buffer_config(self, params: PreviewParams | None, seconds: int) -> None:
        """Called by the window whenever the preview settings change."""
        changed = params != self.buffer_params
        self.buffer_params, self.buffer_seconds = params, int(seconds)
        self._buffered.set_lookahead(seconds)
        if self._feeder is None:
            return
        if not self._playing:
            # Paused: a new look restarts the disk cache from here (after the last change
            # settles); an unchanged one just makes sure it is still filling.
            if not self._use_buffer():
                self._buffered.halt()
            elif changed:
                self._buffered.invalidate(params)
            else:
                self._buffered.prefill(params, self._clock.now())
            return
        usable = self._use_buffer()
        if self._mode_buffered and usable and changed:
            self._buffered.invalidate(params)
        elif self._mode_buffered and not usable:
            self._to_live()
        elif not self._mode_buffered and usable:
            self._to_buffered()

    # -- transport -----------------------------------------------------------

    def _to_buffered(self) -> None:
        """Playing live and the 3D preview came on: hand over to the look-ahead renderer."""
        now = self._clock.now()
        self._feeder.pause()
        self._audio.pause()
        self._mode_buffered = True
        self._buffered.begin(self.buffer_params, now)

    def _to_live(self) -> None:
        """The 3D preview went off (or the buffer was disabled): carry on live from here."""
        self._buffered.halt()
        self._mode_buffered = False
        now = self._clock.now()
        self._clock.start(now)
        if self._has_audio:
            self._audio.setPosition(int(now))
            self._audio.play()
        self.buffering_changed.emit(False, 1.0)
        self._feeder.seek(now, True)
        self._feeder.play()

    def play(self) -> None:
        if self._feeder is None or self._playing:
            return
        self._playing = True
        if self._use_buffer():
            self._mode_buffered = True
            self._feeder.pause()
            self._buffered.begin(self.buffer_params, self._clock.now())
        else:
            self._mode_buffered = False
            self._clock.start(self._clock.now())
            if self._has_audio:
                self._audio.setPosition(int(self._clock.now()))
                self._audio.play()
            self._feeder.play()
        self.state_changed.emit(True)

    def pause(self) -> None:
        if self._feeder is None or not self._playing:
            return
        self._playing = False
        was_buffered, self._mode_buffered = self._mode_buffered, False
        if was_buffered:
            self._buffered.settle()                      # it keeps rendering ahead, into the disk cache
        else:
            self._buffered.halt()
        self._feeder.pause()
        self._audio.pause()
        self._clock.stop()
        if was_buffered:
            self.buffering_changed.emit(False, 1.0)
            self._feeder.seek(self._clock.now(), True)   # redraw this exact frame through the live path
        else:
            self._prefill()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek_ms(self, ms: int, precise: bool = True) -> None:
        if self._feeder is None:
            return
        self._clock.set(ms)
        if self._has_audio:
            self._audio.setPosition(int(ms))
        self._feeder.seek(ms, precise)                   # the live still, immediately
        if self._mode_buffered:
            self._buffered.seek(ms)                      # and refill the buffer from there
        elif not self._playing and self._use_buffer():
            self._buffered.prefill_later(self.buffer_params, ms)   # paused: prepare from the new spot

    def shutdown(self) -> None:
        self.close()
        self._buffered.shutdown()
