"""The video side of the VR player: decode, sound and timing, handing the newest picture to the VR thread.

Like the Library's player (``librarypage.LibraryPlayer``) and the preview, the picture is decoded on a thread of
its own and Qt plays the sound, whose position is the clock the pictures are paced against. The difference is
what comes out: a full-size ``uint8`` frame (the whole two-eye frame, as stored) that the VR thread uploads to
the GPU, rather than a small picture for a window. This object lives on the UI thread; the VR thread only calls
:meth:`take_frame`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal

from .. import hdr, video
from ..player import Clock, FrameFeeder
from . import geometry as geo

#: Largest frame handed to the GPU. A headset eye shows well under 4K of a screen, so more is wasted work.
CAPS = {"sbs": (3840, 2160), "tb": (3840, 2160), "flat": (3840, 2160)}


def target_size(layout: str, width: int, height: int) -> tuple[int, int]:
    """The decoded frame's size: the source's, scaled down only if it is larger than a headset can use."""
    key = "sbs" if layout in ("sbs_half", "sbs_full") else "tb" if layout in ("tb_half", "tb_full") else "flat"
    cap_w, cap_h = CAPS[key]
    scale = min(1.0, cap_w / max(1, width), cap_h / max(1, height))
    w, h = max(2, round(width * scale)), max(2, round(height * scale))
    return w - w % 2, h - h % 2


class _Feeder(FrameFeeder):
    """The preview's frame feeder, but delivering ``uint8`` frames at a size chosen for the headset."""

    size: tuple[int, int] = (0, 0)

    def _to_proxy(self, frame, high: bool) -> np.ndarray:
        info = self._info
        if info.hdr != video.SDR:                                  # tone-map to SDR for the screen
            rgb = video.frame_to_rgb(frame, True, self.size)
            return (np.clip(hdr.tonemap_to_sdr(rgb, info.hdr), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        full = int(frame.color_range or 0) == 2
        return frame.to_ndarray(
            format="rgb24", width=self.size[0], height=self.size[1], interpolation="AREA",
            src_colorspace=video._src_matrix(frame), src_color_range="JPEG" if full else "MPEG",
            dst_color_range="JPEG",
        )


class VRMedia(QObject):
    position_changed = Signal(int)          # milliseconds
    state_changed = Signal(bool)            # True = playing
    ended = Signal()
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        self._audio = QMediaPlayer(self)
        self._output = QAudioOutput(self)
        self._audio.setAudioOutput(self._output)
        self._audio.positionChanged.connect(self._audio_position)
        self._clock = Clock()
        self._feeder: _Feeder | None = None
        self.info: video.VideoInfo | None = None
        self.layout = ""
        self.path: Path | None = None
        self.loop = False
        self._playing = False
        self._has_audio = False
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_ms = 0
        self._taken = True

    # -- facts about what is open ------------------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def is_open(self) -> bool:
        return self._feeder is not None

    @property
    def duration_ms(self) -> int:
        return int((self.info.duration if self.info else 0.0) * 1000)

    @property
    def position_ms(self) -> int:
        return int(self._clock.now())

    @property
    def aspect(self) -> float:
        """The screen's shape: that of one eye's picture."""
        return geo.eye_aspect(self.layout, self.info.width, self.info.height) if self.info else 16 / 9

    # -- opening ----------------------------------------------------------------------------------

    def open(self, path: Path, layout: str, loop: bool = False, muted: bool = False) -> bool:
        self.close()
        try:
            info = video.probe(path)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"Could not open {Path(path).name}: {error}")
            return False
        self.info, self.layout, self.path, self.loop = info, layout, Path(path), loop
        self._has_audio = info.has_audio
        self._output.setMuted(muted)
        self._audio.setLoops(self._audio.Loops.Infinite if loop else 1)
        self._clock.stop()
        self._clock.set(0.0)
        self._audio.setSource(QUrl.fromLocalFile(str(path)))
        feeder = _Feeder(Path(path), info, self._clock, self._delivered)
        feeder.size = target_size(layout, info.width, info.height)
        self._feeder = feeder
        feeder.start()
        feeder.seek(0.0, True)                               # the first picture; stay paused
        self.state_changed.emit(False)
        return True

    def close(self, wait: bool = False) -> None:
        feeder, self._feeder = self._feeder, None
        if feeder is not None:
            feeder.shutdown()
        self._audio.stop()
        self._audio.setSource(QUrl())
        self._playing = False
        with self._lock:
            self._latest, self._taken = None, True
        if wait and feeder is not None:
            feeder.join(3.0)

    def set_muted(self, muted: bool) -> None:
        self._output.setMuted(muted)

    @property
    def muted(self) -> bool:
        return self._output.isMuted()

    def set_audio_device(self, device) -> None:
        """Play through this Qt audio output device (UI thread). A change takes effect at once, even mid-film."""
        self._output.setDevice(device)

    def set_volume(self, volume: float) -> None:
        """0..1 (UI thread). Qt's volume is linear in amplitude; the ear is not, so square it for an even-feeling slider."""
        self._output.setVolume(min(1.0, max(0.0, volume)) ** 2)

    # -- transport (UI thread) --------------------------------------------------------------------

    def play(self) -> None:
        if self._feeder is None or self._playing:
            return
        if not self.loop and self.duration_ms and self._clock.now() >= self.duration_ms - 200:
            self.seek_ms(0)
        self._playing = True
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
        self._feeder.pause()
        self._audio.pause()
        self._clock.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek_ms(self, ms: int, precise: bool = True) -> None:
        if self._feeder is None:
            return
        ms = max(0, min(int(ms), max(0, self.duration_ms - 50)))
        self._clock.set(ms)
        if self._has_audio:
            self._audio.setPosition(ms)
        self._feeder.seek(ms, precise)

    def seek_relative(self, delta_ms: int) -> None:
        self.seek_ms(self.position_ms + delta_ms)

    # -- pictures out (any thread) ----------------------------------------------------------------

    def take_frame(self) -> np.ndarray | None:
        """The newest decoded frame if there is one the caller has not seen yet, else None."""
        with self._lock:
            if self._taken or self._latest is None:
                return None
            self._taken = True
            return self._latest

    def _delivered(self, rgb: np.ndarray, index: int, ms: float) -> None:
        with self._lock:
            self._latest, self._latest_ms, self._taken = rgb, ms, False
        feeder = self._feeder
        if feeder is not None and feeder.approximate:
            return
        self.position_changed.emit(int(ms))
        if not self.loop and self._playing and self.duration_ms and ms >= self.duration_ms - 120:
            self.ended.emit()

    def _audio_position(self, ms: int) -> None:
        if self._has_audio and self._playing:
            self._clock.set(ms)
