"""The video side of the VR player: decode, sound and timing, handing the newest picture to the VR thread.

Like the Library's player (``librarypage.LibraryPlayer``) and the preview, the picture is decoded on a thread of
its own and Qt plays the sound, whose position is the clock the pictures are paced against. The difference is
what comes out: a full-size ``uint8`` frame (the whole two-eye frame, as stored) that the VR thread uploads to
the GPU, rather than a small picture for a window. This object lives on the UI thread; the VR thread only calls
:meth:`take_frame`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, QUrl, Signal

from .. import bars, hdr, video
from ..player import Clock, FrameFeeder
from . import geometry as geo
from .log import log as vrlog

#: The picture runs on the system clock and the sound only steers it. Qt's audio position was used directly before, and when the
#: headset's sound device hiccuped (it reports the same position over and over, or jumps) the picture followed it: judder that no
#: pause or seek cured. Now small differences are slewed away gently, a big one (a seek, real drift) is followed at once, and a sound
#: that has stopped moving is restarted without the picture noticing.
RESYNC_MS = 300.0        # further than this from the sound: jump to it
SLEW = 0.1               # otherwise move this share of the difference per report (about 10 reports a second)
SLEW_BELOW_MS = 25.0     # and ignore differences this small (they are the sound buffer's own latency)
STALL_SECONDS = 2.0      # sound that has not moved for this long while playing is restarted

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
        self.trim_bars = True                           # show only the real picture of a film that has black bars burned into its frame
        self._detected = bars.NONE                      # what the scan of the open file found (None until it finishes)
        self._scan_token = 0
        self._bar_cache: dict[tuple, bars.Crop] = {}
        self._audio_seen = 0.0                          # the last time the sound's position moved, and where it was
        self._audio_last_ms = -1
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(1000)
        self._stall_timer.timeout.connect(self._check_audio)

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
    def crop(self) -> bars.Crop:
        """The black bars to leave out of one eye's picture (nothing, when trimming is off or the scan has not finished)."""
        return self._detected if self.trim_bars else bars.NONE

    @property
    def aspect(self) -> float:
        """The shape of the picture to put on the screen: one eye's picture, without its black bars."""
        base = geo.eye_aspect(self.layout, self.info.width, self.info.height) if self.info else 16 / 9
        return base * self.crop.aspect_factor

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
        self._scan_bars(Path(path), layout)
        self.state_changed.emit(False)
        return True

    def _scan_bars(self, path: Path, layout: str) -> None:
        """Look for burned-in black bars in the background (a few seconds on a big file); the picture is whole until it is done."""
        self._scan_token += 1
        token = self._scan_token
        self._detected = bars.NONE
        try:
            key = (str(path), path.stat().st_mtime_ns, layout)
        except OSError:
            return
        if key in self._bar_cache:
            self._detected = self._bar_cache[key]
            return

        def work() -> None:
            found = bars.detect(path, layout)
            self._bar_cache[key] = found
            if token == self._scan_token:                    # still the same film
                self._detected = found

        threading.Thread(target=work, name="uv3d-bars", daemon=True).start()

    def close(self, wait: bool = False) -> None:
        self._scan_token += 1
        self._detected = bars.NONE
        feeder, self._feeder = self._feeder, None
        if feeder is not None:
            feeder.shutdown()
        self._audio.stop()
        self._audio.setSource(QUrl())
        self._stall_timer.stop()
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
            self._audio_seen, self._audio_last_ms = time.monotonic(), -1
            self._stall_timer.start()
        self._feeder.play()
        self.state_changed.emit(True)

    def pause(self) -> None:
        if self._feeder is None or not self._playing:
            return
        self._playing = False
        self._stall_timer.stop()
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

    def stats_text(self) -> str:
        """How the picture side is doing, for the diary: frames shown, dropped for being late, average lateness of the shown ones."""
        feeder = self._feeder
        if feeder is None:
            return "no film"
        s = feeder.stats
        shown = max(1, s["shown"])
        return f"shown {s['shown']}, dropped {s['dropped']}, on average {s['behind_ms'] / shown:.0f} ms behind"

    def _audio_position(self, ms: int) -> None:
        if not (self._has_audio and self._playing):
            return
        if ms != self._audio_last_ms:
            self._audio_last_ms, self._audio_seen = ms, time.monotonic()
        drift = ms - self._clock.now()                  # positive: the sound is ahead of the picture clock
        if abs(drift) > RESYNC_MS:
            self._clock.set(ms)
        elif abs(drift) > SLEW_BELOW_MS:
            self._clock.nudge(drift * SLEW)

    def _check_audio(self) -> None:
        """Playing, but the sound's position has not moved: the device stalled (a headset that slept, a driver reset). Restart the
        sound where the picture is; the picture keeps running on its own clock meanwhile."""
        if not (self._playing and self._has_audio) or self._feeder is None:
            return
        if time.monotonic() - self._audio_seen < STALL_SECONDS:
            return
        vrlog(f"the sound stopped moving at {self._audio_last_ms} ms; restarting it")
        self._audio_seen = time.monotonic()
        self._audio.setPosition(int(self._clock.now()))
        self._audio.play()
