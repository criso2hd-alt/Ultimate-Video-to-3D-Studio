"""The desktop application: window, video page, workers and first-run setup."""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .discdialog import DiscDialog
from . import library as librarymod
from . import zones as zonesmod
from . import resume as resumemod
from .folderdialog import FolderDialog
from .librarypage import LibraryPage
from .vr.controller import VRController
from . import APP_TITLE, __version__, bootstrap, cloud, models, onboarding, support, updates, depth as depthmod, encoders, paths, pipeline, stereo, sysinfo, video, warmup
from .controls import PreviewView, SystemGauge
from .panels import ColourPanel, ExportPanel, LutPanel, ThreeDPanel, open_folder
from .player import PreviewPlayer
from .preview import (
    PREVIEW_MODES,
    PREVIEW_MODE_TIPS,
    PROXY_CHOICES,
    PreviewEngine,
    PreviewJob,
    PreviewParams,
    PreviewResult,
)
from .settings import AppSettings
from .theme import DEFAULT_THEME, PALETTES, qpalette_for, style_for
from .widgets import (
    CardColumns,
    ElidedLabel,
    FONT_DISPLAY,
    FONT_MONO,
    DownloadDialog,
    ModuleCard,
    StageHost,
    TimelineWidget,
    apply_font,
    format_duration,
    stage_placeholder,
)

#: The sidebar's own width. The scroll area around it adds room for a bar.
SIDEBAR_WIDTH = 344

GITHUB_URL = "https://github.com/criso2hd-alt/Ultimate-Video-to-3D-Studio"
#: A second, separate pin of the support banner's digest (see support.py): the banner's link and
#: message live there masked, and a copy whose two halves disagree is treated as modified.
_SUPPORT_PIN = "b03503fc3fd11920"


# --- small helpers ---------------------------------------------------------


class _EtaTracker:
    """Smoothed seconds-per-frame turned into a time-remaining estimate.

    The first gap of a run is dominated by start-up (model load, encoder open),
    which is nothing like steady state, so the average starts at the second frame
    and ``remaining`` returns None until there is a real rate to quote.
    """

    def __init__(self, smoothing: float = 0.2) -> None:
        self._alpha = smoothing
        self._ema: float | None = None
        self._prev: float | None = None

    def tick(self) -> None:
        now = time.monotonic()
        if self._prev is not None:
            dt = now - self._prev
            self._ema = dt if self._ema is None else self._alpha * dt + (1 - self._alpha) * self._ema
        self._prev = now

    def remaining(self, frames_left: int) -> float | None:
        if self._ema is None or frames_left <= 0:
            return None
        return self._ema * frames_left


def _short_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    hours, rest = divmod(int(round(seconds)), 3600)
    return f"{hours}h {rest // 60:02d}m"


class WheelGuard(QObject):
    """The mouse wheel scrolls the panel; it never changes a control it merely passes over.

    Qt lets the wheel adjust any slider or drop-down under the cursor, so scrolling
    the side rail moves whatever the pointer happens to cross - quietly changing
    settings. Here a slider, drop-down or spin box inside a scrolling panel never takes
    the wheel at all: it is handed to the scroll area that contains it. (An earlier
    version let a control keep the wheel once it had been clicked - but a clicked control
    stays focused, so the next scroll past it changed it. Arrow keys still work.)
    """

    _CONTROLS = (QSlider, QComboBox, QAbstractSpinBox)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt name
        if event.type() != QEvent.Type.Wheel or not isinstance(obj, self._CONTROLS):
            return False
        parent = obj.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is None:
            return False                       # not in a scrolling panel: nothing to hand it to
        QApplication.sendEvent(parent.viewport(), event)
        return True


# --- workers ---------------------------------------------------------------


class ProbeWorker(QObject):
    """First-launch hardware detection and encoder probe (a few seconds)."""

    status = Signal(str)
    finished = Signal(object, object)      # list[GpuInfo], ProbeReport
    failed = Signal(str)

    def run(self) -> None:
        try:
            self.status.emit("Detecting your graphics hardware…")
            gpus = sysinfo.detect_gpus()
            report = encoders.probe_all(gpus, self.status.emit)
        except Exception as error:  # noqa: BLE001 - reported in the UI
            self.failed.emit(str(error))
            return
        self.finished.emit(gpus, report)


class UpdateCheckWorker(QObject):
    """Asks GitHub whether a newer version exists, off the UI thread."""

    finished = Signal(object)                  # UpdateInfo or None

    def run(self) -> None:
        self.finished.emit(updates.check_for_update())


class VideoDownloadWorker(QObject):
    """Fetches PyAV the first time video is used."""

    progress = Signal(object, object)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            bootstrap.install_av(on_bytes=lambda d, t: self.progress.emit(d, t), on_text=self.status.emit)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))
            return
        self.finished.emit()


class ModelDownloadWorker(QObject):
    """Fetches the depth model from the project's GitHub release if the app does not have it."""

    progress = Signal(object, object)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            models.install(on_bytes=lambda d, t: self.progress.emit(d, t), on_text=self.status.emit)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))
            return
        self.finished.emit()


class ConvertWorker(QObject):
    """Drives one conversion off the UI thread."""

    progress = Signal(str)
    frame_done = Signal(object)
    finished = Signal(object)
    stopped = Signal(int, int, bool)          # frames done, total frames, whether it can be resumed
    failed = Signal(str)

    def __init__(self, source: Path, destination: Path, settings: AppSettings, start: int,
                 limit: int | None, gpus: list, resume: bool = False) -> None:
        super().__init__()
        self._args = (source, destination, settings, start, limit)
        self._gpus = gpus
        self._resume = resume
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        source, destination, settings, start, limit = self._args
        try:
            for update in pipeline.convert_video(
                source, destination, settings, start, limit, self.progress.emit,
                lambda: self._stop, self._gpus, self._resume,
            ):
                if update.stage == "stopped":
                    self.stopped.emit(update.index, update.total, update.message == "paused")
                    return
                self.frame_done.emit(update)
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(destination)


class BatchWorker(QObject):
    """Converts a list of videos back to back. One bad file is reported, not fatal."""

    progress = Signal(str)
    started = Signal(int, int, object)
    frame_done = Signal(object)
    done = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal()

    def __init__(self, jobs: list[tuple], settings: AppSettings, gpus: list, skip_existing: bool) -> None:
        # a job is (source, output) or (source, output, audio stream index, subtitle stream index)
        super().__init__()
        self._jobs, self._settings, self._gpus, self._skip = jobs, settings, gpus, skip_existing
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        total = len(self._jobs)
        for index, job in enumerate(self._jobs):
            source, output = job[0], job[1]
            settings = copy.deepcopy(self._settings)                 # each file has its own track choice
            settings.export.audio_track, settings.export.subtitle_track = (job[2], job[3]) if len(job) > 3 else (-1, -1)
            if self._stop:
                break
            if self._skip and output.exists():
                self.failed.emit(source, "Skipped: output already exists.")
                continue
            self.started.emit(index, total, source)
            try:
                for update in pipeline.convert_video(
                    source, output, settings, 0, None, self.progress.emit,
                    lambda: self._stop, self._gpus,
                ):
                    if update.stage == "stopped":
                        break
                    self.frame_done.emit(update)
            except Exception as error:  # noqa: BLE001 - reported, the queue moves on
                self.failed.emit(source, str(error))
                continue
            if not self._stop:
                self.done.emit(source, output)
        self.finished.emit()


class BatchDialog(QDialog):
    """Queue several videos and convert them with the current settings."""

    converted = Signal(object, object)         # (source, output) for each finished video

    def __init__(self, parent: QWidget, settings: AppSettings, gpus: list, style: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert several videos")
        self.setStyleSheet(style)
        self.setMinimumSize(560, 460)
        self._settings, self._gpus = settings, gpus
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._out_dir = Path(settings.last_video_dir) if settings.last_video_dir else paths.output_dir()

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 16, 18, 16)
        col.setSpacing(10)
        note = QLabel("Each video is converted whole with the settings on the main screen. "
                      "One that fails is reported and the queue carries on.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        col.addWidget(note)
        self.list = QListWidget()
        col.addWidget(self.list, 1)
        row = QHBoxLayout()
        for label, slot in (("Add videos…", self._add), ("Remove", self._remove), ("Clear", self.list.clear)):
            b = QPushButton(label)
            b.setObjectName("secondary")
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)
        out = QHBoxLayout()
        self.out_label = QLabel(f"Save to: {self._out_dir}")
        self.out_label.setObjectName("hint")
        out.addWidget(self.out_label, 1)
        b = QPushButton("Folder…")
        b.setObjectName("secondary")
        b.clicked.connect(self._pick_folder)
        out.addWidget(b)
        col.addLayout(out)
        self.skip = QCheckBox("Skip videos already converted")
        self.skip.setChecked(True)
        col.addWidget(self.skip)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        col.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        col.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start = QPushButton("Start")
        self.start.clicked.connect(self._start)
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("secondary")
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.start)
        col.addLayout(buttons)

    def _add(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(video.INPUT_SUFFIXES))
        files, _ = QFileDialog.getOpenFileNames(self, "Add videos", self._settings.last_video_dir, f"Video ({exts})")
        for f in files:
            self.add_job(Path(f), -1, -1)

    def add_job(self, path: Path, audio_index: int = -1, subtitle_index: int = -1) -> None:
        """Queue a video; the two numbers are the audio / subtitle stream to keep (-1: the best audio, no subtitles)."""
        item = QListWidgetItem(str(path))
        item.setData(Qt.ItemDataRole.UserRole, (audio_index, subtitle_index))
        try:
            audio, subtitles = video.list_tracks(path)
            wanted = audio_index if audio_index >= 0 else video.best_audio_index(audio)
            sound = next((t.label() for t in audio if t.index == wanted), "no audio")
            subs = next((t.label() for t in subtitles if t.index == subtitle_index), "none")
            item.setToolTip(f"Audio: {sound}\nSubtitles: {subs}")
        except Exception:  # noqa: BLE001 - the tooltip is a courtesy
            pass
        self.list.addItem(item)

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _pick_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Where should the videos go?", str(self._out_dir))
        if chosen:
            self._out_dir = Path(chosen)
            self.out_label.setText(f"Save to: {chosen}")

    def _start(self) -> None:
        if self._thread is not None:
            self._worker.stop()
            self.status.setText("Stopping after this frame…")
            return
        queue = [(Path(self.list.item(i).text()), *(self.list.item(i).data(Qt.ItemDataRole.UserRole) or (-1, -1)))
                 for i in range(self.list.count())]
        if not queue:
            return
        jobs = [(s, pipeline.output_path_for(s, self._settings, self._out_dir), a, t) for s, a, t in queue]
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.start.setText("Stop")
        self._thread = QThread(self)
        self._worker = BatchWorker(jobs, copy.deepcopy(self._settings), self._gpus, self.skip.isChecked())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._file_started)
        self._worker.frame_done.connect(self._frame)
        self._worker.done.connect(lambda s, o: self.status.setText(f"Finished {Path(s).name}"))
        self._worker.done.connect(self.converted)
        self._worker.failed.connect(lambda s, m: self.status.setText(f"{Path(s).name}: {m}"))
        self._worker.finished.connect(self._finished)
        self._thread.start()

    def _file_started(self, index: int, total: int, source) -> None:
        self.status.setText(f"Video {index + 1} of {total}: {Path(source).name}")

    def _frame(self, update) -> None:
        if update.stage == "converting" and update.total:
            self.bar.setRange(0, update.total)
            self.bar.setValue(update.index)

    def _finished(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = self._worker = None
        self.bar.setVisible(False)
        self.start.setText("Start")
        self.status.setText("All done.")

    def reject(self) -> None:  # noqa: D102
        if self._worker is not None:
            self._worker.stop()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(20000)
        super().reject()


# --- the video page --------------------------------------------------------


class VideoPage(QWidget):
    """Same shape as the DLSS app's Video tab: a bordered stage on the left with a
    permanent view bar, transport and timeline under it, and a rail of cards
    on the right with the convert actions pinned at its foot."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source: Path | None = None
        self.disc_stream = None                 # a title converted straight from a disc (source is then its preview sample)
        self.info: video.VideoInfo | None = None
        self.output_path: Path | None = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # -- stage: empty state, or the live preview
        self.placeholder = stage_placeholder(
            "🎞", "Drop a video here, or choose one",
            "MP4 · MOV · MKV — then scrub, tune the 3D, and convert",
        )
        self.preview = PreviewView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.preview)

        # The view bar is permanent, under the picture and stretched across it: 2D first, then the
        # 3D views. Picking any 3D view is what turns the effect on - there is no separate switch.
        self.mode_buttons: dict[str, QPushButton] = {}
        self.mode_bar = QFrame()
        self.mode_bar.setObjectName("modeBar")
        bar_row = QHBoxLayout(self.mode_bar)
        bar_row.setContentsMargins(5, 5, 5, 5)
        bar_row.setSpacing(2)
        for key, label in PREVIEW_MODES:
            button = QPushButton(label)
            button.setObjectName("viewChip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(PREVIEW_MODE_TIPS.get(key, ""))
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # Without this the bar's minimum width is its labels' full width, which - with a larger system
            # font - would push the window wider than its own minimum. Squeezed, the chips share the room.
            button.setMinimumWidth(28)
            bar_row.addWidget(button)
            self.mode_buttons[key] = button
        self.stage_host = StageHost(self.stack)
        stage = QFrame()
        stage.setObjectName("stage")
        stage_v = QVBoxLayout(stage)
        stage_v.setContentsMargins(0, 0, 0, 0)
        stage_v.addWidget(self.stage_host, 1)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("secondary")
        self.play_button.setFixedWidth(80)
        self.play_button.setEnabled(False)
        transport.addWidget(self.play_button)
        self.timeline = TimelineWidget()
        transport.addWidget(self.timeline, 1)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(stage, 1)
        left.addWidget(self.mode_bar)
        left.addLayout(transport)
        left.addWidget(self.bar)
        outer.addLayout(left, 1)

        # -- rail
        source_card = ModuleCard("Source")
        self.pick_video = QPushButton("Choose video file…")
        # Copies a DVD / Blu-ray from your own drive to a video file first (discdialog.py), then that file
        # is opened like any other.
        self.pick_disc = QPushButton("Choose external disc…")
        self.pick_disc.setObjectName("secondary")
        self.pick_disc.setToolTip(
            "Copy a DVD or Blu-ray from your own drive to a video file on this computer, "
            "then turn it into 3D.")
        # A folder of videos, typically a ripped DVD or Blu-ray: lists every file with its audio and subtitle tracks (folderdialog.py).
        self.pick_folder = QPushButton("Open a folder of videos…")
        self.pick_folder.setObjectName("secondary")
        self.pick_folder.setToolTip(
            "Open a folder full of videos - for example a ripped disc, one .mkv per title - see each one's sound and "
            "subtitle tracks, and turn one or all of them into 3D.")
        self.source_label = ElidedLabel("No video chosen")
        self.source_label.setObjectName("hint")
        self.pick_output = QPushButton("Output…")
        self.pick_output.setObjectName("secondary")
        self.output_label = ElidedLabel("Output: choose a video first")
        self.output_label.setObjectName("hint")
        for w in (self.pick_video, self.pick_folder, self.pick_disc, self.source_label, self.pick_output, self.output_label):
            source_card.add(w)
        self.source_card = source_card

        self.threed = ThreeDPanel(settings.threed)
        self.colour = ColourPanel(settings.colour)
        self.lut = LutPanel(settings.colour)
        self.export = ExportPanel(settings.export)

        self.info_label = QLabel("")
        self.info_label.setObjectName("hint")
        self.info_label.setWordWrap(True)
        apply_font(self.info_label, family=FONT_MONO, size=8.5)

        column = QWidget()
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(12)
        for w in (source_card, self.threed, self.colour, self.lut, self.export, self.info_label):
            col.addWidget(w)
        col.addStretch(1)
        scroller = QScrollArea()
        scroller.setWidget(column)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.rail_scroller = scroller                   # the tutorial scrolls it to reach each card

        self.queue_button = QPushButton("Convert several…")
        self.queue_button.setObjectName("secondary")
        self.queue_button.setToolTip("Queue several videos and convert them back to back with these settings.")
        self.start = QPushButton("Convert video")
        self.start.setObjectName("convert")
        self.start.setEnabled(False)
        apply_font(self.start, family=FONT_DISPLAY, size=11.5, spacing=1.8, caps=True)
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("secondary")
        self.stop.setVisible(False)

        rail = QWidget()
        rail.setFixedWidth(SIDEBAR_WIDTH + 18)
        rail_col = QVBoxLayout(rail)
        rail_col.setContentsMargins(0, 0, 0, 0)
        rail_col.setSpacing(10)
        rail_col.addWidget(scroller, 1)
        rail_col.addWidget(self.queue_button)
        rail_col.addWidget(self.stop)
        rail_col.addWidget(self.start)
        outer.addWidget(rail)

    def show_preview(self) -> None:
        self.stack.setCurrentWidget(self.preview)

    def show_placeholder(self) -> None:
        self.stack.setCurrentWidget(self.placeholder)


# --- settings page ---------------------------------------------------------


class SettingsPage(QWidget):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)     # a bare QWidget page paints nothing, which shows as white
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        cards: list[QWidget] = []

        look = ModuleCard("Appearance")
        self.theme_box = QComboBox()
        for name in PALETTES:
            self.theme_box.addItem(name)
        self.theme_box.setCurrentText(settings.theme if settings.theme in PALETTES else DEFAULT_THEME)
        row = QHBoxLayout()
        row.addWidget(QLabel("Theme"))
        row.addStretch(1)
        self.theme_box.setMinimumWidth(180)
        row.addWidget(self.theme_box)
        look.add_layout(row)
        cards.append(look)

        perf = ModuleCard("Performance")
        self.cpu_depth = QCheckBox("Run depth on the CPU (slow; for troubleshooting)")
        self.cpu_depth.setChecked(settings.force_cpu_depth)
        perf.add(self.cpu_depth)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview resolution"))
        row.addStretch(1)
        self.preview_box = QComboBox()
        for h in PROXY_CHOICES:
            tag = {360: "  (fastest)", 480: "", 540: "  (recommended)", 720: "  (sharpest)"}[h]
            self.preview_box.addItem(f"{h}p{tag}", h)
        self.preview_box.setCurrentIndex(max(0, self.preview_box.findData(settings.preview_height)))
        self.preview_box.setMinimumWidth(150)
        row.addWidget(self.preview_box)
        perf.add_layout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview buffer"))
        row.addStretch(1)
        self.buffer_box = QComboBox()
        for label, seconds in (("Off (render live)", 0), ("5 seconds", 5), ("10 seconds", 10)):
            self.buffer_box.addItem(label, seconds)
        self.buffer_box.setCurrentIndex(max(0, self.buffer_box.findData(settings.preview_buffer)))
        self.buffer_box.setToolTip(
            "While the 3D preview plays, render this far ahead so playback stays smooth. "
            "Longer rides out slower moments; the buffer refills when you change a setting. "
            "(While paused it works further ahead, into a cache on disk.)")
        self.buffer_box.setMinimumWidth(150)
        row.addWidget(self.buffer_box)
        perf.add_layout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame rate"))
        row.addStretch(1)
        self.fps_box = QComboBox()
        for label, value in (("Auto (recommended)", 0), ("30 fps", 30), ("24 fps", 24)):
            self.fps_box.addItem(label, value)
        self.fps_box.setCurrentIndex(max(0, self.fps_box.findData(settings.preview_fps)))
        self.fps_box.setToolTip(
            "Auto measures this computer and picks 30 or 24 fps - and steps down if playback ever "
            "catches up with the buffer. Only the preview is affected; exports render every frame.")
        self.fps_box.setMinimumWidth(150)
        row.addWidget(self.fps_box)
        perf.add_layout(row)
        note = QLabel("The live preview runs on a small proxy so it stays smooth while playing; "
                      "lower it if playback stutters. Exports always use the full-size picture.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        perf.add(note)
        cards.append(perf)

        hw = ModuleCard("Hardware")
        self.hw_label = QLabel("Detecting…")
        self.hw_label.setObjectName("hint")
        self.hw_label.setWordWrap(True)
        self.redetect = QPushButton("Re-detect hardware and encoders")
        self.redetect.setObjectName("secondary")
        hw.add(self.hw_label)
        hw.add(self.redetect)
        cards.append(hw)

        about = ModuleCard("Updates")
        self.update_check = QCheckBox("Check for a new version when the app starts")
        self.update_check.setChecked(settings.check_updates)
        self.update_check.setToolTip(
            "Sends one anonymous request to GitHub and, if a newer version exists, shows a notice with "
            "a link. Nothing is downloaded or installed automatically.")
        about.add(self.update_check)
        self.update_now = QPushButton("Check now")
        self.update_now.setObjectName("secondary")
        self.update_label = QLabel(f"You have version {__version__}.")
        self.update_label.setObjectName("hint")
        self.update_label.setWordWrap(True)
        about.add(self.update_now)
        about.add(self.update_label)
        cards.append(about)

        help_card = ModuleCard("Help")
        self.replay_tutorial = QPushButton("Replay the tutorial…")
        self.replay_tutorial.setObjectName("secondary")
        self.replay_tutorial.setToolTip("Show the welcome and the guided tour of the window again.")
        help_card.add(self.replay_tutorial)
        cards.append(help_card)

        vr_card = ModuleCard("Virtual reality")
        self.vr_card = vr_card
        self.vr_status = QLabel("")
        self.vr_status.setObjectName("hint")
        self.vr_status.setWordWrap(True)
        self.vr_room_label = QLabel("")
        self.vr_room_label.setObjectName("hint")
        self.vr_room_label.setWordWrap(True)
        self.vr_auto = QCheckBox("Also look for my headset by itself while the Library is open")
        self.vr_auto.setChecked(settings.vr_auto_connect)
        self.vr_auto.setToolTip("Off by default: use the Start VR button in the Library. With this on, the app looks for a headset every "
                                "few seconds while the Library is open and connects to it by itself. That depends on the headset "
                                "software behaving, so if starting VR ever fails or sticks, leave this off. "
                                "Leaving VR from the headset's menu pauses it until you ask for VR again.")
        self.vr_trim = QCheckBox("Cut the black bars off wide films")
        self.vr_trim.setChecked(settings.vr_trim_bars)
        self.vr_trim.setToolTip("Many films are stored as a 16:9 picture with black bars burned in above and below. The VR screen then "
                                "shows only the real picture, so a wide film fills a wide screen. Nothing in the file is changed.")
        self.vr_beam = QCheckBox("Show the projector's light beam in the room")
        self.vr_beam.setChecked(settings.vr_beam)
        self.vr_beam.setToolTip("The faint beam of light from the ceiling projector to the screen, while a film plays with the lights "
                                "down. The projector's lens keeps glowing either way. You can also change this in the headset's menu.")
        self.vr_beam_box = QComboBox()
        for level in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            self.vr_beam_box.addItem(f"{round(level * 100)}%", level)
        self.vr_beam_box.setCurrentIndex(max(0, self.vr_beam_box.findData(min((0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                                                                                key=lambda v: abs(v - settings.vr_beam_level)))))
        self.vr_beam_box.setToolTip("How bright the beam is: 100% is the normal strength.")
        self.vr_beam_box.setMinimumWidth(110)
        beam_row = QHBoxLayout()
        beam_row.addWidget(QLabel("Beam brightness"))
        beam_row.addStretch(1)
        beam_row.addWidget(self.vr_beam_box)
        self.vr_rooms_folder = QPushButton("Open the rooms folder")
        self.vr_rooms_folder.setObjectName("secondary")
        self.vr_rooms_folder.setToolTip("Every folder in here that holds a room.glb is a room you can pick in the headset. Put a "
                                        "thumbnail.jpg in a room's folder to give it a picture.")
        for widget in (self.vr_status, self.vr_auto, self.vr_trim, self.vr_beam):
            vr_card.add(widget)
        vr_card.add_layout(beam_row)
        for widget in (self.vr_room_label, self.vr_rooms_folder):
            vr_card.add(widget)
        cards.append(vr_card)

        folders = ModuleCard("Folders")
        self.open_output = QPushButton("Open output folder")
        self.open_output.setObjectName("secondary")
        self.open_luts = QPushButton("Open LUT folder")
        self.open_luts.setObjectName("secondary")
        folders.add(self.open_output)
        folders.add(self.open_luts)
        cards.append(folders)

        # The cards flow into as many columns as the window is wide enough for.
        holder = CardColumns(cards)
        scroller = QScrollArea()
        scroller.setWidget(holder)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.columns = holder
        outer.addWidget(scroller)


# --- the window ------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        icon_path = paths.package_dir() / "assets" / "icon.png"
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, available.width() - 40), min(860, available.height() - 60))
        self.setMinimumSize(980, 600)
        self.setAcceptDrops(True)
        self._wheel_guard = WheelGuard(self)
        QApplication.instance().installEventFilter(self._wheel_guard)

        self.settings = AppSettings.load(paths.settings_path())
        self.gpus: list[sysinfo.GpuInfo] = []
        self.report: encoders.ProbeReport | None = (
            encoders.ProbeReport(**self.settings.encoder_probe) if self.settings.encoder_probe else None
        )
        self._thread: QThread | None = None
        self._worker: ConvertWorker | None = None
        self._dl_thread: QThread | None = None
        self._dl_worker = None
        self._dl_after = None
        self._dl_retry = None
        self._dl_failed_title = ""
        self._probe_thread: QThread | None = None
        self._probe_dialog: DownloadDialog | None = None
        self._eta = _EtaTracker()
        self._convert_began, self._convert_note, self._convert_fps = 0.0, "", 0.0
        self._converting = False
        #: (proxy frame, id) written by the decode thread as one atomic assignment.
        self._current: tuple[np.ndarray, int] | None = None
        self._frame_counter = 0
        #: Immutable snapshot of the preview settings, rebuilt on the UI thread and
        #: read by the decode thread - so that thread never touches live settings.
        self._params = PreviewParams()
        self._monitor: sysinfo.LoadMonitor | None = None
        self._live_note = ""                       # the preview's caption, and what is currently drawn
        self._shown_note = ""
        self._update: updates.UpdateInfo | None = None
        self._update_thread: QThread | None = None
        self._update_manual = False

        self._support_strikes = 0
        self._tour_overlay: onboarding.SpotlightOverlay | None = None
        self._welcome_open = False
        central = QWidget()
        root = self._root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._command_bar())

        self.video_page = VideoPage(self.settings)
        self.settings_page = SettingsPage(self.settings)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.library = librarymod.Library()
        self.library_page = LibraryPage(self.library)
        self.library_page.message.connect(lambda text: self.status.showMessage(text))
        self.vr = VRController(self)
        self.vr.room_provider = self._vr_room_dir                   # a session that starts by itself still sits in the chosen room
        self.vr.notice.connect(lambda text: self.status.showMessage(text))
        self.vr.room_chosen.connect(self._vr_room_chosen)
        from .vr import unavailable_reason

        reason = unavailable_reason()
        self.library_page.set_vr_available(not reason, reason)
        self.vr.failed.connect(self._vr_failed)
        self.library_page.vr_requested.connect(self._play_in_vr)
        self.library_page.vr_start_requested.connect(self._start_vr)
        self.library_page.vr_stop_requested.connect(lambda: self.vr.stop())
        self.vr.state_changed.connect(lambda state: self.library_page.set_vr_state(state, reason))
        self.library_page.space_hook = self._space_for_vr
        QApplication.instance().installEventFilter(self)
        self.tabs.addTab(self.video_page, "Video")
        self.tabs.addTab(self.library_page, "Library")
        self.tabs.addTab(self.settings_page, "Settings")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tab_apply = QPushButton("+  Convert several")
        self.tab_apply.setObjectName("tabAction")
        self.tab_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tab_apply.clicked.connect(self.open_batch)
        corner = QFrame()
        corner.setObjectName("tabCorner")
        corner_row = QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 0, 0, 0)
        corner_row.addWidget(self.tab_apply)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        root.addWidget(self.tabs, 1)
        # The status line lives inside the window (not QMainWindow's own status bar)
        # so the support banner can sit below it, at the very bottom.
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)
        root.addWidget(self.status)
        self.banner = support.SupportBanner()
        root.addWidget(self.banner)
        self.setCentralWidget(central)
        self._build_footer()
        self.setStyleSheet(style_for(self.settings.theme))

        self._guard_timer = QTimer(self)                 # the banner is re-checked every few seconds
        self._guard_timer.setInterval(7000)
        self._guard_timer.timeout.connect(self._guard_support)
        self._guard_timer.start()
        QTimer.singleShot(1500, self._guard_support)

        self._setup_player()
        self._setup_preview()
        self._wire()
        self._sync_mode_buttons()
        self.export_refresh()

        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.pick_video)
        QTimer.singleShot(0, self._start_initial_setup)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        if self._tour_overlay is not None:
            self._tour_overlay.setGeometry(self.rect())          # the dimming layer follows the window
        # The command bar holds the wordmark, the gauge and up to two pills; below this width
        # the gauge would overflow into its neighbours, so it drops its optional cells.
        need = 1050 + (155 if self.update_pill.isVisible() else 0)
        self.gauge.set_compact(event.size().width() < need)

    # -- chrome ------------------------------------------------------------

    def _command_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("commandBar")
        bar.setFixedHeight(64)
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 10, 20, 10)
        row.setSpacing(16)

        glyph = QLabel()
        icon = paths.package_dir() / "assets" / "icon.png"
        if icon.is_file():
            glyph.setPixmap(QPixmap(str(icon)).scaled(
                44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        glyph.setFixedSize(44, 44)
        row.addWidget(glyph)

        self.wordmark = QLabel()
        self.wordmark.setObjectName("wordmark")
        apply_font(self.wordmark, family=FONT_DISPLAY, size=12.5, spacing=2.6)
        self._set_wordmark()
        row.addWidget(self.wordmark)
        row.addStretch(1)

        self.gauge = SystemGauge()
        row.addWidget(self.gauge)
        # Appears only when a newer version exists; clicking opens the download page.
        self.update_pill = QPushButton()
        self.update_pill.setObjectName("updatePill")
        self.update_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_pill.setVisible(False)
        self.update_pill.clicked.connect(self._open_update_page)
        apply_font(self.update_pill, family=FONT_MONO, size=8.5, spacing=1.2)
        row.addWidget(self.update_pill)
        self.encoder_pill = QLabel("●  DETECTING")
        self.encoder_pill.setObjectName("encoderPill")
        apply_font(self.encoder_pill, family=FONT_MONO, size=8.5, spacing=1.4)
        row.addWidget(self.encoder_pill)
        return bar

    def _set_wordmark(self) -> None:
        signal = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])["signal"]
        self.wordmark.setText(
            f'ULTIMATE&nbsp;VIDEO&nbsp;TO&nbsp;<span style="color:{signal};">3D</span>&nbsp;STUDIO')

    # -- the support banner ------------------------------------------------
    # It can be closed for the session with its X (support.SupportBanner) and returns at the next
    # launch. What it says and where it points is checked here and in support.py - see that module.

    def _guard_support(self) -> bool:
        """Is the banner intact? If not, put the official one back; a copy that keeps failing is locked."""
        try:
            ok = support.DIGEST.startswith(_SUPPORT_PIN) and support.intact() and self.banner.verify()
        except Exception:  # noqa: BLE001 - a deleted or broken banner is a failed check
            ok = False
        if ok:
            return True
        self._support_strikes += 1
        self._install_banner()
        if self._support_strikes >= 2:
            self._lock_modified_copy()
        return False

    def _install_banner(self) -> None:
        try:
            self._root.removeWidget(self.banner)
            self.banner.hide()
            self.banner.deleteLater()
        except Exception:  # noqa: BLE001
            pass
        self.banner = support.SupportBanner()            # a fresh one, built from the official values
        self._root.addWidget(self.banner)
        self.banner.show()

    def _lock_modified_copy(self) -> None:
        page = self.video_page
        for button in (page.start, page.queue_button, self.tab_apply):
            button.setEnabled(False)
        self.status.showMessage(
            "This copy has been modified (its support banner was changed or removed), so converting is "
            f"switched off. Please get the official version: {GITHUB_URL}")

    def _build_footer(self) -> None:
        github = QPushButton("GitHub")
        github.setObjectName("link")
        github.setCursor(Qt.CursorShape.PointingHandCursor)
        github.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        sep = QLabel("·")
        sep.setObjectName("linkSep")
        version = QLabel(f"v{__version__}")
        version.setObjectName("footerVersion")
        self.status.addPermanentWidget(version)
        self.status.addPermanentWidget(sep)
        self.status.addPermanentWidget(github)
        self.status.showMessage("Choose a video to begin.")

    # -- player and preview ------------------------------------------------

    def _setup_player(self) -> None:
        page = self.video_page
        self.player = PreviewPlayer(self)
        self.player.proxy_height = self.settings.preview_height
        self.player.on_frame = self._on_proxy_frame            # called on the decode thread
        self.player.position_changed.connect(self._position_changed)
        self.player.state_changed.connect(self._playback_state)
        self.player.buffered_image.connect(self._on_buffered_image)
        self.player.buffering_changed.connect(self._on_buffering)
        self.player.notice.connect(self._on_player_notice)
        self.player.smaller_preview_needed.connect(self._smaller_preview)
        self.player.set_preview_fps(self.settings.preview_fps)
        # The timeline's buffer bar: refreshed a few times a second, only while relevant.
        self._buffer_bar_timer = QTimer(self)
        self._buffer_bar_timer.setInterval(120)
        self._buffer_bar_timer.timeout.connect(self._update_buffer_bar)
        self._buffer_bar_timer.start()
        self.player.failed.connect(lambda m: self.status.showMessage(f"Preview: {m}"))
        self._scrub_target = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(70)
        self._scrub_timer.timeout.connect(self._scrub_preview)
        page.timeline.seeked.connect(self._scrub)
        page.timeline.scrub_finished.connect(self._scrub_done)
        page.play_button.clicked.connect(self.player.toggle)

    def _setup_preview(self) -> None:
        self.engine = PreviewEngine(self)
        self.engine.rendered.connect(self._on_rendered)
        self.engine.device_ready.connect(lambda d: self.status.showMessage(f"Depth running on: {d}"))
        self.engine.failed.connect(lambda m: self.status.showMessage(f"Preview: {m}"))
        # The 3D orbit view animates on its own clock rather than per video frame.
        self._orbit_src: tuple[np.ndarray, np.ndarray] | None = None
        self._orbit_phase = 0.0
        self._orbit_timer = QTimer(self)
        self._orbit_timer.setInterval(45)
        self._orbit_timer.timeout.connect(self._tick_orbit)
        self._rebuild_params()

    def _effective_mode(self) -> str:
        """What the preview shows: the plain video unless the 3D effect has been switched on."""
        s = self.settings
        if not (s.preview_3d and s.threed.enabled):
            return "original"
        if s.preview_mode in dict(PREVIEW_MODES) and s.preview_mode != "original":
            return s.preview_mode
        return "anaglyph"

    def _rebuild_params(self) -> None:
        s = self.settings
        self._params = PreviewParams(
            mode=self._effective_mode(), layout=s.threed.layout, stereo=s.threed.params(),
            temporal=s.threed.temporal, edge_aware=s.threed.edge_aware, depth_blur=s.threed.depth_blur,
            depth_mode=s.threed.depth_quality,
            colour=s.colour.copy(), force_cpu=s.force_cpu_depth,
        )
        self.player.set_buffer_config(self._params if self._params.needs_depth else None, s.preview_buffer)

    def _on_proxy_frame(self, rgb: np.ndarray, index: int, ms: float) -> None:
        """Runs on the decode thread: straight to the preview engine, never via the UI thread."""
        if self._converting:
            return
        self._frame_counter += 1
        self._current = (rgb, self._frame_counter)
        self.engine.submit(PreviewJob(rgb, self._frame_counter, True, self._params))

    def _submit(self, new_frame: bool = False) -> None:
        """Re-render the current picture with new settings (UI thread)."""
        self._rebuild_params()
        current = self._current
        if current is None or self._converting:
            return
        self.engine.submit(PreviewJob(current[0], current[1], new_frame, self._params))

    def _on_rendered(self, result: PreviewResult) -> None:
        if self._converting or self.player.buffer_playing:
            return                               # the look-ahead cache is driving the picture
        page = self.video_page
        if page.stack.currentWidget() is not page.preview:
            page.show_preview()
        if page.colour.is_open():
            page.colour.scope.set_frame(result.scope)
        if self._effective_mode() == "orbit" and result.depth is not None:
            self._orbit_src = (result.colour, result.depth)
            if not self._orbit_timer.isActive():
                self._orbit_timer.start()
            self._tick_orbit()
            return
        self._orbit_timer.stop()
        page.preview.set_images(result.images, result.badge, result.wiggle)
        stats = result.stats
        if stats.get("zones"):
            page.threed.set_depth_stats(zonesmod.DepthStats.from_dict(stats["zones"]))
        if stats:
            speed = f"{stats.get('fps', 0):.0f} fps · " if self.player.playing else ""
            self._live_note = (f"{result.device} · {result.images[0].shape[0]}p preview · {speed}"
                               f"{stats.get('hole_pct', 0):.1f}% of pixels filled")
        else:
            self._live_note = ""
        self._shown_note = self._live_note
        page.preview.set_note(self._live_note)

    def _on_buffered_image(self, payload) -> None:
        """A frame from the look-ahead cache, shown on the playback clock."""
        if self._converting:
            return
        images, badge, wiggle, hole_pct, scope = payload
        page = self.video_page
        if scope is not None and page.colour.is_open():
            page.colour.scope.set_frame(scope)            # the scopes keep following the picture while it plays
        page.preview.set_images(images, badge, wiggle)
        page.preview.set_note(
            f"buffered · {images[0].shape[0]}p preview · {self.player.buffer_ahead_seconds():.1f}s ahead · "
            f"{hole_pct:.1f}% of pixels filled")

    def _update_buffer_bar(self) -> None:
        page = self.video_page
        if not self.player.buffer_visible():
            page.timeline.set_buffered(0, 0)
            return
        first, last, waiting = self.player.buffered_frames()
        page.timeline.set_buffered(first, last, waiting)
        if not self.player.playing and not self._converting and self._live_note:
            # Paused with a 3D view: the renderer keeps working ahead, and the note says how far.
            ahead = self.player.buffer_ahead_seconds()
            text = f"{self._live_note} · {ahead:.0f}s cached, ready to play" if ahead >= 1 else self._live_note
            if text != self._shown_note:
                self._shown_note = text
                page.preview.set_note(text)

    def _on_player_notice(self, message: str) -> None:
        self.status.showMessage(message, 12000)

    def _smaller_preview(self) -> None:
        """Even the lowest frame rate ran dry: step the preview down a resolution."""
        current = self.settings.preview_height
        lower = [h for h in PROXY_CHOICES if h < current]
        if not lower:
            return
        height = lower[-1]
        self.status.showMessage(f"Still catching up, so the preview is now {height}p.", 12000)
        self.settings_page.preview_box.setCurrentIndex(self.settings_page.preview_box.findData(height))

    def _on_buffering(self, waiting: bool, progress: float) -> None:
        page = self.video_page
        if waiting:
            page.preview.set_note(f"Buffering the 3D preview…  {int(progress * 100)}%")
            self.status.showMessage("Buffering the 3D preview — it plays smoothly once a second is ready.")
        else:
            self.status.clearMessage()

    def _tick_orbit(self) -> None:
        if self._orbit_src is None or self._effective_mode() != "orbit":
            self._orbit_timer.stop()
            return
        colour, depth = self._orbit_src
        self._orbit_phase += 0.09
        h, w = colour.shape[:2]
        frame = cloud.render(colour, depth, w, h, self._orbit_phase)
        self.video_page.preview.set_images([frame], "3D orbit · depth as geometry", False)

    # -- wiring ------------------------------------------------------------

    def _wire(self) -> None:
        page = self.video_page
        page.pick_video.clicked.connect(self.pick_video)
        page.pick_disc.clicked.connect(self.pick_disc)
        page.pick_folder.clicked.connect(self.pick_folder)
        page.export.audio_track_chosen.connect(self.player.set_audio_track)
        page.pick_output.clicked.connect(self._pick_output)
        page.start.clicked.connect(self._start_convert)
        page.stop.clicked.connect(self._stop_convert)
        page.queue_button.clicked.connect(self.open_batch)
        page.threed.changed.connect(self._params_changed)
        page.colour.changed.connect(self._params_changed)
        page.lut.changed.connect(self._params_changed)
        page.export.changed.connect(self._export_changed)
        page.export.range_box.currentIndexChanged.connect(self._range_mode)
        page.export.set_rate_context(self._rate_context)
        page.timeline.in_changed.connect(lambda _f: page.export.refresh_rates())
        page.timeline.out_changed.connect(lambda _f: page.export.refresh_rates())
        for key, button in page.mode_buttons.items():
            button.clicked.connect(lambda _c=False, k=key: self._set_mode(k))
        page.preview.clicked.connect(self._picture_clicked)
        sp = self.settings_page
        sp.theme_box.currentTextChanged.connect(self._theme_changed)
        sp.cpu_depth.toggled.connect(self._cpu_depth)
        sp.preview_box.currentIndexChanged.connect(self._preview_height_changed)
        sp.buffer_box.currentIndexChanged.connect(self._buffer_changed)
        sp.fps_box.currentIndexChanged.connect(self._fps_changed)
        sp.update_check.toggled.connect(self._update_setting_changed)
        sp.update_now.clicked.connect(lambda: self._check_updates(manual=True))
        sp.replay_tutorial.clicked.connect(self.replay_tutorial)
        sp.redetect.clicked.connect(lambda: self._run_probe(force=True))
        sp.vr_auto.toggled.connect(self._vr_auto_changed)
        self.vr.audio_choice = self.settings.vr_audio_device
        self.vr.audio_choice_changed.connect(self._vr_audio_chosen)
        self.vr.trim_bars = self.settings.vr_trim_bars
        sp.vr_trim.toggled.connect(self._vr_trim_changed)
        self.vr.beam_on, self.vr.beam_level = self.settings.vr_beam, self.settings.vr_beam_level
        self.vr.beam_changed.connect(self._vr_beam_from_headset)
        sp.vr_beam.toggled.connect(self._vr_beam_from_settings)
        sp.vr_beam_box.currentIndexChanged.connect(self._vr_beam_from_settings)
        sp.vr_rooms_folder.clicked.connect(self._open_rooms_folder)
        self._refresh_vr_settings()
        sp.open_output.clicked.connect(lambda: open_folder(paths.output_dir()))
        sp.open_luts.clicked.connect(lambda: open_folder(paths.luts_dir()))
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(lambda: self.settings.save(paths.settings_path()))

    def _set_mode(self, key: str) -> None:
        """A view chip: any 3D view switches the effect on, "2D" switches it off."""
        if key == "original":
            self.settings.preview_3d = False
        else:
            self.settings.preview_mode = key
            self._enable_preview_3d()
        self._sync_mode_buttons()
        self._submit(new_frame=False)
        self._save_timer.start()

    def _enable_preview_3d(self) -> None:
        self.settings.preview_3d = True
        if not self.settings.threed.enabled:
            self.video_page.threed.setChecked(True)          # previewing 3D implies converting to 3D
            self.settings.threed.enabled = True

    def _picture_clicked(self) -> None:
        """A click anywhere on the picture plays or pauses, like a streaming player."""
        if self.video_page.play_button.isEnabled():
            self.player.toggle()

    def _sync_mode_buttons(self) -> None:
        mode = self._effective_mode()
        page = self.video_page
        for key, button in page.mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(False)

    # -- welcome and tutorial ---------------------------------------------------
    # The look and the flow are the DLSS converter's: a welcome, then a dimmed window with one real
    # control lit at a time. It is offered once (settings.onboarding_version) and can be replayed
    # from Settings.

    def _offer_tour(self) -> None:
        """First launch only, once set-up dialogs are out of the way."""
        if os.environ.get("UV3D_SKIP_TOUR"):
            return
        if (self.settings.onboarding_version >= onboarding.ONBOARDING_VERSION or self._welcome_open
                or self._tour_overlay is not None or self._probe_thread is not None):
            return
        self._welcome_open = True
        QTimer.singleShot(600, self.show_welcome)

    def replay_tutorial(self) -> None:
        self.show_welcome()

    def show_welcome(self) -> None:
        self._welcome_open = True
        palette = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])
        dialog = onboarding.WelcomeDialog(paths.package_dir() / "assets" / "icon.png", self, palette=palette)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._welcome_open = False
        if accepted:
            QTimer.singleShot(150, self.start_tutorial)
        else:
            self._complete_onboarding()

    def _show_video_tab(self) -> None:
        self.tabs.setCurrentWidget(self.video_page)

    def _reveal(self, widget: QWidget) -> None:
        """Scroll the side rail so ``widget`` is on screen (and put the Video tab in front)."""
        self._show_video_tab()
        self.video_page.rail_scroller.ensureWidgetVisible(widget, 0, 12)

    def _tour_steps(self) -> list[onboarding.TourStep]:
        page = self.video_page
        step = onboarding.TourStep
        video = self._show_video_tab
        steps = [
            step("Your video",
                 "Choose a video (or drop one here) and it opens paused on the plain picture. Click "
                 "anywhere on it to play or pause. The mouse wheel zooms, right-drag pans, and "
                 "double-click fits it again. This is a small preview - exports use the full-size video.",
                 page.stack, video),
            step("Pick a view",
                 "2D is the plain video. Output, Anaglyph and Side by side show the 3D the way it will be "
                 "delivered. Wiggle, Depth, Parallax and Holes let you check the 3D without any glasses, "
                 "and 3D orbit flies a camera around the depth as geometry. Zones paints the picture by "
                 "how far each part sticks out of or sinks into the screen, so you can see what is too much "
                 "for comfortable viewing. Picking a 3D view is what turns the effect on.",
                 page.mode_bar, video),
            step("Timeline",
                 "Drag to scrub. While a 3D view is chosen the app keeps rendering ahead in the "
                 "background - the red bar under the track shows how far - so pressing Play starts at "
                 "once and stays smooth. To convert only part of the video, choose Select In/Out under Export "
                 "and drag the two markers here.",
                 page.timeline, video),
            step("Source",
                 "Choose the video file to convert, and where the result is saved; the name is filled in "
                 "for you, e.g. name_SBS.mp4. \"Choose external disc\" reads a DVD or Blu-ray you own from a "
                 "drive: copy a title to disk first, or convert straight from the disc without keeping a copy. "
                 "Copy-protected discs are not supported - the app tells you when it meets one.",
                 page.source_card, lambda: self._reveal(page.source_card)),
            step("3D conversion",
                 "Output is the format: Side-by-Side and Top-and-Bottom (half or full), red-cyan "
                 "Anaglyph, or Colour + Depth. 3D strength is how deep it feels; Depth position sets what "
                 "sits in front of the screen and what behind it. Depth cleanup smooths the depth map "
                 "for exports (standard, smooth or smoother). Advanced adds foreground and background depth, "
                 "depth stability and softening, edge snapping, and a left/right swap. The gauge and the "
                 "marks on the sliders show how much of the picture lands in the comfortable zone.",
                 page.threed, lambda: self._reveal(page.threed)),
            step("Professional colour correction",
                 "Tick to enable, click to open. The controls a colourist expects: temperature and tint, "
                 "exposure, contrast, highlights, shadows, curves and colour wheels - all live on the "
                 "preview - with a histogram, waveform and vectorscope to judge by. The grade is baked "
                 "into the export as well.",
                 page.colour, lambda: self._reveal(page.colour)),
            step("Looks (.cube LUTs)",
                 "Load a .cube LUT for a ready-made film look and set how strongly it applies. Drop "
                 "LUT files in the LUT folder (Settings > Folders) to find them here.",
                 page.lut, lambda: self._reveal(page.lut)),
            step("Export",
                 "Pick the codec (H.264, H.265, AV1, VP9, ProRes, DNxHR, FFV1), quality, output size and what "
                 "to do with the audio. 10-bit HDR video can stay HDR. The encoders were tested on your "
                 "graphics card the first time you opened the app, so the fast one is already chosen.",
                 page.export, lambda: self._reveal(page.export)),
            step("Convert",
                 "Convert video runs the whole clip with the settings above, on your GPU, and shows "
                 "progress and time left. Press Pause to stop whenever you like - nothing already encoded "
                 "is lost, and converting the same video again offers to continue where it stopped. "
                 "Convert several... queues a list of videos back to back with the same settings.",
                 page.start, video),
            step("Your hardware",
                 "Live GPU, video-encoder, memory and CPU load while you work, and the video encoder "
                 "chosen for this computer. It works on NVIDIA, AMD and Intel graphics alike - no CUDA "
                 "needed.",
                 self.gauge, video),
            step("Library",
                 "Every video you convert lands here, with a thumbnail. Click one to watch it in the "
                 "original layout or as red-cyan anaglyph, add videos you already have with Add videos, "
                 "and remove one with Delete (it goes to the Recycle Bin, not away for good).",
                 self.library_page, lambda: self.tabs.setCurrentWidget(self.library_page)),
            step("Watch in VR",
                 "With a VR headset (Quest via Link or Virtual Desktop, SteamVR, or anything that speaks "
                 "OpenXR) the Library gets a Play in VR button. While the Library is open the app finds your "
                 "headset by itself, so you can just put it on: the menu offers your videos, the lights dim "
                 "when a film plays, and the sound goes to the headset. In the menu (left menu button, or B / Y) "
                 "you can pick videos, set the lights, change seat (push the right stick toward a seat), "
                 "choose the sound, take a screenshot and leave; pressing the right stick resets the view. "
                 "Choose the room here. The headset is optional; everything else works without it.",
                 self.settings_page.vr_card, lambda: self._reveal_setting(self.settings_page.vr_card)),
            step("Settings",
                 "Preview size and smoothness, how far ahead to buffer, appearance, update checks and "
                 "folders live here - and this is where you can replay the tutorial.",
                 self.settings_page.columns, lambda: self.tabs.setCurrentWidget(self.settings_page)),
        ]
        if self.banner.isVisible():
            steps.append(step(
                "Free, and staying that way",
                "This app is free and always will be. If it earns a place in your workflow, please "
                "consider buying me a coffee - it takes a moment, and it helps keep the software free for "
                "everyone, for good. (The X hides this strip until you next open the app.)",
                self.banner, video))
        return steps

    def _reveal_setting(self, card: QWidget) -> None:
        """Show the Settings tab with one of its cards in view."""
        self.tabs.setCurrentWidget(self.settings_page)
        scroller = self.settings_page.findChild(QScrollArea)
        if scroller is not None:
            scroller.ensureWidgetVisible(card, 0, 40)

    def start_tutorial(self) -> None:
        """Spotlight the real controls in the order of a first conversion."""
        if self._tour_overlay is not None:
            return
        self._show_video_tab()
        page = self.video_page
        page.rail_scroller.verticalScrollBar().setValue(0)
        palette = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])
        overlay = onboarding.SpotlightOverlay(self, self._tour_steps(), palette=palette)
        self._tour_overlay = overlay
        overlay.finished.connect(self._complete_onboarding)
        overlay.skipped.connect(self._complete_onboarding)
        overlay.start()

    def _complete_onboarding(self) -> None:
        self.settings.onboarding_version = onboarding.ONBOARDING_VERSION
        self.settings.save(paths.settings_path())
        self._tour_overlay = None
        self._show_video_tab()
        self.video_page.rail_scroller.verticalScrollBar().setValue(0)
        self.status.showMessage("Tutorial done - open Settings > Help to replay it any time.", 12000)

    # -- update notice -----------------------------------------------------

    def _check_updates(self, manual: bool = False) -> None:
        if self._update_thread is not None:
            return
        self._update_manual = manual
        if manual:
            self.settings_page.update_label.setText("Checking…")
        self._update_thread = QThread(self)
        worker = UpdateCheckWorker()
        self._update_worker = worker
        worker.moveToThread(self._update_thread)
        self._update_thread.started.connect(worker.run)
        worker.finished.connect(self._update_checked)
        self._update_thread.start()

    def _update_checked(self, info) -> None:
        if self._update_thread is not None:
            self._update_thread.quit()
            self._update_thread.wait(3000)
            self._update_thread = None
        manual, self._update_manual = self._update_manual, False
        label = self.settings_page.update_label
        if info is None:
            if manual:
                label.setText(f"You have version {__version__}, which is the newest one "
                              "(or GitHub could not be reached).")
            return
        self._update = info
        self.update_pill.setText(f"⬆  UPDATE · v{info.version}")
        self.update_pill.setToolTip(f"Version {info.version} is available (you have {__version__}). "
                                    "Click to open the download page.")
        self.update_pill.setVisible(True)
        self.gauge.set_compact(self.width() < 1050 + 155)
        label.setText(f"Version {info.version} is available (you have {__version__}).")
        self.status.showMessage(f"A new version is available: v{info.version} — click UPDATE at the top to get it.", 20000)

    def _open_update_page(self) -> None:
        url = self._update.url if self._update else updates.RELEASES_PAGE
        QDesktopServices.openUrl(QUrl(url))

    def _update_setting_changed(self, on: bool) -> None:
        self.settings.check_updates = bool(on)
        self._save_timer.start()

    def _fps_changed(self) -> None:
        self.settings.preview_fps = int(self.settings_page.fps_box.currentData())
        self.player.set_preview_fps(self.settings.preview_fps)
        self._save_timer.start()

    def _buffer_changed(self) -> None:
        self.settings.preview_buffer = int(self.settings_page.buffer_box.currentData())
        self._rebuild_params()
        self._save_timer.start()

    def _preview_height_changed(self) -> None:
        height = self.settings_page.preview_box.currentData()
        self.settings.preview_height = int(height)
        self.player.set_proxy_height(int(height))
        self._save_timer.start()

    def _rate_context(self):
        """What the export panel needs to turn quality into a bitrate and a file size: the output picture, its rate, its length, the sound."""
        page = self.video_page
        info = page.disc_stream.info if page.disc_stream is not None else page.info
        if info is None or not info.fps:
            return None
        frames = info.frames or round(info.duration * info.fps)
        if self.settings.export.range_mode == "range" and page.disc_stream is None:
            first, last = page.timeline.in_out()
            frames = max(1, last - first + 1)
        audio = 0
        if info.has_audio:
            audio = pipeline.audio_bps_for_export(page.source, self.settings.export, page.export._audio_tracks)
        return pipeline.frame_size_for(info, self.settings), info.fps, frames / info.fps, audio

    def _params_changed(self) -> None:
        self.video_page.export.refresh_rates()                       # the 3D layout changes the frame, and so the file size
        self._submit(new_frame=False)
        self._sync_mode_buttons()
        self._save_timer.start()

    def _export_changed(self) -> None:
        self._save_timer.start()
        self.export_refresh()
        page = self.video_page
        if not self._converting and page.info is not None:
            page.info_label.setText(page.info.describe())   # not the last run's "Encoding with H.264 ..." for a format no longer chosen

    def export_refresh(self) -> None:
        self._update_encoder_pill()
        self._refresh_output_name()

    def _range_mode(self) -> None:
        self.video_page.timeline.set_range_mode(self.video_page.export.range_box.currentData() == "range")
        self.video_page.export.refresh_rates()

    def _theme_changed(self, name: str) -> None:
        self.settings.theme = name
        app = QApplication.instance()
        app.setPalette(qpalette_for(name))
        self.setStyleSheet(style_for(name))
        self._set_wordmark()
        self._save_timer.start()

    def _cpu_depth(self, on: bool) -> None:
        self.settings.force_cpu_depth = on
        self._submit(new_frame=True)
        self._save_timer.start()

    # -- first-run setup ---------------------------------------------------

    def _start_initial_setup(self) -> None:
        resumemod.cleanup_stale()                        # paused conversions nobody came back to
        warmup.start(self.settings.force_cpu_depth)      # kernels and the depth model, once per launch
        self.gpus = sysinfo.detect_gpus()
        primary = sysinfo.primary_gpu(self.gpus)
        self.gauge.set_gpu_name(primary.short_name if primary else "No GPU")
        self._monitor = sysinfo.LoadMonitor(primary)
        self._gauge_timer = QTimer(self)
        self._gauge_timer.timeout.connect(lambda: self.gauge.update_load(self._monitor.read()))
        self._gauge_timer.start(1200)
        self._describe_hardware()

        if self.settings.check_updates:
            QTimer.singleShot(3000, self._check_updates)       # after the window is up; never blocks start-up
        if not video.is_available():
            self._download_video_support(then=self._after_video_support)
            return
        self._after_video_support()

    def _after_video_support(self) -> None:
        if not depthmod.is_installed():
            # Not shipped with this copy (running from source, or the file was removed): the app fetches
            # it from the project's own GitHub release - nobody is sent off to download it by hand.
            self._download_depth_model(then=self._depth_model_arrived)
            return
        self._after_depth_model()

    def _depth_model_arrived(self) -> None:
        warmup.start(self.settings.force_cpu_depth)      # the launch-time warm-up found nothing to load: do it now
        self._after_depth_model()

    def _after_depth_model(self) -> None:
        stale = self.report is None or self.report.gpu_signature != encoders.signature(self.gpus)
        if stale:
            self._run_probe(force=False)
        else:
            self.video_page.export.set_probe(self.report)
            self._update_encoder_pill()
        if not stale:
            self._offer_tour()

    def _run_probe(self, force: bool) -> None:
        if self._probe_thread is not None or not video.is_available():
            return
        self._probe_dialog = DownloadDialog(
            "Setting up", "Detecting your graphics card and the video encoders it can use. "
                          "This happens once, and again if your GPU changes.", self)
        self._probe_dialog.setStyleSheet(style_for(self.settings.theme))
        self._probe_dialog.set_busy()
        self._probe_thread = QThread(self)
        worker = ProbeWorker()
        self._probe_worker = worker
        worker.moveToThread(self._probe_thread)
        self._probe_thread.started.connect(worker.run)
        worker.status.connect(self._probe_dialog.set_status)
        worker.finished.connect(self._probe_done)
        worker.failed.connect(self._probe_failed)
        self._probe_thread.start()
        self._probe_dialog.show()

    def _end_probe(self) -> None:
        if self._probe_thread is not None:
            self._probe_thread.quit()
            self._probe_thread.wait()
            self._probe_thread = None
        if self._probe_dialog is not None:
            self._probe_dialog.close()
            self._probe_dialog = None

    def _probe_done(self, gpus, report) -> None:
        self._end_probe()
        self.gpus, self.report = gpus, report
        from dataclasses import asdict

        self.settings.encoder_probe = asdict(report)
        self.settings.save(paths.settings_path())
        self.video_page.export.set_probe(report)
        self._update_encoder_pill()
        self._describe_hardware()
        self._offer_tour()

    def _probe_failed(self, message: str) -> None:
        self._end_probe()
        QMessageBox.warning(self, "Could not test the encoders",
                            f"{message}\n\nSoftware encoding will be used, which always works.")
        self._offer_tour()

    def _describe_hardware(self) -> None:
        lines = []
        for g in self.gpus:
            mem = f", {sysinfo.format_bytes(g.vram_bytes)} VRAM" if g.vram_bytes else ""
            lines.append(f"{g.name} ({g.vendor}{mem})")
        text = "\n".join(lines) or "No GPU detected - software encoding and CPU depth will be used."
        if self.report:
            ok = [encoders.encoder_label(n) for n, works in self.report.results.items() if works and encoders.is_hardware(n)]
            text += "\n\nHardware encoders: " + (", ".join(sorted(set(ok))) if ok else "none - using the CPU")
        self.settings_page.hw_label.setText(text)

    def _update_encoder_pill(self) -> None:
        family = self.video_page.export._s.family
        chosen = self.video_page.export._s.encoder
        best = chosen if chosen != "auto" else (self.report.defaults.get(family) if self.report else None)
        pill = self.encoder_pill
        if not best:
            pill.setText("●  CPU ENCODING" if self.report else "●  DETECTING")
            state = "software" if self.report else "software"
        else:
            hw = encoders.is_hardware(best)
            label = encoders.encoder_label(best).replace(" (CPU)", "")
            pill.setText(f"●  {label.upper()}" if hw else f"●  CPU · {best.upper()}")
            state = "ready" if hw else "software"
        pill.setProperty("state", state)
        pill.style().unpolish(pill)
        pill.style().polish(pill)

    # -- opening a video ---------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if any(Path(u.toLocalFile()).suffix.lower() in video.INPUT_SUFFIXES for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt name
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in video.INPUT_SUFFIXES:
                self._open_path(path)
                return

    def pick_video(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(video.INPUT_SUFFIXES))
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose video", self.settings.last_video_dir, f"Video ({exts})")
        if chosen:
            self._open_path(Path(chosen))

    def pick_disc(self) -> None:
        if self._converting:
            return
        if not video.is_available():
            self._download_video_support(then=self.pick_disc)
            return
        dialog = DiscDialog(self)
        dialog.setStyleSheet(style_for(self.settings.theme))
        if not dialog.exec():
            return
        if dialog.copied is not None:
            self._open_path(dialog.copied)
        elif dialog.stream is not None and dialog.sample is not None:
            self._open_disc_stream(dialog.stream, dialog.sample)

    def pick_folder(self) -> None:
        """Open a folder of videos (a ripped disc): look at every file's tracks, then open one or convert several."""
        if self._converting:
            return
        if not video.is_available():
            self._download_video_support(then=self.pick_folder)
            return
        start = Path(self.settings.last_video_dir) if self.settings.last_video_dir else None
        dialog = FolderDialog(self, start if start and start.is_dir() else None, self.settings.export.container == "mkv")
        dialog.setStyleSheet(style_for(self.settings.theme))
        if not dialog.exec() or dialog.action is None:
            return
        export = self.video_page.export
        if dialog.save_mkv != (self.settings.export.container == "mkv"):
            export.set_container("mkv" if dialog.save_mkv else "auto")
        if dialog.action == "open":
            path, audio_index, subtitle_index = dialog.jobs[0]
            self._open_path(path)
            if self.video_page.source == path:
                export.choose_tracks(audio_index, subtitle_index)
        else:
            self.open_batch(jobs=dialog.jobs)

    def _open_disc_stream(self, stream, sample: Path) -> None:
        """Direct mode: the picture is a short sample from the film; Convert reads the whole title from the disc."""
        self._open_path(sample)
        page = self.video_page
        if page.source != sample:                       # the sample could not be opened
            return
        page.disc_stream = stream
        page.source_label.setText(f"Disc: {stream.stem}")
        page.info_label.setText(
            f"{stream.info.describe()}\nThe preview is a short sample; Convert reads the whole title from the disc, "
            "so leave the disc in.")
        page.export.range_box.setCurrentIndex(max(0, page.export.range_box.findData("whole")))
        page.export.range_box.setEnabled(False)
        page.export.range_box.setToolTip("A title converted straight from the disc is always converted whole.")
        self._refresh_output_name()
        self.status.showMessage("Disc ready: adjust the 3D and colour on the sample, then press Convert.")

    def _open_path(self, path: Path) -> None:
        if self._converting:
            return
        if not video.is_available():
            self._download_video_support(then=lambda: self._open_path(path))
            return
        try:
            info = video.probe(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open video", str(error))
            return
        page = self.video_page
        page.source, page.info = path, info
        page.disc_stream = None
        page.export.range_box.setEnabled(True)
        page.export.range_box.setToolTip("")
        self.settings.last_video_dir = str(path.parent)
        page.source_label.setText(path.name)
        page.info_label.setText(info.describe())
        page.start.setEnabled(True)
        page.play_button.setEnabled(True)
        page.export.set_hdr_visible(info.is_hdr)
        page.timeline.set_duration(info.frames or max(1, round(info.duration * info.fps)), info.fps)
        page.timeline.set_range_mode(page.export.range_box.currentData() == "range")
        self._current = None
        try:
            audio, subtitles = video.list_tracks(path)
        except Exception:  # noqa: BLE001 - tracks are a convenience; a file that opens must open
            audio, subtitles = [], []
        page.export.set_tracks(audio, subtitles)
        # Opens paused on the first frame: the video only plays when asked to.
        self.player.open(path, info)
        page.show_preview()
        page.output_path = None
        self._refresh_output_name()
        self.status.showMessage(
            f"Opened {path.name} — click the picture to play or pause, and pick a 3D view below it to see the effect.")

    def _refresh_output_name(self) -> None:
        page = self.video_page
        if page.source is None:
            return
        folder = page.output_path.parent if page.output_path else None
        named = Path(page.disc_stream.stem + ".mkv") if page.disc_stream is not None else page.source
        page.output_path = pipeline.output_path_for(named, self.settings, folder)
        page.output_label.setText(f"Output: {page.output_path}")

    def _pick_output(self) -> None:
        page = self.video_page
        if page.source is None:
            return
        fam = encoders.FAMILIES_BY_KEY[self.settings.export.family]
        start = str(page.output_path or paths.output_dir())
        chosen, _ = QFileDialog.getSaveFileName(self, "Output video", start, f"{fam.label} (*{fam.suffix})")
        if chosen:
            page.output_path = Path(chosen)
            page.output_label.setText(f"Output: {chosen}")

    # -- playback ----------------------------------------------------------

    def _fps(self) -> float:
        info = self.video_page.info
        return info.fps if info and info.fps else 24.0

    def _playback_state(self, playing: bool) -> None:
        self.video_page.play_button.setText("Pause" if playing else "Play")
        self.video_page.preview.set_playing(playing)

    def _position_changed(self, ms: int) -> None:
        if not self.video_page.timeline._drag:
            self.video_page.timeline.set_playhead(int(round(ms / 1000 * self._fps())))

    def _scrub(self, frame: int) -> None:
        # While dragging, jump to the nearest keyframe only: decoding forward from
        # it on a long-GOP 4K clip is the slow part, and the drag needs to feel live.
        # Not on the very first event, though: a plain click is a press and a release a moment
        # apart, and its keyframe-only picture (frame 0 on a short clip) would flash up just before
        # the right one. So this waits a beat, and the release cancels it.
        self._scrub_target = frame
        if not self._scrub_timer.isActive():
            self._scrub_timer.start()

    def _scrub_preview(self) -> None:
        self.player.seek_ms(int(round(self._scrub_target / self._fps() * 1000)), precise=False)

    def _scrub_done(self) -> None:
        self._scrub_timer.stop()
        frame = self.video_page.timeline._playhead
        self.player.seek_ms(int(round(frame / self._fps() * 1000)), precise=True)

    # -- converting --------------------------------------------------------

    def _start_convert(self) -> None:
        page = self.video_page
        if page.source is None or self._thread is not None:
            return
        if not self._guard_support() and self._support_strikes >= 2:
            return
        if not video.is_available():
            self._download_video_support(then=self._start_convert)
            return
        if page.output_path is None:
            self._pick_output()
            if page.output_path is None:
                return

        settings = copy.deepcopy(self.settings)
        start, limit = 0, None
        if settings.export.range_mode == "range" and page.disc_stream is None:
            in_frame, out_frame = page.timeline.in_out()
            start, limit = in_frame, max(1, out_frame - in_frame + 1)
        resume = False
        if page.disc_stream is None:
            answer = self._ask_resume(page, settings, start, limit)
            if answer is None:
                return
            resume = answer
        self.player.pause()
        self._converting = True
        page.show_preview()
        page.stop.setText("Stop" if page.disc_stream is not None else "Pause")
        page.stop.setToolTip("" if page.disc_stream is not None else
                             "Keeps what is converted so far. Press Convert again to carry on from there.")
        page.start.setEnabled(False)
        page.pick_video.setEnabled(False)
        page.pick_disc.setEnabled(False)
        page.pick_folder.setEnabled(False)
        page.play_button.setEnabled(False)
        page.stop.setVisible(True)
        page.bar.setVisible(True)
        page.bar.setValue(0)
        whole = page.disc_stream.info.frames if page.disc_stream is not None else (page.info.frames if page.info else 0)
        total = limit if limit is not None else whole
        page.bar.setMaximum(total or 0)
        self._eta = _EtaTracker()
        self._convert_began = time.monotonic()
        self._convert_note = ""                           # "Encoding with ... at N Mb/s", from the encoder's start-up message
        self._convert_fps = page.info.fps if page.info else 0.0
        page.preview.set_converting([f"Starting the conversion of {page.source.name if page.source else 'the video'}..."])

        self._thread = QThread(self)
        self._worker = ConvertWorker(page.disc_stream or page.source, page.output_path, settings, start, limit, self.gpus, resume)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.showMessage)
        self._worker.frame_done.connect(self._frame_done)
        self._worker.finished.connect(self._finished)
        self._worker.stopped.connect(self._was_stopped)
        self._worker.failed.connect(self._failed)
        self._thread.start()

    def _ask_resume(self, page, settings: AppSettings, start: int, limit: int | None) -> bool | None:
        """True to resume an earlier paused run of this same job, False to start fresh, None to cancel."""
        total = limit if limit is not None else max(0, (page.info.frames if page.info else 0) - start)
        try:
            left = resumemod.find(page.source, settings, start, limit, total)
        except Exception:  # noqa: BLE001 - a damaged state is the same as none
            left = None
        if left is None:
            return False
        box = QMessageBox(QMessageBox.Icon.Question, "Continue this conversion?",
                          f"This conversion was paused at {left.percent}%.\n\nContinue from there, or start over?",
                          parent=self)
        go_on = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        again = box.addButton("Start over", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(go_on)
        box.exec()
        if box.clickedButton() is go_on:
            return True
        if box.clickedButton() is again:
            left.clear()
            return False
        return None

    def _stop_convert(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.video_page.info_label.setText("Pausing…" if self.video_page.disc_stream is None
                                               else "Stopping after this frame…")

    def _frame_done(self, update) -> None:
        page = self.video_page
        if update.stage == "starting":
            page.info_label.setText(update.message)
            self._convert_note = update.message
        elif update.stage == "converting":
            page.bar.setValue(update.index)
            self._eta.tick()
            left = self._eta.remaining(update.total - update.index if update.total else 0)
            speed = f"{update.fps:.1f} fps"
            page.bar.setFormat(f"%v of %m frames · {speed}" + (f" · ~{_short_duration(left)} left" if left else ""))
            elapsed = time.monotonic() - self._convert_began
            first = f"Elapsed {_short_duration(elapsed)}   ·   " + (f"~{_short_duration(left)} left" if left else "working out the time left...")
            if update.total:
                first += f"   ·   {100.0 * update.index / update.total:.0f}%"
            second = f"Frame {update.index:,}" + (f" of {update.total:,}" if update.total else "") + f"   ·   {speed}"
            if self._convert_fps and update.fps:
                second += f"   ·   {update.fps / self._convert_fps:.2f}x real time"
            lines = [first, second]
            if self._convert_note:
                lines.append(self._convert_note)
            if page.output_path is not None:
                lines.append(f"Saving to {page.output_path.name}")
            page.preview.set_converting(lines)
            if update.preview is not None:
                page.preview.set_images([(np.clip(update.preview, 0, 1) * 255).astype(np.uint8)],
                                        f"Converting · frame {update.index}", False)
        elif update.stage == "muxing":
            page.info_label.setText("Adding audio…")
            page.preview.set_converting([update.message or "Adding the sound...",
                                         f"Elapsed {_short_duration(time.monotonic() - self._convert_began)}"])

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(20000)
            self._thread = None
        self._worker = None
        self._converting = False
        page = self.video_page
        page.preview.set_converting(None)
        page.start.setEnabled(page.source is not None)
        page.pick_video.setEnabled(True)
        page.pick_disc.setEnabled(True)
        page.pick_folder.setEnabled(True)
        page.play_button.setEnabled(page.source is not None)
        page.stop.setVisible(False)
        page.bar.setVisible(False)

    # -- virtual reality ---------------------------------------------------

    def _vr_room_dir(self) -> Path | None:
        """The room to sit in: the one chosen in Settings, else the first folder with a room.glb in the data folder."""
        from .vr.rooms import PLAIN

        if self.settings.vr_room_dir == PLAIN:               # the plain built-in room was chosen: no baked room at all
            return None
        chosen = Path(self.settings.vr_room_dir) if self.settings.vr_room_dir else None
        if chosen is not None and (chosen / "room.glb").is_file():
            return chosen
        rooms = paths.data_dir() / "rooms"
        if rooms.is_dir():
            for folder in sorted(p for p in rooms.iterdir() if p.is_dir()):
                if (folder / "room.glb").is_file():
                    return folder
        return None

    def _refresh_vr_settings(self) -> None:
        sp = self.settings_page
        from .vr import unavailable_reason

        reason = unavailable_reason()
        sp.vr_status.setText("A VR headset can be used on this computer." if not reason else reason)
        room = self._vr_room_dir()
        name = "the plain room"
        if room is not None:
            from .vr import rooms as roomsmod

            name = next((r.name for r in roomsmod.find_rooms(room.parent) if r.folder == room), room.name)
        sp.vr_room_label.setText(f"You start in {name}. Pick another room from the Rooms page of the headset's menu; your pick is remembered.")

    def _open_rooms_folder(self) -> None:
        folder = paths.data_dir() / "rooms"
        folder.mkdir(parents=True, exist_ok=True)
        open_folder(folder)

    def _vr_failed(self, message: str) -> None:
        """The headset session could not start or stopped: say so in a box (a status-bar line is too easy to miss)."""
        QMessageBox.information(self, "VR headset", message)

    def _push_vr_videos(self) -> None:
        """Give the headset's menu the Library, newest first, so a video can be picked without taking the headset off."""
        if self.vr is None or not self.library_page._vr_available:
            return
        from PySide6.QtGui import QImage

        from .vr.menu import Tile

        tiles = []
        for entry in self.library.entries[:60]:
            if entry.missing:
                continue
            thumb = self.library.thumb_path(entry)
            picture = QImage(str(thumb)) if thumb.is_file() else None
            tiles.append(Tile(entry.title, entry.path, entry.layout, entry.detail, picture))
        self.vr.set_videos(tiles)
        self._push_vr_rooms()

    def _push_vr_rooms(self) -> None:
        """Give the headset's menu the rooms it can sit in: every folder with a room.glb under the data folder's rooms, with its picture."""
        if not hasattr(self.vr, "set_rooms"):
            return
        from PySide6.QtGui import QImage

        from .vr import rooms as roomsmod
        from .vr.menu import Tile

        tiles = []
        for entry in roomsmod.find_rooms(paths.data_dir() / "rooms"):
            picture = QImage(str(entry.thumbnail)) if entry.thumbnail is not None else None
            tiles.append(Tile(entry.name, str(entry.folder), "", "", picture))
        tiles.append(Tile(roomsmod.PLAIN_NAME, roomsmod.PLAIN, "", roomsmod.PLAIN_NOTE, roomsmod.plain_thumbnail()))   # always on offer
        self.vr.set_rooms(tiles)

    def _vr_room_chosen(self, folder: str) -> None:
        """A room was picked in the headset: it is the one to start in next time, too."""
        self.settings.vr_room_dir = folder
        self.settings.save(paths.settings_path())
        self._refresh_vr_settings()

    def _play_in_vr(self, entry) -> None:
        """Show a Library video in the headset, in the chosen room."""
        self._push_vr_videos()
        if not self.vr.available() and self.vr.session is None:
            QMessageBox.information(self, "VR is not available", self.settings_page.vr_status.text())
            return
        if self.vr.room is None or str(self.vr.room.folder) != str(self._vr_room_dir() or ""):
            self.vr.use_room(self._vr_room_dir())
        if entry is None:                                    # nothing chosen on the PC: go in and pick from the headset's menu
            if self.vr.start_vr():
                self.status.showMessage("Put the headset on and pick a video from the menu.", 12000)
            return
        self.vr.show(entry.file, entry.layout, preview=False)

    def _start_vr(self) -> None:
        """The Library's Start VR button: connect to the headset now (the room is the chosen one), pick the film in the headset."""
        self._play_in_vr(None)

    def _space_for_vr(self) -> bool:
        """Space plays and pauses the video in the headset while one is running. True if it did."""
        if self.vr.running and self.vr.media is not None and self.vr.media.is_open:
            self.vr.toggle()
            return True
        return False

    def eventFilter(self, obj, event):  # noqa: N802 - Qt name
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit, QTextEdit

        if (event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space and not event.modifiers()
                and self.vr.running and self.vr.media is not None and self.vr.media.is_open):
            focus = QApplication.focusWidget()
            if not isinstance(focus, (QLineEdit, QTextEdit, QAbstractSpinBox)):     # never steal a space from a text field
                self.vr.toggle()
                return True
        return super().eventFilter(obj, event)

    def _record_conversion(self, output, source_name: str = "") -> None:
        """List a finished video in the Library (its layout is whatever 3D layout was chosen)."""
        try:
            layout = self.settings.threed.layout if self.settings.threed.enabled else ""
            self.library.add_converted(Path(output), layout, source_name)
            self.library_page.refresh()
            self._push_vr_videos()
        except Exception:  # noqa: BLE001 - the Library is a convenience; a finished video is never lost to it
            pass

    def _vr_auto_changed(self, on: bool) -> None:
        self.settings.vr_auto_connect = on
        self.settings.save(paths.settings_path())
        self._update_vr_watch()

    def _vr_trim_changed(self, on: bool) -> None:
        self.settings.vr_trim_bars = on
        self.settings.save(paths.settings_path())
        self.vr.set_trim_bars(on)

    def _vr_beam_from_settings(self, *_) -> None:
        sp = self.settings_page
        self.settings.vr_beam = sp.vr_beam.isChecked()
        self.settings.vr_beam_level = float(sp.vr_beam_box.currentData() or 1.0)
        self.settings.save(paths.settings_path())
        self.vr.set_beam(self.settings.vr_beam, self.settings.vr_beam_level)

    def _vr_beam_from_headset(self, on: bool, level: float) -> None:
        """The beam was switched or dimmed in the headset's menu: remember it and show it in Settings too."""
        self.settings.vr_beam, self.settings.vr_beam_level = on, level
        self.settings.save(paths.settings_path())
        sp = self.settings_page
        for widget in (sp.vr_beam, sp.vr_beam_box):
            widget.blockSignals(True)
        sp.vr_beam.setChecked(on)
        sp.vr_beam_box.setCurrentIndex(max(0, sp.vr_beam_box.findData(level)))
        for widget in (sp.vr_beam, sp.vr_beam_box):
            widget.blockSignals(False)

    def _vr_audio_chosen(self, choice: str) -> None:
        self.settings.vr_audio_device = choice            # "" = automatic
        self.settings.save(paths.settings_path())

    def _update_vr_watch(self) -> None:
        """Look for a headset while the Library is open (if the person wants that and VR can work here at all)."""
        on = (self.settings.vr_auto_connect and self.tabs.currentWidget() is self.library_page
              and self.library_page._vr_available)
        self.vr.watch(on)

    def _tab_changed(self, index: int) -> None:
        self._push_vr_videos()                           # additions and removals are made in the Library tab
        self._update_vr_watch()
        if self.tabs.widget(index) is not self.library_page:
            self.library_page.pause()                    # leaving the Library stops its video

    def _finished(self, output) -> None:
        self._teardown()
        output = Path(output)
        page = self.video_page
        self._record_conversion(output, page.disc_stream.stem if page.disc_stream is not None
                                else (page.source.name if page.source else ""))
        self.video_page.info_label.setText(f"Saved {output.name}")
        self.status.showMessage(f"Video saved: {output}")
        box = QMessageBox(QMessageBox.Icon.Information, "Conversion complete", f"Saved to:\n{output}", parent=self)
        box.addButton(QMessageBox.StandardButton.Ok)
        reveal = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.exec()
        if box.clickedButton() is reveal:
            open_folder(output.parent)

    def _was_stopped(self, done: int = 0, total: int = 0, resumable: bool = False) -> None:
        self._teardown()
        if resumable and total:
            text = f"Paused at {round(100 * done / total)}%. Press Convert to continue from there."
        else:
            text = "Stopped."
        self.video_page.info_label.setText(text)
        self.status.showMessage(text)

    def _failed(self, message: str) -> None:
        self._teardown()
        QMessageBox.warning(self, "Conversion failed", message)
        self.video_page.info_label.setText("Conversion failed")

    def open_batch(self, *_, jobs=None) -> None:
        """The several-videos converter; ``jobs`` pre-fills it with (file, audio stream index, subtitle stream index)."""
        if not self._guard_support() and self._support_strikes >= 2:
            return
        if not video.is_available():
            self._download_video_support(then=lambda: self.open_batch(jobs=jobs))
            return
        dialog = BatchDialog(self, self.settings, self.gpus, style_for(self.settings.theme))
        for path, audio_index, subtitle_index in (jobs or []):
            dialog.add_job(path, audio_index, subtitle_index)
        dialog.converted.connect(lambda source, output: self._record_conversion(output, Path(source).name))
        dialog.exec()

    # -- PyAV download -----------------------------------------------------

    def _download_video_support(self, then) -> None:
        self._run_download("Adding video support", "Downloads FFmpeg (about 35 MB) once. Nothing else is fetched.",
                           VideoDownloadWorker(), then, "Could not add video support")

    def _download_depth_model(self, then) -> None:
        self._run_download("Downloading the depth model",
                           "The AI model that estimates depth (about 50 MB), once, from this project's own "
                           "GitHub page. It is kept in the models folder next to the app.",
                           ModelDownloadWorker(), then, "Could not download the depth model")

    def _run_download(self, title: str, subtitle: str, worker, then, failed_title: str) -> None:
        if self._dl_thread is not None:
            return
        self._dl_after = then
        self._dl_retry = lambda: self._run_download(title, subtitle, type(worker)(), then, failed_title)
        self._dl_failed_title = failed_title
        dialog = DownloadDialog(title, subtitle, self)
        dialog.setStyleSheet(style_for(self.settings.theme))
        self._dl_dialog = dialog
        self._dl_thread = QThread(self)
        self._dl_worker = worker
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.progress.connect(lambda d, t: dialog.update_bytes(int(d), int(t)))
        self._dl_worker.status.connect(dialog.set_status)
        self._dl_worker.finished.connect(self._dl_done)
        self._dl_worker.failed.connect(self._dl_failed)
        self._dl_thread.start()
        dialog.show()

    def _end_download(self) -> None:
        if self._dl_thread is not None:
            self._dl_thread.quit()
            self._dl_thread.wait()
            self._dl_thread = None
        self._dl_worker = None
        self._dl_dialog.close()

    def _dl_done(self) -> None:
        self._end_download()
        then, self._dl_after = self._dl_after, None
        if then is not None:
            then()

    def _dl_failed(self, message: str) -> None:
        self._end_download()
        self._dl_after = None
        retry, self._dl_retry = self._dl_retry, None
        box = QMessageBox(QMessageBox.Icon.Warning, self._dl_failed_title, message, parent=self)
        again = box.addButton("Try again", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is again and retry is not None:
            retry()

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.settings.save(paths.settings_path())
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(15000)
        self.player.shutdown()
        self.library_page.shutdown()
        self.vr.watch(False)
        self.vr.stop(blocking=True)                      # a few seconds at most, so the runtime is left clean
        self.engine.shutdown()
        if self._monitor is not None:
            self._monitor.close()
        super().closeEvent(event)


# --- entry point -------------------------------------------------------------


def apply_app_theme(app: QApplication, name: str) -> None:
    """Fusion is the one built-in style that honours a custom palette on every platform."""
    app.setStyle("Fusion")
    app.setPalette(qpalette_for(name))


def _load_bundled_fonts(app: QApplication) -> None:
    """Register Archivo + IBM Plex so the stylesheet's font names resolve.

    A missing font must never be fatal: the stylesheet's fallbacks keep it readable.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    try:
        for ttf in sorted(paths.fonts_dir().glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(ttf))
        if "IBM Plex Sans" in QFontDatabase.families():
            base = QFont("IBM Plex Sans")
            base.setPointSize(app.font().pointSize())
            app.setFont(base)
    except Exception:  # noqa: BLE001 - styling is never worth a failed launch
        pass


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    _load_bundled_fonts(app)
    try:
        theme = AppSettings.load(paths.settings_path()).theme
    except Exception:  # noqa: BLE001 - a bad settings file must not block launch
        theme = DEFAULT_THEME
    apply_app_theme(app, theme)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
