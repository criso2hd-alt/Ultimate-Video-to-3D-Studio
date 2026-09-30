"""The "Choose external disc" dialog: pick a drive and a title, copy it to a file, hand the file back.

Two steps in one window. First the drive and the title list; then, once "Copy" is pressed, a progress bar with
the speed and time left while :func:`disc.copy_title` runs on a worker thread. The copy is the slow part (it is
as fast as the drive), so it can be cancelled and never leaves a half-written file behind.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from . import disc, paths


class DiscScanWorker(QObject):
    """Lists the titles of a disc off the UI thread: reading tables from a spinning-up drive can take seconds."""

    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, root: Path, kind: str) -> None:
        super().__init__()
        self._root, self._kind = root, kind

    def run(self) -> None:
        try:
            titles = disc.list_titles(self._root, self._kind)
            if titles:
                try:
                    titles[0].scrambled = disc.is_scrambled(titles[0])     # the main film; reading 4 MB from the drive
                except OSError:
                    pass
            self.finished.emit(titles)
        except Exception as error:  # noqa: BLE001 - shown in the dialog
            self.failed.emit(str(error))


class DiscCopyWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, title: disc.Title, destination: Path) -> None:
        super().__init__()
        self._title, self._destination = title, destination
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            out = disc.copy_title(self._title, self._destination, self.progress.emit, lambda: self._stop)
        except disc.Stopped:
            self.stopped.emit()
            return
        except disc.DiscError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"The copy failed: {error}")
            return
        self.finished.emit(out)


def copies_dir() -> Path:
    path = paths.data_dir() / "discs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def enough_space(folder: Path, needed: int) -> bool:
    try:
        return shutil.disk_usage(folder).free > needed * 1.05 + (256 << 20)
    except OSError:
        return True                                             # cannot tell: let the copy try


class DiscDialog(QDialog):
    """Result: :attr:`copied` is the path of the finished file, or None if the dialog was cancelled."""

    NOTICE = ("Copy only discs you own. What you do with the copy is your responsibility. "
              "The disc is copied to your disk as it is, without changing the picture, and then converted.")

    def __init__(self, parent: QWidget | None = None, folder: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose external disc")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.copied: Path | None = None
        self._folder = folder or copies_dir()
        self._titles: list[disc.Title] = []
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._started = 0.0
        self._on_result = None
        self._on_stopped = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        heading = QLabel("Copy a DVD or Blu-ray to a video file")
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        note = QLabel(self.NOTICE)
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        row = QHBoxLayout()
        self.drives = QComboBox()
        self.drives.currentIndexChanged.connect(self._drive_changed)
        self.refresh = QPushButton("Refresh")
        self.refresh.setObjectName("secondary")
        self.refresh.clicked.connect(self.reload_drives)
        self.browse = QPushButton("Choose folder…")
        self.browse.setObjectName("secondary")
        self.browse.setToolTip("Pick a disc that is already copied to a folder (the one that holds VIDEO_TS or BDMV).")
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.drives, 1)
        row.addWidget(self.refresh)
        row.addWidget(self.browse)
        layout.addLayout(row)

        self.titles = QListWidget()
        self.titles.setMinimumHeight(170)
        self.titles.currentRowChanged.connect(self._title_changed)
        layout.addWidget(self.titles)

        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel = QPushButton("Cancel")
        self.cancel.setObjectName("secondary")
        self.cancel.clicked.connect(self._cancel)
        self.copy = QPushButton("Copy and open")
        self.copy.setEnabled(False)
        self.copy.clicked.connect(self._start_copy)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.copy)
        layout.addLayout(buttons)

        self._manual: list[disc.Drive] = []
        QTimer.singleShot(0, self.reload_drives)

    # -- choosing ---------------------------------------------------------------------------------

    def reload_drives(self) -> None:
        self.drives.blockSignals(True)
        self.drives.clear()
        found = disc.find_drives() + self._manual
        for drive in found:
            self.drives.addItem(drive.text, drive)
        self.drives.blockSignals(False)
        if not found:
            self.status.setText("No disc found. Put a DVD or Blu-ray in the drive and press Refresh, or choose a folder.")
            self.titles.clear()
            self.copy.setEnabled(False)
            return
        self._drive_changed()

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose the disc folder")
        if not chosen:
            return
        found = disc.disc_root(Path(chosen))
        if found is None:
            self.status.setText("That folder has no VIDEO_TS or BDMV folder in it, so it is not a DVD or Blu-ray.")
            return
        drive = disc.Drive(found[0], found[0].name, found[1])
        self._manual.append(drive)
        self.reload_drives()
        self.drives.setCurrentIndex(self.drives.findData(drive))

    def _drive_changed(self) -> None:
        self.titles.clear()
        self.copy.setEnabled(False)
        drive = self.drives.currentData()
        if drive is None:
            return
        if not drive.kind:
            self.status.setText("There is no DVD or Blu-ray video on this disc.")
            return
        self.status.setText("Reading the disc…")
        self._run(DiscScanWorker(drive.root, drive.kind), done=self._scanned)

    def _scanned(self, titles: list) -> None:
        self._titles = titles
        self.titles.clear()
        for title in titles:
            self.titles.addItem(QListWidgetItem(title.text))
        if not titles:
            self.status.setText("No titles were found on this disc.")
            return
        self.titles.setCurrentRow(0)
        self.status.setText("The longest title is selected: that is normally the main film.")
        if titles[0].scrambled:
            self.copy.setEnabled(False)
            self.status.setText(disc.PROTECTED)

    def _title_changed(self, row: int) -> None:
        self.copy.setEnabled(0 <= row < len(self._titles) and self._thread is None)

    # -- copying ----------------------------------------------------------------------------------

    def _start_copy(self) -> None:
        row = self.titles.currentRow()
        drive = self.drives.currentData()
        if row < 0 or drive is None:
            return
        title = self._titles[row]
        if not enough_space(self._folder, title.nbytes):
            self.status.setText(f"Not enough free space in {self._folder} for a {title.nbytes / 1e9:.1f} GB copy.")
            return
        name = f"{disc.safe_name(drive.label)} - {disc.safe_name(title.name)}.mkv"
        destination = self._folder / name
        for control in (self.drives, self.refresh, self.browse, self.titles, self.copy):
            control.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self._started = time.monotonic()
        self.status.setText("Starting…")
        worker = DiscCopyWorker(title, destination)
        worker.progress.connect(self._progress)
        self._run(worker, done=self._copied, stopped=self._stopped)

    def _progress(self, done: int, total: int) -> None:
        self.bar.setValue(int(1000 * done / max(1, total)))
        elapsed = max(0.1, time.monotonic() - self._started)
        rate = done / elapsed
        left = (total - done) / rate if rate > 0 and done else 0
        m, s = divmod(int(left), 60)
        self.status.setText(f"{done / 1e9:.2f} of {total / 1e9:.2f} GB  -  {rate / 1e6:.0f} MB/s"
                            + (f"  -  about {m}:{s:02d} left" if done > total * 0.02 else ""))

    def _copied(self, path) -> None:
        self.copied = Path(path)
        self.accept()

    def _stopped(self) -> None:
        self.reject()

    def _cancel(self) -> None:
        if isinstance(self._worker, DiscCopyWorker):
            self.status.setText("Stopping…")
            self.cancel.setEnabled(False)
            self._worker.stop()                                 # the worker then emits `stopped`
            return
        self.reject()

    def reject(self) -> None:  # noqa: D102 - Qt name; also the window's X and Escape
        if isinstance(self._worker, DiscCopyWorker):
            self._cancel()
            return
        self._end()
        super().reject()

    # -- one worker at a time ---------------------------------------------------------------------

    def _run(self, worker, done, stopped=None) -> None:
        self._end()
        self._on_result, self._on_stopped = done, stopped
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        # Bound methods of this dialog, not lambdas: a lambda would run on the worker's thread, and the
        # handlers stop that very thread.
        worker.finished.connect(self._worker_finished)
        worker.failed.connect(self._worker_failed)
        if hasattr(worker, "stopped"):
            worker.stopped.connect(self._worker_stopped)
        self._thread.start()

    def _current(self) -> bool:
        """False for a late signal from a worker that has already been replaced (e.g. an earlier scan)."""
        return self.sender() is self._worker

    def _worker_finished(self, result) -> None:
        if not self._current():
            return
        done = self._on_result
        self._end()
        done(result)

    def _worker_stopped(self) -> None:
        if not self._current():
            return
        stopped = self._on_stopped
        self._end()
        if stopped:
            stopped()

    def _worker_failed(self, message: str) -> None:
        if not self._current():
            return
        self._end()
        self.bar.setVisible(False)
        self.status.setText(message)
        for control in (self.drives, self.refresh, self.browse, self.titles):
            control.setEnabled(True)
        self.cancel.setEnabled(True)
        self._title_changed(self.titles.currentRow())

    def _end(self) -> None:
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.quit()
            thread.wait(20000)

    def done(self, result: int) -> None:  # noqa: D102 - Qt name
        self._end()
        super().done(result)
