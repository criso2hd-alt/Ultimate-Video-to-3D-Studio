"""Open a folder of videos - typically a ripped DVD or Blu-ray (a folder of .mkv titles) - and see what is in each file.

The folder is scanned on a worker thread (every file is probed: its length, picture, and all its audio and subtitle tracks), then
the person picks a file and, for that file, which audio track and which subtitles to keep. From there a single video is opened in the
main window, or several are sent to the batch converter with each file's own track choices.

Nothing is changed on disk and nothing is copied: the files are only read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import video

MAX_DEPTH = 3                                    # a rip is usually Movie/title_t00.mkv, or Movie/BDMV/STREAM/00001.m2ts
MIN_BYTES = 2_000_000                            # smaller files are menus, thumbnails or leftovers


@dataclass
class Entry:
    """What was found for one file."""

    path: Path
    size: int = 0
    info: video.VideoInfo | None = None
    audio: list = field(default_factory=list)       # list[video.Track]
    subtitles: list = field(default_factory=list)
    error: str = ""

    @property
    def length(self) -> float:
        return self.info.duration if self.info else 0.0


def find_videos(root: Path, recursive: bool = True) -> list[Path]:
    """The video files under ``root`` (to MAX_DEPTH folders down), without opening them."""
    found: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_file() and entry.suffix.lower() in video.INPUT_SUFFIXES and entry.stat().st_size >= MIN_BYTES:
                    found.append(entry)
                elif entry.is_dir() and recursive and depth < MAX_DEPTH:
                    walk(entry, depth + 1)
            except OSError:
                continue

    walk(root, 0)
    return found


def scan_file(path: Path) -> Entry:
    entry = Entry(path)
    try:
        entry.size = path.stat().st_size
        entry.info = video.probe(path)
        entry.audio, entry.subtitles = video.list_tracks(path)
    except Exception as error:  # noqa: BLE001 - one unreadable file must not hide the others
        entry.error = str(error) or type(error).__name__
    return entry


class FolderScanWorker(QObject):
    found = Signal(object)                       # an Entry, as each file is read
    progress = Signal(int, int)
    finished = Signal()

    def __init__(self, root: Path, recursive: bool) -> None:
        super().__init__()
        self._root, self._recursive = root, recursive
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        files = find_videos(self._root, self._recursive)
        for i, path in enumerate(files):
            if self._stop:
                break
            self.progress.emit(i, len(files))
            self.found.emit(scan_file(path))
        self.progress.emit(len(files), len(files))
        self.finished.emit()


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


def _size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _summary(tracks: list, none: str) -> str:
    if not tracks:
        return none
    names = []
    for t in tracks:
        name = t.language_name or "?"
        if name not in names:
            names.append(name)
    return f"{len(tracks)}: " + ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")


class FolderDialog(QDialog):
    """Pick videos from a folder. Afterwards read ``action`` ("open", "convert" or None), ``jobs`` and ``save_mkv``."""

    def __init__(self, parent: QWidget | None = None, folder: Path | None = None, save_mkv: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open a folder of videos")
        self.setMinimumSize(980, 560)
        self.action: str | None = None
        self.jobs: list[tuple[Path, int, int]] = []
        self.save_mkv = save_mkv
        self.entries: dict[Path, Entry] = {}
        self.choice: dict[Path, tuple[int, int]] = {}            # per file: (audio stream index, subtitle stream index)
        self._thread: QThread | None = None
        self._worker: FolderScanWorker | None = None
        self._root = folder

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 16, 18, 16)
        col.setSpacing(10)
        top = QHBoxLayout()
        self.folder_label = QLabel("Choose the folder that holds the videos (for example a ripped disc).")
        self.folder_label.setObjectName("hint")
        self.folder_label.setWordWrap(True)
        top.addWidget(self.folder_label, 1)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        self.recursive.toggled.connect(self._rescan)
        top.addWidget(self.recursive)
        self.browse = QPushButton("Choose folder…")
        self.browse.clicked.connect(self._browse)
        top.addWidget(self.browse)
        col.addLayout(top)

        self.tree = QTreeWidget()
        self.tree.setObjectName("folderTree")
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["File", "Length", "Picture", "Size", "Audio", "Subtitles"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setSortingEnabled(False)
        for column, width in enumerate((300, 70, 120, 70, 200, 180)):
            self.tree.setColumnWidth(column, width)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self._finish("open"))
        col.addWidget(self.tree, 1)

        details = QFrame()
        details.setObjectName("folderDetails")
        grid = QVBoxLayout(details)
        grid.setContentsMargins(12, 10, 12, 10)
        self.detail_title = QLabel("Select a video to choose its sound and subtitles.")
        self.detail_title.setWordWrap(True)
        grid.addWidget(self.detail_title)
        row = QHBoxLayout()
        row.addWidget(QLabel("Audio track"))
        self.audio_box = QComboBox()
        self.audio_box.setMinimumWidth(340)
        self.audio_box.currentIndexChanged.connect(self._choice_changed)
        row.addWidget(self.audio_box, 1)
        row.addWidget(QLabel("Subtitles"))
        self.subtitle_box = QComboBox()
        self.subtitle_box.setMinimumWidth(300)
        self.subtitle_box.currentIndexChanged.connect(self._choice_changed)
        row.addWidget(self.subtitle_box, 1)
        grid.addLayout(row)
        self.mkv = QCheckBox("Save the 3D videos as MKV (keeps subtitles, and lossless audio such as TrueHD or DTS-HD, as it is)")
        self.mkv.setChecked(save_mkv)
        grid.addWidget(self.mkv)
        col.addWidget(details)

        self.status = QLabel("")
        self.status.setObjectName("hint")
        col.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("secondary")
        self.close_button.clicked.connect(self.reject)
        self.open_button = QPushButton("Open the selected video")
        self.open_button.clicked.connect(lambda: self._finish("open"))
        self.convert_button = QPushButton("Convert the selected…")
        self.convert_button.clicked.connect(lambda: self._finish("convert"))
        self.all_button = QPushButton("Convert all…")
        self.all_button.setObjectName("secondary")
        self.all_button.clicked.connect(self._convert_all)
        for button in (self.close_button, self.all_button, self.convert_button, self.open_button):
            buttons.addWidget(button)
        col.addLayout(buttons)
        self._selection_changed()
        if folder is not None:
            self._scan(folder)

    # -- scanning ----------------------------------------------------------------------------------------

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder of videos", str(self._root or ""))
        if chosen:
            self._scan(Path(chosen))

    def _rescan(self) -> None:
        if self._root is not None:
            self._scan(self._root)

    def _scan(self, root: Path) -> None:
        self._stop_scan()
        self._root = root
        self.entries.clear()
        self.tree.clear()
        self.folder_label.setText(f"Folder: {root}")
        self.status.setText("Reading the folder…")
        self._thread = QThread(self)
        self._worker = FolderScanWorker(root, self.recursive.isChecked())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.found.connect(self._add_entry)
        self._worker.progress.connect(self._scan_progress)
        self._worker.finished.connect(self._scan_done)
        self._thread.start()

    def _stop_scan(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(15000)
        self._thread = self._worker = None

    def _scan_progress(self, done: int, total: int) -> None:
        if done < total:
            self.status.setText(f"Reading {done + 1} of {total}…")

    def _add_entry(self, entry: Entry) -> None:
        self.entries[entry.path] = entry
        item = QTreeWidgetItem()
        item.setData(0, Qt.ItemDataRole.UserRole, str(entry.path))
        relative = entry.path.name
        if self._root is not None:
            try:
                relative = str(entry.path.relative_to(self._root))
            except ValueError:
                pass
        item.setText(0, relative)
        item.setToolTip(0, str(entry.path))
        if entry.error:
            item.setText(1, "?")
            item.setText(2, "could not be read")
            item.setToolTip(2, entry.error)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        else:
            info = entry.info
            item.setText(1, _clock(info.duration))
            item.setText(2, f"{info.width}×{info.height} {info.codec}" + (" HDR" if info.is_hdr else ""))
            item.setText(3, _size(entry.size))
            item.setText(4, _summary(entry.audio, "no audio"))
            item.setToolTip(4, "\n".join(t.label() for t in entry.audio) or "no audio")
            item.setText(5, _summary(entry.subtitles, "none"))
            item.setToolTip(5, "\n".join(t.label() for t in entry.subtitles) or "none")
            self.choice[entry.path] = (video.best_audio_index(entry.audio), -1)
        # longest first: the film, then the extras (as a disc's own title list does)
        position = 0
        while position < self.tree.topLevelItemCount():
            other = self.entries.get(Path(self.tree.topLevelItem(position).data(0, Qt.ItemDataRole.UserRole)))
            if other is not None and other.length >= entry.length:
                position += 1
            else:
                break
        self.tree.insertTopLevelItem(position, item)

    def _scan_done(self) -> None:
        readable = sum(1 for e in self.entries.values() if not e.error)
        self.status.setText(f"{readable} video{'s' if readable != 1 else ''} found." if self.entries
                            else "No videos found in that folder.")
        if self._thread is not None:
            self._thread.quit()
        if self.tree.topLevelItemCount() and not self.tree.selectedItems():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._selection_changed()

    # -- choosing ----------------------------------------------------------------------------------------

    def _selected_paths(self) -> list[Path]:
        return [Path(i.data(0, Qt.ItemDataRole.UserRole)) for i in self.tree.selectedItems()]

    def _selection_changed(self) -> None:
        paths = self._selected_paths()
        self.open_button.setEnabled(len(paths) == 1)
        self.convert_button.setEnabled(bool(paths))
        readable = [p for p, e in self.entries.items() if not e.error]
        self.all_button.setEnabled(bool(readable))
        self.all_button.setText(f"Convert all ({len(readable)})…" if readable else "Convert all…")
        self._fill_details(paths[0] if len(paths) == 1 else None, len(paths))

    def _fill_details(self, path: Path | None, count: int) -> None:
        for box in (self.audio_box, self.subtitle_box):
            box.blockSignals(True)
            box.clear()
        entry = self.entries.get(path) if path else None
        enabled = entry is not None and not entry.error
        if enabled:
            audio_index, sub_index = self.choice.get(path, (-1, -1))
            self.audio_box.addItem("No audio" if not entry.audio else "Best available (automatic)", -1)
            for track in entry.audio:
                self.audio_box.addItem(track.label(), track.index)
            self.audio_box.setCurrentIndex(max(0, self.audio_box.findData(audio_index)))
            self.subtitle_box.addItem("None", -1)
            for track in entry.subtitles:
                self.subtitle_box.addItem(track.label(), track.index)
            self.subtitle_box.setCurrentIndex(max(0, self.subtitle_box.findData(sub_index)))
            info = entry.info
            self.detail_title.setText(f"{path.name} — {_clock(info.duration)}, {info.width}×{info.height}, {info.codec}"
                                      + (f", {len(entry.audio)} audio / {len(entry.subtitles)} subtitle tracks"))
        else:
            self.detail_title.setText(f"{count} videos selected: each is converted with its own best audio track, no subtitles "
                                      "(select one video to choose)." if count > 1 else
                                      "Select a video to choose its sound and subtitles.")
        for box in (self.audio_box, self.subtitle_box):
            box.setEnabled(enabled)
            box.blockSignals(False)

    def _choice_changed(self) -> None:
        paths = self._selected_paths()
        if len(paths) == 1:
            self.choice[paths[0]] = (self.audio_box.currentData() if self.audio_box.currentData() is not None else -1,
                                     self.subtitle_box.currentData() if self.subtitle_box.currentData() is not None else -1)
            if self.subtitle_box.currentData() not in (None, -1):
                self.mkv.setChecked(True)                      # subtitles can only be kept in an MKV

    def _convert_all(self) -> None:
        self.tree.selectAll()
        self._finish("convert")

    def _finish(self, action: str) -> None:
        paths = self._selected_paths()
        if not paths or (action == "open" and len(paths) != 1):
            return
        self.jobs = [(p, *self.choice.get(p, (-1, -1))) for p in paths]
        self.action = action
        self.save_mkv = self.mkv.isChecked()
        self.accept()

    def done(self, result: int) -> None:  # noqa: D102 - Qt name
        self._stop_scan()
        super().done(result)
