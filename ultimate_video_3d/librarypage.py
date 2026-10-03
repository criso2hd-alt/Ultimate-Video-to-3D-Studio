"""The Library tab: your 3D videos in a grid, and a player that shows them on an ordinary screen.

A 3D video holds two pictures in one frame, so on a flat monitor it looks like two squashed copies. The
player therefore offers just the two views that are useful there: **Original** (the video as it is stored,
which is what a headset or a 3D TV wants) and **Anaglyph** (red/cyan, for cheap glasses on any screen).

Pictures are decoded on a thread of their own and the sound is played by Qt, whose position is the clock
(the same arrangement as the preview, see ``player.py``): the UI thread only ever draws finished pictures.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from . import library as lib
from . import paths, video
from .controls import PreviewView
from .player import Clock, FrameFeeder, proxy_from_frame
from .widgets import FONT_DISPLAY, apply_font

ORIGINAL, ANAGLYPH = "original", "anaglyph"
PROXY_HEIGHT = 720


class LibraryPlayer(QObject):
    """Plays one library video: a decode thread for the picture, Qt for the sound and the clock."""

    frame_ready = Signal(object)            # a uint8 picture, ready to paint
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
        self._feeder: FrameFeeder | None = None
        self._info: video.VideoInfo | None = None
        self._layout = ""
        self._mode = ORIGINAL
        self._playing = False
        self._has_audio = False
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def duration_ms(self) -> int:
        return int((self._info.duration if self._info else 0.0) * 1000)

    def set_volume(self, fraction: float) -> None:
        self._output.setVolume(max(0.0, min(1.0, fraction)))

    def set_mode(self, mode: str) -> None:
        """Original or anaglyph; redraws the current picture."""
        self._mode = mode
        if self._feeder is not None and not self._playing:
            self._feeder.refresh()

    # -- opening ----------------------------------------------------------------------------------

    def open(self, entry: lib.Entry) -> bool:
        self.close()
        try:
            info = video.probe(entry.file)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"Could not open {entry.file.name}: {error}")
            return False
        self._info, self._layout, self._has_audio = info, entry.layout, info.has_audio
        self._clock.stop()
        self._clock.set(0.0)
        self._audio.setSource(QUrl.fromLocalFile(str(entry.file)))
        feeder = FrameFeeder(entry.file, info, self._clock, self._delivered)
        feeder.proxy_height = PROXY_HEIGHT
        self._feeder = feeder
        feeder.start()
        feeder.seek(0.0, True)                           # the first picture; stay paused
        self.state_changed.emit(False)
        return True

    def close(self, wait: bool = False) -> None:
        """Stop and let go of the file. ``wait`` blocks until the decoder has really closed it, which
        a delete needs (Windows will not move a file that is still open)."""
        feeder, self._feeder = self._feeder, None
        if feeder is not None:
            feeder.shutdown()
        self._audio.stop()
        self._audio.setSource(QUrl())                    # the audio backend keeps the file open until told otherwise
        self._playing = False
        if wait and feeder is not None:
            feeder.join(3.0)

    def shutdown(self) -> None:
        self.close()

    # -- transport --------------------------------------------------------------------------------

    def play(self) -> None:
        if self._feeder is None or self._playing:
            return
        if self.duration_ms and self._clock.now() >= self.duration_ms - 200:
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
        self._clock.set(ms)
        if self._has_audio:
            self._audio.setPosition(int(ms))
        self._feeder.seek(ms, precise)

    # -- pictures and time ------------------------------------------------------------------------

    def _delivered(self, rgb: np.ndarray, index: int, ms: float) -> None:
        """On the decode thread: shape the picture for the chosen view, then hand it over."""
        picture = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        self.frame_ready.emit(lib.view_of(picture, self._layout, self._mode))
        feeder = self._feeder
        if feeder is not None and feeder.approximate:
            return                                       # a keyframe-only scrub picture is not a position
        self.position_changed.emit(int(ms))
        if self._playing and self.duration_ms and ms >= self.duration_ms - 120:
            self.ended.emit()

    def _audio_position(self, ms: int) -> None:
        if self._has_audio and self._playing:
            self._clock.set(ms)                          # the sound is the master clock while it plays


class _ThumbSignals(QObject):
    ready = Signal(str)


class LibraryPage(QWidget):
    """The tab: grid on the left, player on the right."""

    message = Signal(str)                    # a line for the window's status bar
    vr_requested = Signal(object)            # the entry to play in the headset

    def __init__(self, library: lib.Library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.library = library
        self._current: lib.Entry | None = None
        self._vr_available = False
        #: Set by the window: called when Space is pressed; returns True if it used the key (a headset is playing).
        self.space_hook = None
        self._dragging = False
        self._thumbs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uv3d-thumb")
        #: Held while a thumbnail is being made, and while a file is deleted: a file being read cannot be moved.
        self._file_io = threading.Lock()
        self._thumb_signals = _ThumbSignals()
        self._thumb_signals.ready.connect(self._thumb_ready)
        self.player = LibraryPlayer(self)
        self.player.frame_ready.connect(self._show_frame)
        self.player.position_changed.connect(self._position)
        self.player.state_changed.connect(self._state)
        self.player.ended.connect(self._ended)
        self.player.failed.connect(self.message)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(14)

        # -- the grid
        left = QVBoxLayout()
        left.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Your 3D videos")
        title.setObjectName("cardTitle")
        apply_font(title, family=FONT_DISPLAY, size=11, spacing=1.2, caps=True)
        head.addWidget(title)
        head.addStretch(1)
        self.add_button = QPushButton("Add videos…")
        self.add_button.setObjectName("secondary")
        self.add_button.setToolTip("Add 3D videos you already have, from anywhere. They are not copied or changed.")
        self.add_button.clicked.connect(self.add_videos)
        head.addWidget(self.add_button)
        left.addLayout(head)

        self.grid = QListWidget()
        self.grid.setObjectName("libraryGrid")
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setIconSize(QSize(220, 124))
        self.grid.setGridSize(QSize(244, 190))
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setWordWrap(True)
        self.grid.setSpacing(4)
        self.grid.setUniformItemSizes(True)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        self.grid.itemDoubleClicked.connect(lambda item: self.play_entry(self._entry_of(item)))
        self.grid.currentItemChanged.connect(lambda item, _: self._selected(self._entry_of(item)))
        left.addWidget(self.grid, 1)

        self.empty = QLabel("Videos you convert appear here.\nTo add 3D videos you already have, press “Add videos…”.")
        self.empty.setObjectName("hint")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        left.addWidget(self.empty)

        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setMinimumWidth(330)
        outer.addWidget(left_box, 5)

        # -- the player
        right = QVBoxLayout()
        right.setSpacing(8)
        self.view = PreviewView()
        self.view.setMinimumSize(360, 240)
        self.view.clicked.connect(self.player.toggle)
        right.addWidget(self.view, 1)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("secondary")
        self.play_button.setFixedWidth(80)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.player.toggle)
        transport.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._drag_start)
        self.slider.sliderMoved.connect(lambda ms: self.player.seek_ms(ms, precise=False))
        self.slider.sliderReleased.connect(self._drag_end)
        transport.addWidget(self.slider, 1)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("hint")
        transport.addWidget(self.time_label)
        right.addLayout(transport)

        options = QHBoxLayout()
        self.original_button = QPushButton("Original")
        self.anaglyph_button = QPushButton("Anaglyph")
        for button, mode, tip in (
            (self.original_button, ORIGINAL, "The video as stored: what a headset or a 3D TV expects."),
            (self.anaglyph_button, ANAGLYPH, "Red/cyan, for cheap 3D glasses on any screen."),
        ):
            button.setCheckable(True)
            button.setObjectName("secondary")
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            options.addWidget(button)
        self.original_button.setChecked(True)
        options.addStretch(1)
        volume_label = QLabel("Volume")
        volume_label.setObjectName("hint")
        options.addWidget(volume_label)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(110)
        self.volume.valueChanged.connect(lambda v: self.player.set_volume(v / 100.0))
        self.player.set_volume(0.8)
        options.addWidget(self.volume)
        right.addLayout(options)

        self.caption = QLabel("Choose a video to play it here.")
        self.caption.setObjectName("hint")
        self.caption.setWordWrap(True)
        right.addWidget(self.caption)

        actions = QHBoxLayout()
        self.folder_button = QPushButton("Show in folder")
        self.folder_button.setObjectName("secondary")
        self.folder_button.clicked.connect(self.show_in_folder)
        self.remove_button = QPushButton("Remove from list")
        self.remove_button.setObjectName("secondary")
        self.remove_button.setToolTip("Forget this video here. The file is not touched.")
        self.remove_button.clicked.connect(self.remove_selected)
        self.vr_button = QPushButton("Play in VR")
        self.vr_button.setObjectName("secondary")
        self.vr_button.setToolTip("Watch this video in your VR headset (put the headset on when it says so).")
        self.vr_button.clicked.connect(self._play_in_vr)
        self.vr_button.setToolTip("Watch this video in your VR headset (put the headset on when it says so).")
        self.delete_button = QPushButton("Delete…")
        self.delete_button.setObjectName("secondary")
        self.delete_button.setToolTip("Move the file to the Recycle Bin.")
        self.delete_button.clicked.connect(self.delete_selected)
        for button in (self.folder_button, self.remove_button, self.delete_button):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addWidget(self.vr_button)
        actions.addStretch(1)
        right.addLayout(actions)

        right_box = QWidget()
        right_box.setLayout(right)
        outer.addWidget(right_box, 7)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._space,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.refresh()

    # -- the list ---------------------------------------------------------------------------------

    def refresh(self) -> None:
        keep = self._current.id if self._current else None
        self.grid.blockSignals(True)
        self.grid.clear()
        for entry in self.library.entries:
            item = QListWidgetItem(self._icon(entry), f"{entry.title}\n{entry.detail}" if not entry.missing
                                   else f"{entry.title}\n(file not found)")
            item.setData(Qt.ItemDataRole.UserRole, entry.id)
            item.setToolTip(entry.path + ("" if entry.kind == lib.CONVERTED else "\nAdded by you"))
            item.setSizeHint(QSize(236, 182))
            self.grid.addItem(item)
            if keep == entry.id:
                self.grid.setCurrentItem(item)
        self.grid.blockSignals(False)
        self.empty.setVisible(not self.library.entries)
        self.grid.setVisible(bool(self.library.entries))

    def _icon(self, entry: lib.Entry) -> QIcon:
        thumb = self.library.thumb_path(entry)
        if thumb.exists():
            return QIcon(QPixmap(str(thumb)))
        self._thumbs.submit(self._make_thumb, entry)
        placeholder = QPixmap(320, 180)
        placeholder.fill(Qt.GlobalColor.black)
        return QIcon(placeholder)

    def _make_thumb(self, entry: lib.Entry) -> None:
        with self._file_io:
            if self.library.get(entry.id) is None:       # removed or deleted while it waited its turn
                return
            made = self.library.make_thumbnail(entry)
        if made is not None:
            self._thumb_signals.ready.emit(entry.id)

    def _thumb_ready(self, entry_id: str) -> None:
        entry = self.library.get(entry_id)
        if entry is None:
            return
        for row in range(self.grid.count()):
            item = self.grid.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == entry_id:
                item.setIcon(QIcon(QPixmap(str(self.library.thumb_path(entry)))))
                return

    def _entry_of(self, item: QListWidgetItem | None) -> lib.Entry | None:
        return self.library.get(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _selected(self, entry: lib.Entry | None) -> None:
        for button in (self.folder_button, self.remove_button, self.delete_button):
            button.setEnabled(entry is not None)
        self.delete_button.setEnabled(entry is not None and not entry.missing)
        self.vr_button.setEnabled(self._vr_available and entry is not None and not entry.missing)
        if entry is not None:
            self.caption.setText(f"{entry.title}  ·  {entry.detail}"
                                 + ("" if entry.kind == lib.CONVERTED else "  ·  added by you"))

    def selected_entry(self) -> lib.Entry | None:
        return self._entry_of(self.grid.currentItem())

    # -- adding, removing, deleting ---------------------------------------------------------------

    def add_videos(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(video.INPUT_SUFFIXES))
        files, _ = QFileDialog.getOpenFileNames(self, "Add 3D videos", "", f"Video ({exts})")
        if files:
            self.add_files([Path(f) for f in files])

    def add_files(self, files: list[Path]) -> int:
        added = [self.library.add_external(f) for f in files]
        count = sum(1 for e in added if e is not None)
        self.refresh()
        if count:
            self.message.emit(f"Added {count} video{'s' if count != 1 else ''}. If the 3D layout was guessed wrong, "
                              "right-click the video and choose “3D layout”.")
        return count

    def remove_selected(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if self._current and self._current.id == entry.id:
            self._stop_playback()
        self.library.remove(entry.id)
        self.refresh()
        self.message.emit(f"Removed “{entry.title}” from the list. The file is untouched.")

    def delete_selected(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        box = QMessageBox(QMessageBox.Icon.Question, "Delete this video?",
                          f"Move “{entry.file.name}” to the Recycle Bin?\n\nYou can still restore it from there.",
                          parent=self)
        yes = box.addButton("Move to Recycle Bin", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not yes:
            return
        if self._current and self._current.id == entry.id:
            self._stop_playback(wait=True)               # a playing file is open: it cannot be moved
        if self._delete_keeping_the_window_alive(entry.id):
            self.message.emit(f"Moved “{entry.file.name}” to the Recycle Bin.")
        else:
            QMessageBox.warning(self, "Could not delete", "The file could not be moved to the Recycle Bin, "
                                "so it was left where it is.")
        self.refresh()

    def _delete_keeping_the_window_alive(self, entry_id: str) -> bool:
        """Move the file to the Recycle Bin, trying again for a couple of seconds if something still has it open.

        The waiting happens with the event loop running: Qt's audio backend closes a file it was playing only
        when its events are delivered, so a plain sleep here would keep the very lock it is waiting out."""
        for _ in range(10):
            with self._file_io:                          # a file being cut into a thumbnail cannot be moved either
                if self.library.delete_file(entry_id, attempts=1) or self.library.get(entry_id) is None:
                    return True
            end = time.monotonic() + 0.25
            while time.monotonic() < end:
                QApplication.processEvents()
                time.sleep(0.01)
        return False

    def show_in_folder(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        from .panels import open_folder

        open_folder(entry.file.parent)

    def _menu(self, pos) -> None:
        entry = self._entry_of(self.grid.itemAt(pos))
        if entry is None:
            return
        self.grid.setCurrentItem(self.grid.itemAt(pos))
        menu = QMenu(self)
        menu.addAction("Play", lambda: self.play_entry(entry))
        layouts = menu.addMenu("3D layout")
        for key, label in lib.LAYOUT_CHOICES:
            action = QAction(label, layouts, checkable=True, checked=(entry.layout == key))
            action.triggered.connect(lambda _=False, k=key: self._set_layout(entry, k))
            layouts.addAction(action)
        menu.addSeparator()
        menu.addAction("Show in folder", self.show_in_folder)
        menu.addAction("Remove from list", self.remove_selected)
        menu.addAction("Delete (Recycle Bin)…", self.delete_selected)
        menu.exec(self.grid.viewport().mapToGlobal(pos))

    def _set_layout(self, entry: lib.Entry, layout: str) -> None:
        self.library.set_layout(entry.id, layout)
        self.refresh()
        if self._current and self._current.id == entry.id:
            self.play_entry(self.library.get(entry.id))

    # -- playing ----------------------------------------------------------------------------------

    def play_entry(self, entry: lib.Entry | None, autoplay: bool = True) -> None:
        if entry is None:
            return
        if entry.missing:
            self.message.emit(f"“{entry.title}” was moved or deleted: {entry.path}")
            return
        self._current = entry
        self._selected(entry)
        self.anaglyph_button.setEnabled(entry.is_stereo)
        if not entry.is_stereo and self.anaglyph_button.isChecked():
            self._set_mode(ORIGINAL)
        if self.player.open(entry):
            self.slider.setRange(0, self.player.duration_ms)
            self.play_button.setEnabled(True)
            if autoplay:
                self.player.play()

    def _stop_playback(self, wait: bool = False) -> None:
        self.player.close(wait)
        self._current = None
        self.view.clear()
        self.play_button.setEnabled(False)
        self.slider.setRange(0, 0)
        self.time_label.setText("0:00 / 0:00")

    def pause(self) -> None:
        self.player.pause()

    def shutdown(self) -> None:
        self.player.shutdown()
        self._thumbs.shutdown(wait=False, cancel_futures=True)

    def set_vr_available(self, available: bool, reason: str = "") -> None:
        """The Play in VR button is always there; it is faded out, with the reason as its tooltip, when this computer
        cannot do VR (the window says so). A headset that is merely not connected is reported when it is clicked."""
        self._vr_available = available
        self.vr_button.setToolTip(
            "Watch this video in your VR headset (put the headset on when it says so)." if available
            else reason or "VR is not available on this computer.")
        self._selected(self.selected_entry())

    def _play_in_vr(self) -> None:
        entry = self.selected_entry()
        if entry is not None and not entry.missing:
            self.player.pause()                           # the picture moves to the headset
            self.vr_requested.emit(entry)

    def _space(self) -> None:
        if self.space_hook is not None and self.space_hook():
            return                                        # a headset is playing: Space controls that
        if self._current is not None:
            self.player.toggle()

    def _set_mode(self, mode: str) -> None:
        self.original_button.setChecked(mode == ORIGINAL)
        self.anaglyph_button.setChecked(mode == ANAGLYPH)
        self.player.set_mode(mode)

    def _show_frame(self, picture) -> None:
        self.view.set_images([picture], ANAGLYPH.capitalize() if self.anaglyph_button.isChecked() else "", False)

    def _state(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        self.view.set_playing(playing)

    def _position(self, ms: int) -> None:
        if not self._dragging:
            self.slider.setValue(ms)
        self.time_label.setText(f"{_clock(ms)} / {_clock(self.player.duration_ms)}")

    def _ended(self) -> None:
        self.player.pause()
        self.player.seek_ms(0)

    def _drag_start(self) -> None:
        self._dragging = True

    def _drag_end(self) -> None:
        self._dragging = False
        self.player.seek_ms(self.slider.value(), precise=True)


def _clock(ms: int) -> str:
    s = max(0, int(ms // 1000))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
