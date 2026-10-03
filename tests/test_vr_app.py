"""VR in the app's own screens: the Library's Play in VR button, Space going to the headset, and picking a room."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("av")

from test_video import _make_clip  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402
from ultimate_video_3d import library as lib  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Media:
    def __init__(self, is_open=True):
        self.is_open = is_open


class FakeVR:
    """Stands in for VRController so no headset is needed."""

    def __init__(self, running=False, open_media=False):
        self.session = None
        self.room = None
        self.running = running
        self.media = _Media(open_media)
        self.calls = []

    @staticmethod
    def available():
        return True

    def use_room(self, folder):
        self.calls.append(("use_room", folder))
        return True

    def show(self, path, layout, preview=False):
        self.calls.append(("show", Path(path).name, layout, preview))
        return True

    def toggle(self):
        self.calls.append("toggle")

    def set_videos(self, tiles):
        self.calls.append(("videos", len(tiles)))

    def stop(self):
        self.calls.append("stop")


@pytest.fixture()
def win(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(appmod.paths, "settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(appmod.paths, "data_dir", lambda: tmp_path)
    window = appmod.MainWindow()
    window.vr = FakeVR()
    window.show()
    qapp.processEvents()
    yield window
    window.vr = FakeVR()                                            # so closing does not touch a real session
    window.close()


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "film_SBS.mp4"
    _make_clip(path, frames=30, audio=False)
    return path


def test_the_library_fades_play_in_vr_out_when_vr_cannot_work_here(win, clip):
    page = win.library_page
    page.add_files([clip])
    page.grid.setCurrentRow(0)
    page.set_vr_available(False, "No OpenXR runtime is installed.")
    assert not page.vr_button.isHidden() and not page.vr_button.isEnabled()    # always there, faded out
    assert "OpenXR runtime" in page.vr_button.toolTip()                        # and it says why
    page.set_vr_available(True)
    assert not page.vr_button.isHidden() and page.vr_button.isEnabled()
    page.set_vr_available(True)
    page.grid.clearSelection()
    page.grid.setCurrentItem(None)
    page._selected(None)
    assert not page.vr_button.isEnabled()                                      # nothing chosen: nothing to play


def test_play_in_vr_pauses_the_local_player_and_hands_the_video_to_the_window(win, clip, monkeypatch):
    page = win.library_page
    page.add_files([clip])
    page.grid.setCurrentRow(0)
    page.set_vr_available(True)
    asked = []
    page.vr_requested.connect(asked.append)
    page.vr_button.click()
    assert len(asked) == 1 and asked[0].file == clip
    win._play_in_vr(asked[0])
    assert ("show", "film_SBS.mp4", "sbs_half", False) in win.vr.calls
    kinds = [c[0] if isinstance(c, tuple) else c for c in win.vr.calls]
    assert "videos" in kinds and kinds.index("use_room") < kinds.index("show")      # the Library and the room are set first


def test_space_goes_to_the_headset_when_one_is_playing_and_not_otherwise(win):
    win.vr = FakeVR(running=True, open_media=True)
    assert win._space_for_vr() is True and win.vr.calls == ["toggle"]
    win.vr = FakeVR(running=True, open_media=False)                            # a session but nothing loaded
    assert win._space_for_vr() is False and win.vr.calls == []
    win.vr = FakeVR(running=False, open_media=True)
    assert win._space_for_vr() is False


def test_the_application_key_filter_takes_space_for_vr_but_never_from_a_text_box(win, qapp, monkeypatch):
    win.vr = FakeVR(running=True, open_media=True)
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ")
    assert win.eventFilter(win, press) is True and win.vr.calls == ["toggle"]
    box = QLineEdit(win)
    win.vr.calls.clear()
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: box))     # an offscreen window never takes focus
    assert win.eventFilter(box, press) is False and win.vr.calls == []         # typing a space in a field still works
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: None))
    win.vr = FakeVR(running=False, open_media=False)
    assert win.eventFilter(win, press) is False                                # no headset playing: Space is left alone


def test_a_room_folder_must_hold_a_room_and_is_remembered_in_the_settings(win, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    empty = tmp_path / "not_a_room"
    empty.mkdir()
    monkeypatch.setattr(appmod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(empty))
    win._choose_vr_room()
    assert warned and win.settings.vr_room_dir == ""
    good = tmp_path / "a_room"
    good.mkdir()
    (good / "room.glb").write_bytes(b"x")
    monkeypatch.setattr(appmod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(good))
    win._choose_vr_room()
    assert win.settings.vr_room_dir == str(good) and win._vr_room_dir() == good
    assert str(good) in win.settings_page.vr_room_label.text()
    win._clear_vr_room()
    assert win.settings.vr_room_dir == "" and win._vr_room_dir() is None


def test_a_room_in_the_data_folder_is_found_without_choosing_it(win, tmp_path):
    rooms = tmp_path / "rooms" / "home_theatre"
    rooms.mkdir(parents=True)
    (rooms / "room.glb").write_bytes(b"x")
    assert win._vr_room_dir() == rooms


def test_the_chosen_room_is_saved_and_comes_back_after_a_restart(tmp_path):
    from ultimate_video_3d.settings import AppSettings

    settings = AppSettings()
    settings.vr_room_dir = "D:/rooms/theatre"
    settings.save(tmp_path / "s.json")
    assert AppSettings.load(tmp_path / "s.json").vr_room_dir == "D:/rooms/theatre"


def test_a_missing_headset_is_reported_as_please_connect_it(win, monkeypatch):
    from ultimate_video_3d.vr import session as sessionmod

    class FormFactorUnavailable(Exception):
        pass

    assert sessionmod.friendly_error(FormFactorUnavailable("XR_ERROR_FORM_FACTOR_UNAVAILABLE")) == sessionmod.NO_HEADSET
    assert "connect" in sessionmod.NO_HEADSET.lower()
    assert sessionmod.friendly_error(ValueError("boom")) == "ValueError: boom"        # anything else keeps its detail
    shown = []
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a: shown.append(a[2]))
    win._vr_failed(sessionmod.NO_HEADSET)
    assert shown == [sessionmod.NO_HEADSET]
