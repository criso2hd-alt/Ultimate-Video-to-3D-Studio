"""The VR controller's logic, with stand-ins for the headset session and the media player (no headset needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d.vr.controller import VRController  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeMedia(QObject):
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opened, self.is_open, self.playing, self.muted_state, self.calls = None, False, False, False, []

    def open(self, path, layout, loop=False, muted=False):
        self.opened, self.is_open, self.muted_state = (path, layout, loop), True, muted
        self.calls.append(("open", loop, muted))
        return True

    def play(self):
        self.playing = True
        self.calls.append("play")

    def pause(self):
        self.playing = False
        self.calls.append("pause")

    def toggle(self):
        self.playing = not self.playing
        self.calls.append("toggle")

    def set_muted(self, muted):
        self.muted_state = muted
        self.calls.append(("muted", muted))

    def close(self, wait=False):
        self.is_open = False
        self.calls.append("close")


class FakeSession(QObject):
    worn_changed = Signal(bool)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, media, parent=None):
        super().__init__(parent)
        self.media, self.running, self.rooms, self.seats = media, False, [], []

    def start(self):
        self.running = True

    def stop(self, wait=5.0):
        self.running = False

    def set_room(self, room):
        self.rooms.append(room)

    def seat_names(self):
        return [("Seat_A", "A")]

    def set_seat(self, name):
        self.seats.append(name)


@pytest.fixture()
def vr(qapp):
    return VRController(session_factory=FakeSession, media_factory=FakeMedia)


def test_a_video_does_not_play_until_the_headset_is_on(vr):
    states, notes = [], []
    vr.state_changed.connect(states.append)
    vr.notice.connect(notes.append)
    assert vr.show(Path("film_SBS.mp4"), "sbs_half")
    assert vr.session.running and "starting" in states
    assert not vr.media.playing and "Put the headset on" in notes[-1]
    vr.session.worn_changed.emit(True)
    assert vr.media.playing and vr.state == "worn"                           # worn: it starts
    vr.session.worn_changed.emit(False)
    assert not vr.media.playing and vr.state == "waiting"                    # taken off: it pauses


def test_with_the_headset_already_on_a_new_video_starts_at_once(vr):
    vr.show(Path("a.mp4"), "sbs_half")
    vr.session.worn_changed.emit(True)
    vr.media.calls.clear()
    vr.show(Path("b.mp4"), "tb_half")
    assert vr.media.calls[0][0] == "open" and "play" in vr.media.calls


def test_a_preview_loops_muted_and_the_first_press_brings_the_sound_in(vr):
    vr.show(Path("clip.mp4"), "sbs_half", preview=True)
    assert vr.media.opened[2] is True and vr.media.muted_state is True                  # loop, muted
    vr.session.worn_changed.emit(True)
    assert vr.media.playing
    vr.toggle()
    assert vr.media.muted_state is False and vr.media.playing                          # sound in, still playing
    vr.toggle()
    assert not vr.media.playing                                                         # now it is a normal pause


def test_a_full_video_plays_with_sound_and_toggles_normally(vr):
    vr.show(Path("film.mp4"), "sbs_full")
    assert vr.media.opened[2] is False and vr.media.muted_state is False
    vr.session.worn_changed.emit(True)
    vr.toggle()
    assert not vr.media.playing


def test_a_failed_session_is_reported_and_the_state_goes_off(vr):
    vr.show(Path("a.mp4"), "sbs_half")
    notes, failures = [], []
    vr.notice.connect(notes.append)
    vr.failed.connect(failures.append)
    vr.session.failed.emit("SessionLost: gone")
    assert vr.state == "off" and failures == ["SessionLost: gone"] and "VR stopped" in notes[-1]


def test_stopping_closes_everything_and_forgets_that_the_headset_was_on(vr):
    vr.show(Path("a.mp4"), "sbs_half")
    vr.session.worn_changed.emit(True)
    vr.stop()
    assert not vr.session.running and "close" in vr.media.calls and not vr.worn and vr.state == "off"


def test_a_room_that_cannot_be_loaded_says_so_and_falls_back_to_the_plain_room(vr, tmp_path):
    notes = []
    vr.notice.connect(notes.append)
    assert vr.use_room(tmp_path) is False and vr.room is None and "no room.glb" in notes[-1]
    assert vr.use_room(None) is True


def test_seats_are_passed_to_the_session(vr):
    vr.show(Path("a.mp4"), "sbs_half")
    seen = []
    vr.seats_changed.connect(seen.append)
    vr.set_seat("Seat_A")
    assert vr.session.seats == ["Seat_A"] and vr.seat_names() == [("Seat_A", "A")]
