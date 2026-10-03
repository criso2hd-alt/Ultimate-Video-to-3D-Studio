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

    def set_audio_device(self, device):
        self.calls.append(("device", device.description()))

    def set_volume(self, volume):
        self.calls.append(("volume", volume))

    def close(self, wait=False):
        self.is_open = False
        self.calls.append("close")


class FakeSession(QObject):
    worn_changed = Signal(bool)
    failed = Signal(str)
    stopped = Signal()
    menu_action = Signal(object)

    def __init__(self, media, parent=None):
        super().__init__(parent)
        self.media, self.running, self.rooms, self.seats, self.videos = media, False, [], [], []

    def set_videos(self, tiles):
        self.videos = list(tiles)

    def set_audio_label(self, label):
        self.audio_label = label

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


def test_the_menus_requests_are_carried_out(vr):
    from ultimate_video_3d.vr.menu import Tile

    vr.set_videos([Tile("A film", "a_film.mp4", "sbs_half")])                  # given before the headset starts...
    vr.show(Path("first.mp4"), "sbs_half")
    assert [t.title for t in vr.session.videos] == ["A film"]                     # ...and handed over when it does
    vr.set_videos([Tile("B", "b.mp4"), Tile("C", "c.mp4")])
    assert len(vr.session.videos) == 2                                            # and kept current afterwards
    vr.session.worn_changed.emit(True)
    vr.media.calls.clear()
    vr.session.menu_action.emit(("video", Tile("B", "b.mp4", "tb_half")))
    assert vr.media.opened[0] == Path("b.mp4") and vr.media.opened[1] == "tb_half" and "play" in vr.media.calls
    vr.session.menu_action.emit(("volume", 0.4))
    assert ("volume", 0.4) in vr.media.calls
    vr.session.menu_action.emit(("exit",))
    assert not vr.session.running and vr.state == "off"


class _Device:
    def __init__(self, name):
        self._name = name

    def description(self):
        return self._name


SPEAKERS = _Device("Speakers (Realtek(R) Audio)")
HEADSET = _Device("Headphones (Oculus Virtual Audio Device)")
MONITOR = _Device("Acer XR382CQK (NVIDIA High Definition Audio)")


def _controller(qapp, probe=lambda: True, devices=(SPEAKERS, MONITOR, HEADSET)):
    return VRController(session_factory=FakeSession, media_factory=FakeMedia, probe=probe, audio_outputs=lambda: list(devices))


def _wait(qapp, condition, seconds=3.0):
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_watching_starts_the_session_by_itself_when_a_headset_appears(qapp):
    vr = _controller(qapp)
    notes = []
    vr.notice.connect(notes.append)
    vr.watch(True)
    assert _wait(qapp, lambda: vr.session is not None and vr.session.running)
    assert any("headset found" in n.lower() for n in notes)
    vr.watch(False)


def test_watching_stays_quiet_and_does_nothing_when_there_is_no_headset(qapp):
    vr = _controller(qapp, probe=lambda: False)
    notes = []
    vr.notice.connect(notes.append)
    vr.watch(True)
    _wait(qapp, lambda: False, 0.4)
    assert vr.session is None and notes == []
    vr.watch(False)


def test_a_session_that_finds_no_headset_while_watching_is_not_reported_as_an_error(qapp):
    from ultimate_video_3d.vr import NO_HEADSET

    vr = _controller(qapp, probe=lambda: False)
    failures, notes = [], []
    vr.failed.connect(failures.append)
    vr.notice.connect(notes.append)
    vr.watch(True)
    vr.start()
    vr.session.failed.emit(NO_HEADSET)
    assert failures == [] and not any("VR stopped" in n for n in notes) and vr.state == "off"
    vr.watch(False)
    vr.session.failed.emit(NO_HEADSET)                                        # not watching: the person asked for VR, so say so
    assert failures == [NO_HEADSET]


def test_leaving_vr_pauses_watching_until_vr_is_asked_for_again(qapp):
    vr = _controller(qapp)
    vr.watch(True)
    assert _wait(qapp, lambda: vr.session is not None and vr.session.running)
    vr._on_menu_action(("exit",))
    assert not vr.session.running
    _wait(qapp, lambda: False, 0.3)
    vr._poll()
    _wait(qapp, lambda: False, 0.3)
    assert not vr.session.running                                             # not pulled straight back in
    vr.watch(True)                                                            # coming back to the Library asks again
    assert _wait(qapp, lambda: vr.session.running)
    vr.watch(False)


def test_the_sound_goes_to_the_headset_by_itself(qapp):
    vr = _controller(qapp)
    vr.show(Path("film.mp4"), "sbs_half")
    assert ("device", "Headphones (Oculus Virtual Audio Device)") in vr.media.calls
    assert vr.session.audio_label == "Automatic: Headphones (Oculus Virtual Audio Device)"


def test_without_a_headset_device_the_default_is_left_alone(qapp):
    vr = _controller(qapp, devices=(SPEAKERS, MONITOR))
    vr.show(Path("film.mp4"), "sbs_half")
    assert not any(isinstance(c, tuple) and c[0] == "device" for c in vr.media.calls)
    assert "default" in vr.session.audio_label


def test_the_menus_sound_row_steps_through_the_devices_and_the_choice_is_reported(qapp):
    vr = _controller(qapp)
    vr.show(Path("film.mp4"), "sbs_half")
    chosen = []
    vr.audio_choice_changed.connect(chosen.append)
    vr.session.menu_action.emit(("audio", 1))                                  # automatic -> the first device
    assert chosen == ["Speakers (Realtek(R) Audio)"] and ("device", "Speakers (Realtek(R) Audio)") in vr.media.calls
    vr.session.menu_action.emit(("audio", -1))                                 # and back to automatic
    assert chosen[-1] == "" and vr.session.audio_label.startswith("Automatic")
    vr.session.menu_action.emit(("audio", -1))                                 # backwards from automatic wraps to the last one
    assert chosen[-1] == "Headphones (Oculus Virtual Audio Device)"


def test_a_saved_choice_wins_over_automatic_when_that_device_is_still_there(qapp):
    vr = _controller(qapp)
    vr.audio_choice = "Acer XR382CQK (NVIDIA High Definition Audio)"
    vr.show(Path("film.mp4"), "sbs_half")
    assert ("device", "Acer XR382CQK (NVIDIA High Definition Audio)") in vr.media.calls
    vr.audio_choice = "A device that was unplugged"
    assert vr.audio_device().description() == "Headphones (Oculus Virtual Audio Device)"          # falls back to automatic
