"""The VR controller's logic, with stand-ins for the headset session and the media player (no headset needed)."""

from __future__ import annotations

import sys
import time
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

    def set_beam(self, on, level):
        self.beam = (on, level)

    def set_rooms(self, tiles, current=None):
        self.room_tiles, self.room_current = list(tiles), current

    def set_current_room(self, folder):
        self.room_current = folder

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


def test_the_projector_beam_setting_reaches_the_headset_and_the_headsets_choice_comes_back(vr):
    vr.beam_on, vr.beam_level = False, 0.5                                     # what the app's settings said before VR started
    vr.start()
    assert vr.session.beam == (False, 0.5)
    vr.set_beam(True, 1.5)
    assert vr.session.beam == (True, 1.5)
    heard = []
    vr.beam_changed.connect(lambda on, level: heard.append((on, level)))
    vr._on_menu_action(("beam", False, 0.25))                                  # the person switched it in the headset
    assert heard == [(False, 0.25)] and (vr.beam_on, vr.beam_level) == (False, 0.25)


def test_a_session_that_starts_by_itself_still_sits_in_the_chosen_room(qapp, tmp_path):
    vr = _controller(qapp)
    loaded = []

    def fake_use_room(folder):
        loaded.append(folder)
        vr.room = object()
        return True

    vr.use_room = fake_use_room
    vr.room_provider = lambda: tmp_path / "home_theatre"
    vr.watch(True)                                                             # a headset appears: the session starts without anyone pressing Play in VR
    assert _wait(qapp, lambda: vr.session is not None and vr.session.running)
    assert loaded == [tmp_path / "home_theatre"] and vr.session.rooms == [vr.room]
    vr.watch(False)


def test_without_a_room_to_offer_a_session_starts_in_the_plain_room_as_before(qapp):
    vr = _controller(qapp)
    vr.room_provider = lambda: None
    vr.watch(True)
    assert _wait(qapp, lambda: vr.session is not None and vr.session.running)
    assert vr.session.rooms == [] and vr.room is None
    vr.watch(False)


def test_a_real_failure_is_shown_once_and_the_app_does_not_keep_trying_by_itself(qapp):
    probes = []
    vr = _controller(qapp, probe=lambda: probes.append(1) or True)
    failures = []
    vr.failed.connect(failures.append)
    vr.watch(True)
    assert _wait(qapp, lambda: vr.session is not None and vr.session.running)
    vr.session.running = False
    vr.session.failed.emit("LimitReachedError: the runtime supports no more of the requested resource.")
    for _ in range(5):                                                          # the timer would have probed again, and again
        vr._poll()
        qapp.processEvents()
    time.sleep(0.2)
    qapp.processEvents()
    assert len(failures) == 1 and len(probes) == 1                              # one box, and no second try that would pile on
    vr.watch(True)                                                              # asked for again (the Library tab opened): it tries once more
    assert _wait(qapp, lambda: len(probes) == 2)
    vr.watch(False)


def test_the_headset_probe_leaves_the_runtime_alone_while_a_session_is_starting(monkeypatch):
    import sys
    import types

    from ultimate_video_3d import vr as vrpkg

    monkeypatch.setattr(vrpkg, "available", lambda: True)

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError("the runtime was touched during a start-up")

    monkeypatch.setitem(sys.modules, "xr", Untouchable())
    with vrpkg.XR_LOCK:
        assert vrpkg.headset_present() is False

    calls = []
    fake = types.SimpleNamespace(
        InstanceCreateInfo=lambda **k: k, ApplicationInfo=lambda **k: k, XR_CURRENT_API_VERSION=1,
        SystemGetInfo=lambda **k: k, FormFactor=types.SimpleNamespace(HEAD_MOUNTED_DISPLAY=1),
        create_instance=lambda info: calls.append("create") or "instance",
        get_system=lambda instance, info: calls.append("system"),
        destroy_instance=lambda instance: calls.append("destroy"))
    monkeypatch.setitem(sys.modules, "xr", fake)
    assert vrpkg.headset_present() is True and calls == ["create", "system", "destroy"]
    assert vrpkg.XR_LOCK.acquire(blocking=False)                                # and the lock is free again afterwards
    vrpkg.XR_LOCK.release()


def test_a_session_start_is_guarded_and_a_failed_one_is_cleaned_up_so_the_next_try_is_not_refused():
    from ultimate_video_3d import vr as vrpkg
    from ultimate_video_3d.vr.session import GuardedContext

    class Context:
        def __init__(self, fail):
            self.fail, self.entered, self.exited = fail, False, []
            self.locked_inside = None

        def __enter__(self):
            self.locked_inside = not vrpkg.XR_LOCK.acquire(blocking=False)     # True: the lock was held during the start-up
            if not self.locked_inside:
                vrpkg.XR_LOCK.release()
            if self.fail:
                raise RuntimeError("GraphicsRequirementsCallMissingError")
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited.append(args)

    good = Context(fail=False)
    with GuardedContext(good) as ctx:
        assert ctx is good and good.locked_inside is True
        assert vrpkg.XR_LOCK.acquire(blocking=False)                             # the lock is let go once the session exists
        vrpkg.XR_LOCK.release()
    assert len(good.exited) == 1

    bad = Context(fail=True)
    with pytest.raises(RuntimeError, match="GraphicsRequirements"):
        with GuardedContext(bad):
            pass
    assert len(bad.exited) == 1                                                  # what the failed start-up made was freed
    assert vrpkg.XR_LOCK.acquire(blocking=False)                                 # and the lock was released for the next try
    vrpkg.XR_LOCK.release()


def test_the_rooms_to_choose_from_reach_the_headset_and_a_choice_made_there_is_carried_out_and_reported(vr, tmp_path):
    from ultimate_video_3d.vr.menu import Tile

    tiles = [Tile("Home theatre", str(tmp_path / "a")), Tile("Grand cinema", str(tmp_path / "b"))]
    vr.set_rooms(tiles)                                                         # before VR has started: kept for when it does
    vr.start()
    assert vr.session.room_tiles == tiles
    used, chosen = [], []
    vr.use_room = lambda folder: used.append(folder) or True
    vr.room_chosen.connect(chosen.append)
    vr._on_menu_action(("room", tiles[1]))
    assert used == [tmp_path / "b"] and chosen == [str(tmp_path / "b")]


def test_a_room_that_cannot_be_loaded_is_not_reported_as_chosen(vr, tmp_path):
    from ultimate_video_3d.vr.menu import Tile

    chosen = []
    vr.room_chosen.connect(chosen.append)
    vr.start()
    vr._on_menu_action(("room", Tile("Broken", str(tmp_path))))                  # a folder with no room.glb: use_room says so and fails
    assert chosen == []


def test_the_plain_room_is_a_choice_like_any_other(vr, tmp_path):
    from ultimate_video_3d.vr.menu import Tile
    from ultimate_video_3d.vr.rooms import PLAIN

    vr.start()
    used, chosen = [], []
    vr.use_room = lambda folder: used.append(folder) or True
    vr.room_chosen.connect(chosen.append)
    vr._on_menu_action(("room", Tile("Plain room", PLAIN)))
    assert used == [None] and chosen == [PLAIN]                                  # no folder at all: the built-in room
    assert vr.room_key() == PLAIN                                                # and that is how the Rooms page names it before any room is loaded
