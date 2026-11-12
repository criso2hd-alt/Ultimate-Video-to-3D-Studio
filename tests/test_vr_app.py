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

    def stop(self, blocking=False):
        self.calls.append("stop")

    def start(self):
        self.calls.append("start")
        return True

    def start_vr(self):
        self.calls.append("start_vr")
        return True

    def watch(self, on):
        self.calls.append(("watch", on))

    def set_beam(self, on, level):
        self.calls.append(("beam", on, level))

    def set_rooms(self, tiles):
        self.calls.append(("rooms", [t.title for t in tiles]))


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
    assert page.vr_button.isEnabled()                                          # nothing chosen: still goes into VR, to pick a video there


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


def test_settings_say_which_room_you_start_in_and_leave_the_choosing_to_the_headset(win, tmp_path, monkeypatch):
    import json

    sp = win.settings_page
    assert not hasattr(sp, "vr_room_button") and not hasattr(sp, "vr_room_clear")             # the old room-folder buttons are gone
    room = tmp_path / "rooms" / "home_theatre"
    room.mkdir(parents=True)
    (room / "room.glb").write_bytes(b"x")
    (room / "room.json").write_text(json.dumps({"name": "Home theatre"}))
    win._refresh_vr_settings()
    assert "start in Home theatre" in sp.vr_room_label.text() and "Rooms page" in sp.vr_room_label.text()
    win._vr_room_chosen("plain")
    assert "start in the plain room" in sp.vr_room_label.text()
    opened = []
    monkeypatch.setattr(appmod, "open_folder", lambda folder: opened.append(folder))
    sp.vr_rooms_folder.click()
    assert opened == [tmp_path / "rooms"] and (tmp_path / "rooms").is_dir()


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


def test_play_in_vr_with_nothing_selected_goes_in_and_leaves_the_choice_to_the_headset(win, clip):
    page = win.library_page
    page.add_files([clip])
    page.grid.clearSelection()
    page.grid.setCurrentItem(None)
    page._selected(None)
    page.set_vr_available(True)
    asked = []
    page.vr_requested.connect(asked.append)
    page.vr_button.click()
    assert asked == [None]
    win._play_in_vr(None)
    assert "start_vr" in win.vr.calls and not any(isinstance(c, tuple) and c[0] == "show" for c in win.vr.calls)
    assert ("videos", 1) in win.vr.calls                                       # the Library was handed to the headset's menu


def test_the_library_tab_gets_ready_for_the_headset_and_leaving_it_stops_looking(win, qapp):
    win.library_page.set_vr_available(True)
    win.settings.vr_watch = True
    win.tabs.setCurrentWidget(win.library_page)
    assert ("watch", True) in win.vr.calls
    win.tabs.setCurrentWidget(win.video_page)
    assert win.vr.calls[-1] == ("watch", False)
    win.settings.vr_watch = False                                       # the person switched it off in Settings
    win.vr.calls.clear()
    win.tabs.setCurrentWidget(win.library_page)
    assert ("watch", True) not in win.vr.calls and ("watch", False) in win.vr.calls


def test_the_settings_switch_for_getting_ready_is_saved(win):
    win.settings_page.vr_auto.setChecked(False)
    assert win.settings.vr_watch is False
    win.settings_page.vr_auto.setChecked(True)
    assert win.settings.vr_watch is True


def test_the_projector_beam_switch_and_brightness_are_saved_and_sent_to_the_headset(win):
    sp = win.settings_page
    sp.vr_beam.setChecked(False)
    assert win.settings.vr_beam is False and ("beam", False, 1.0) in win.vr.calls
    sp.vr_beam.setChecked(True)
    sp.vr_beam_box.setCurrentIndex(sp.vr_beam_box.findData(1.5))
    assert win.settings.vr_beam_level == 1.5 and win.vr.calls[-1] == ("beam", True, 1.5)


def test_a_beam_change_made_in_the_headset_shows_in_settings_without_echoing_back(win):
    win.vr.calls.clear()
    win._vr_beam_from_headset(False, 0.5)
    sp = win.settings_page
    assert win.settings.vr_beam is False and win.settings.vr_beam_level == 0.5
    assert not sp.vr_beam.isChecked() and sp.vr_beam_box.currentData() == 0.5
    assert not any(isinstance(c, tuple) and c[0] == "beam" for c in win.vr.calls)      # the headset already knows


def test_the_projector_beam_choice_is_saved_and_a_damaged_value_is_ignored(tmp_path):
    import json

    from ultimate_video_3d.settings import AppSettings

    settings = AppSettings()
    assert settings.vr_beam is True and settings.vr_beam_level == 1.0               # on, at the normal strength, for everyone to begin with
    settings.vr_beam, settings.vr_beam_level = False, 0.5
    settings.save(tmp_path / "s.json")
    loaded = AppSettings.load(tmp_path / "s.json")
    assert loaded.vr_beam is False and loaded.vr_beam_level == 0.5
    raw = json.loads((tmp_path / "s.json").read_text())
    raw["vr_beam"], raw["vr_beam_level"] = "yes", 99
    (tmp_path / "s.json").write_text(json.dumps(raw))
    damaged = AppSettings.load(tmp_path / "s.json")
    assert damaged.vr_beam is True and damaged.vr_beam_level == 2.0                  # not a switch: ignored; too bright: held at the top


def test_the_headsets_room_page_is_filled_from_the_rooms_folder_and_a_pick_is_remembered(win, tmp_path):
    import json

    for folder, name in (("home_theatre", "Home theatre"), ("grand_cinema", "Grand cinema")):
        (tmp_path / "rooms" / folder).mkdir(parents=True)
        (tmp_path / "rooms" / folder / "room.glb").write_bytes(b"glTF")
        (tmp_path / "rooms" / folder / "room.json").write_text(json.dumps({"name": name}))
    win.vr.calls.clear()
    win._push_vr_rooms()
    assert win.vr.calls[-1] == ("rooms", ["Grand cinema", "Home theatre", "Plain room"])        # the plain room is always one of the choices
    win._vr_room_chosen(str(tmp_path / "rooms" / "grand_cinema"))
    assert win.settings.vr_room_dir == str(tmp_path / "rooms" / "grand_cinema")
    assert win._vr_room_dir() == tmp_path / "rooms" / "grand_cinema"                 # and it is the room the next session starts in


def test_the_plain_room_can_be_chosen_and_stays_chosen(win, tmp_path):
    from ultimate_video_3d.vr.rooms import PLAIN

    (tmp_path / "rooms" / "home_theatre").mkdir(parents=True)
    (tmp_path / "rooms" / "home_theatre" / "room.glb").write_bytes(b"glTF")
    assert win._vr_room_dir() == tmp_path / "rooms" / "home_theatre"                # nothing chosen: the first baked room
    win._vr_room_chosen(PLAIN)                                                      # picked on the headset's Rooms page
    assert win.settings.vr_room_dir == PLAIN and win._vr_room_dir() is None         # no baked room: the plain built-in one
    win._vr_room_chosen(str(tmp_path / "rooms" / "home_theatre"))                   # and back again
    assert win._vr_room_dir() == tmp_path / "rooms" / "home_theatre"


def test_the_start_vr_button_connects_without_choosing_a_video_and_stop_vr_leaves(win):
    page = win.library_page
    page.set_vr_available(True)
    page.vr_start.click()
    assert "start_vr" in win.vr.calls and not any(isinstance(c, tuple) and c[0] == "show" for c in win.vr.calls)
    page.set_vr_state("waiting")
    page.vr_start.click()
    assert "stop" in win.vr.calls


def test_automatic_headset_watching_is_off_unless_the_person_turns_it_on(win):
    assert win.settings.vr_watch is False and not win.settings_page.vr_auto.isChecked()


def test_changing_the_format_replaces_the_last_runs_encoder_message_with_the_files_description(win):
    from ultimate_video_3d import video as videomod

    page = win.video_page
    page.info = videomod.VideoInfo(1920, 1080, 24.0, 240, False, 10.0)
    page.info_label.setText("Encoding with NVIDIA H.264 at 12.0 Mb/s")        # left over from the last conversion
    win._export_changed()
    assert "H.264" not in page.info_label.text() and page.info_label.text() == page.info.describe()


def test_a_running_conversion_puts_its_times_on_the_picture_and_clears_them_after(win):
    from ultimate_video_3d.pipeline import VideoProgress

    page = win.video_page
    win._converting = True
    win._frame_done(VideoProgress("starting", message="Encoding with NVIDIA H.265 at 9.0 Mb/s"))
    win._frame_done(VideoProgress("converting", 50, 200, 20.0, None, "hevc_nvenc"))
    assert page.preview._stats and any("Elapsed" in line for line in page.preview._stats)
    assert any("H.265" in line for line in page.preview._stats)
    win._teardown()
    assert page.preview._stats is None


def test_the_exit_watchdog_ends_a_process_that_would_otherwise_linger():
    ended = []
    thread = appmod._exit_watchdog(grace=0.05, kill=ended.append)
    thread.join(2.0)
    assert ended == [0] and thread.daemon                              # a daemon thread, so it also runs while Python waits for the others


def test_fit_to_comfort_scans_the_film_and_sets_one_strength_for_all_of_it(win, qapp, monkeypatch):
    import time as timemod

    from ultimate_video_3d import comfort as comfortmod
    from ultimate_video_3d import video as videomod

    page = win.video_page
    page.source, page.info = Path("film.mp4"), videomod.VideoInfo(1920, 1080, 24.0, 2400, False, 100.0)
    page.disc_stream = None
    win.settings.threed.strength = 50.0
    seen = {}

    def fake_scan(path, threed, start_ms, end_ms, progress=None, should_stop=None, **kw):
        seen["args"] = (str(path), threed.strength, start_ms, end_ms)
        progress(0.5)
        return comfortmod.ComfortScan(strength=50.0, samples=[comfortmod.Sample(60_000, 2.5, 0.2), comfortmod.Sample(9000, 0.9, 0.1)])

    monkeypatch.setattr(comfortmod, "scan", fake_scan)
    page.threed.fit_comfortable.click()
    deadline = timemod.time() + 5
    while win._fit is not None and timemod.time() < deadline:
        qapp.processEvents()
        timemod.sleep(0.01)
    qapp.processEvents()
    assert win._fit is None and seen["args"][1:] == (50.0, 0.0, None)
    assert win.settings.threed.strength == pytest.approx(20.0, abs=0.6)          # 2.5% -> 1.0% is 50 * 1.0 / 2.5
    assert page.threed.strength.value() == pytest.approx(20.0, abs=0.6)
    assert "strongest moment" in page.threed.fit_note.text() and page.threed.fit_comfortable.text() == "Fit to comfort"


def test_the_gauge_is_updated_from_the_playing_frames_a_few_times_a_second(win):
    from ultimate_video_3d import zones as zonesmod

    page = win.video_page
    seen = []
    page.threed.set_depth_stats = seen.append
    zones_now = zonesmod.DepthStats(out_pct=1.7, valid=True).to_dict()
    import numpy as np

    picture = np.zeros((36, 64, 3), np.uint8)
    win._gauge_at = 0.0
    win._on_buffered_image(([picture], "", False, 1.0, None, zones_now))
    win._on_buffered_image(([picture], "", False, 1.0, None, zones_now))      # inside the same 150 ms: not again
    assert len(seen) == 1 and seen[0].out_pct == pytest.approx(1.7)
    win._on_buffered_image(([picture], "", False, 1.0, None))                  # an older 5-part payload still works


def test_choosing_high_quality_asks_before_the_download_and_declining_keeps_fast(win, monkeypatch):
    asked = []
    monkeypatch.setattr(appmod.depthmod, "missing_files", lambda model, nvidia=False: ["Depth-Anything-3-Mono-Large.onnx"])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No)
    box = win.video_page.threed.model_box
    box.setCurrentIndex(box.findData("quality"))
    assert asked and "1.3 GB" in asked[0]                                       # says how big, before anything happens
    assert win.settings.threed.depth_model == "fast" and box.currentData() == "fast"
    assert win._dl_thread is None


def test_choosing_an_installed_model_switches_the_preview_and_loads_it_in_the_background(win, monkeypatch):
    warmed = []
    monkeypatch.setattr(appmod.depthmod, "missing_files", lambda model, nvidia=False: [])
    monkeypatch.setattr(appmod.warmup, "start", lambda *a: warmed.append(a))
    box = win.video_page.threed.model_box
    box.setCurrentIndex(box.findData("quality"))
    assert win.settings.threed.depth_model == "quality" and win._params.depth_model == "quality"
    assert warmed and warmed[-1][1] == "quality"
    box.setCurrentIndex(box.findData("fast"))
    assert win.settings.threed.depth_model == "fast" and win._params.depth_model == "fast"


def test_the_nvidia_switch_uses_the_add_on_when_it_is_there(win, monkeypatch):
    monkeypatch.setattr(appmod.accelmod, "is_installed", lambda: True)
    monkeypatch.setattr(appmod.accelmod, "status", lambda: "Ready.")
    monkeypatch.setattr(appmod.accelmod, "has_nvidia", lambda gpus: True)
    monkeypatch.setattr(appmod.depthmod, "missing_files", lambda model, nvidia=False: [])
    monkeypatch.setattr(appmod.warmup, "start", lambda *a: None)
    sp = win.settings_page
    sp.nvidia_box.setChecked(True)
    assert win.settings.nvidia_accel and win._params.nvidia and sp.nvidia_note.text() == "Ready."
    sp.nvidia_box.setChecked(False)
    assert not win.settings.nvidia_accel and not win._params.nvidia


def test_without_an_nvidia_card_the_switch_says_why_it_is_off(win, monkeypatch):
    monkeypatch.setattr(appmod.accelmod, "has_nvidia", lambda gpus: False)
    win._update_nvidia_note()
    assert not win.settings_page.nvidia_box.isEnabled() and "NVIDIA" in win.settings_page.nvidia_note.text()


def test_a_failed_add_on_download_leaves_the_switch_off(win, monkeypatch):
    monkeypatch.setattr(appmod.accelmod, "is_installed", lambda: False)
    monkeypatch.setattr(appmod.accelmod, "has_nvidia", lambda gpus: True)
    monkeypatch.setattr(appmod.warmup, "start", lambda *a: None)

    def fake_download(title, subtitle, worker, then, failed_title, on_fail=None):
        assert isinstance(worker, appmod.AccelDownloadWorker)
        on_fail()

    monkeypatch.setattr(win, "_run_download", fake_download)
    win.settings_page.nvidia_box.setChecked(True)
    assert not win.settings.nvidia_accel and not win.settings_page.nvidia_box.isChecked()
