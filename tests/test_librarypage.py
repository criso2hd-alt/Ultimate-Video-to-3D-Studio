"""The Library tab: grid, adding, playing on a normal screen (original / anaglyph), removing, deleting."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("av")

from test_video import _make_clip  # noqa: E402

from ultimate_video_3d import library as lib  # noqa: E402
from ultimate_video_3d.librarypage import ANAGLYPH, ORIGINAL, LibraryPage  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _bin(path):
    """A stand-in Recycle Bin that, like the real one, fails (rather than raises) while the file is still open."""
    try:
        path.unlink()
        return True
    except OSError:
        return False


def _wait(app, condition, seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


@pytest.fixture()
def page(qapp, tmp_path):
    library = lib.Library(tmp_path / "library.json", tmp_path / "thumbs")
    widget = LibraryPage(library)
    widget.resize(1100, 640)
    widget.show()
    yield widget
    widget.shutdown()
    widget.close()


@pytest.fixture()
def sbs_clip(tmp_path):
    """A 3D-looking clip: a 640 x 180 side-by-side frame whose left half is red and right half is blue."""
    import av
    from fractions import Fraction

    path = tmp_path / "scene_FSBS.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 640, 180, "yuv420p"
        for i in range(48):
            img = np.zeros((180, 640, 3), np.uint8)
            img[:, :320] = (230, 40, 40)
            img[:, 320:] = (40, 40, 230)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_an_empty_library_says_what_to_do(page):
    assert page.empty.isVisible() and not page.grid.isVisible()
    assert "Add videos" in page.empty.text()


def test_adding_videos_lists_them_with_a_guessed_layout_and_makes_thumbnails(page, qapp, sbs_clip):
    assert page.add_files([sbs_clip]) == 1
    assert page.add_files([sbs_clip]) == 0                                   # already there
    assert page.grid.count() == 1 and not page.empty.isVisible()
    entry = page.library.entries[0]
    assert entry.layout == "sbs_full" and entry.kind == lib.ADDED
    assert _wait(qapp, lambda: page.library.thumb_path(entry).exists())


def test_playing_shows_the_original_frame_or_the_red_cyan_mix(page, qapp, sbs_clip):
    page.add_files([sbs_clip])
    entry = page.library.entries[0]
    shown = []
    page.player.frame_ready.connect(shown.append)
    page.play_entry(entry, autoplay=False)
    assert _wait(qapp, lambda: len(shown) >= 1)
    original = shown[-1]
    assert original.shape[1] == 2 * original.shape[0] * 640 // 640 // 1 or original.shape[1] > original.shape[0] * 3
    assert page.anaglyph_button.isEnabled()
    shown.clear()
    page._set_mode(ANAGLYPH)
    assert _wait(qapp, lambda: len(shown) >= 1)
    mixed = shown[-1]
    assert mixed.shape[1] < original.shape[1]                                  # one eye wide now
    centre = mixed[mixed.shape[0] // 2, mixed.shape[1] // 2]
    assert centre[0] > 100 and centre[2] > 100                                 # red from one eye, cyan/blue from the other
    page._set_mode(ORIGINAL)


def test_play_pause_and_the_position_follow_the_video(page, qapp, sbs_clip):
    page.add_files([sbs_clip])
    page.play_entry(page.library.entries[0])
    assert _wait(qapp, lambda: page.player.playing and page.slider.value() > 100)
    page.player.pause()
    assert not page.player.playing and page.play_button.text() == "Play"
    assert page.time_label.text().startswith("0:0")
    page.player.seek_ms(1200)
    assert _wait(qapp, lambda: page.slider.value() >= 1000)


def test_a_flat_video_cannot_be_viewed_as_anaglyph(page, qapp, tmp_path):
    flat = tmp_path / "holiday.mp4"
    _make_clip(flat, frames=30, audio=False)
    page.add_files([flat])
    assert page.library.entries[0].layout == ""
    page.play_entry(page.library.entries[0], autoplay=False)
    assert not page.anaglyph_button.isEnabled()


def test_removing_keeps_the_file_and_deleting_asks_then_uses_the_bin(page, qapp, sbs_clip, monkeypatch):
    page.add_files([sbs_clip])
    page.grid.setCurrentRow(0)
    page.remove_selected()
    assert page.library.entries == [] and sbs_clip.exists()

    page.add_files([sbs_clip])
    page.grid.setCurrentRow(0)
    answers = iter(["Cancel", "Move to Recycle Bin"])

    def fake_exec(box):
        want = next(answers)
        box._clicked = next(b for b in box.buttons() if b.text() == want)

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._clicked)
    monkeypatch.setattr(lib, "send_to_trash", _bin)
    page.delete_selected()
    assert sbs_clip.exists() and len(page.library.entries) == 1                # cancelled: nothing happened
    page.delete_selected()
    assert not sbs_clip.exists() and page.library.entries == []


def test_deleting_the_playing_video_stops_it_first(page, qapp, sbs_clip, monkeypatch):
    page.add_files([sbs_clip])
    page.grid.setCurrentRow(0)
    page.play_entry(page.library.entries[0])
    assert _wait(qapp, lambda: page.player.playing)
    closed = []
    real_close = page.player.close
    monkeypatch.setattr(page.player, "close", lambda wait=False: (closed.append(wait), real_close(wait))[1])
    monkeypatch.setattr(QMessageBox, "exec", lambda box: setattr(box, "_c", next(b for b in box.buttons() if "Recycle" in b.text())))
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._c)
    monkeypatch.setattr(lib, "send_to_trash", _bin)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))     # a modal box would hang a test
    page.delete_selected()
    assert not warned and closed == [True] and not sbs_clip.exists()                      # it waited for the decoder to let go


def test_a_moved_file_is_flagged_and_does_not_play(page, qapp, sbs_clip):
    page.add_files([sbs_clip])
    entry = page.library.entries[0]
    assert _wait(qapp, lambda: page.library.thumb_path(entry).exists())        # the worker has let go of the file
    sbs_clip.unlink()
    page.refresh()
    assert "not found" in page.grid.item(0).text()
    page.play_entry(page.library.entries[0])
    assert not page.player.playing


def test_changing_the_layout_updates_the_entry(page, qapp, sbs_clip):
    page.add_files([sbs_clip])
    entry = page.library.entries[0]
    page._set_layout(entry, "tb_half")
    assert page.library.entries[0].layout == "tb_half" and "Top-and-Bottom" in page.grid.item(0).text()


def test_the_start_vr_button_follows_the_session_and_asks_for_start_or_stop(page):
    page.set_vr_available(True)
    asked = []
    page.vr_start_requested.connect(lambda: asked.append("start"))
    page.vr_stop_requested.connect(lambda: asked.append("stop"))
    assert page.vr_start.text() == "Start VR" and "Press Start VR" in page.vr_state.text()
    page.vr_start.click()
    page.set_vr_state("starting")
    assert page.vr_start.text() == "Stop VR" and "Starting" in page.vr_state.text()
    page.vr_start.click()
    page.set_vr_state("waiting")
    assert "Put your headset on" in page.vr_state.text()
    page.set_vr_state("stopping")
    assert not page.vr_start.isEnabled()
    page.set_vr_state("off")
    assert page.vr_start.isEnabled() and asked == ["start", "stop"]
    page.set_vr_available(False, "No OpenXR runtime is installed.")
    assert not page.vr_start.isEnabled() and "OpenXR" in page.vr_state.text()
