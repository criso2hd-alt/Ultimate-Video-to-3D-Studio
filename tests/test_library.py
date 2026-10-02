"""The Library: what it remembers, the layout guess for videos from elsewhere, the eye split, and deleting safely."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("av")

from test_video import _make_clip  # noqa: E402

from ultimate_video_3d import library  # noqa: E402
from ultimate_video_3d.library import Library  # noqa: E402


@pytest.fixture()
def lib(tmp_path):
    return Library(tmp_path / "library.json", tmp_path / "thumbs")


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "movie_SBS.mp4"
    _make_clip(path, frames=60, audio=False)
    return path


def test_layout_is_guessed_from_the_name_then_the_shape():
    g = library.guess_layout
    assert g(Path("Film.3D.HSBS.mkv"), 1920, 1080) == "sbs_half"
    assert g(Path("Film_SBS.mp4"), 1920, 1080) == "sbs_half"
    assert g(Path("Film_FSBS.mp4"), 3840, 1080) == "sbs_full"
    assert g(Path("Film.Half-OU.mkv"), 1920, 1080) == "tb_half"
    assert g(Path("Film_FTB.mp4"), 1920, 2160) == "tb_full"
    assert g(Path("Film_anaglyph.mp4"), 1920, 1080) == "anaglyph"
    assert g(Path("holiday.mp4"), 3840, 1080) == "sbs_full"                 # 32:9: two pictures side by side
    assert g(Path("holiday.mp4"), 1920, 2160) == "tb_full"                  # as tall as wide: stacked
    assert g(Path("holiday.mp4"), 1920, 1080) == ""                         # an ordinary flat video
    assert g(Path("Basketball.mp4"), 1920, 1080) == ""                      # "ball" must not read as a tag


def test_eyes_are_split_back_to_their_true_shape():
    frame = np.zeros((100, 400, 3), np.uint8)
    frame[:, :200] = (255, 0, 0)
    frame[:, 200:] = (0, 0, 255)
    left, right = library.split_eyes(frame, "sbs_half")
    assert left.shape == right.shape == (100, 400, 3) and left[0, 0, 0] == 255 and right[0, 0, 2] == 255
    left, right = library.split_eyes(frame, "sbs_full")
    assert left.shape == (100, 200, 3)
    stacked = np.zeros((200, 100, 3), np.uint8)
    top, bottom = library.split_eyes(stacked, "tb_half")
    assert top.shape == (200, 100, 3)                                         # each squeezed half is stretched back
    assert library.split_eyes(stacked, "")[0] is stacked


def test_anaglyph_view_mixes_the_two_eyes_and_original_leaves_the_frame_alone():
    frame = np.zeros((40, 80, 3), np.uint8)
    frame[:, :40] = 255                                                       # left eye white, right eye black
    out = library.view_of(frame, "sbs_full", "anaglyph")
    assert out.shape == (40, 40, 3) and out[0, 0, 0] > 100 and out[0, 0, 2] < 40   # red from the left eye only
    assert library.view_of(frame, "sbs_full", "original") is frame
    assert library.view_of(frame, "", "anaglyph") is frame                    # a flat video has no eyes to mix


def test_a_converted_video_is_listed_with_its_details_and_survives_a_restart(lib, clip, tmp_path):
    entry = lib.add_converted(clip, "sbs_half", "holiday.mp4")
    assert entry.kind == "converted" and entry.layout == "sbs_half" and entry.duration > 1 and entry.size > 0
    assert "Side-by-Side" in entry.detail and "MB" in entry.detail
    again = Library(tmp_path / "library.json", tmp_path / "thumbs")
    assert [e.path for e in again.entries] == [str(clip)] and again.entries[0].source == "holiday.mp4"


def test_reconverting_the_same_file_refreshes_it_instead_of_listing_it_twice(lib, clip):
    lib.add_converted(clip, "sbs_half")
    lib.add_converted(clip, "tb_half")
    assert len(lib.entries) == 1 and lib.entries[0].layout == "tb_half"


def test_a_video_from_elsewhere_can_be_added_once_and_gets_a_guessed_layout(lib, clip):
    entry = lib.add_external(clip)
    assert entry.kind == "added" and entry.layout == "sbs_half"
    assert lib.add_external(clip) is None and len(lib.entries) == 1
    assert lib.add_external(clip.with_name("nope.mp4")) is None
    lib.set_layout(entry.id, "tb_half")
    assert lib.get(entry.id).layout == "tb_half"


def test_newest_first(lib, clip, tmp_path):
    other = tmp_path / "b.mp4"
    other.write_bytes(clip.read_bytes())
    lib.add_converted(clip, "sbs_half")
    lib.add_external(other, "")
    assert [e.title for e in lib.entries] == ["b", "movie_SBS"]


def test_removing_forgets_the_entry_but_keeps_the_file(lib, clip):
    entry = lib.add_converted(clip, "sbs_half")
    lib.remove(entry.id)
    assert lib.entries == [] and clip.exists()


def test_deleting_sends_the_file_to_the_bin_and_does_nothing_if_that_fails(lib, clip, monkeypatch):
    entry = lib.add_converted(clip, "sbs_half")
    monkeypatch.setattr(library, "send_to_trash", lambda path: False)
    assert lib.delete_file(entry.id) is False and clip.exists() and len(lib.entries) == 1      # nothing is lost
    monkeypatch.setattr(library, "send_to_trash", lambda path: (path.unlink(), True)[1])
    assert lib.delete_file(entry.id) is True and not clip.exists() and lib.entries == []


def test_the_real_recycle_bin_takes_a_file(tmp_path):
    victim = tmp_path / "uv3d-test-delete-me.txt"
    victim.write_text("test")
    assert library.send_to_trash(victim) is True and not victim.exists()


def test_a_thumbnail_is_made_once_from_one_eye_and_dropped_when_the_layout_changes(lib, clip):
    entry = lib.add_converted(clip, "sbs_half")
    path = lib.make_thumbnail(entry)
    assert path is not None and path.exists() and path.stat().st_size > 500
    import cv2

    assert cv2.imread(str(path)).shape[:2] == (180, 320)
    lib.set_layout(entry.id, "tb_half")
    assert not path.exists()
    assert lib.make_thumbnail(lib.get(entry.id)) is not None


def test_a_missing_file_is_flagged_not_fatal(lib, clip):
    entry = lib.add_converted(clip, "sbs_half")
    clip.unlink()
    assert entry.missing and lib.make_thumbnail(entry) is None


def test_a_damaged_list_file_starts_empty(tmp_path):
    (tmp_path / "library.json").write_text("{not json")
    assert Library(tmp_path / "library.json", tmp_path / "t").entries == []
