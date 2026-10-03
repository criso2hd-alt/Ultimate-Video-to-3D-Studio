"""Finding and cutting the black bars burned into a film's frame, so a wide film can fill a wide screen."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d import bars  # noqa: E402
from ultimate_video_3d.bars import Crop  # noqa: E402


def frame(h=90, w=160, rows=(0, 90), cols=(0, 160), level=120, noise=True, seed=0, spread=30):
    """A grey picture with bars: black (16, a little noisy like a real encode) outside ``rows`` x ``cols``, ``level`` inside."""
    rng = np.random.default_rng(seed)
    out = np.full((h, w), 16, np.uint8)
    if noise:
        out += rng.integers(0, 4, (h, w)).astype(np.uint8)
    inside = rng.integers(level - spread, level + spread, (rows[1] - rows[0], cols[1] - cols[0])).astype(np.uint8)
    out[rows[0]:rows[1], cols[0]:cols[1]] = inside
    return out


def test_letterbox_bars_are_found_with_a_hair_of_black_left_so_no_picture_is_cut():
    crop = bars.crop_from_gray([frame(rows=(11, 79), seed=i) for i in range(6)])                    # 2.35:1 in a 16:9 frame (68 of 90 rows)
    assert crop.left == 0 and crop.right == 0
    assert crop.top == pytest.approx(10 / 90) and crop.bottom == pytest.approx(10 / 90)           # 11 black rows, one kept
    assert 1.0 - crop.top - crop.bottom > 68 / 90                                                  # the whole picture is inside what is left
    assert crop.aspect_factor > 1.0


def test_pillarbox_bars_are_found_too():
    crop = bars.crop_from_gray([frame(cols=(20, 140), seed=i) for i in range(4)])
    assert crop.top == 0 and crop.bottom == 0 and crop.left == pytest.approx(19 / 160) and crop.right == pytest.approx(19 / 160)
    assert crop.aspect_factor < 1.0


def test_a_picture_without_bars_and_one_with_hair_thin_bars_are_left_alone():
    assert bars.crop_from_gray([frame(seed=i) for i in range(4)]).none
    assert bars.crop_from_gray([frame(rows=(1, 89), seed=i) for i in range(4)]).none               # 1 row of 90: noise, not a bar


def test_a_dark_scene_cannot_make_the_bars_look_bigger_only_smaller():
    dark = frame(rows=(11, 79), level=26, spread=8, seed=1)                                                   # a night scene: hardly anything bright
    bright = frame(rows=(11, 79), seed=2)
    assert bars.crop_from_gray([dark, dark, dark, bright, bright]).top == pytest.approx(10 / 90)  # the bright ones show where the picture is
    assert bars.crop_from_gray([dark, dark, dark]).none                                            # nothing to go on: no cut


def test_burned_in_subtitles_in_a_bar_stop_that_bar_being_cut_only_if_they_show_up_more_than_once():
    def with_text(seed):
        g = frame(rows=(11, 79), seed=seed)
        g[82:86, 40:120] = 220                                                                      # white subtitle text inside the lower bar
        return g

    clean = [frame(rows=(11, 79), seed=i) for i in range(5)]
    once = bars.crop_from_gray(clean + [with_text(9)])
    assert once.bottom == pytest.approx(10 / 90)                                                    # one stray frame is ignored
    twice = bars.crop_from_gray(clean + [with_text(8), with_text(9)])
    assert twice.bottom < 10 / 90                                                                   # a subtitle that comes back: the bar is kept


def test_a_result_that_would_leave_hardly_any_picture_is_a_mistake_and_shows_everything():
    sliver = [frame(rows=(40, 45), seed=i) for i in range(4)]
    assert bars.crop_from_gray(sliver).none
    assert bars.crop_from_gray([]).none


def test_crop_rect_cuts_inside_the_area_one_eye_sees():
    crop = Crop(top=0.1, bottom=0.2, left=0.05, right=0.0)
    assert bars.crop_rect((0, 0, 1, 1), crop) == pytest.approx((0.05, 0.1, 1.0, 0.8))
    assert bars.crop_rect((0.5, 0, 1, 1), Crop(left=0.1)) == pytest.approx((0.55, 0, 1, 1))        # the right half of a side-by-side frame
    assert bars.crop_rect((0, 0.5, 1, 1), Crop(top=0.2)) == pytest.approx((0, 0.6, 1, 1))          # the bottom half of a top-and-bottom one
    assert bars.crop_rect((0, 0, 1, 1), bars.NONE) == (0, 0, 1, 1)


# --- real files ---------------------------------------------------------------------------------------------------


av = pytest.importorskip("av")


def make_clip(path: Path, bars_rows=(45, 315), size=(640, 360), seconds=3, layout="", level=130):
    """A short H.264 clip whose picture sits between black bars, so the whole thing has gone through a real encode."""
    w, h = size
    out = av.open(str(path), "w")
    stream = out.add_stream("libx264", rate=24)
    eye_w = w // 2 if layout == "sbs_half" else w
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    rng = np.random.default_rng(3)
    for i in range(int(24 * seconds)):
        eye = np.full((h, eye_w, 3), 16, np.uint8)
        eye[bars_rows[0]:bars_rows[1]] = rng.integers(level - 40, level + 40, (bars_rows[1] - bars_rows[0], eye_w, 3)).astype(np.uint8)
        picture = np.concatenate([eye, eye], axis=1) if layout == "sbs_half" else eye
        for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return path


def test_a_letterboxed_clip_is_measured_and_a_clean_one_is_not(tmp_path):
    boxed = bars.detect(make_clip(tmp_path / "boxed.mp4"))
    assert boxed.top == pytest.approx(44 / 360, abs=0.02) and boxed.bottom == pytest.approx(44 / 360, abs=0.02)
    assert boxed.left == 0 and boxed.right == 0
    assert bars.detect(make_clip(tmp_path / "clean.mp4", bars_rows=(0, 360))).none


def test_each_eye_of_a_side_by_side_clip_is_measured_on_its_own(tmp_path):
    clip = make_clip(tmp_path / "sbs.mp4", size=(1280, 360), layout="sbs_half")
    crop = bars.detect(clip, "sbs_half")
    assert crop.top == pytest.approx(44 / 360, abs=0.02) and crop.left == 0 and crop.right == 0      # not the seam between the eyes


def test_a_file_that_cannot_be_read_just_shows_everything(tmp_path):
    (tmp_path / "junk.mp4").write_bytes(b"not a video" * 50)
    assert bars.detect(tmp_path / "junk.mp4").none
    assert bars.detect(tmp_path / "missing.mp4").none


# --- the VR player ----------------------------------------------------------------------------------------------------


def test_the_vr_screen_takes_the_shape_of_the_real_picture_and_trimming_can_be_switched_off(tmp_path):
    pytest.importorskip("PySide6.QtMultimedia")
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d.vr.media import VRMedia

    app = QApplication.instance() or QApplication([])
    media = VRMedia()
    clip = make_clip(tmp_path / "wide.mp4")
    assert media.open(clip, "")
    end = time.monotonic() + 15
    while media.crop.none and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
    assert media.aspect == pytest.approx(640 / 270, rel=0.03)                   # 640 x 270 of picture: about 2.37:1, not 16:9
    media.trim_bars = False
    assert media.aspect == pytest.approx(16 / 9, rel=0.01) and media.crop.none
    media.trim_bars = True
    media.close(wait=True)
    assert media.crop.none                                                      # nothing open: nothing to cut
