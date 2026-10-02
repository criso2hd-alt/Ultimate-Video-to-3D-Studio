"""Depth zones: the numbers and colours that tell a person where things will sit relative to the screen."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d import stereo, zones  # noqa: E402


def _scene(convergence=0.5, strength=50.0, near_scale=1.0, far_scale=1.0, width=400, height=200):
    """Left half far (depth 0.1), right half near (depth 0.9): the disparity map for those settings."""
    depth = np.full((height, width), 0.1, np.float32)
    depth[:, width // 2:] = 0.9
    params = stereo.StereoParams(strength=strength, convergence=convergence, near_scale=near_scale, far_scale=far_scale)
    return depth, stereo.disparity_map(depth, params, width), width


def _m(**kwargs):
    """The stats for the two-halves scene with these settings."""
    depth, disp, _ = _scene(**kwargs)
    return zones.measure(disp, depth)


def test_near_things_are_in_front_and_far_things_behind():
    depth, disp, width = _scene(convergence=0.5)
    pct = zones.to_percent(disp, width)
    assert pct[100, 300] > 0 and pct[100, 100] < 0                  # near right: out of the screen; far left: behind
    stats = zones.measure(disp, depth)
    assert stats.valid and stats.out_pct > 0 and stats.back_pct > 0
    assert stats.out_pct == pytest.approx(stats.back_pct, rel=0.05)  # symmetric about the middle
    assert stats.share_out == pytest.approx(0.5, abs=0.05) and stats.share_back == pytest.approx(0.5, abs=0.05)


def test_the_numbers_follow_the_strength_and_the_convergence():
    weak = _m(strength=20)
    strong = _m(strength=80)
    assert strong.out_pct == pytest.approx(weak.out_pct * 4, rel=0.05)
    behind = _m(convergence=1.0)             # the screen at the very front: nothing comes out
    assert behind.out_pct == 0.0 and behind.share_out == 0.0 and behind.back_pct > 0
    front = _m(convergence=0.0)
    assert front.back_pct == 0.0 and front.out_pct > 0
    only_near = _m(far_scale=0.0)             # background depth switched off
    assert only_near.back_pct == 0.0 and only_near.out_pct > 0


def test_the_verdict_grades_comfort():
    assert _m(strength=15).verdict == "comfortable"       # ~0.6% of the width
    assert _m(strength=50).verdict in ("strong", "comfortable")
    assert _m(strength=100, convergence=0.0).verdict == "too strong"   # all in front, 4.5% of the width: far beyond comfort
    assert _m(strength=100).verdict == "strong"                        # centred, the strongest shift is exactly 2%
    assert zones.DepthStats().verdict == ""                                       # nothing measured yet


def test_the_zone_picture_is_warm_in_front_cool_behind_and_marks_the_screen_line():
    depth, disp, width = _scene(convergence=0.5, strength=60)
    rgb = np.full(depth.shape + (3,), 0.5, np.float32)
    image = zones.zone_image(rgb, disp, label=False)
    near, far = image[100, 330].astype(int), image[100, 70].astype(int)
    assert near[0] > near[2] + 40                                    # red/orange dominates where it comes out
    assert far[2] > far[0] + 40                                      # blue dominates where it sinks back
    seam = image[100, 190:211].astype(int)                           # the seam between them is the screen plane
    assert seam.min(axis=1).max() > 180                              # a bright white pixel there (all three channels high)
    assert image.dtype == np.uint8 and image.shape == rgb.shape


def test_a_picture_on_the_screen_stays_clear():
    flat = np.zeros((120, 200), np.float32)                          # everything exactly at the screen
    rgb = np.full((120, 200, 3), 0.6, np.float32)
    image = zones.zone_image(rgb, flat, label=False).astype(int)
    assert np.abs(image[:, :, 0] - image[:, :, 2]).max() < 6         # no tint: red and blue equal


def test_the_legend_is_drawn_on_the_picture():
    depth, disp, width = _scene()
    rgb = np.full(depth.shape + (3,), 0.5, np.float32)
    with_label = zones.zone_image(rgb, disp, label=True)
    without = zones.zone_image(rgb, disp, label=False)
    assert (with_label != without).any() and (with_label[-40:, -200:] != without[-40:, -200:]).any()   # bottom right


def test_the_slider_marks_say_where_the_picture_splits_across_the_screen():
    depth, disp, width = _scene()
    stats = zones.measure(disp, depth)
    lo, hi, mid = zones.convergence_marks(stats)
    assert lo == pytest.approx(10.0, abs=2.0) and hi == pytest.approx(90.0, abs=2.0)   # the bulk spans depth 0.1 .. 0.9
    assert 10.0 <= mid <= 90.0                                                          # the median sits between
    assert zones.convergence_marks(zones.DepthStats()) is None
    # and they agree with what the disparity does: just inside the band the scene is split, just outside it is not
    inside = _m(convergence=0.5)
    assert inside.out_pct > 0 and inside.back_pct > 0
    below = _m(convergence=0.05)                              # slider below lo: all in front
    assert below.back_pct == 0.0
    above = _m(convergence=0.95)                              # slider above hi: all behind
    assert above.out_pct == 0.0


def test_stats_survive_the_trip_through_a_dict():
    depth, disp, width = _scene()
    stats = zones.measure(disp, depth)
    again = zones.DepthStats.from_dict(stats.to_dict())
    assert again.out_pct == stats.out_pct and np.array_equal(again.hist, stats.hist) and again.valid


# --- the Qt side: the Zones view, the gauge and the slider marks ------------------------------------------


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_zones_is_a_view_with_a_tip_and_it_paints_a_picture():
    from ultimate_video_3d import preview

    keys = [k for k, _ in preview.PREVIEW_MODES]
    assert "zones" in keys and keys.index("zones") == keys.index("depth") + 1 and "zones" in preview.PREVIEW_MODE_TIPS
    depth, disp, width = _scene()
    pair = stereo.Pair(np.zeros(depth.shape + (3,), np.float32), np.zeros(depth.shape + (3,), np.float32),
                       np.zeros(depth.shape, np.uint8), np.zeros(depth.shape, np.uint8), disp,
                       np.zeros(depth.shape + (3,), np.float32))
    params = preview.PreviewParams(mode="zones")
    images, badge, wiggle = preview.build_views(params, pair, depth, np.full(depth.shape + (3,), 0.5, np.float32))
    assert len(images) == 1 and images[0].dtype == np.uint8 and "Zones" in badge and not wiggle


def test_the_gauge_shows_a_hint_until_it_has_numbers_then_draws_them():
    from ultimate_video_3d.controls import DepthGauge

    _app()
    gauge = DepthGauge()
    gauge.resize(280, 92)
    empty = gauge.grab().toImage()
    depth, disp, _ = _scene(strength=40)
    gauge.set_stats(zones.measure(disp, depth))
    drawn = gauge.grab().toImage()
    assert empty != drawn                                              # the numbers and bars appeared
    colours = {drawn.pixelColor(x, y).name() for x in range(10, 270, 3) for y in range(25, 70, 3)}
    assert len(colours) > 6                                            # bars, bands and text, not a flat box


def test_the_depth_position_slider_gets_a_coloured_strip_where_the_picture_splits():
    from ultimate_video_3d.controls import ParamSlider

    _app()
    slider = ParamSlider("Depth position", 0, 100, 55, 55, 0)
    slider.resize(300, 44)
    slider.show()
    before = slider.grab().toImage()
    slider.set_zones(10.0, 90.0)
    after = slider.grab().toImage()
    assert before != after
    assert slider._slider._zones == pytest.approx((0.1, 0.9))
    slider.set_zones(None, None)
    assert slider._slider._zones is None


def test_the_3d_panel_passes_the_numbers_to_both_the_gauge_and_the_slider():
    from ultimate_video_3d import panels, settings

    _app()
    panel = panels.ThreeDPanel(settings.ThreeDSettings())
    depth, disp, _ = _scene()
    panel.set_depth_stats(zones.measure(disp, depth))
    lo, hi, mid = zones.convergence_marks(zones.measure(disp, depth))
    assert panel.gauge._stats.valid and panel.convergence._slider._zones == pytest.approx((lo / 100, hi / 100))
    assert panel.convergence._slider._mid == pytest.approx(mid / 100)
    panel.set_depth_stats(zones.DepthStats())                          # nothing measured (e.g. back on the 2D view)
    assert panel.convergence._slider._zones is None


def test_the_strength_marks_are_exact_because_disparity_is_proportional_to_strength():
    depth, disp, _ = _scene(strength=40)
    stats = zones.measure(disp, depth)
    comfort, strong = zones.strength_marks(stats, 40.0)
    assert comfort == pytest.approx(50.0, rel=0.03) and strong == pytest.approx(100.0, rel=0.03)   # 0.8% now: 1% at 50, 2% at 100
    # at the "comfortable until" strength the strongest part really is at the comfort limit
    at_limit = _m(strength=comfort)
    assert max(at_limit.out_pct, at_limit.back_pct) == pytest.approx(zones.COMFORT, rel=0.03)
    at_strong = _m(strength=strong)
    assert max(at_strong.out_pct, at_strong.back_pct) == pytest.approx(zones.STRONG, rel=0.03)
    assert zones.strength_marks(zones.DepthStats(), 40.0) is None                 # nothing measured
    assert zones.strength_marks(_m(convergence=1.0, far_scale=0.0), 40.0) is None  # nothing sticks out at all


def test_the_strength_slider_gets_a_green_amber_red_strip_and_loses_it_with_no_picture():
    from ultimate_video_3d import panels, settings

    _app()
    panel = panels.ThreeDPanel(settings.ThreeDSettings())
    depth, disp, _ = _scene(strength=panel.strength.value())
    panel.set_depth_stats(zones.measure(disp, depth))
    strip = panel.strength._slider._strip
    assert [round(a, 3) for a, _, _ in strip] == sorted(round(a, 3) for a, _, _ in strip) and len(strip) == 3
    assert strip[0][2] == panel.strength._slider.GOOD and strip[2][2] == panel.strength._slider.TOO_MUCH
    panel.set_depth_stats(zones.DepthStats())
    assert panel.strength._slider._strip == []
