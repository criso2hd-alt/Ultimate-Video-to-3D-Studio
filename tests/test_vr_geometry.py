"""The VR player's maths: eye mapping, matrices, the virtual screen and pointing at it. No GPU or headset needed."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d.vr import geometry as g  # noqa: E402

IDENT = (0.0, 0.0, 0.0, 1.0)


def test_each_eye_sees_its_own_part_of_the_frame():
    assert g.eye_uv_rect("sbs_half", 0) == (0.0, 0.0, 0.5, 1.0)
    assert g.eye_uv_rect("sbs_full", 1) == (0.5, 0.0, 1.0, 1.0)
    assert g.eye_uv_rect("tb_half", 0) == (0.0, 0.0, 1.0, 0.5)
    assert g.eye_uv_rect("tb_full", 1) == (0.0, 0.5, 1.0, 1.0)
    assert g.eye_uv_rect("", 0) == g.eye_uv_rect("", 1) == (0.0, 0.0, 1.0, 1.0)         # flat: both eyes see all
    assert g.eye_uv_rect("anaglyph", 1) == (0.0, 0.0, 1.0, 1.0)


def test_the_screen_has_the_shape_of_one_eye_not_of_the_frame():
    assert g.eye_aspect("sbs_half", 1920, 1080) == pytest.approx(16 / 9)          # squeezed halves are stretched back
    assert g.eye_aspect("sbs_full", 3840, 1080) == pytest.approx(16 / 9)
    assert g.eye_aspect("tb_half", 1920, 1080) == pytest.approx(16 / 9)
    assert g.eye_aspect("tb_full", 1920, 2160) == pytest.approx(16 / 9)
    assert g.eye_aspect("", 1920, 800) == pytest.approx(2.4)                      # a scope film stays wide
    assert g.eye_aspect("sbs_half", 0, 0) == pytest.approx(16 / 9)


def test_quaternions_turn_things_the_right_way():
    s = math.sqrt(0.5)
    turn_left_90 = (0.0, s, 0.0, s)                                               # +90 degrees about the up axis
    m = g.quat_to_matrix(turn_left_90)
    assert np.allclose(m @ np.array([0, 0, -1.0]), [-1, 0, 0], atol=1e-9)         # forward becomes "to the left"
    assert np.allclose(g.quat_to_matrix(IDENT), np.eye(3))
    assert g.yaw_of(turn_left_90) == pytest.approx(math.pi / 2)
    assert g.yaw_of(IDENT) == pytest.approx(0.0)


def test_view_matrix_is_the_inverse_of_the_pose():
    pos, ori = (1.0, 1.7, -2.0), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    assert np.allclose(g.view_matrix(pos, ori) @ g.pose_matrix(pos, ori), np.eye(4), atol=1e-9)
    # a point straight ahead of the eye lands on the view's -z axis
    ahead = g.pose_matrix(pos, ori) @ np.array([0, 0, -2.0, 1])
    assert np.allclose((g.view_matrix(pos, ori) @ ahead)[:3], [0, 0, -2.0], atol=1e-9)


def test_projection_maps_the_view_edges_to_the_clip_edges():
    half = math.radians(45)
    p = g.projection_matrix(-half, half, half, -half, near=0.1, far=50)
    for x, y in ((1, 1), (-1, 1), (1, -1), (-1, -1)):                              # a point on the 45 degree edge, 1 m out
        clip = p @ np.array([x, y, -1.0, 1.0])
        assert abs(clip[0] / clip[3]) == pytest.approx(1.0) and abs(clip[1] / clip[3]) == pytest.approx(1.0)
    near = p @ np.array([0, 0, -0.1, 1.0])
    far = p @ np.array([0, 0, -50.0, 1.0])
    assert near[2] / near[3] == pytest.approx(-1.0) and far[2] / far[3] == pytest.approx(1.0)


def test_an_asymmetric_field_of_view_shifts_the_centre():
    p = g.projection_matrix(math.radians(-50), math.radians(40), math.radians(45), math.radians(-35))
    centre = p @ np.array([0, 0, -1.0, 1.0])
    assert centre[0] / centre[3] != 0 and centre[1] / centre[3] != 0               # the eye's axis is off-centre


def test_the_screen_is_placed_in_front_of_the_viewer_and_keeps_its_shape():
    s = g.Screen(aspect=2.0, distance=3.0, width=4.0, origin=(0.0, 1.6, 0.0), heading=0.0)
    assert np.allclose(s.centre(), [0.0, 1.6, -3.0])
    assert s.height == pytest.approx(2.0)
    corner = s.model_matrix() @ np.array([0.5, 0.5, 0.0, 1.0])
    assert np.allclose(corner[:3], [2.0, 2.6, -3.0])
    turned = g.Screen(aspect=2.0, distance=3.0, width=4.0, origin=(1.0, 1.5, 1.0), heading=math.pi / 2)
    assert np.allclose(turned.centre(), [-2.0, 1.5, 1.0], atol=1e-9)                # facing left of -z: out along -x


def test_resizing_and_moving_stay_within_sane_limits_and_keep_the_apparent_size():
    s = g.Screen(distance=3.0, width=3.2)
    angle = math.atan((s.width / 2) / s.distance)
    s.push(+3.0)
    assert math.atan((s.width / 2) / s.distance) == pytest.approx(angle)             # farther, but it looks the same size
    s.resize(2.0)
    assert math.atan((s.width / 2) / s.distance) > angle                              # resize: it looks bigger
    for _ in range(40):
        s.resize(1.5)
    assert s.width == g.Screen.MAX_WIDTH
    for _ in range(40):
        s.resize(0.5)
        s.push(-2.0)
    assert s.width >= g.Screen.MIN_WIDTH and s.distance == g.Screen.MIN_DISTANCE
    s.raise_by(10)
    assert s.lift == 2.5


def test_recentring_puts_the_screen_where_the_viewer_is_looking():
    s = g.Screen()
    ori = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))                                  # head turned 90 degrees left
    s.recentre((0.5, 1.4, 0.2), ori)
    assert np.allclose(s.centre(), [0.5 - s.distance, 1.4, 0.2], atol=1e-9)


def test_a_pointing_ray_finds_where_on_the_picture_it_lands():
    s = g.Screen(aspect=2.0, distance=3.0, width=4.0, origin=(0.0, 1.6, 0.0))
    hit = g.ray_hits_screen(s, (0, 1.6, 0), (0, 0, -1))
    assert hit is not None and hit[0] == pytest.approx(3.0) and hit[1] == pytest.approx(0.5) and hit[2] == pytest.approx(0.5)
    top_left = g.ray_hits_screen(s, (0, 1.6, 0), (-2.0, 1.0, -3.0))                    # towards the top-left corner
    assert top_left[1] == pytest.approx(0.0, abs=1e-9) and top_left[2] == pytest.approx(0.0, abs=1e-9)
    assert g.ray_hits_screen(s, (0, 1.6, 0), (-2.5, 0, -3.0)) is None                 # off the side
    assert g.ray_hits_screen(s, (0, 1.6, 0), (0, 0, 1)) is None                       # pointing away
    assert g.ray_hits_screen(s, (0, 1.6, 0), (1, 0, 0)) is None                       # parallel to it
