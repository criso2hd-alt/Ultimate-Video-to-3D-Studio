"""The VR renderer drawn into an ordinary off-screen framebuffer: what each eye sees, without a headset.

Needs an OpenGL context (any desktop GPU); the tests skip themselves where there is none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

glfw = pytest.importorskip("glfw")
GL = pytest.importorskip("OpenGL.GL")

from ultimate_video_3d.vr import geometry as geo  # noqa: E402
from ultimate_video_3d.vr import renderer as rend  # noqa: E402

W, H = 320, 200


@pytest.fixture(scope="module")
def gl():
    if not glfw.init():
        pytest.skip("no OpenGL available")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "uv3d-test", None, None)
    if not window:
        glfw.terminate()
        pytest.skip("could not create an OpenGL context")
    glfw.make_context_current(window)
    fbo = GL.glGenFramebuffers(1)
    colour = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, colour)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, W, H, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    depth = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, colour, 0)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth)
    r = rend.Renderer()
    r.init_gl()
    yield r
    glfw.destroy_window(window)
    glfw.terminate()


def _shot(r, frame, layout, eye, scene=rend.ROOM, screen=None):
    """Render one eye of a frame; returns an H x W x 3 uint8 image, top row first."""
    r.upload(frame)
    screen = screen or geo.Screen(aspect=16 / 9, distance=3.0, width=3.2, origin=(0.0, 1.6, 0.0))
    half = np.radians(45)
    proj = geo.projection_matrix(-half, half, half * H / W, -half * H / W)
    view = geo.view_matrix((0.0, 1.6, 0.0), (0, 0, 0, 1))
    GL.glViewport(0, 0, W, H)
    r.draw_eye(view, proj, eye, (0.0, 1.6, 0.0), screen, layout, scene)
    data = GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    return np.frombuffer(data, np.uint8).reshape(H, W, 3)[::-1]


def _sbs(red_left=(220, 30, 30), blue_right=(30, 30, 220), size=(64, 128)):
    frame = np.zeros((size[0], size[1], 3), np.uint8)
    frame[:, : size[1] // 2] = red_left
    frame[:, size[1] // 2:] = blue_right
    return frame


def test_the_left_eye_sees_the_left_half_and_the_right_eye_the_right_half(gl):
    frame = _sbs()
    left = _shot(gl, frame, "sbs_half", 0)[H // 2, W // 2]
    right = _shot(gl, frame, "sbs_half", 1)[H // 2, W // 2]
    assert left[0] > 150 and left[2] < 60                                      # red
    assert right[2] > 150 and right[0] < 60                                    # blue


def test_top_and_bottom_gives_the_top_to_the_left_eye(gl):
    frame = np.zeros((128, 64, 3), np.uint8)
    frame[:64] = (30, 220, 30)                                                  # top: green
    frame[64:] = (220, 220, 30)                                                 # bottom: yellow
    assert _shot(gl, frame, "tb_half", 0)[H // 2, W // 2][1] > 150
    right = _shot(gl, frame, "tb_half", 1)[H // 2, W // 2]
    assert right[0] > 150 and right[1] > 150


def test_a_flat_video_shows_the_same_picture_to_both_eyes(gl):
    frame = _sbs()
    a, b = _shot(gl, frame, "", 0), _shot(gl, frame, "", 1)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 4          # the same picture, bar the +-1 level dither noise


def test_the_picture_is_the_right_way_up_and_not_mirrored(gl):
    frame = np.zeros((64, 128, 3), np.uint8)
    frame[:32, :64] = (250, 20, 20)                                             # top-left red
    frame[:32, 64:] = (20, 250, 20)                                             # top-right green
    frame[32:, :64] = (20, 20, 250)                                             # bottom-left blue
    frame[32:, 64:] = (250, 250, 20)                                            # bottom-right yellow
    img = _shot(gl, frame, "", 0, scene=rend.VOID)
    cx, cy, dx, dy = W // 2, H // 2, 40, 25
    assert img[cy - dy, cx - dx][0] > 150 and img[cy - dy, cx - dx][1] < 80     # top-left: red
    assert img[cy - dy, cx + dx][1] > 150 and img[cy - dy, cx + dx][0] < 80     # top-right: green
    assert img[cy + dy, cx - dx][2] > 150                                       # bottom-left: blue
    assert img[cy + dy, cx + dx][0] > 150 and img[cy + dy, cx + dx][1] > 150    # bottom-right: yellow


def test_the_room_is_lit_by_the_pictures_colour(gl):
    red = np.full((64, 128, 3), (240, 20, 20), np.uint8)
    blue = np.full((64, 128, 3), (20, 20, 240), np.uint8)
    wall_red = _shot(gl, red, "", 0)[H // 2, 8]                                 # the left wall, near the edge of view
    wall_blue = _shot(gl, blue, "", 0)[H // 2, 8]
    assert wall_red[0] > wall_red[2] and wall_blue[2] > wall_blue[0]
    assert int(wall_red.sum()) > 12                                             # and it is lit at all, not black


def test_the_floor_mirrors_a_bright_picture(gl):
    white = np.full((64, 128, 3), 255, np.uint8)
    dark = np.full((64, 128, 3), 4, np.uint8)
    lit = _shot(gl, white, "", 0)[H - 20, W // 2].astype(int).sum()             # the floor below the screen
    unlit = _shot(gl, dark, "", 0)[H - 20, W // 2].astype(int).sum()
    assert lit > unlit + 60


def test_without_a_room_the_background_is_black(gl):
    img = _shot(gl, _sbs(), "sbs_half", 0, scene=rend.VOID)
    assert img[5, 5].astype(int).sum() < 20 and img[H - 5, W - 5].astype(int).sum() < 20


def test_pointer_beams_draw_in_their_colour(gl):
    gl.upload(_sbs())
    half = np.radians(45)
    proj = geo.projection_matrix(-half, half, half * H / W, -half * H / W)
    view = geo.view_matrix((0.0, 1.6, 0.0), (0, 0, 0, 1))
    GL.glViewport(0, 0, W, H)
    beam = ((0.0, 1.55, -0.3), (0.0, 0.0, -1.0), 1.5, (1.0, 1.0, 0.0, 1.0))
    gl.draw_eye(view, proj, 0, (0.0, 1.6, 0.0), geo.Screen(distance=3.0), "sbs_half", rend.VOID, (beam,))
    img = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
    yellow = (img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] < 80)
    assert yellow.sum() > 10
