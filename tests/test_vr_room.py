"""A baked VR room, end to end: a tiny GLB built here, loaded by ``vr.room`` and drawn by the renderer off-screen.

The room has a floor, a back wall behind the screen, the screen quad, the LED-glow quad and one seat, with three
single-colour bakes (lit grey, dim near-black, screen-light white) so the mixing is easy to read from pixels.
Needs an OpenGL context; the GL tests skip themselves where there is none.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from ultimate_video_3d.vr import geometry as geo  # noqa: E402
from ultimate_video_3d.vr import room as roommod  # noqa: E402


def _quad(corners, normal):
    """Positions, UVs, indices for one quad given its four corners (bottom-left, bottom-right, top-right, top-left)."""
    positions = np.array(corners, np.float32)
    uv = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], np.float32)
    return positions, uv, np.array([0, 1, 2, 0, 2, 3], np.uint16), np.array([normal] * 4, np.float32)


def _make_room(folder: Path, lit=(128, 128, 128), dim=(20, 20, 20), screen=(255, 255, 255)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meshes = {
        "Floor": _quad([(-4, 0, 9), (4, 0, 9), (4, 0, 0), (-4, 0, 0)], (0, 1, 0)),
        "Wall_Front": _quad([(-4, 0, 0), (4, 0, 0), (4, 5, 0), (-4, 5, 0)], (0, 0, 1)),
        "Screen": _quad([(-2, 0.7, 0.04), (2, 0.7, 0.04), (2, 2.95, 0.04), (-2, 2.95, 0.04)], (0, 0, 1)),
        "BiasGlow": _quad([(-3, 0.1, 0.01), (3, 0.1, 0.01), (3, 4.0, 0.01), (-3, 4.0, 0.01)], (0, 0, 1)),
    }
    blob, views, accessors, mesh_defs, nodes = bytearray(), [], [], [], []

    def add(array: np.ndarray, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes, **({"target": target} if target else {})})
        blob.extend(array.tobytes())
        return len(views) - 1

    for name, (positions, uv, indices, normals) in meshes.items():
        pos_acc = len(accessors)
        accessors.append({"bufferView": add(positions, 34962), "componentType": 5126, "count": 4, "type": "VEC3",
                          "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()})
        accessors.append({"bufferView": add(normals, 34962), "componentType": 5126, "count": 4, "type": "VEC3"})
        accessors.append({"bufferView": add(uv, 34962), "componentType": 5126, "count": 4, "type": "VEC2"})
        accessors.append({"bufferView": add(indices, 34963), "componentType": 5123, "count": 6, "type": "SCALAR"})
        material = None if name in ("Screen", "BiasGlow") else 0
        prim = {"attributes": {"POSITION": pos_acc, "NORMAL": pos_acc + 1, "TEXCOORD_0": pos_acc + 2}, "indices": pos_acc + 3}
        if material is not None:
            prim["material"] = material
        mesh_defs.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(mesh_defs) - 1})
    nodes.append({"name": "Seat_Centre", "translation": [0, 1.2, 6.0], "extras": {"label": "Centre"}})

    ok, png = cv2.imencode(".png", np.full((8, 8, 3), lit[::-1], np.uint8))                 # the lit bake, embedded
    image_view = add(np.frombuffer(png.tobytes(), np.uint8))
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
                "meshes": mesh_defs, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
                "images": [{"bufferView": image_view, "mimeType": "image/png"}], "textures": [{"source": 0}],
                "materials": [{"name": "Bake_all", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    while len(blob) % 4:
        blob.append(0)
    glb = (b"glTF" + struct.pack("<II", 2, 12 + 8 + len(payload) + 8 + len(blob))
           + struct.pack("<I4s", len(payload), b"JSON") + payload + struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    (folder / "room.glb").write_bytes(glb)
    cv2.imwrite(str(folder / "all_dim.png"), np.full((8, 8, 3), dim[::-1], np.uint8))
    cv2.imwrite(str(folder / "all_screen.png"), np.full((8, 8, 3), screen[::-1], np.uint8))
    (folder / "room.json").write_text(json.dumps({
        "name": "Test room", "default_seat": "Seat_Centre", "light_level_paused": 0.5,
        "groups": {"all": {"material": "Bake_all", "lit": "all_lit.png", "dim": "all_dim.png", "screen": "all_screen.png",
                           "screen_ev": 0.0}}}))
    return folder


# --- loading ---------------------------------------------------------------------------------------------


def test_a_room_loads_with_its_groups_seats_screen_and_glow(tmp_path):
    room = roommod.load(_make_room(tmp_path / "r"))
    assert room.name == "Test room" and len(room.groups) == 1 and room.groups[0].name == "all"
    group = room.groups[0]
    assert group.triangles == 4 and group.lit.shape[2] == 3 and group.dim is not None and group.screen is not None
    assert room.first_seat().name == "Seat_Centre" and room.first_seat().label == "Centre"
    assert room.screen.width == pytest.approx(4.0) and room.screen.height == pytest.approx(2.25)
    assert room.screen.normal.tolist() == pytest.approx([0, 0, 1]) and room.screen.centre[1] == pytest.approx(1.825)
    assert room.glow is not None and room.glow.positions.shape == (4, 3)


def test_a_missing_or_broken_room_says_what_is_wrong(tmp_path):
    with pytest.raises(roommod.RoomError, match="no room.glb"):
        roommod.load(tmp_path)
    (tmp_path / "room.glb").write_bytes(b"nope")
    with pytest.raises(roommod.RoomError):
        roommod.load(tmp_path)
    folder = _make_room(tmp_path / "r2")
    (folder / "room.json").write_text("{not json")
    with pytest.raises(roommod.RoomError, match="not valid JSON"):
        roommod.load(folder)


def test_a_room_without_the_extra_bakes_still_loads(tmp_path):
    folder = _make_room(tmp_path / "r3")
    (folder / "all_dim.png").unlink()
    (folder / "all_screen.png").unlink()
    group = roommod.load(folder).groups[0]
    assert group.dim is None and group.screen is None                       # the renderer falls back to lit / no glow


# --- drawing ---------------------------------------------------------------------------------------------

glfw = pytest.importorskip("glfw")
GL = pytest.importorskip("OpenGL.GL")
from ultimate_video_3d.vr import renderer as rend  # noqa: E402

W, H = 320, 200


@pytest.fixture(scope="module")
def gl():
    if not glfw.init():
        pytest.skip("no OpenGL available")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "uv3d-room-test", None, None)
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
    yield
    glfw.destroy_window(window)
    glfw.terminate()


def _shot(room, picture, light):
    """Draw the room from its seat, looking at the screen, with this picture on it and these room lights."""
    r = rend.Renderer()
    r.init_gl()
    r.set_room(room)
    r.room_light = light
    r.upload(picture)
    r.gain = 1.0
    seat = room.first_seat()
    half = np.radians(45)
    proj = geo.projection_matrix(-half, half, half * H / W, -half * H / W)
    view = geo.view_matrix(tuple(seat.position), (0, 0, 0, 1))
    screen = geo.Screen.from_quad(room.screen.matrix, room.screen.width, room.screen.height)
    GL.glViewport(0, 0, W, H)
    r.draw_eye(view, proj, 0, tuple(seat.position), screen, "", rend.BAKED)
    pixels = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
    return pixels.astype(int)


@pytest.fixture()
def room(tmp_path):
    return roommod.load(_make_room(tmp_path / "gl"))


def test_the_picture_appears_on_the_screen(gl, room):
    red = np.zeros((90, 160, 3), np.uint8)
    red[...] = (230, 30, 30)
    img = _shot(room, red, light=0.0)
    centre = img[H // 2, W // 2]
    assert centre[0] > 180 and centre[1] < 70 and centre[2] < 70


def test_lights_up_brightens_the_room_and_lights_down_darkens_it(gl, room):
    black = np.zeros((90, 160, 3), np.uint8)
    up = _shot(room, black, light=1.0)
    down = _shot(room, black, light=0.0)
    floor = (H - 12, W // 2)
    assert up[floor].sum() > down[floor].sum() + 60                           # the lit bake against the dim one
    assert down[floor].sum() < 120                                            # film lighting is dark


def test_the_screens_light_takes_the_pictures_colour_when_the_room_is_dark(gl, room):
    red = np.full((90, 160, 3), (230, 20, 20), np.uint8)
    blue = np.full((90, 160, 3), (20, 20, 230), np.uint8)
    # the back wall at x = -3.5 m, y = 3.5 m (outside the glow quad, so only the screen's own light reaches it)
    wall_red = _shot(room, red, light=0.0)[39, 67]
    wall_blue = _shot(room, blue, light=0.0)[39, 67]
    assert wall_red[0] > wall_red[2] and wall_blue[2] > wall_blue[0]


def test_the_glow_behind_the_screen_follows_the_nearest_edge_colour(gl, room):
    picture = np.zeros((90, 160, 3), np.uint8)
    picture[:, :16] = (230, 20, 20)                                           # the left edge is red
    picture[:, -16:] = (20, 20, 230)                                          # the right edge is blue
    img = _shot(room, picture, light=0.0)
    left_of_screen, right_of_screen = img[84, 103], img[84, 217]               # the wall at x = -2.15 m and +2.15 m
    assert left_of_screen[0] > left_of_screen[2]                              # the wall left of the screen glows red
    assert right_of_screen[2] > right_of_screen[0]                            # and right of it blue
