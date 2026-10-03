"""The glTF reader, on the clean template room that is committed in the repository (no third-party content)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("cv2")

from ultimate_video_3d.vr import gltf  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "room-template" / "room_template.glb"


@pytest.fixture(scope="module")
def room():
    return gltf.read(TEMPLATE)


def test_the_named_objects_the_app_looks_for_are_all_there(room):
    names = {n.name for n in room.nodes}
    assert {"Screen", "BiasGlow", "Spawn", "Floor", "Ceiling", "Wall_Front"} <= names
    assert {n.name for n in room.nodes_named("Seat_")} >= {"Seat_Centre", "Seat_Left", "Seat_Right"}


def test_seats_carry_their_labels_and_stand_at_eye_height_behind_the_screen(room):
    centre = room.node("Seat_Centre")
    assert centre.extras["label"] == "Centre"
    x, y, z = centre.position
    assert (x, y) == pytest.approx((0.0, 1.15), abs=1e-4) and z == pytest.approx(4.2, abs=1e-4)   # +z is toward the viewers
    assert room.node("Seat_Left").position[0] < 0 < room.node("Seat_Right").position[0]


def test_the_screen_is_a_120_inch_16_by_9_quad_facing_the_seats(room):
    screen = room.node("Screen")
    mesh = room.mesh_of(screen)
    prim = mesh.primitives[0]
    points = (screen.matrix @ np.c_[prim.positions, np.ones(len(prim.positions))].T).T[:, :3]
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    assert width == pytest.approx(2.656, abs=0.01) and height == pytest.approx(1.494, abs=0.01)
    assert width / height == pytest.approx(16 / 9, abs=0.01)
    assert prim.normals is not None and prim.normals[0][2] > 0.99              # the normal points to +z: toward the seats
    assert prim.uv is not None and prim.uv.min() >= 0 and prim.uv.max() <= 1


def test_meshes_come_out_as_triangle_lists_with_matching_array_sizes(room):
    for node in room.nodes:
        mesh = room.mesh_of(node)
        if mesh is None:
            continue
        for prim in mesh.primitives:
            assert prim.indices.size % 3 == 0 and prim.indices.max() < len(prim.positions)
            assert prim.normals is None or len(prim.normals) == len(prim.positions)
            assert prim.uv is None or len(prim.uv) == len(prim.positions)


def test_a_file_that_is_not_a_glb_is_refused_plainly(tmp_path):
    bad = tmp_path / "x.glb"
    bad.write_bytes(b"not a glb at all, just bytes")
    with pytest.raises(gltf.GltfError):
        gltf.read(bad)
    wrong_version = tmp_path / "v1.glb"
    wrong_version.write_bytes(b"glTF" + struct.pack("<II", 1, 20) + b"\0" * 8)
    with pytest.raises(gltf.GltfError, match="version"):
        gltf.read(wrong_version)


def test_node_transforms_compose_down_the_tree(tmp_path):
    """A child moved 1 m forward of a parent turned 90 degrees about the vertical lands where the maths says."""
    import json

    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"name": "parent", "translation": [1, 0, 0], "rotation": [0, 0.70710678, 0, 0.70710678], "children": [1]},
                          {"name": "child", "translation": [0, 0, -1], "extras": {"label": "x"}}]}
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    glb = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(payload)) + struct.pack("<I4s", len(payload), b"JSON") + payload
    path = tmp_path / "tree.glb"
    path.write_bytes(glb)
    tree = gltf.read(path)
    child = tree.node("child")
    # turning +90 degrees about Y takes -z to -x, so the child sits at (1 - 1, 0, 0) = (0, 0, 0)... from the parent at x = 1
    assert np.allclose(child.position, [0.0, 0.0, 0.0], atol=1e-6) and child.extras == {"label": "x"} and child.parent == 0
