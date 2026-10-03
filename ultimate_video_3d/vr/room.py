"""A baked VR room: the geometry and the three lighting bakes the Blender export produced, ready for the renderer.

A room is a folder with ``room.glb`` and ``room.json``, and per baked group of objects three pictures that share one UV
layout: ``<group>_lit.jpg`` (embedded in the GLB as its base colour), ``<group>_dim.jpg`` (the lights as during a film)
and ``<group>_screen.jpg`` (only the screen lighting the room; stored brightened, see ``screen_ev``). The renderer mixes
them live: lights up or dimmed, plus the screen's glow in whatever colour the picture is. See ``docs/vr-room-spec.md``.

Nothing here touches OpenGL, so a room can be loaded and checked anywhere.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gltf


class RoomError(Exception):
    """The folder is not a usable room (the message says what is missing)."""


@dataclass
class Group:
    """One baked set of objects: world-space triangles and the three pictures that colour them."""

    name: str
    positions: np.ndarray                    # (N, 3) float32, already in world space
    uv: np.ndarray                           # (N, 2) float32, (0, 0) = top-left of the atlas
    indices: np.ndarray                      # (M,) uint32
    lit: np.ndarray                          # uint8 (H, W, 3)
    dim: np.ndarray | None = None            # same layout as lit; None -> the renderer uses lit
    screen: np.ndarray | None = None         # screen-only light; None -> no screen glow on this group
    screen_ev: float = 0.0                   # exposure it was brightened by when stored (linear * 2**ev)

    @property
    def triangles(self) -> int:
        return len(self.indices) // 3


@dataclass
class Seat:
    name: str
    label: str
    position: np.ndarray                     # eye position in the room
    yaw: float                               # heading in radians; 0 looks down -z (at the screen wall)


@dataclass
class ScreenQuad:
    """The room's screen: where it is, how big, and which way it faces."""

    matrix: np.ndarray                       # unit quad (-0.5..0.5, facing +z) -> world, size included
    centre: np.ndarray
    width: float
    height: float
    normal: np.ndarray

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1e-6)


@dataclass
class GlowQuad:
    """The halo behind the screen: the wall area the app lights like an LED backlight."""

    positions: np.ndarray                    # (4, 3) world
    indices: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        """Which way the surface faces (unit length), from its first triangle."""
        p = self.positions[self.indices[:3]].astype(np.float64)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(n)
        return n / length if length else np.array([0.0, 0.0, 1.0])


@dataclass
class Room:
    name: str
    folder: Path
    groups: list[Group] = field(default_factory=list)
    seats: list[Seat] = field(default_factory=list)
    spawn: Seat | None = None
    screen: ScreenQuad | None = None
    glow: GlowQuad | None = None             # the plane on the screen wall (named exactly BiasGlow)
    glows: list[GlowQuad] = field(default_factory=list)      # every BiasGlow* plane: that one, plus floor, top and sides if the owner made them
    default_seat: str = ""
    light_level_paused: float = 0.6
    screen_emission: float = 8.0

    @property
    def triangles(self) -> int:
        return sum(g.triangles for g in self.groups)

    def seat(self, name: str) -> Seat | None:
        return next((s for s in self.seats if s.name == name), None)

    def first_seat(self) -> Seat | None:
        return self.seat(self.default_seat) or (self.seats[0] if self.seats else self.spawn)


def _world_points(node: gltf.Node, positions: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[positions.astype(np.float64), np.ones(len(positions))]
    return (node.matrix @ homogeneous.T).T[:, :3].astype(np.float32)


def _seat_of(node: gltf.Node) -> Seat:
    forward = node.matrix[:3, :3] @ np.array([0.0, 0.0, -1.0])
    return Seat(node.name, str(node.extras.get("label") or node.name.replace("Seat_", "").replace("_", " ")),
                node.position.copy(), math.atan2(-forward[0], -forward[2]))


def _screen_of(node: gltf.Node, mesh: gltf.Mesh) -> ScreenQuad:
    prim = mesh.primitives[0]
    points = _world_points(node, prim.positions).astype(np.float64)
    normal = node.matrix[:3, :3] @ (prim.normals[0].astype(np.float64) if prim.normals is not None else [0, 0, 1.0])
    normal /= np.linalg.norm(normal) or 1.0
    up = np.array([0.0, 1.0, 0.0]) - normal * normal[1]                    # the world's up, made perpendicular to it
    up /= np.linalg.norm(up) or 1.0
    right = np.cross(up, normal)
    centre = points.mean(axis=0)
    width = float(np.ptp(points @ right))
    height = float(np.ptp(points @ up))
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right * width, up * height, normal, centre
    return ScreenQuad(matrix, centre, width, height, normal)


def load(folder: str | Path) -> Room:
    """Read a room folder. Raises :class:`RoomError` with a plain message if it is unusable."""
    import cv2

    folder = Path(folder)
    glb = folder / "room.glb"
    if not glb.is_file():
        raise RoomError(f"{folder} has no room.glb")
    manifest = {}
    if (folder / "room.json").is_file():
        try:
            manifest = json.loads((folder / "room.json").read_text(encoding="utf8"))
        except ValueError as error:
            raise RoomError(f"room.json is not valid JSON: {error}") from error
    try:
        scene = gltf.read(glb)
    except gltf.GltfError as error:
        raise RoomError(str(error)) from error

    by_material = {spec["material"]: (name, spec) for name, spec in manifest.get("groups", {}).items()}

    def picture(file: str | None):
        if not file or not (folder / file).is_file():
            return None
        data = cv2.imread(str(folder / file), cv2.IMREAD_COLOR)
        return None if data is None else cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    room = Room(manifest.get("name", folder.name), folder, default_seat=manifest.get("default_seat", ""),
                light_level_paused=float(manifest.get("light_level_paused", 0.6)),
                screen_emission=float(manifest.get("screen_emission", 8.0)))
    pieces: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for node in scene.nodes:
        mesh = scene.mesh_of(node)
        if mesh is None:
            continue
        if node.name == "Screen":
            room.screen = _screen_of(node, mesh)
            continue
        if node.name.startswith("BiasGlow"):
            prim = mesh.primitives[0]
            quad = GlowQuad(_world_points(node, prim.positions), prim.indices)
            room.glows.append(quad)
            if node.name == "BiasGlow" or room.glow is None:
                room.glow = quad
            continue
        for prim in mesh.primitives:
            if prim.material is None or prim.uv is None:
                continue
            material = scene.materials[prim.material]
            if material.name in by_material:
                pieces.setdefault(material.name, []).append((_world_points(node, prim.positions), prim.uv, prim.indices))

    for material_name, parts in pieces.items():
        name, spec = by_material[material_name]
        lit_index = scene.materials[[m.name for m in scene.materials].index(material_name)].base_colour_image
        lit = scene.images[lit_index] if lit_index is not None else picture(spec.get("lit"))
        if lit is None:
            raise RoomError(f"group {name} has no lit picture")
        offset, positions, uvs, indices = 0, [], [], []
        for p, uv, idx in parts:
            positions.append(p)
            uvs.append(uv)
            indices.append(idx + offset)
            offset += len(p)
        ev = spec.get("screen_ev")
        room.groups.append(Group(
            name, np.concatenate(positions), np.concatenate(uvs).astype(np.float32), np.concatenate(indices).astype(np.uint32),
            lit, picture(spec.get("dim")), picture(spec.get("screen")), float(ev) if ev is not None else 0.0))

    seats = [_seat_of(n) for n in scene.nodes_named("Seat_")]
    room.seats = sorted(seats, key=lambda s: s.name)
    spawn = scene.node("Spawn")
    room.spawn = _seat_of(spawn) if spawn is not None else None
    if room.screen is None:
        raise RoomError("the room has no object named Screen")
    if not room.groups:
        raise RoomError("the room has no baked geometry (no material matches a group in room.json)")
    return room
