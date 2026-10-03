"""The maths of the VR player, with no OpenGL and no OpenXR so it can be tested anywhere.

Conventions (the same as OpenXR's): metres; x right, y up, -z forward (into the scene). A *pose* is a position
plus a unit quaternion (x, y, z, w). Matrices are 4x4 NumPy arrays written the usual (row-major) way, and are
transposed when handed to OpenGL, which wants them column-major.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import library as lib

# --- how a video's frame maps to the two eyes -----------------------------------------------------


def eye_aspect(layout: str, width: int, height: int) -> float:
    """Width / height of *one eye's* picture, the shape the virtual screen must have.

    A half layout squeezes each eye to half the frame, so it is stretched back; a full layout (and a flat video)
    already holds each eye at its true shape.
    """
    if height <= 0 or width <= 0:
        return 16 / 9
    if layout == "sbs_full":
        return (width / 2) / height
    if layout == "tb_full":
        return width / (height / 2)
    return width / height                    # sbs_half, tb_half and flat: the frame's own shape


def eye_uv_rect(layout: str, eye: int) -> tuple[float, float, float, float]:
    """The part of the frame one eye sees, as (u0, v0, u1, v1) with the origin at the frame's top-left.

    ``eye`` is 0 for the left eye and 1 for the right. A flat (2D) or anaglyph video shows the whole frame to both.
    """
    if layout in lib.SIDE_BY_SIDE:
        return (0.5 * eye, 0.0, 0.5 * eye + 0.5, 1.0)
    if layout in lib.TOP_BOTTOM:
        return (0.0, 0.5 * eye, 1.0, 0.5 * eye + 0.5)
    return (0.0, 0.0, 1.0, 1.0)


# --- poses and matrices ---------------------------------------------------------------------------


def quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def pose_matrix(position, orientation) -> np.ndarray:
    """The 4x4 matrix that carries a thing from its own space into the world."""
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(tuple(orientation))
    m[:3, 3] = position
    return m


def view_matrix(position, orientation) -> np.ndarray:
    """The matrix that carries the world into an eye's space (the inverse of that eye's pose)."""
    m = np.eye(4)
    rot = quat_to_matrix(tuple(orientation))
    m[:3, :3] = rot.T
    m[:3, 3] = -rot.T @ np.asarray(position, np.float64)
    return m


def projection_matrix(angle_left: float, angle_right: float, angle_up: float, angle_down: float,
                      near: float = 0.05, far: float = 100.0) -> np.ndarray:
    """An OpenGL projection from OpenXR's field of view (four angles in radians; left and down are negative)."""
    left, right = math.tan(angle_left), math.tan(angle_right)
    up, down = math.tan(angle_up), math.tan(angle_down)
    w, h = right - left, up - down
    m = np.zeros((4, 4))
    m[0, 0], m[0, 2] = 2.0 / w, (right + left) / w
    m[1, 1], m[1, 2] = 2.0 / h, (up + down) / h
    m[2, 2], m[2, 3] = -(far + near) / (far - near), -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


def yaw_matrix(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def yaw_of(orientation) -> float:
    """The heading (rotation about the vertical axis) of a pose, in radians: 0 looks down -z, positive turns left."""
    fwd = quat_to_matrix(tuple(orientation)) @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-fwd[0], -fwd[2])


# --- the virtual screen ---------------------------------------------------------------------------


@dataclass
class Screen:
    """Where the virtual screen is, in the same world as the room.

    ``distance`` is how far in front of where the viewer was when it was placed (``origin`` + ``heading``),
    ``width`` is the picture's width in metres, and ``lift`` raises its centre above the viewer's eye height.
    Its height follows from the picture's shape (``aspect``), so it is never squashed.
    """

    aspect: float = 16 / 9
    distance: float = 3.0
    width: float = 3.2
    lift: float = 0.0
    curve: float = 0.0                       # 0 flat; > 0 bends the sides toward the viewer (radians of arc)
    #: In a room, the gap between the floor and the picture's lower edge (metres). The screen then sits on the room,
    #: not at the viewer's eye height, so it looks like a cinema screen whatever the person's height. None: eye height.
    floor_clearance: float | None = None
    #: A screen that is part of a room (a physical object): its unit-quad -> world matrix. Set, it cannot be moved or resized.
    fixed: np.ndarray | None = None
    #: With a fixed screen: the shape of the picture shown on it. A wider or narrower film is fitted inside the physical
    #: screen with black bars, never stretched. None = the picture fills the screen.
    picture_aspect: float | None = None
    origin: tuple[float, float, float] = (0.0, 1.6, 0.0)       # the viewer's head when it was placed
    heading: float = 0.0                     # which way the viewer faced then (yaw)

    MIN_WIDTH, MAX_WIDTH = 0.6, 14.0
    MIN_DISTANCE, MAX_DISTANCE = 0.8, 12.0

    @property
    def height(self) -> float:
        return self.width / max(0.2, self.aspect)

    @classmethod
    def from_quad(cls, matrix: np.ndarray, width: float, height: float) -> Screen:
        """The screen a room provides: placed by ``matrix`` (unit quad -> world, size included)."""
        return cls(aspect=width / max(height, 1e-6), width=width, fixed=np.array(matrix, np.float64))

    def centre(self) -> np.ndarray:
        if self.fixed is not None:
            return self.fixed[:3, 3].copy()
        forward = yaw_matrix(self.heading)[:3, :3] @ np.array([0.0, 0.0, -1.0])
        centre = np.asarray(self.origin, np.float64) + forward * self.distance
        centre[1] = (self.origin[1] + self.lift if self.floor_clearance is None
                     else self.floor_clearance + self.height / 2 + self.lift)
        return centre

    def model_matrix(self) -> np.ndarray:
        """Unit quad (-0.5..0.5 in x and y, facing +z) -> world: sized, turned to face the viewer, and placed."""
        if self.fixed is not None:
            return self.picture_matrix()
        m = yaw_matrix(self.heading)
        m[:3, 3] = self.centre()
        scale = np.diag([self.width, self.height, 1.0, 1.0])
        return m @ scale

    def physical_matrix(self) -> np.ndarray:
        """The whole screen (for a room: the black panel), whatever picture is on it."""
        return self.fixed if self.fixed is not None else self.model_matrix()

    def picture_matrix(self) -> np.ndarray:
        """Where the picture is drawn: the screen itself, or for a room screen the picture fitted inside it."""
        if self.fixed is None or not self.picture_aspect:
            return self.fixed if self.fixed is not None else self.model_matrix()
        screen_aspect = self.aspect
        sx, sy = (1.0, screen_aspect / self.picture_aspect) if self.picture_aspect > screen_aspect \
            else (self.picture_aspect / screen_aspect, 1.0)
        return self.fixed @ np.diag([sx, sy, 1.0, 1.0])

    def resize(self, factor: float) -> None:
        """Scale the picture about the viewer, keeping the angle it fills (so it looks bigger, not just nearer)."""
        if self.fixed is not None:
            return
        self.width = float(np.clip(self.width * factor, self.MIN_WIDTH, self.MAX_WIDTH))

    def push(self, delta: float) -> None:
        """Move it nearer (negative) or farther (positive), scaling with the distance so its size on the eye holds."""
        if self.fixed is not None:
            return
        new = float(np.clip(self.distance + delta, self.MIN_DISTANCE, self.MAX_DISTANCE))
        self.width = float(np.clip(self.width * new / self.distance, self.MIN_WIDTH, self.MAX_WIDTH))
        self.distance = new

    def raise_by(self, delta: float) -> None:
        if self.fixed is not None:
            return
        self.lift = float(np.clip(self.lift + delta, -1.5, 2.5))

    def recentre(self, head_position, head_orientation) -> None:
        """Put it straight in front of where the viewer is now looking, at their eye height."""
        if self.fixed is not None:
            return
        self.origin = tuple(float(v) for v in head_position)
        self.heading = yaw_of(head_orientation)


def ray_hits_screen(screen: Screen, origin, direction) -> tuple[float, float, float] | None:
    """Where a ray (a controller pointing) meets the screen: (distance, u, v) with u, v in 0..1 from the top left.

    None if it misses, points away, or is parallel. The screen is treated as flat; a mild curve changes the hit
    point by a few centimetres at the edges, which does not matter for pointing.
    """
    return ray_hits_matrix(screen.physical_matrix(), origin, direction)


def ray_hits_matrix(model: np.ndarray, origin, direction) -> tuple[float, float, float] | None:
    """Where a ray meets a unit quad placed by ``model`` (unit quad -> world): (metres, u, v), u and v from its top left."""
    inverse = np.linalg.inv(model)
    o = (inverse @ np.append(np.asarray(origin, np.float64), 1.0))[:3]
    d = (inverse @ np.append(np.asarray(direction, np.float64), 0.0))[:3]
    if abs(d[2]) < 1e-9:
        return None
    t = -o[2] / d[2]
    if t <= 0:
        return None
    x, y = o[0] + t * d[0], o[1] + t * d[1]
    if abs(x) > 0.5 or abs(y) > 0.5:
        return None
    # t is in model units; convert to metres along the ray (the direction was transformed with the scale)
    world = np.asarray(origin, np.float64) + np.asarray(direction, np.float64) * t
    metres = float(np.linalg.norm(world - np.asarray(origin, np.float64)))
    return metres, x + 0.5, 0.5 - y


def panel_matrix(head_position, forward, width: float, aspect: float, distance: float = 1.5, drop: float = 0.2) -> np.ndarray:
    """A floating panel in front of a person: ``distance`` ahead along ``forward`` (a direction; only its heading is
    used), ``drop`` below their eyes, turned to face them and tilted up toward their eyes. Unit quad -> world."""
    heading = math.atan2(-forward[0], -forward[2])
    tilt = math.atan2(drop, distance)
    c, s = math.cos(-tilt), math.sin(-tilt)
    pitch = np.eye(4)
    pitch[1, 1], pitch[1, 2], pitch[2, 1], pitch[2, 2] = c, -s, s, c
    m = yaw_matrix(heading) @ pitch
    centre = np.asarray(head_position, np.float64) + yaw_matrix(heading)[:3, :3] @ np.array([0.0, 0.0, -distance])
    centre[1] = head_position[1] - drop
    m[:3, 3] = centre
    return m @ np.diag([width, width / aspect, 1.0, 1.0])


# --- sitting in a room ----------------------------------------------------------------------------------


def translation(offset) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def rig_to_seat(seat_position, seat_yaw: float, head_position, head_orientation) -> np.ndarray:
    """The matrix that carries the headset's tracked space (the "stage") into the room so that the person's head lands
    exactly on a seat's eye position, facing the way the seat faces.

    The turn is about the head (so they are not swung around the room), and only about the vertical: the tilt of the
    head and the lean of the body stay theirs. Walking around within the real room moves them the same way in the
    virtual one.
    """
    turn = yaw_matrix(seat_yaw - yaw_of(head_orientation))
    return translation(seat_position) @ turn @ translation(-np.asarray(head_position, np.float64))


def through_rig(rig: np.ndarray, point) -> np.ndarray:
    return (rig @ np.append(np.asarray(point, np.float64), 1.0))[:3]


def direction_through_rig(rig: np.ndarray, direction) -> np.ndarray:
    return rig[:3, :3] @ np.asarray(direction, np.float64)
