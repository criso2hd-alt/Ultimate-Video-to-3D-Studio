"""Small pieces of VR player state with no OpenXR or OpenGL in them, so they can be tested on their own."""

from __future__ import annotations

import math


class LightFader:
    """The room lights: up when nothing is playing, down while the film runs, fading smoothly both ways.

    ``paused_level`` is how far they come up when the film is paused (the room's own setting; a pause is not an
    interval, so they only come up part way). With nothing loaded they are fully up.
    """

    def __init__(self, level: float = 1.0, seconds_for_full_fade: float = 1.4) -> None:
        self.level = level
        self._rate = 1.0 / max(seconds_for_full_fade, 0.1)
        #: "auto" follows the film; "on" and "off" are the person's own choice and hold until they change it.
        self.mode = "auto"

    def target(self, playing: bool, has_media: bool, paused_level: float) -> float:
        if self.mode == "on":
            return 1.0
        if self.mode == "off":
            return 0.0
        if not has_media:
            return 1.0
        return 0.0 if playing else paused_level

    def update(self, dt: float, playing: bool, has_media: bool, paused_level: float = 0.6) -> float:
        goal = self.target(playing, has_media, paused_level)
        step = max(0.0, dt) * self._rate
        if self.level < goal:
            self.level = min(goal, self.level + step)
        else:
            self.level = max(goal, self.level - step)
        return self.level


def pick_seat(seats: list[tuple], current: str | None, push: tuple[float, float], min_cos: float = 0.5) -> str | None:
    """The seat a stick push points at: ``push`` is (right, forward) as seen by someone sitting in ``current`` facing the way
    that seat faces (forward = toward the screen). Of the seats within 60 degrees of that direction, the nearest wins, so
    pushing forward from the back row goes to the seat in front of you, right goes to the one on your right, back to the one
    behind. None if there is no seat that way (the person stays where they are).

    ``seats`` is a list of (name, position, yaw); only x and z of a position matter.
    """
    me = next((s for s in seats if s[0] == current), None)
    if me is None or math.hypot(*push) < 1e-6:
        return None
    _, origin, yaw = me
    s, c = math.sin(yaw), math.cos(yaw)
    forward, right = (-s, -c), (c, -s)                    # in (x, z): yaw 0 looks down -z
    d = (push[0] * right[0] + push[1] * forward[0], push[0] * right[1] + push[1] * forward[1])
    d_len = math.hypot(*d)
    best = None
    for name, position, _ in seats:
        if name == current:
            continue
        ox, oz = position[0] - origin[0], position[2] - origin[2]
        dist = math.hypot(ox, oz)
        if dist < 1e-6:
            continue
        cos = (ox * d[0] + oz * d[1]) / (dist * d_len)
        if cos >= min_cos:
            key = (round(dist, 2), -cos)
            if best is None or key < best[0]:
                best = (key, name)
    return best[1] if best else None


class SeatCycle:
    """Which seat the person is in, and stepping to the next or previous one."""

    def __init__(self, names: list[str], current: str | None = None) -> None:
        self.names = list(names)
        self.index = self.names.index(current) if current in self.names else 0

    @property
    def current(self) -> str | None:
        return self.names[self.index] if self.names else None

    def select(self, name: str) -> bool:
        if name in self.names:
            self.index = self.names.index(name)
            return True
        return False

    def step(self, direction: int) -> str | None:
        """Move along the seats, wrapping round at either end."""
        if not self.names:
            return None
        self.index = (self.index + (1 if direction >= 0 else -1)) % len(self.names)
        return self.current
