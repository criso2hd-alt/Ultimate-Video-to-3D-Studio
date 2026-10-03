"""Small pieces of VR player state with no OpenXR or OpenGL in them, so they can be tested on their own."""

from __future__ import annotations


class LightFader:
    """The room lights: up when nothing is playing, down while the film runs, fading smoothly both ways.

    ``paused_level`` is how far they come up when the film is paused (the room's own setting; a pause is not an
    interval, so they only come up part way). With nothing loaded they are fully up.
    """

    def __init__(self, level: float = 1.0, seconds_for_full_fade: float = 1.4) -> None:
        self.level = level
        self._rate = 1.0 / max(seconds_for_full_fade, 0.1)

    def target(self, playing: bool, has_media: bool, paused_level: float) -> float:
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
