"""The small pieces of VR state: the room lights fading, and stepping between seats."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d.vr.state import LightFader, SeatCycle  # noqa: E402


def test_lights_fade_down_when_the_film_starts_and_up_when_it_pauses():
    lights = LightFader(1.0, seconds_for_full_fade=1.0)
    assert lights.update(0.5, playing=True, has_media=True) == pytest.approx(0.5)          # halfway down after half a second
    assert lights.update(0.7, playing=True, has_media=True) == pytest.approx(0.0)          # and no lower than off
    assert lights.update(0.3, playing=False, has_media=True, paused_level=0.6) == pytest.approx(0.3)
    assert lights.update(5.0, playing=False, has_media=True, paused_level=0.6) == pytest.approx(0.6)   # a pause only half-ups them
    assert lights.update(5.0, playing=False, has_media=False) == pytest.approx(1.0)        # nothing loaded: fully up


def test_a_long_frame_cannot_make_the_lights_jump_past_their_goal():
    lights = LightFader(0.0, seconds_for_full_fade=1.0)
    assert lights.update(30.0, playing=False, has_media=True, paused_level=0.6) == pytest.approx(0.6)
    assert lights.update(-1.0, playing=True, has_media=True) == pytest.approx(0.6)          # a negative time step changes nothing


def test_seats_step_round_in_both_directions():
    seats = SeatCycle(["A", "B", "C"], current="B")
    assert seats.current == "B" and seats.step(+1) == "C" and seats.step(+1) == "A" and seats.step(-1) == "C"
    assert seats.select("A") and seats.current == "A" and not seats.select("nope") and seats.current == "A"
    assert SeatCycle([]).step(+1) is None and SeatCycle([]).current is None
    assert SeatCycle(["X", "Y"], current="missing").current == "X"
