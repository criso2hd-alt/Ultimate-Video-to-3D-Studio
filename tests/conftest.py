"""Test-wide setup: run Qt offscreen, and never pop the first-run welcome dialog over a test."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("UV3D_SKIP_TOUR", "1")

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _vr_diary_in_a_temp_folder(monkeypatch):
    """The VR diary (vr.log) is real user data: tests must write their own, not into the person's."""
    from ultimate_video_3d.vr import log

    monkeypatch.setattr(log, "_path", lambda: Path(tempfile.gettempdir()) / "uv3d-test-vr.log")


@pytest.fixture(autouse=True)
def _fresh_shared_depth_engine(monkeypatch):
    """Conversions use the one resident depth engine (``depth.shared_engine``); each test gets its own, so a stand-in
    a test patches in is what that test really runs (not an engine some earlier test left loaded)."""
    from ultimate_video_3d import depth

    monkeypatch.setattr(depth, "_shared", None)
