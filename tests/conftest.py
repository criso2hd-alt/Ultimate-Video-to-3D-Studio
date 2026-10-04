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
