"""Test-wide setup: run Qt offscreen, and never pop the first-run welcome dialog over a test."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("UV3D_SKIP_TOUR", "1")
