"""Compile the Numba kernels in the background so the first video does not stall.

Numba compiles a kernel the first time it is called (a few seconds each) and
caches the machine code on disk. Doing that on a tiny throwaway frame while the
window is still being looked at means the first real preview or export finds
everything ready - and on every later launch it is a fast cache load.

The depth model gets the same treatment: it is opened, and run once, at launch and then
stays resident for the whole session (``depth.shared_engine`` hands every later caller the
same session). Otherwise the first time 3D is used - or the first Play after choosing it -
pays about a second to load it plus a slow first inference.
"""

from __future__ import annotations

import threading

import numpy as np


def _run(force_cpu: bool = False) -> None:
    _warm_kernels()
    _warm_model(force_cpu)


def _warm_model(force_cpu: bool) -> None:
    try:
        from . import depth

        engine = depth.shared_engine(force_cpu)
        engine.infer(np.zeros((64, 64, 3), np.float32))       # the first real call is slower than the rest
    except Exception:  # noqa: BLE001 - a missing model is reported when 3D is first used, not at launch
        pass


def _warm_kernels() -> None:
    try:
        from . import depth, grade, hdr, stereo
        from .kernels import LOCK

        rgb = np.random.default_rng(0).random((48, 64, 3), dtype=np.float32)
        d = np.random.default_rng(1).random((48, 64), dtype=np.float32)
        pair = stereo.synthesize(rgb, d, stereo.StereoParams())
        for bits in (8, 16):                       # each integer width is its own compiled version
            for layout in stereo.LAYOUTS_BY_KEY:
                stereo.pack_frame(layout, pair, rgb, d, bits)
            stereo.pack_frame("copy", None, rgb, None, bits)
        engine = grade.GradeEngine()
        engine.apply(rgb, grade.ColourSettings(enabled=True, exposure=0.1, vignette=10, sharpen=5))
        cube = np.zeros((2, 2, 2, 3), np.float32)
        with LOCK:
            grade._grade_kernel(rgb, np.empty_like(rgb), grade.tone_tables(grade.ColourSettings(), grade.SDR),
                                grade._params(grade.ColourSettings(), 64, 48, True), cube,
                                np.zeros(3, np.float32), np.ones(3, np.float32), False)
        hdr.tonemap_to_sdr(rgb, "pq")
        hdr.tonemap_to_sdr(rgb, "hlg")
        depth._to_nchw(np.zeros((8, 8, 3), np.float32), np.empty((1, 3, 8, 8), np.float32),
                       depth._MEAN1, depth._INV_STD1)
    except Exception:  # noqa: BLE001 - warm-up is an optimisation, never a requirement
        pass


def start(force_cpu: bool = False) -> threading.Thread:
    thread = threading.Thread(target=_run, args=(force_cpu,), daemon=True, name="uv3d-warmup")
    thread.start()
    return thread
