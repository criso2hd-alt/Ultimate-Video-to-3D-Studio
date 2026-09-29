"""``--selftest``: what this machine can actually do, in a form to paste into a bug report.

"The window opened" proves very little. This runs the real things - finds the
GPU, opens every encoder, loads the depth model, and times a depth + stereo
frame - and prints what happened, so most questions answer themselves.

    UltimateVideo3DStudio.exe --selftest 2> report.txt
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

from . import __version__, bootstrap, encoders, paths, stereo, sysinfo


def run_selftest() -> int:
    out = sys.stderr
    problems: list[str] = []

    def line(text: str = "") -> None:
        print(text, file=out, flush=True)

    line(f"Ultimate Video to 3D Studio {__version__}")
    line(f"{platform.platform()} · Python {platform.python_version()} · {platform.machine()}")
    line(f"data folder: {paths.data_dir()}")
    line()

    line("== GPUs ==")
    gpus = sysinfo.detect_gpus()
    if not gpus:
        line("none detected (software encoding and CPU depth will be used)")
    for g in gpus:
        line(f"{g.name}  [{g.vendor}]  {sysinfo.format_bytes(g.vram_bytes) if g.vram_bytes else 'unified memory'}")

    line()
    line("== Video component ==")
    bootstrap.activate_av()
    try:
        import av

        line(f"PyAV {av.__version__}")
    except Exception as error:  # noqa: BLE001
        line(f"PyAV unavailable: {error}")
        return 1

    line()
    line("== Encoders (each opened and fed a frame) ==")
    report = encoders.probe_all(gpus)
    for family in encoders.FAMILIES:
        working = [n for n in encoders.ladder(family, gpus) if report.results.get(n)]
        default = report.defaults.get(family.key, "—")
        line(f"{family.label:26} default: {default:16} working: {', '.join(working) or 'none'}")
        if not working:
            problems.append(f"no working encoder for {family.label}")

    line()
    line("== Depth ==")
    try:
        from . import depth

        engine = depth.DepthEngine()
        line(f"model: {depth.locate()}")
        line(f"running on: {engine.load()}")
        frame = np.random.default_rng(0).random((720, 1280, 3), dtype=np.float32)
        engine.infer(frame)
        began = time.perf_counter()
        runs = 10
        for _ in range(runs):
            raw = engine.infer(frame)
        line(f"depth model: {(time.perf_counter() - began) / runs * 1000:.1f} ms / frame")
        steady = depth.DepthStabilizer(0.5)(raw, frame)
        refined = depth.refine(steady, frame)
        stereo.synthesize(frame, refined, stereo.StereoParams())
        began = time.perf_counter()
        for _ in range(runs):
            stereo.synthesize(frame, refined, stereo.StereoParams())
        line(f"stereo warp (720p): {(time.perf_counter() - began) / runs * 1000:.1f} ms / frame")
    except Exception as error:  # noqa: BLE001
        line(f"FAILED: {type(error).__name__}: {error}")
        problems.append("depth model did not run")

    line()
    if problems:
        line("PROBLEMS:")
        for p in problems:
            line(f"  - {p}")
        return 1
    line("All checks passed.")
    return 0
