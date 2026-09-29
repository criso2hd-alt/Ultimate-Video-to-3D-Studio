"""Measure live-preview playback in the real window: frames per second, and UI-thread stalls.

    python tests/perf_preview.py <video> [seconds]

For each configuration (plain 2D, then each 3D view) it plays the clip and reports
how many preview pictures reached the screen per second, and the worst gap the UI
thread saw between two 10 ms timer ticks - a stall there is what "the whole thing
is unresponsive" feels like.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("UV3D_SKIP_TOUR", "1")            # a fresh settings file would otherwise open the welcome dialog
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402


def main() -> None:
    video = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    qt = QApplication(sys.argv)
    appmod._load_bundled_fonts(qt)
    appmod.apply_app_theme(qt, appmod.DEFAULT_THEME)
    win = appmod.MainWindow()
    win.resize(1500, 900)
    win.show()

    state = {"frames": 0, "last_tick": time.perf_counter(), "worst": 0.0, "gaps": []}

    def on_rendered(_r) -> None:
        state["frames"] += 1

    win.engine.rendered.connect(on_rendered)

    def tick() -> None:
        now = time.perf_counter()
        gap = (now - state["last_tick"]) * 1000
        state["last_tick"] = now
        state["worst"] = max(state["worst"], gap)
        state["gaps"].append(gap)

    ticker = QTimer()
    ticker.setInterval(10)
    ticker.timeout.connect(tick)
    ticker.start()

    every = [("2D (original video)", None), ("3D: anaglyph", "anaglyph"), ("3D: output", "output"),
             ("3D: side by side", "sbs"), ("3D: wiggle", "wiggle"), ("3D: depth", "depth"),
             ("3D: parallax", "parallax"), ("3D: holes", "holes"), ("3D: orbit", "orbit")]
    wanted = sys.argv[3:]
    configs = [c for c in every if not wanted or (c[1] or "2d") in wanted]
    steps = []

    def configure(mode):
        def go():
            win.player.pause()
            if mode is None:
                win.settings.preview_3d = False
            else:
                win.settings.preview_mode = mode
                win._enable_preview_3d()
            win._sync_mode_buttons()
            win._submit(False)
            win.player.seek_ms(2000, True)
            QTimer.singleShot(1200, win.player.play)
        return go

    def begin(name):
        def go():
            state.update(frames=0, worst=0.0, gaps=[])
            state["t0"] = time.perf_counter()
            state["name"] = name
        return go

    def end():
        dt = time.perf_counter() - state["t0"]
        gaps = sorted(state["gaps"])
        p95 = gaps[int(len(gaps) * 0.95)] if gaps else 0
        print(f"{state['name']:22} {state['frames'] / dt:5.1f} fps   UI worst stall {state['worst']:5.0f} ms   "
              f"p95 tick {p95:4.0f} ms", flush=True)

    t = 2500
    QTimer.singleShot(800, lambda: win._open_path(video))
    for name, mode in configs:
        QTimer.singleShot(t, configure(mode))
        QTimer.singleShot(t + 2600, begin(name))
        QTimer.singleShot(t + 2600 + int(seconds * 1000), end)
        t += 2600 + int(seconds * 1000) + 300
    QTimer.singleShot(t, qt.quit)
    qt.exec()


if __name__ == "__main__":
    main()
