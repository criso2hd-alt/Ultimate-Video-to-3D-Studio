"""Exercise the look-ahead buffer in the real window and report what the viewer would see.

    python tests/perf_buffer.py <video>

Scenarios, in order: press Play with the 3D preview on (start-up buffering, then
smooth playback), change a setting mid-play (the buffer should refill and resume),
pause and resume, and seek. Prints displayed frames per second, time spent waiting
on the buffer, the worst gap between two displayed pictures, and UI-thread stalls.
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
    qt = QApplication(sys.argv)
    appmod._load_bundled_fonts(qt)
    appmod.apply_app_theme(qt, appmod.DEFAULT_THEME)
    win = appmod.MainWindow()
    win.resize(1500, 900)
    win.show()

    t0 = time.perf_counter()
    log = {"shown": [], "wait": 0.0, "waiting_since": None, "ui": [], "last_tick": time.perf_counter()}

    def stamp() -> str:
        return f"[{time.perf_counter() - t0:5.1f}s]"

    def on_image(_p) -> None:
        log["shown"].append(time.perf_counter())

    def on_buffering(waiting: bool, progress: float) -> None:
        now = time.perf_counter()
        if waiting and log["waiting_since"] is None:
            log["waiting_since"] = now
            print(f"{stamp()} buffering...", flush=True)
        elif not waiting and log["waiting_since"] is not None:
            log["wait"] += now - log["waiting_since"]
            print(f"{stamp()} resumed after {now - log['waiting_since']:.2f}s", flush=True)
            log["waiting_since"] = None

    win.player.buffered_image.connect(on_image)
    win.player.buffering_changed.connect(on_buffering)

    def tick() -> None:
        now = time.perf_counter()
        log["ui"].append((now - log["last_tick"]) * 1000)
        log["last_tick"] = now

    ticker = QTimer()
    ticker.setInterval(10)
    ticker.timeout.connect(tick)
    ticker.start()

    def window_report(label: str, seconds: float) -> None:
        cutoff = time.perf_counter() - seconds
        shown = [t for t in log["shown"] if t >= cutoff]
        gaps = sorted((b - a) * 1000 for a, b in zip(shown, shown[1:]))
        worst = gaps[-1] if gaps else 0
        ui = sorted(log["ui"][-int(seconds * 100):] or [0])
        print(f"{stamp()} {label:34} {len(shown) / seconds:5.1f} fps shown | worst gap between pictures "
              f"{worst:4.0f} ms | UI worst stall {ui[-1]:4.0f} ms | ahead {win.player.buffer_ahead_seconds():.1f}s",
              flush=True)

    def step(delay_ms, fn):
        QTimer.singleShot(delay_ms, fn)

    def enable_3d():
        win.settings.preview_mode = "anaglyph"
        win._enable_preview_3d()
        win._sync_mode_buttons()
        win._submit(False)

    def change_setting():
        print(f"{stamp()} >> changing 3D strength mid-play", flush=True)
        win.video_page.threed.strength.set_value(75, emit=True)

    step(600, lambda: win._open_path(video))
    step(2200, enable_3d)
    step(3000, lambda: (print(f"{stamp()} >> Play (3D preview on, 5 s buffer)", flush=True), win.player.play()))
    step(9500, lambda: window_report("steady playback (5.5 s)", 5.5))
    step(9600, change_setting)
    step(15500, lambda: window_report("after a setting change (5.9 s)", 5.5))
    step(15600, lambda: (print(f"{stamp()} >> Pause", flush=True), win.player.pause()))
    step(17000, lambda: (print(f"{stamp()} >> Play again", flush=True), win.player.play()))
    step(21500, lambda: window_report("after pause/resume (4 s)", 4.0))
    step(21600, lambda: (print(f"{stamp()} >> Seek to 20 s", flush=True), win.player.seek_ms(20000, True)))
    step(27500, lambda: window_report("after a seek (5.5 s)", 5.0))
    step(27600, lambda: (win.player.pause(), print(f"{stamp()} done; total buffering wait {log['wait']:.1f}s", flush=True), qt.quit()))
    qt.exec()


if __name__ == "__main__":
    main()
