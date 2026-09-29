"""Watch the paused 3D preview fill its disk cache, then play from it. A harness, not a unit test.

    python tests/perf_paused_fill.py <video>

Scenarios, in order: pick a 3D view while paused (the cache should start growing on disk), let it
run, press Play (it should start at once, from the cache), pause again (it should keep filling),
scrub to somewhere else (it should restart from there), and change a setting (it should empty and
refill). Prints how far ahead the cache is, its size on disk, and the worst UI-thread stall.
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
    ui = {"last": time.perf_counter(), "worst": 0.0}
    shown = []
    waited = {"since": None, "total": 0.0}

    def stamp() -> str:
        return f"[{time.perf_counter() - t0:5.1f}s]"

    def tick() -> None:
        now = time.perf_counter()
        ui["worst"] = max(ui["worst"], (now - ui["last"]) * 1000)
        ui["last"] = now

    ticker = QTimer()
    ticker.setInterval(10)
    ticker.timeout.connect(tick)
    ticker.start()
    win.player.buffered_image.connect(lambda _p: shown.append(time.perf_counter()))

    def on_buffering(waiting: bool, _progress: float) -> None:
        now = time.perf_counter()
        if waiting and waited["since"] is None:
            waited["since"] = now
        elif not waiting and waited["since"] is not None:
            waited["total"] += now - waited["since"]
            print(f"{stamp()} resumed after waiting {now - waited['since']:.2f}s", flush=True)
            waited["since"] = None

    win.player.buffering_changed.connect(on_buffering)

    def cache(label: str) -> None:
        c = win.player._buffered.cache
        folder = getattr(c, "_dir", None)
        files = len(list(folder.iterdir())) if folder is not None and folder.exists() else 0
        print(f"{stamp()} {label:30} ahead {win.player.buffer_ahead_seconds():5.1f}s | {len(c):4d} frames | "
              f"{c.nbytes / 1e6:6.1f} MB | {files:4d} files | worst UI stall {ui['worst']:4.0f} ms", flush=True)
        ui["worst"] = 0.0

    def say(text: str) -> None:
        print(f"{stamp()} >> {text}", flush=True)

    def play_and_watch() -> None:
        say("Play")
        shown.clear()
        win.player.play()

    def first_picture() -> None:
        if shown:
            print(f"{stamp()}    first picture after {(shown[0] - t_play[0]) * 1000:.0f} ms", flush=True)
        else:
            print(f"{stamp()}    no picture yet", flush=True)

    t_play = [0.0]
    original = [50.0]

    def do_play() -> None:
        t_play[0] = time.perf_counter()
        play_and_watch()

    def strength() -> None:
        say("changing 3D strength (it is restored afterwards)")
        original[0] = win.settings.threed.strength
        win.video_page.threed.strength.set_value(90 if win.settings.threed.strength != 90 else 30, emit=True)

    def step(ms, fn):
        QTimer.singleShot(ms, fn)

    step(500, lambda: win._open_path(video))
    step(2000, lambda: (say("pick Anaglyph while paused"), win._set_mode("anaglyph")))
    for at in (3000, 5000, 8000, 12000, 16000):
        step(at, lambda a=at: cache(f"paused, {a / 1000:.0f}s in"))
    step(16500, do_play)
    step(17300, first_picture)
    step(20000, lambda: cache("playing"))
    step(20100, lambda: (say("Pause"), win.player.pause()))
    for at in (22000, 26000):
        step(at, lambda a=at: cache(f"paused again, {a / 1000:.0f}s in"))
    step(26100, lambda: (say("scrub to 40 s"), win.player.seek_ms(40000, True)))
    for at in (27000, 30000, 34000):
        step(at, lambda a=at: cache(f"after scrub, {a / 1000:.0f}s in"))
    step(34100, strength)
    for at in (34700, 38000):
        step(at, lambda a=at: cache(f"after setting change, {a / 1000:.0f}s"))
    step(38050, lambda: win.video_page.threed.strength.set_value(original[0], emit=True))
    step(38100, lambda: (print(f"{stamp()} done; waited on the buffer {waited['total']:.1f}s in total", flush=True), qt.quit()))
    qt.exec()


if __name__ == "__main__":
    main()
