"""Take the README screenshots at 2560 x 1440 from the real window. A harness, not a unit test.

    python tests/readme_shots.py <video> <out_dir>

Use the demo clip from ``scripts/make_demo_scene.py`` so the pictures show footage that belongs to the
project. The preview is set to its highest quality first, so the picture is crisp at this size.
"""

from __future__ import annotations

import os
import sys

os.environ["UV3D_SKIP_TOUR"] = "1"                       # we open the welcome and the tour ourselves
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402
from ultimate_video_3d import onboarding, paths  # noqa: E402

WIDTH, HEIGHT = 2560, 1440


def main() -> None:
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    qt = QApplication(sys.argv)
    appmod._load_bundled_fonts(qt)
    appmod.apply_app_theme(qt, appmod.DEFAULT_THEME)
    win = appmod.MainWindow()
    win.resize(WIDTH, HEIGHT)
    win.show()
    page = win.video_page
    keep = []

    def save(name: str, widget=None) -> None:
        win.status.clearMessage()                        # transient notes (e.g. the auto frame-rate message) stay out of the pictures
        # The real output path contains the Windows user name; a public picture should not.
        page.output_label.setText("Output: ...\\UltimateVideo3DStudio\\output\\demo_scene_SBS.mp4")
        pix = (widget or win).grab()
        pix.save(str(out / f"{name}.png"))
        print(f"saved {name}: {pix.width()}x{pix.height()}", flush=True)

    def rail_to(widget) -> None:
        bar = page.rail_scroller.verticalScrollBar()
        bar.setValue(max(0, widget.mapTo(page.rail_scroller.widget(), widget.rect().topLeft()).y() - 8))

    def enable_colour() -> None:
        colour = page.colour
        win.settings.colour.enabled = True
        colour.set_enabled_state(True)
        colour.set_open(True)
        for attr, value in (("temperature", 14), ("tint", -6), ("contrast", 18), ("highlights", -12),
                            ("shadows", 16), ("saturation", 14), ("vibrance", 12)):
            slider = colour._sliders[attr]
            slider.set_value(value, emit=True)

    def welcome() -> None:
        dialog = onboarding.WelcomeDialog(paths.package_dir() / "assets" / "icon.png", win,
                                          palette=appmod.PALETTES[appmod.DEFAULT_THEME])
        dialog.setStyleSheet(dialog.styleSheet())
        keep.append(dialog)
        dialog.show()
        QTimer.singleShot(500, lambda: (save("welcome", dialog), dialog.close()))

    def tour() -> None:
        win.start_tutorial()
        overlay = win._tour_overlay
        overlay._index = 1                                # "Pick a view"
        overlay._show_step(animate=False)
        QTimer.singleShot(1500, lambda: (save("tutorial"), overlay._skip()))

    steps = [
        (600, lambda: win._open_path(video)),
        (2200, lambda: page and win.settings_page.preview_box.setCurrentIndex(win.settings_page.preview_box.findData(720))),
        (3200, lambda: win.player.seek_ms(1500, True)),
        (4500, lambda: win._set_mode("sbs")),
        (9500, lambda: save("hero-side-by-side")),
        (9600, lambda: win._set_mode("anaglyph")),
        (13000, lambda: save("anaglyph")),
        (13100, lambda: win._set_mode("depth")),
        (16500, lambda: save("depth-map")),
        (16600, lambda: win._set_mode("parallax")),
        (20000, lambda: save("parallax")),
        (20100, lambda: win._set_mode("sbs")),
        (22500, tour),
        (25500, lambda: win.tabs.setCurrentWidget(win.settings_page)),
        (26800, lambda: save("settings")),
        (26900, lambda: win.tabs.setCurrentWidget(win.video_page)),
        (27500, lambda: rail_to(page.export)),
        (28800, lambda: save("export")),
        (28900, enable_colour),
        (31500, lambda: rail_to(page.colour)),
        (33000, lambda: save("colour-correction")),
        (33500, welcome),
        (35500, qt.quit),
    ]
    timers = []
    for delay, fn in steps:
        timer = QTimer()
        timer.setSingleShot(True)
        timer.setTimerType(appmod.Qt.TimerType.PreciseTimer)
        timer.setInterval(delay)
        timer.timeout.connect(fn)
        timer.start()
        timers.append(timer)
    qt.exec()


if __name__ == "__main__":
    main()
