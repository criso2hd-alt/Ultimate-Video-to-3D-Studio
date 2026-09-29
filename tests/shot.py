"""Launch the real window, open a clip, and save screenshots. Not a unit test - a visual check.

    python tests/shot.py <video> <out_dir> [mode ...]

Each mode (output, anaglyph, sbs, wiggle, depth, parallax, holes, original) is
shown for a moment and grabbed to ``<out_dir>/<mode>.png``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("UV3D_SKIP_TOUR", "1")            # a fresh settings file would otherwise open the welcome dialog
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402


def main() -> None:
    video = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    modes = sys.argv[3:] or ["anaglyph"]
    qt = QApplication(sys.argv)
    appmod._load_bundled_fonts(qt)
    appmod.apply_app_theme(qt, appmod.DEFAULT_THEME)
    win = appmod.MainWindow()
    win.resize(1500, 900)
    win.show()

    steps = [(1500, lambda: win._open_path(video))]
    t = 6500
    for mode in modes:
        steps.append((t, lambda m=mode: win._set_mode(m)))
        steps.append((t + 2500, lambda m=mode: (win.grab().save(str(out / f"{m}.png")), print("saved", m, flush=True))))
        t += 3500
    steps.append((t, qt.quit))
    for delay, fn in steps:
        QTimer.singleShot(delay, fn)
    qt.exec()


if __name__ == "__main__":
    main()
