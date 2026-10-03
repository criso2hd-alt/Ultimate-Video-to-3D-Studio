"""Take the README screenshots of the app window: real widgets, drawn offscreen, nothing from the desktop.

    python scripts/make_screenshots.py <demo_clip.mp4> <out_dir> [--work <scratch_dir>]

The clip is the scene generated for this project (``scripts/make_demo_scene.py``), so no one else's footage is shown.
Everything runs against a throwaway data folder (settings, Library, thumbnails), never the person's own. The window is
drawn at twice the logical size so the pictures are sharp (a 1600 x 900 window grabs as 3200 x 1800).

Saves: zones, library, library-anaglyph, settings-vr, welcome, tour-library, tour-vr, tour-3d.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")
os.environ["UV3D_SKIP_TOUR"] = "1"
if sys.platform == "win32":
    # The offscreen platform draws with a bare-bones font database: no emoji or symbol fallback, so the coffee cup
    # and the encoder badge would come out as empty boxes. Point it at the system fonts.
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

LOGICAL = (1600, 900)


def spin(qt: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qt.processEvents()
        time.sleep(0.01)


def make_sbs(demo: Path, destination: Path) -> None:
    """Convert the demo clip to side-by-side with the app's own pipeline (so the Library has a real 3D video)."""
    from ultimate_video_3d import pipeline
    from ultimate_video_3d.settings import AppSettings

    settings = AppSettings()
    settings.threed.enabled = True
    settings.threed.layout = "sbs_half"
    for _ in pipeline.convert_video(demo, destination, settings):       # raises on failure
        pass


def main() -> int:
    args = sys.argv[1:]
    demo, out = Path(args[0]), Path(args[1])
    work = Path(args[args.index("--work") + 1]) if "--work" in args else Path(tempfile.mkdtemp(prefix="uv3d-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    from ultimate_video_3d import app as appmod
    from ultimate_video_3d import onboarding, paths

    data = work / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir()
    paths.data_dir = lambda: data                                     # the Library, thumbnails and settings go here
    paths.settings_path = lambda: data / "settings.json"
    appmod.paths.data_dir, appmod.paths.settings_path = paths.data_dir, paths.settings_path

    qt = QApplication(sys.argv)
    appmod._load_bundled_fonts(qt)
    appmod.apply_app_theme(qt, appmod.DEFAULT_THEME)

    sbs = work / "Glossy spheres - SBS.mp4"
    if not sbs.exists():
        print("converting the demo clip to side by side ...", flush=True)
        make_sbs(demo, sbs)
    for name in ("Corridor in the fog - SBS.mp4", "Camera trucking sideways - SBS.mp4"):
        if not (work / name).exists():
            shutil.copy(sbs, work / name)

    win = appmod.MainWindow()
    win.resize(*LOGICAL)
    win.show()
    spin(qt, 2.0)

    def grab(widget, name: str) -> None:
        spin(qt, 0.4)
        widget.grab().save(str(out / f"{name}.png"))
        print("saved", name, flush=True)

    # -- the video page: the Zones view, with the depth gauge and the coloured slider strips
    win._open_path(demo)
    spin(qt, 7.0)
    win._set_mode("zones")
    spin(qt, 4.0)
    grab(win, "zones")

    # -- the Library, with a 3D video playing, then as anaglyph
    for name in ("Glossy spheres - SBS.mp4", "Corridor in the fog - SBS.mp4", "Camera trucking sideways - SBS.mp4"):
        win.library.add_converted(work / name, "sbs_half", "demo scene")
    win.library_page.refresh()
    win.tabs.setCurrentWidget(win.library_page)
    spin(qt, 1.0)
    win.library_page.grid.setCurrentRow(0)
    spin(qt, 5.0)                                                       # thumbnails are cut in the background
    win.library_page.play_entry(win.library_page.selected_entry(), autoplay=False)   # selecting only picks; this opens it
    win.library_page.player.seek_ms(1100)                               # a frame with all the depth in it
    spin(qt, 2.5)
    grab(win, "library")
    win.library_page.anaglyph_button.click()
    spin(qt, 1.5)
    grab(win, "library-anaglyph")

    # -- Settings > Virtual reality
    win.tabs.setCurrentWidget(win.settings_page)
    win._reveal_setting(win.settings_page.vr_card)
    spin(qt, 1.0)
    grab(win, "settings-vr")

    # -- the welcome and the tour steps that are new in 0.2
    win.tabs.setCurrentWidget(win.video_page)
    welcome = onboarding.WelcomeDialog(paths.package_dir() / "assets" / "icon.png", win, palette=None)
    welcome.show()
    spin(qt, 0.8)
    welcome.grab().save(str(out / "welcome.png"))
    print("saved welcome", flush=True)
    welcome.close()

    win.start_tutorial()
    overlay = win._tour_overlay
    wanted = {"Library": "tour-library", "Watch in VR": "tour-vr", "3D conversion": "tour-3d"}
    for index, step in enumerate(overlay._steps):
        if step.title in wanted:
            overlay._index = index
            overlay._show_step(animate=False)
            spin(qt, 0.8)
            grab(win, wanted[step.title])
    overlay._skip()
    spin(qt, 0.3)
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
