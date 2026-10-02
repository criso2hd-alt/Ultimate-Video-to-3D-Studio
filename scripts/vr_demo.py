"""Play a 3D video in the headset without the rest of the app: a manual test for the VR player.

    python scripts/vr_demo.py <video> [layout] [seconds]

``layout`` is one of sbs_half sbs_full tb_half tb_full "" (flat); it is guessed from the file name when left out.
Put the headset on; playback starts when it is worn. Controls are listed in ``ultimate_video_3d/vr/session.py``.
Stops after ``seconds`` (default 120) or on Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ultimate_video_3d import bootstrap, library, video  # noqa: E402
from ultimate_video_3d.vr import available  # noqa: E402
from ultimate_video_3d.vr.media import VRMedia  # noqa: E402
from ultimate_video_3d.vr.session import VRSession  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1])
    bootstrap.activate_av()
    info = video.probe(path)
    layout = sys.argv[2] if len(sys.argv) > 2 else library.guess_layout(path, info.width, info.height)
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    if not available():
        print("VR is not available here (needs Windows, an OpenXR runtime and pyopenxr / glfw / PyOpenGL).")
        return 1
    app = QApplication(sys.argv)
    media = VRMedia()
    session = VRSession(media)
    print(f"{path.name}: {info.width}x{info.height}, layout {layout or 'flat'}")
    if not media.open(path, layout, loop=info.duration < 30, muted=False):         # a short clip (a test card) repeats
        return 1
    session.state_changed.connect(lambda s: print("session:", s, flush=True))
    session.worn_changed.connect(lambda worn: (print("headset worn" if worn else "headset removed", flush=True),
                                               media.play() if worn else media.pause()))
    session.failed.connect(lambda m: (print("FAILED:", m, flush=True), app.quit()))
    session.stopped.connect(app.quit)
    ticker = QTimer()
    ticker.timeout.connect(lambda: print(f"  {session.state:12s} {session.fps:5.1f} fps  "
                                         f"{media.position_ms / 1000:6.1f}s / {media.duration_ms / 1000:.0f}s", flush=True))
    ticker.start(3000)
    QTimer.singleShot(int(seconds * 1000), session.stop)
    session.start()
    code = app.exec()
    media.close(wait=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
