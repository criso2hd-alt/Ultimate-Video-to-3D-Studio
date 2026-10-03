"""Play a 3D video in the headset without the rest of the app: a manual test for the VR player.

    python scripts/vr_demo.py <video> [layout] [seconds] [--room <room_folder>] [--seat Seat_R2_Centre]

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
    args = sys.argv[1:]
    room_folder = args[args.index("--room") + 1] if "--room" in args else None
    seat = args[args.index("--seat") + 1] if "--seat" in args else None
    for flag in ("--room", "--seat"):
        if flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    path = Path(args[0])
    bootstrap.activate_av()
    info = video.probe(path)
    layout = args[1] if len(args) > 1 else library.guess_layout(path, info.width, info.height)
    seconds = float(args[2]) if len(args) > 2 else 120.0
    if not available():
        print("VR is not available here (needs Windows, an OpenXR runtime and pyopenxr / glfw / PyOpenGL).")
        return 1
    app = QApplication(sys.argv)
    media = VRMedia()
    session = VRSession(media)
    if room_folder:
        from ultimate_video_3d.vr import room as roommod

        room = roommod.load(room_folder)
        session.set_room(room)
        if seat:
            session.set_seat(seat)
        print(f"room: {room.name} ({room.triangles} triangles); seats: {[s.name for s in room.seats]}")
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
