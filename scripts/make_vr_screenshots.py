"""Take the README pictures of the VR room and its menu, drawn with the same renderer the headset uses (no headset needed).

    python scripts/make_vr_screenshots.py <demo_clip.mp4> <room_folder> <out_dir>

The picture on the screen is a frame of the scene generated for this project (``scripts/make_demo_scene.py``). One eye is
drawn, so they are flat pictures of what a person sees, not stereo. Saves 1920 x 1080 PNGs:
``vr-room-lights-up``, ``vr-room-paused``, ``vr-room-film``, ``vr-room-front-row``, ``vr-menu-main``, ``vr-menu-videos``,
``vr-menu-help``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

W, H = 1920, 1080


def frame_at(clip: Path, seconds: float) -> np.ndarray:
    from ultimate_video_3d import bootstrap

    bootstrap.activate_av()
    import av

    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        container.seek(int(seconds / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise SystemExit("could not read the demo clip")


def main() -> int:
    import cv2
    import glfw
    from OpenGL import GL
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    clip, room_folder, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    qt = QApplication.instance() or QApplication([])

    from ultimate_video_3d.vr import geometry as geo
    from ultimate_video_3d.vr import menu as menumod
    from ultimate_video_3d.vr import renderer as rend
    from ultimate_video_3d.vr import room as roommod
    from ultimate_video_3d.vr.session import MENU_DISTANCE, MENU_DROP, MENU_WIDTH

    room = roommod.load(room_folder)
    picture = frame_at(clip, 1.2)

    if not glfw.init():
        raise SystemExit("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "shots", None, None)
    glfw.make_context_current(window)
    fbo, colour, depth = GL.glGenFramebuffers(1), GL.glGenTextures(1), GL.glGenRenderbuffers(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, colour)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_SRGB8_ALPHA8, W, H, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, colour, 0)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth)
    GL.glViewport(0, 0, W, H)

    r = rend.Renderer()
    r.init_gl()
    r.set_room(room)
    r.upload(picture)
    screen = geo.Screen.from_quad(room.screen.matrix, room.screen.width, room.screen.height)
    screen.picture_aspect = picture.shape[1] / picture.shape[0]
    half = np.radians(46)
    proj = geo.projection_matrix(-half, half, half * H / W, -half * H / W)

    def look_from(seat, lower: float = 0.0):
        """Looking at the middle of the screen, or ``lower`` radians below that (a person looking at the menu)."""
        forward = room.screen.centre - seat.position
        pitch = float(np.arctan2(forward[1], np.linalg.norm(forward[[0, 2]]))) - lower
        return geo.view_matrix(tuple(seat.position), (float(np.sin(pitch / 2)), 0.0, 0.0, float(np.cos(pitch / 2))))

    def shoot(name: str, seat, light: float, panel=None, lower: float = 0.0) -> None:
        r.room_light = light
        r.draw_eye(look_from(seat, lower), proj, 0, tuple(seat.position), screen, "", rend.BAKED, (), panel)
        pixels = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
        cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        print("saved", name, flush=True)

    centre = room.seat("Seat_R2_Centre") or room.first_seat()
    front = room.seat("Seat_R1_Centre") or centre
    shoot("vr-room-lights-up", centre, 1.0)
    shoot("vr-room-paused", centre, room.light_level_paused)
    shoot("vr-room-film", centre, 0.0)
    shoot("vr-room-front-row", front, 0.0)

    # the menu, hung in front of the seat as the headset hangs it, over the paused room
    thumbs = []
    for i, t in enumerate((0.2, 0.9, 1.6, 2.3, 2.7, 0.5)):
        rgb = np.ascontiguousarray(cv2.resize(frame_at(clip, t), (320, 180), interpolation=cv2.INTER_AREA))
        thumbs.append(QImage(rgb.data, 320, 180, 320 * 3, QImage.Format.Format_RGB888).copy())
    names = ("Glossy spheres", "Corridor in the fog", "Camera trucking sideways", "Behind the scenes",
             "Holiday in 3D", "Sunday cartoons")
    menu = menumod.Menu()
    menu.sync(playing=False, title="Glossy spheres", has_media=True, has_lights=True, seat_label=centre.label, has_seats=True)
    menu.set_tiles([menumod.Tile(n, "x.mp4", "sbs_half", "Side by side - 3:12", t) for n, t in zip(names, thumbs)])
    to_screen = room.screen.centre - centre.position
    panel = geo.panel_matrix(centre.position, to_screen, MENU_WIDTH, menumod.ASPECT, MENU_DISTANCE, MENU_DROP)   # as in the headset
    for page in ("main", "videos", "help"):
        menu.show(page)
        menu.set_hover({"main": "play", "videos": "video:1", "help": "close"}[page])
        image = menu.image()
        pixels = np.frombuffer(bytes(image.constBits()), np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
        r.set_panel_image(pixels[:, :image.width()])
        shoot(f"vr-menu-{page}", centre, room.light_level_paused, panel, lower=0.17)
    glfw.terminate()
    del qt
    return 0


if __name__ == "__main__":
    sys.exit(main())
