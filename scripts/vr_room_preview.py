"""Draw a baked VR room offscreen from a seat, with a picture on the screen, and save the pictures: no headset needed.

    python scripts/vr_room_preview.py <room_folder> <out_dir> [--seat Seat_R2_Centre] [--picture some.mp4|image.png] [--menu main|videos|help]

Writes ``seat_<name>_<state>.png`` for the room lights up, half up (a pause) and down (the film), plus a sheet of them
all. With ``--menu`` the headset's menu panel is hung in front of the seat and drawn too (``<seat>_menu.png``). The picture is a frame from the given video or image, or a built-in colour test picture. Uses the same renderer the
headset uses, so what shows here is what the headset draws, minus the stereo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

W, H = 1600, 900


def test_picture() -> np.ndarray:
    """A colourful 16:9 test picture (made here, not taken from anything)."""
    import cv2

    h, w = 540, 960
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (255 * x / w).astype(np.uint8)
    img[..., 1] = (255 * (1 - y / h)).astype(np.uint8)
    img[..., 2] = (160 + 90 * np.sin(x / 60.0)).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (w // 2, h // 2), 150, (250, 240, 200), -1)
    cv2.putText(img, "TEST PICTURE", (w // 2 - 210, h // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 40), 4)
    return img


def load_picture(path: str | None) -> np.ndarray:
    import cv2

    if not path:
        return test_picture()
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    from ultimate_video_3d import bootstrap

    bootstrap.activate_av()
    import av

    with av.open(str(p)) as container:
        stream = container.streams.video[0]
        container.seek(int(min(30.0, (container.duration or 0) / 2e6) / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise SystemExit("could not read a frame from the picture")


def main() -> int:
    import cv2
    import glfw
    from OpenGL import GL

    args = sys.argv[1:]
    room_folder, out = Path(args[0]), Path(args[1])
    seat_name = args[args.index("--seat") + 1] if "--seat" in args else None
    picture = load_picture(args[args.index("--picture") + 1] if "--picture" in args else None)
    out.mkdir(parents=True, exist_ok=True)

    from ultimate_video_3d.vr import geometry as geo
    from ultimate_video_3d.vr import renderer as rend
    from ultimate_video_3d.vr import room as roommod

    room = roommod.load(room_folder)
    seat = room.seat(seat_name) if seat_name else room.first_seat()
    print(f"{room.name}: {len(room.groups)} groups, {room.triangles} triangles, {len(room.seats)} seats; from {seat.name}")

    if not glfw.init():
        raise SystemExit("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "preview", None, None)
    glfw.make_context_current(window)
    fbo, colour, depth = GL.glGenFramebuffers(1), GL.glGenTextures(1), GL.glGenRenderbuffers(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, colour)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_SRGB8_ALPHA8, W, H, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, colour, 0)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth)

    r = rend.Renderer()
    r.init_gl()
    r.set_room(room)
    r.upload(picture)
    screen = geo.Screen.from_quad(room.screen.matrix, room.screen.width, room.screen.height)
    screen.picture_aspect = picture.shape[1] / picture.shape[0]
    half = np.radians(50)
    proj = geo.projection_matrix(-half, half, half * H / W, -half * H / W)
    # look at the middle of the screen from the seat, as a person would
    forward = room.screen.centre - seat.position
    pitch = float(np.arctan2(forward[1], np.linalg.norm(forward[[0, 2]])))
    quat = (float(np.sin(pitch / 2)), 0.0, 0.0, float(np.cos(pitch / 2)))
    view = geo.view_matrix(tuple(seat.position), quat)
    GL.glViewport(0, 0, W, H)
    if "--menu" in args:
        from PySide6.QtWidgets import QApplication

        from ultimate_video_3d.vr import menu as menumod

        _app = QApplication.instance() or QApplication([])
        menu = menumod.Menu()
        menu.sync(playing=True, title="Behind the scenes", has_media=True, has_lights=True, seat_label=seat.label, has_seats=True)
        menu.show(args[args.index("--menu") + 1])
        image = menu.image()
        pixels = np.frombuffer(bytes(image.constBits()), np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
        r.set_panel_image(pixels[:, :image.width()])
        eye_to_screen = room.screen.centre - seat.position
        panel = geo.panel_matrix(seat.position, eye_to_screen, 1.3, menumod.ASPECT)
        r.room_light = room.light_level_paused
        r.draw_eye(view, proj, 0, tuple(seat.position), screen, "", rend.BAKED, (), panel)
        shown = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
        cv2.imwrite(str(out / f"{seat.name}_menu.png"), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        print("wrote the menu view")
    shots = []
    for label, light in (("lights_up", 1.0), ("paused", room.light_level_paused), ("film", 0.0)):
        r.room_light = light
        r.draw_eye(view, proj, 0, tuple(seat.position), screen, "", rend.BAKED)
        pixels = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
        path = out / f"{seat.name}_{label}.png"
        cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        print("wrote", path)
        shots.append(cv2.resize(pixels, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(out / f"{seat.name}_sheet.png"), cv2.cvtColor(np.vstack([np.hstack(shots[:2]), np.hstack([shots[2], shots[2] * 0])]), cv2.COLOR_RGB2BGR))
    glfw.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
