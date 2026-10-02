"""Make a stereo test card: half-SBS video with squares at known depths, to check a 3D player without any film.

    python scripts/make_stereo_test.py [out.mp4]

Seen correctly in a headset or on a 3D display the squares float at different depths: the one marked SCREEN sits on
the screen, OUT ones come toward you, BEHIND ones sink away. The numbers are the separation between the eyes in
pixels of a 1920-wide picture. If everything looks flat, the player (or the file) has no stereo; if OUT and BEHIND are
swapped, the eyes are swapped. Drawn here, not taken from anything, so it is safe to publish.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

W, H = 1920, 1080


def eye_picture(shift_sign: int) -> np.ndarray:
    """One eye's 1920 x 1080 picture. ``shift_sign`` is -1 for the left eye, +1 for the right."""
    img = np.full((H, W, 3), (28, 28, 34), np.uint8)
    for x in range(0, W, 120):                                                   # a zero-depth grid: the screen plane
        cv2.line(img, (x, 0), (x, H), (52, 52, 62), 2)
    for y in range(0, H, 120):
        cv2.line(img, (0, y), (W, y), (52, 52, 62), 2)
    items = [(-40, "OUT 40", (60, 90, 255)), (-20, "OUT 20", (80, 170, 255)), (0, "SCREEN", (240, 240, 240)),
             (20, "BEHIND 20", (120, 220, 120)), (40, "BEHIND 40", (60, 190, 60))]
    for slot, (disparity, label, colour) in enumerate(items):
        cx = 220 + slot * 370
        # ``disparity`` is where the right eye's copy sits relative to the left's: positive = to its right (uncrossed,
        # the object is behind the screen), negative = to its left (crossed, it comes out of the screen).
        dx = int(round(disparity / 2 * shift_sign))
        x0, x1, y0, y1 = cx - 130 + dx, cx + 130 + dx, 400, 660
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 4)
        text_x = x0 + (260 - cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]) // 2
        cv2.putText(img, label, (text_x, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (cx + dx, 300), 26, colour, -1)
    cv2.putText(img, "STEREO TEST CARD  -  squares at known depths", (330, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 3, cv2.LINE_AA)
    return img


def main() -> int:
    from ultimate_video_3d import bootstrap, paths

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.output_dir() / "stereo-test_SBS.mp4"
    bootstrap.activate_av()
    import av

    left, right = eye_picture(-1), eye_picture(+1)
    half = np.hstack([cv2.resize(left, (W // 2, H), interpolation=cv2.INTER_AREA),
                      cv2.resize(right, (W // 2, H), interpolation=cv2.INTER_AREA)])          # half side-by-side
    with av.open(str(out), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "10", "preset": "medium"}
        for _ in range(24 * 6):
            frame = av.VideoFrame.from_ndarray(half, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
