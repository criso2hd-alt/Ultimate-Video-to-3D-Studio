"""Finding the black bars burned into a video, so a film that is 2.35:1 inside a 16:9 frame can fill a cinema screen.

Many discs and rips store a wide film as a 16:9 picture with black bars above and below (or a 4:3 one with bars at the sides).
The bars are part of the video, so a player that shows the whole frame shows them too. :func:`detect` looks at a couple of dozen
frames spread through the film and reports how much of each side is black in all of them; the VR player then shows only the
real picture (:func:`crop_rect`) and fits *that* shape to the screen.

Conservative on purpose: a row or column counts as picture if a few of the sampled frames have anything bright in it, so a dark
scene, a fade or burned-in subtitles can only make the bars look smaller (nothing is cut that is not black everywhere).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLES = 24                 # frames looked at, spread over the middle of the film
PROXY_WIDTH = 320            # they are shrunk to this wide first: a few milliseconds each
BRIGHT = 40                  # luma (limited range: black is 16) above which a pixel is picture
ROW_SHARE = 0.005            # a row or column is picture if this share of its pixels is bright ...
MIN_FRAMES = 2               # ... in at least this many of the sampled frames
MIN_BAR = 0.02               # bars thinner than this share of the picture are not worth cutting (and could be noise)
MIN_PICTURE = 0.4            # a result that leaves less than this of a side is a mistake: show everything


@dataclass(frozen=True)
class Crop:
    """How much of ONE eye's picture to cut from each side, as fractions (0..0.5) of its height / width."""

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @property
    def none(self) -> bool:
        return not (self.top or self.bottom or self.left or self.right)

    @property
    def aspect_factor(self) -> float:
        """By how much the picture's width / height ratio changes when this is cut off."""
        return (1.0 - self.left - self.right) / max(1e-6, 1.0 - self.top - self.bottom)


NONE = Crop()


def crop_rect(rect: tuple[float, float, float, float], crop: Crop) -> tuple[float, float, float, float]:
    """The part of the frame to show: ``rect`` (u0, v0, u1, v1; one eye's area) with ``crop`` cut off its sides."""
    if crop.none:
        return rect
    u0, v0, u1, v1 = rect
    w, h = u1 - u0, v1 - v0
    return (u0 + crop.left * w, v0 + crop.top * h, u1 - crop.right * w, v1 - crop.bottom * h)


def crop_from_gray(frames: list[np.ndarray]) -> Crop:
    """The bars of a set of equally sized grey pictures (height x width, uint8): black in every one of them."""
    if not frames:
        return NONE
    height, width = frames[0].shape
    rows = np.zeros(height, int)
    cols = np.zeros(width, int)
    for gray in frames:
        bright = gray > BRIGHT
        rows += bright.mean(axis=1) > ROW_SHARE
        cols += bright.mean(axis=0) > ROW_SHARE
    need = min(MIN_FRAMES, len(frames))
    picture_rows, picture_cols = np.where(rows >= need)[0], np.where(cols >= need)[0]
    if len(picture_rows) == 0 or len(picture_cols) == 0:
        return NONE                                       # all dark: nothing to say
    top, bottom = picture_rows[0], height - 1 - picture_rows[-1]
    left, right = picture_cols[0], width - 1 - picture_cols[-1]

    def side(count: int, size: int) -> float:
        share = max(0, count - 1) / size                  # keep one row of the bar: a hair of black is harmless, a cut picture is not
        return share if count / size >= MIN_BAR else 0.0

    crop = Crop(float(side(top, height)), float(side(bottom, height)), float(side(left, width)), float(side(right, width)))
    if 1.0 - crop.top - crop.bottom < MIN_PICTURE or 1.0 - crop.left - crop.right < MIN_PICTURE:
        return NONE
    return crop


def eye_region(layout: str) -> tuple[float, float, float, float]:
    """Where one eye's picture is within the frame (u0, v0, u1, v1), for the layouts that put two in one frame."""
    if layout in ("sbs_half", "sbs_full"):
        return (0.0, 0.0, 0.5, 1.0)
    if layout in ("tb_half", "tb_full"):
        return (0.0, 0.0, 1.0, 0.5)
    return (0.0, 0.0, 1.0, 1.0)


def detect(path: str | Path, layout: str = "", samples: int = SAMPLES) -> Crop:
    """Look through the file for black bars. Returns NONE when there are none, or when anything goes wrong."""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = float(container.duration or 0) / 1e6
            if duration <= 0 and stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            width, height = stream.codec_context.width, stream.codec_context.height
            u0, v0, u1, v1 = eye_region(layout)
            target_w = PROXY_WIDTH * 2 if u1 - u0 < 1.0 else PROXY_WIDTH
            target_h = max(8, int(round(target_w * height / max(1, width))))
            pictures: list[np.ndarray] = []
            count = max(2, samples)
            for i in range(count):
                if duration > 1.0:
                    when = duration * (0.05 + 0.9 * i / (count - 1))
                    try:
                        container.seek(int(when / stream.time_base), stream=stream, backward=True)
                    except Exception:  # noqa: BLE001 - a file that will not seek: use what we get from the top
                        pass
                got = None
                for frame in container.decode(stream):
                    got = frame
                    break
                if got is None:
                    continue
                gray = got.reformat(width=target_w, height=target_h, format="gray").to_ndarray()
                y0, y1 = int(round(v0 * gray.shape[0])), int(round(v1 * gray.shape[0]))
                x0, x1 = int(round(u0 * gray.shape[1])), int(round(u1 * gray.shape[1]))
                pictures.append(np.ascontiguousarray(gray[y0:y1, x0:x1]))
                if duration <= 1.0 and len(pictures) >= 4:
                    break
            return crop_from_gray(pictures)
    except Exception:  # noqa: BLE001 - bars are a nicety: any trouble means "show the whole frame"
        return NONE
