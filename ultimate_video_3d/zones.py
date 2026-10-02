"""Depth zones: showing where every part of the picture will sit relative to the screen.

The numbers come from ``stereo.disparity_map`` (signed pixels, positive = in front of the screen), turned into a
percentage of the picture's width so they mean the same at any size. Everything here is plain NumPy/OpenCV on the
small preview frame, and is used twice: to colour the picture (the *Zones* view) and to fill in the live gauge
and the marks on the sliders, so what the person sees on the picture and on the controls always agrees.

Colour: warm (orange to red) = comes out of the screen, cool (cyan to blue) = sinks behind it, clear = on the
screen. A thin white line marks the screen plane itself: whatever it runs along is exactly at the screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Disparity beyond these (percent of picture width) is where viewers start to strain. Between COMFORT and STRONG
#: is a strong but fine effect for most; above STRONG it is hard to fuse and tiring.
COMFORT, STRONG = 1.0, 2.0
#: The colour scale tops out here (percent of the width), so a typical film fills the range without clipping.
SCALE = 2.0
BINS = 41                                   # histogram bins from -SCALE to +SCALE; the middle one is the screen plane

#: Warm / cool end colours (RGB 0..1), blended toward by how far from the screen a pixel sits.
WARM = np.array([1.00, 0.42, 0.10], np.float32)
COOL = np.array([0.10, 0.55, 1.00], np.float32)


@dataclass
class DepthStats:
    out_pct: float = 0.0                    # strongest pop-out (99th percentile), % of picture width; 0 if none
    back_pct: float = 0.0                   # strongest push-back, % of picture width; 0 if none
    share_out: float = 0.0                  # fraction of the picture in front of the screen
    share_back: float = 0.0                 # fraction behind it
    hist: np.ndarray = field(default_factory=lambda: np.zeros(BINS, np.float32))
    # Where the bulk of the picture is in depth (0 far .. 1 near): the 10th and 90th percentiles and the median. The
    # depth model rescales every scene to span the whole range, so the extremes (2nd / 98th) would always read 0 / 1.
    depth_lo: float = 0.0
    depth_hi: float = 1.0
    depth_mid: float = 0.5
    valid: bool = False

    @property
    def verdict(self) -> str:
        worst = max(self.out_pct, self.back_pct)
        if not self.valid:
            return ""
        if worst <= COMFORT:
            return "comfortable"
        if worst <= STRONG:
            return "strong"
        return "too strong"

    def to_dict(self) -> dict:
        return {"out_pct": self.out_pct, "back_pct": self.back_pct, "share_out": self.share_out,
                "share_back": self.share_back, "hist": self.hist, "depth_lo": self.depth_lo,
                "depth_hi": self.depth_hi, "depth_mid": self.depth_mid, "valid": self.valid}

    @staticmethod
    def from_dict(data: dict) -> DepthStats:
        stats = DepthStats()
        for key, value in data.items():
            if hasattr(stats, key):
                setattr(stats, key, value)
        return stats


def to_percent(disparity: np.ndarray, width: int) -> np.ndarray:
    """Signed disparity in pixels -> percent of the picture's width."""
    return disparity * np.float32(100.0 / max(1, width))


def measure(disparity: np.ndarray, depth: np.ndarray | None = None) -> DepthStats:
    """The numbers behind the gauge, from one frame's disparity (pixels) and, optionally, its depth (0..1)."""
    height, width = disparity.shape
    pct = to_percent(disparity[::4, ::4], width).ravel()
    if pct.size == 0:
        return DepthStats()
    out = pct[pct > 0.02]
    back = -pct[pct < -0.02]
    stats = DepthStats(
        out_pct=float(np.percentile(out, 99)) if out.size > 20 else 0.0,
        back_pct=float(np.percentile(back, 99)) if back.size > 20 else 0.0,
        share_out=float((pct > 0.02).mean()),
        share_back=float((pct < -0.02).mean()),
        valid=True,
    )
    hist, _ = np.histogram(np.clip(pct, -SCALE, SCALE), bins=BINS, range=(-SCALE, SCALE))
    stats.hist = (hist / max(1, hist.max())).astype(np.float32)
    if depth is not None and depth.size:
        sample = depth[::4, ::4]
        stats.depth_lo, stats.depth_mid, stats.depth_hi = (float(v) for v in np.percentile(sample, (10.0, 50.0, 90.0)))
    return stats


def zone_colours(pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a map of disparity in percent: (rgb tint, opacity), both HxW(x3), from the warm-cool scale."""
    t = np.clip(pct / SCALE, -1.0, 1.0)
    mag = np.abs(t)
    tint = np.where((t > 0)[..., None], WARM, COOL).astype(np.float32)
    # clear at the screen plane, then quickly visible; an eased curve so small shifts still read
    opacity = np.clip(mag, 0.0, 1.0) ** 0.7 * 0.62
    return tint, opacity.astype(np.float32)


def zone_image(rgb: np.ndarray, disparity: np.ndarray, label: bool = True) -> np.ndarray:
    """The picture with the zones painted over it (uint8 HxWx3), a white line on the screen plane and a legend.

    ``rgb`` is float32 0..1; ``disparity`` is the signed pixel map from ``stereo.disparity_map`` at the same size.
    """
    height, width = disparity.shape
    pct = to_percent(disparity, width)
    smooth = cv2.GaussianBlur(pct, (0, 0), max(1.0, width / 400.0))
    tint, opacity = zone_colours(smooth)
    # the picture itself is dimmed and desaturated a little so the colours carry the message
    gray = rgb.mean(axis=-1, keepdims=True)
    base = (rgb * 0.55 + gray * 0.45) * 0.88
    out = base * (1.0 - opacity[..., None]) + tint * opacity[..., None]

    # the screen plane: where the disparity crosses zero
    sign = (smooth > 0).astype(np.uint8)
    edge = (sign[:, 1:] != sign[:, :-1])
    edge = np.pad(edge, ((0, 0), (0, 1)))
    edge |= np.pad(sign[1:, :] != sign[:-1, :], ((0, 1), (0, 0)))
    line = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(np.float32)
    out = out * (1.0 - 0.85 * line[..., None]) + 0.85 * line[..., None]
    image = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if label:
        _legend(image)
    return image


def _legend(image: np.ndarray) -> None:
    """A small gradient bar, bottom left: IN FRONT (warm) - SCREEN - BEHIND (cool)."""
    height, width = image.shape[:2]
    bar_w, bar_h = max(120, width // 4), max(8, height // 50)
    x0, y0 = width - bar_w - max(8, width // 40), height - max(10, height // 14)       # bottom right: the note is bottom left
    ramp = np.linspace(1.0, -1.0, bar_w)                          # left = in front, right = behind
    pct = (ramp * SCALE).astype(np.float32)[None, :]
    tint, opacity = zone_colours(np.repeat(pct, bar_h, axis=0))
    swatch = (np.full((bar_h, bar_w, 3), 0.16, np.float32) * (1 - opacity[..., None]) + tint * opacity[..., None])
    pad = 5
    cv2.rectangle(image, (x0 - pad, y0 - pad - 18), (x0 + bar_w + pad, y0 + bar_h + pad + 16), (0, 0, 0), -1)
    image[y0:y0 + bar_h, x0:x0 + bar_w] = (np.clip(swatch, 0, 1) * 255).astype(np.uint8)
    scale = max(0.35, min(0.6, width / 1600.0 + 0.3))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "IN FRONT", (x0, y0 - 5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "SCREEN", (x0 + bar_w // 2 - int(30 * scale * 1.6), y0 + bar_h + 13), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    text = "BEHIND"
    tw = cv2.getTextSize(text, font, scale, 1)[0][0]
    cv2.putText(image, text, (x0 + bar_w - tw, y0 - 5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (x0 + bar_w // 2, y0 - 3), (x0 + bar_w // 2, y0 + bar_h + 3), (255, 255, 255), 1, cv2.LINE_AA)


def strength_marks(stats: DepthStats, strength: float) -> tuple[float, float] | None:
    """Where the *3D strength* slider (0..100) crosses the comfort limits for this picture.

    Disparity is directly proportional to strength, so from the current numbers it is exact: the strongest part of
    the picture reaches COMFORT percent of the width at ``strength * COMFORT / worst`` and STRONG at ``strength *
    STRONG / worst``. Returns (comfortable_until, strong_until) in slider units (a value over 100 means the limit is
    beyond the slider's end), or None when nothing sticks out of or behind the screen yet.
    """
    worst = max(stats.out_pct, stats.back_pct)
    if not stats.valid or worst < 0.02 or strength <= 0:
        return None
    return strength * COMFORT / worst, strength * STRONG / worst


def convergence_marks(stats: DepthStats) -> tuple[float, float, float] | None:
    """Where the *Depth position* slider (0..100) puts the bulk of the picture in front of, on, or behind the screen.

    Depth runs 0 (far) to 1 (near). A pixel nearer than the slider's value comes out of the screen; a farther one
    sinks behind it (``stereo._disparity``). So with the slider below ``lo`` the whole picture is in front of the
    screen, above ``hi`` all of it is behind, and between the two it is split across the screen. ``lo`` and ``hi`` are
    the 10th and 90th percentiles (the bulk of the picture, not stray extremes) and ``mid`` the median: at that
    setting half the picture is in front. Returns (lo, hi, mid) on the slider's 0..100 scale, or None when there
    is no measurement yet.
    """
    if not stats.valid:
        return None
    return stats.depth_lo * 100.0, stats.depth_hi * 100.0, stats.depth_mid * 100.0
