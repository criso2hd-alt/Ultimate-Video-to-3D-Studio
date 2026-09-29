"""Lumetri-style colour correction, fast enough to run live on video.

Everything a colourist reaches for first, in one pass over the frame:

  Basic      temperature, tint, exposure, contrast, highlights, shadows,
             whites, blacks, saturation
  Creative   faded film, sharpen, vibrance, shadow / highlight tint
  Curves     master + red / green / blue
  Wheels     shadows / midtones / highlights colour balance
  Vignette   amount, midpoint, roundness, feather
  LUT        a 3D ``.cube`` look, before or after the grade, with intensity

Why it is fast: everything that acts on one channel at a time (exposure,
contrast, blacks, whites, fade, the curves) is folded into a single 1D lookup
table per channel, built once per settings change. Everything that needs the
whole pixel (white balance, luminance-weighted highlights and shadows, wheels,
saturation, the 3D LUT, the vignette) then runs in one Numba kernel, one image
row per CPU thread. A 1080p frame takes about ten milliseconds, so the sliders
move the picture as they are dragged.

The maths works on the *encoded* signal - gamma for SDR, PQ or HLG for HDR - and
the tone table is built with the matching transfer curve, so exposure is a true
stop of light either way. The frame is never converted to another colour space
and back per pixel, which is where quality and time both go.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from . import lut as lutmod

if "NUMBA_CACHE_DIR" not in os.environ:
    from . import paths

    try:
        os.environ["NUMBA_CACHE_DIR"] = str(paths.data_dir() / "numba_cache")
    except Exception:  # noqa: BLE001
        pass

from numba import njit, prange

from .kernels import LOCK

TONE_SIZE = 4096

# Curve control points are (x, y) in 0..1. The identity is the two corners.
IDENTITY_CURVE = [[0.0, 0.0], [1.0, 1.0]]


@dataclass
class Wheel:
    """A colour-balance wheel: push the tonal range toward a hue by an amount."""

    hue: float = 0.0          # degrees, 0 = red
    amount: float = 0.0       # 0..100
    luma: float = 0.0         # -100..100, brightness of just this range

    @property
    def neutral(self) -> bool:
        return self.amount == 0.0 and self.luma == 0.0


@dataclass
class ColourSettings:
    """Neutral at the defaults, so an untouched panel costs nothing."""

    enabled: bool = False
    # Basic
    temperature: float = 0.0     # -100 cool .. +100 warm
    tint: float = 0.0            # -100 green .. +100 magenta
    exposure: float = 0.0        # stops, -4 .. +4
    contrast: float = 0.0        # -100 .. +100
    highlights: float = 0.0      # -100 .. +100
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    saturation: float = 0.0      # -100 (greyscale) .. +100
    # Creative
    faded_film: float = 0.0      # 0 .. 100
    sharpen: float = 0.0         # 0 .. 100
    vibrance: float = 0.0        # -100 .. +100
    # Curves (lists of [x, y])
    curve_master: list = field(default_factory=lambda: [list(p) for p in IDENTITY_CURVE])
    curve_red: list = field(default_factory=lambda: [list(p) for p in IDENTITY_CURVE])
    curve_green: list = field(default_factory=lambda: [list(p) for p in IDENTITY_CURVE])
    curve_blue: list = field(default_factory=lambda: [list(p) for p in IDENTITY_CURVE])
    # Colour wheels
    wheel_shadows: Wheel = field(default_factory=Wheel)
    wheel_midtones: Wheel = field(default_factory=Wheel)
    wheel_highlights: Wheel = field(default_factory=Wheel)
    # Vignette
    vignette: float = 0.0        # -100 (lighten edges) .. +100 (darken)
    vignette_midpoint: float = 50.0
    vignette_roundness: float = 0.0
    vignette_feather: float = 50.0
    # LUT
    lut_enabled: bool = False
    lut_path: str = ""
    lut_intensity: float = 100.0
    lut_before: bool = False     # True = "input LUT" ahead of the grade

    def curves_neutral(self) -> bool:
        return all(_is_identity(c) for c in (
            self.curve_master, self.curve_red, self.curve_green, self.curve_blue))

    def grade_neutral(self) -> bool:
        return (
            not any((self.temperature, self.tint, self.exposure, self.contrast, self.highlights,
                     self.shadows, self.whites, self.blacks, self.saturation, self.faded_film,
                     self.vibrance, self.vignette))
            and self.curves_neutral()
            and self.wheel_shadows.neutral and self.wheel_midtones.neutral
            and self.wheel_highlights.neutral
        )

    @property
    def grade_active(self) -> bool:
        """Whether the colour panel itself is doing anything."""
        return self.enabled and (not self.grade_neutral() or bool(self.sharpen))

    @property
    def lut_active(self) -> bool:
        return self.lut_enabled and bool(self.lut_path)

    @property
    def active(self) -> bool:
        """Whether applying these settings would change the picture at all."""
        return self.grade_active or self.lut_active

    def copy(self) -> "ColourSettings":
        import copy as _copy

        return _copy.deepcopy(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ColourSettings":
        """Tolerant load: unknown keys dropped, missing ones keep defaults."""
        settings = cls()
        if not isinstance(raw, dict):
            return settings
        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in known:
                continue
            if key.startswith("wheel_") and isinstance(value, dict):
                wheel = Wheel()
                for wf in fields(Wheel):
                    if wf.name in value:
                        try:
                            setattr(wheel, wf.name, float(value[wf.name]))
                        except (TypeError, ValueError):
                            pass
                value = wheel
            setattr(settings, key, value)
        return settings


def _is_identity(points) -> bool:
    return [list(map(float, p)) for p in points] == IDENTITY_CURVE


# --- transfer curves ------------------------------------------------------

SDR, PQ, HLG = "sdr", "pq", "hlg"


def _srgb_decode(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(v):
    v = np.maximum(v, 0.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


_M1, _M2 = 2610 / 16384, 2523 / 4096 * 128
_C1, _C2, _C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_decode(v):
    """PQ signal 0..1 -> linear light, 1.0 = 10,000 nits."""
    p = np.power(np.clip(v, 0.0, 1.0), 1.0 / _M2)
    return np.power(np.maximum(p - _C1, 0.0) / (_C2 - _C3 * p), 1.0 / _M1)


def pq_encode(y):
    p = np.power(np.clip(y, 0.0, 1.0), _M1)
    return np.power((_C1 + _C2 * p) / (1.0 + _C3 * p), _M2)


_HA, _HB, _HC = 0.17883277, 0.28466892, 0.55991073


def hlg_decode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.5, v * v / 3.0, (np.exp((v - _HC) / _HA) + _HB) / 12.0)


def hlg_encode(e):
    e = np.clip(e, 0.0, 1.0)
    return np.where(e <= 1 / 12, np.sqrt(3.0 * e), _HA * np.log(np.maximum(12.0 * e - _HB, 1e-9)) + _HC)


_CURVES = {SDR: (_srgb_decode, _srgb_encode), PQ: (pq_decode, pq_encode), HLG: (hlg_decode, hlg_encode)}


def decode(signal: np.ndarray, transfer: str) -> np.ndarray:
    return _CURVES[transfer][0](signal)


def encode(linear: np.ndarray, transfer: str) -> np.ndarray:
    return _CURVES[transfer][1](linear)


# --- building the lookup tables ----------------------------------------------


def monotone_curve(points, size: int = TONE_SIZE) -> np.ndarray:
    """A smooth curve through control points, sampled at ``size`` positions.

    Monotone cubic (Fritsch-Carlson), so dragging a point cannot make the curve
    overshoot or fold back on itself - a plain spline would, and a folded tone
    curve inverts part of the picture.
    """
    pts = sorted((float(x), float(y)) for x, y in points)
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    if len(xs) < 2:
        return np.linspace(0.0, 1.0, size, dtype=np.float32)
    dx, dy = np.diff(xs), np.diff(ys)
    dx = np.maximum(dx, 1e-6)
    slopes = dy / dx
    tangents = np.zeros_like(xs)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    for i in range(1, len(xs) - 1):
        if slopes[i - 1] * slopes[i] <= 0:
            tangents[i] = 0.0
        else:
            tangents[i] = (slopes[i - 1] + slopes[i]) / 2.0
    for i in range(len(slopes)):
        if abs(slopes[i]) < 1e-12:
            tangents[i] = tangents[i + 1] = 0.0
            continue
        a, b = tangents[i] / slopes[i], tangents[i + 1] / slopes[i]
        r = a * a + b * b
        if r > 9.0:
            t = 3.0 / np.sqrt(r)
            tangents[i], tangents[i + 1] = t * a * slopes[i], t * b * slopes[i]
    grid = np.linspace(0.0, 1.0, size)
    idx = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, len(xs) - 2)
    h = dx[idx]
    t = np.clip((grid - xs[idx]) / h, 0.0, 1.0)
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    out = h00 * ys[idx] + h10 * h * tangents[idx] + h01 * ys[idx + 1] + h11 * h * tangents[idx + 1]
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tone_tables(s: ColourSettings, transfer: str) -> np.ndarray:
    """One ``(3, TONE_SIZE)`` table: exposure, black/white points, contrast, fade, curves."""
    grid = np.linspace(0.0, 1.0, TONE_SIZE, dtype=np.float64)
    v = grid.copy()

    if s.exposure:
        v = encode(decode(v, transfer) * (2.0 ** s.exposure), transfer)

    # Black and white points: +blacks lifts the floor, +whites lets the top clip earlier.
    black = -s.blacks / 100.0 * 0.12
    white = 1.0 - s.whites / 100.0 * 0.15
    if black or white != 1.0:
        v = (v - black) / max(white - black, 1e-3)

    if s.contrast:
        g = np.exp(s.contrast / 100.0 * 1.2)
        low = 0.5 * np.power(np.clip(2.0 * v, 0.0, 1.0), g)
        high = 1.0 - 0.5 * np.power(np.clip(2.0 * (1.0 - v), 0.0, 1.0), g)
        v = np.where(v < 0.5, low, high)

    if s.faded_film:
        f = s.faded_film / 100.0
        v = v * (1.0 - 0.16 * f) + 0.11 * f          # raise the blacks, soften the whites

    v = np.clip(v, 0.0, 1.0)
    tables = np.empty((3, TONE_SIZE), np.float32)
    master = monotone_curve(s.curve_master)
    for c, points in enumerate((s.curve_red, s.curve_green, s.curve_blue)):
        per = monotone_curve(points)
        idx = np.clip(v * (TONE_SIZE - 1), 0, TONE_SIZE - 1)
        # master then the channel curve, sampled through the tone result
        m = np.interp(idx, np.arange(TONE_SIZE), master)
        tables[c] = np.interp(np.clip(m, 0, 1) * (TONE_SIZE - 1), np.arange(TONE_SIZE), per)
    return tables


def _wheel_rgb(w: Wheel) -> tuple[float, float, float]:
    """A wheel as an RGB offset: hue picks the direction, amount the strength."""
    if w.amount == 0.0:
        return 0.0, 0.0, 0.0
    hue = np.deg2rad(w.hue)
    sat = w.amount / 100.0
    # A hue vector in RGB with zero mean, so the wheel shifts colour, not brightness.
    r = np.cos(hue)
    g = np.cos(hue - 2.0943951)
    b = np.cos(hue + 2.0943951)
    return float(r * sat), float(g * sat), float(b * sat)


def _params(s: ColourSettings, width: int, height: int, has_lut: bool) -> np.ndarray:
    """Every scalar the kernel needs, packed in one array (keeps it one compiled signature)."""
    p = np.zeros(48, np.float32)
    temp, tint = s.temperature / 100.0, s.tint / 100.0
    p[0] = 1.0 + 0.22 * temp - 0.05 * tint                   # red gain
    p[1] = 1.0 - 0.12 * tint                                 # green gain
    p[2] = 1.0 - 0.22 * temp - 0.05 * tint                   # blue gain
    p[3] = s.shadows / 100.0 * 0.30
    p[4] = s.highlights / 100.0 * 0.30
    p[5] = 1.0 + s.saturation / 100.0
    p[6] = s.vibrance / 100.0
    for i, w in enumerate((s.wheel_shadows, s.wheel_midtones, s.wheel_highlights)):
        r, g, b = _wheel_rgb(w)
        p[8 + i * 4: 11 + i * 4] = (r * 0.20, g * 0.20, b * 0.20)
        p[11 + i * 4] = w.luma / 100.0 * 0.20
    p[20] = s.vignette / 100.0
    p[21] = 0.15 + s.vignette_midpoint / 100.0 * 0.85       # where the falloff starts (radius)
    p[22] = s.vignette_roundness / 100.0
    p[23] = 0.05 + s.vignette_feather / 100.0 * 0.95
    p[24] = float(width) / max(height, 1)
    p[25] = s.lut_intensity / 100.0 if has_lut else 0.0
    return p


# --- the kernel -----------------------------------------------------------


@njit(cache=True, fastmath=True, inline="always")
def _tone(v, table):
    x = min(max(v, 0.0), 1.0) * (table.shape[0] - 1)
    i = int(x)
    if i >= table.shape[0] - 1:
        return table[table.shape[0] - 1]
    f = x - i
    return table[i] + f * (table[i + 1] - table[i])


@njit(cache=True, fastmath=True, inline="always")
def _lut3d(r, g, b, table, dmin, dmax, out):
    n = table.shape[0]
    fr = min(max((r - dmin[0]) / (dmax[0] - dmin[0]), 0.0), 1.0) * (n - 1)
    fg = min(max((g - dmin[1]) / (dmax[1] - dmin[1]), 0.0), 1.0) * (n - 1)
    fb = min(max((b - dmin[2]) / (dmax[2] - dmin[2]), 0.0), 1.0) * (n - 1)
    r0, g0, b0 = int(fr), int(fg), int(fb)
    r1, g1, b1 = min(r0 + 1, n - 1), min(g0 + 1, n - 1), min(b0 + 1, n - 1)
    dr, dg, db = fr - r0, fg - g0, fb - b0
    for c in range(3):
        c00 = table[b0, g0, r0, c] * (1 - dr) + table[b0, g0, r1, c] * dr
        c01 = table[b0, g1, r0, c] * (1 - dr) + table[b0, g1, r1, c] * dr
        c10 = table[b1, g0, r0, c] * (1 - dr) + table[b1, g0, r1, c] * dr
        c11 = table[b1, g1, r0, c] * (1 - dr) + table[b1, g1, r1, c] * dr
        out[c] = (c00 * (1 - dg) + c01 * dg) * (1 - db) + (c10 * (1 - dg) + c11 * dg) * db


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _grade_kernel(img, out, tone, p, lut, dmin, dmax, lut_before):
    height, width, _ = img.shape
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    aspect = p[24]
    lut_amount = p[25]
    do_vig = p[20] != 0.0
    for y in prange(height):
        tmp = np.empty(3, np.float32)
        for x in range(width):
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]

            if lut_amount > 0.0 and lut_before:
                _lut3d(r, g, b, lut, dmin, dmax, tmp)
                r += (tmp[0] - r) * lut_amount
                g += (tmp[1] - g) * lut_amount
                b += (tmp[2] - b) * lut_amount

            r, g, b = _tone(r, tone[0]), _tone(g, tone[1]), _tone(b, tone[2])
            r, g, b = r * p[0], g * p[1], b * p[2]

            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            # Smooth masks for the tonal ranges. Shadows fall off by mid-grey,
            # highlights start there, midtones peak in the middle.
            ws = min(max(1.0 - luma * 2.0, 0.0), 1.0)
            wh = min(max(luma * 2.0 - 1.0, 0.0), 1.0)
            ws, wh = ws * ws, wh * wh
            wm = 1.0 - ws - wh
            if wm < 0.0:
                wm = 0.0
            lift = p[3] * ws + p[4] * wh
            r += lift + p[8] * ws + p[12] * wm + p[16] * wh + (p[11] * ws + p[15] * wm + p[19] * wh)
            g += lift + p[9] * ws + p[13] * wm + p[17] * wh + (p[11] * ws + p[15] * wm + p[19] * wh)
            b += lift + p[10] * ws + p[14] * wm + p[18] * wh + (p[11] * ws + p[15] * wm + p[19] * wh)

            if p[5] != 1.0 or p[6] != 0.0:
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                s = p[5]
                if p[6] != 0.0:
                    # Vibrance backs off where colour is already strong, so
                    # skies lift and skin (moderate saturation) is left alone.
                    mx = max(r, max(g, b))
                    mn = min(r, min(g, b))
                    s *= 1.0 + p[6] * (1.0 - min(max(mx - mn, 0.0), 1.0))
                r = luma + (r - luma) * s
                g = luma + (g - luma) * s
                b = luma + (b - luma) * s

            if do_vig:
                dx = (x - cx) / max(cx, 1.0)
                dy = (y - cy) / max(cy, 1.0)
                # roundness 0 follows the frame's shape, 1 is a true circle.
                dx = dx * (1.0 + (aspect - 1.0) * p[22])
                d = np.sqrt(dx * dx + dy * dy) * 0.7071
                t = min(max((d - p[21] * 0.6) / p[23], 0.0), 1.0)
                t = t * t * (3.0 - 2.0 * t)
                k = 1.0 - p[20] * t if p[20] > 0.0 else 1.0 - p[20] * t * 0.6
                r, g, b = r * k, g * k, b * k

            if lut_amount > 0.0 and not lut_before:
                _lut3d(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0),
                       lut, dmin, dmax, tmp)
                r += (tmp[0] - r) * lut_amount
                g += (tmp[1] - g) * lut_amount
                b += (tmp[2] - b) * lut_amount

            out[y, x, 0] = min(max(r, 0.0), 1.0)
            out[y, x, 1] = min(max(g, 0.0), 1.0)
            out[y, x, 2] = min(max(b, 0.0), 1.0)


_NO_LUT = np.zeros((2, 2, 2, 3), np.float32)
_DMIN = np.zeros(3, np.float32)
_DMAX = np.ones(3, np.float32)


class GradeEngine:
    """Applies :class:`ColourSettings` to frames, caching the tables between calls."""

    def __init__(self) -> None:
        self._tone_key: tuple | None = None
        self._tone: np.ndarray | None = None
        self._lut_path = ""
        self._lut: lutmod.CubeLUT | None = None
        self.lut_error = ""

    def _load_lut(self, path: str) -> lutmod.CubeLUT | None:
        if path != self._lut_path:
            self._lut_path, self._lut, self.lut_error = path, None, ""
            if path:
                try:
                    self._lut = lutmod.load_cube(Path(path))
                except ValueError as error:
                    self.lut_error = str(error)
        return self._lut

    def apply(self, rgb: np.ndarray, s: ColourSettings, transfer: str = SDR) -> np.ndarray:
        """Float32 RGB in [0, 1] (encoded) in, float32 out. Settings that change nothing return the input."""
        lut = self._load_lut(s.lut_path if s.lut_active else "")
        if not s.grade_active and lut is None:
            return rgb
        if not s.grade_active:
            # LUT only: run the kernel with a neutral grade rather than a second code path.
            keep = ColourSettings(enabled=True, lut_enabled=True, lut_path=s.lut_path,
                                  lut_intensity=s.lut_intensity, lut_before=s.lut_before)
            s = keep
        rgb = np.ascontiguousarray(rgb, np.float32)
        height, width = rgb.shape[:2]

        key = (transfer, s.exposure, s.blacks, s.whites, s.contrast, s.faded_film,
               tuple(map(tuple, s.curve_master)), tuple(map(tuple, s.curve_red)),
               tuple(map(tuple, s.curve_green)), tuple(map(tuple, s.curve_blue)))
        if key != self._tone_key or self._tone is None:
            self._tone = tone_tables(s, transfer)
            self._tone_key = key

        params = _params(s, width, height, lut is not None)
        out = np.empty_like(rgb)
        with LOCK:
            _grade_kernel(
                rgb, out, self._tone, params,
                lut.table if lut is not None else _NO_LUT,
                lut.domain_min if lut is not None else _DMIN,
                lut.domain_max if lut is not None else _DMAX,
                bool(s.lut_before),
            )
        if s.sharpen:
            amount = s.sharpen / 100.0 * 1.6
            radius = max(0.6, width / 1600.0)
            blurred = cv2.GaussianBlur(out, (0, 0), radius)
            out = np.clip(out + amount * (out - blurred), 0.0, 1.0)
        return out
