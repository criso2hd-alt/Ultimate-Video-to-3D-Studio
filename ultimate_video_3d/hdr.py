"""HDR video: keep it as HDR when asked, tone-map it cleanly when not.

The rule for the HDR path is that nothing is converted unless it has to be.

- **Keep HDR** (default for an HDR source going to H.265 / AV1 / ProRes): frames
  stay in their PQ or HLG *encoded* signal from decode to encode. The stereo warp
  is geometric and the grade works on the encoded signal, so neither needs the
  picture in linear light. The output is written 10-bit with BT.2020 + PQ/HLG
  tags, so a player treats it as the HDR it is.
- **Tone-map** happens only in three places: the depth model's input (it was
  trained on ordinary pictures), the on-screen preview (the display is SDR), and
  an SDR export of an HDR source. All three use the same operator below.

The operator is identity below a knee and extended Reinhard above it, in BT.709
after a proper BT.2020 gamut conversion. Working on luminance keeps hue and
saturation where they were, which per-channel tone mapping does not.
"""

from __future__ import annotations

import os

import numpy as np

if "NUMBA_CACHE_DIR" not in os.environ:
    from . import paths

    try:
        os.environ["NUMBA_CACHE_DIR"] = str(paths.data_dir() / "numba_cache")
    except Exception:  # noqa: BLE001
        pass

from numba import njit, prange

from .kernels import LOCK

#: Reference white for SDR-in-HDR: 203 nits (ITU-R BT.2408).
REFERENCE_WHITE_NITS = 203.0

#: Luminance (1.0 = diffuse white) below which the tone map leaves the picture alone.
KNEE = 0.85

# BT.2020 -> BT.709 primaries, linear light.
_M2020_TO_709 = np.array([
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
], np.float32)

_M1, _M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
_C1, _C2, _C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
_HA, _HB, _HC = 0.17883277, 0.28466892, 0.55991073


@njit(cache=True, fastmath=True, inline="always")
def _pq_to_nits(v):
    p = min(max(v, 0.0), 1.0) ** (1.0 / _M2)
    num = max(p - _C1, 0.0)
    return 10000.0 * (num / (_C2 - _C3 * p)) ** (1.0 / _M1)


@njit(cache=True, fastmath=True, inline="always")
def _hlg_to_nits(v):
    v = min(max(v, 0.0), 1.0)
    e = v * v / 3.0 if v <= 0.5 else (np.exp((v - _HC) / _HA) + _HB) / 12.0
    # HLG's display OOTF at 1000 nits peak: system gamma 1.2.
    return 1000.0 * e ** 1.2


@njit(cache=True, fastmath=True, inline="always")
def _srgb_encode(v):
    v = min(max(v, 0.0), 1.0)
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1.0 / 2.4) - 0.055


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _tonemap(img, out, matrix, is_hlg, white, exposure):
    height, width, _ = img.shape
    for y in prange(height):
        for x in range(width):
            if is_hlg:
                r = _hlg_to_nits(img[y, x, 0])
                g = _hlg_to_nits(img[y, x, 1])
                b = _hlg_to_nits(img[y, x, 2])
            else:
                r = _pq_to_nits(img[y, x, 0])
                g = _pq_to_nits(img[y, x, 1])
                b = _pq_to_nits(img[y, x, 2])
            # Nits -> "SDR white = 1.0", then into BT.709 primaries.
            r, g, b = r / 203.0 * exposure, g / 203.0 * exposure, b / 203.0 * exposure
            r2 = matrix[0, 0] * r + matrix[0, 1] * g + matrix[0, 2] * b
            g2 = matrix[1, 0] * r + matrix[1, 1] * g + matrix[1, 2] * b
            b2 = matrix[2, 0] * r + matrix[2, 1] * g + matrix[2, 2] * b
            r2, g2, b2 = max(r2, 0.0), max(g2, 0.0), max(b2, 0.0)
            luma = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
            # Extended Reinhard on luminance: 1.0 stays near 1.0, `white` maps to 1.0.
            scale = 1.0
            if luma > KNEE:
                # Identity up to the knee, then extended Reinhard on what is above it, so
                # ordinary SDR-level content is untouched and only highlights compress.
                room = 1.0 - KNEE
                over = (luma - KNEE) / room
                peak = (white - KNEE) / room
                mapped = KNEE + room * over * (1.0 + over / (peak * peak)) / (1.0 + over)
                scale = mapped / luma
            out[y, x, 0] = _srgb_encode(r2 * scale)
            out[y, x, 1] = _srgb_encode(g2 * scale)
            out[y, x, 2] = _srgb_encode(b2 * scale)


def tonemap_to_sdr(rgb: np.ndarray, transfer: str, peak_nits: float = 1000.0,
                   exposure: float = 1.0) -> np.ndarray:
    """PQ / HLG encoded BT.2020 float RGB (0..1) -> display-referred BT.709 sRGB (0..1)."""
    rgb = np.ascontiguousarray(rgb, np.float32)
    out = np.empty_like(rgb)
    white = max(peak_nits / REFERENCE_WHITE_NITS, 1.0)
    with LOCK:
        _tonemap(rgb, out, _M2020_TO_709, transfer == "hlg", np.float32(white), np.float32(exposure))
    return out
