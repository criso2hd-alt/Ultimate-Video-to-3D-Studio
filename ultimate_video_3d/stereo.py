"""Synthesise a stereo pair from one image and its depth map.

This is depth-image-based rendering (DIBR): every source pixel is moved
sideways by an amount proportional to how near it is, once for each eye, in
opposite directions. Near things slide further than far things, and the brain
reads that difference as depth.

Why a forward warp, and why a scanline kernel
---------------------------------------------
The cheap way is a backward warp (``cv2.remap``: for each output pixel, look
up where it came from). It is fast and wrong at exactly the places that matter -
the edges of near objects - where it smears foreground colour into the
background that has just been revealed. Moving pixels *forward* instead, with a
z-buffer so nearer pixels win, keeps edges clean, and leaves honest holes where
the background was uncovered. Those holes are then filled from the *background*
side, never the foreground's.

The kernel is compiled with Numba and runs one image row per CPU thread. Rows
are independent (disparity is purely horizontal), so it scales linearly with
cores and needs no GPU - which is what lets one implementation serve NVIDIA, AMD,
Intel and Apple silicon alike. A 1080p pair takes a few milliseconds.

Sign convention: depth is 1.0 at the nearest point and 0.0 at the farthest, the
layout Depth Anything produces. A pixel nearer than the convergence plane is *in
front of the screen*, so the left eye sees it displaced to the right.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Numba compiles on first use and caches the machine code. A frozen app's own
# folder is read-only on macOS and possibly on Windows, so point the cache at the
# user data dir before numba is imported.
if "NUMBA_CACHE_DIR" not in os.environ:
    from . import paths

    try:
        os.environ["NUMBA_CACHE_DIR"] = str(paths.data_dir() / "numba_cache")
    except Exception:  # noqa: BLE001 - a missing cache only costs start-up time
        pass

import cv2
from numba import njit, prange

from .kernels import LOCK


@dataclass
class StereoParams:
    #: 0..100. How much the eyes are pushed apart: the total parallax between the
    #: farthest and nearest point, as a fraction of image width (100 = 5%).
    strength: float = 50.0
    #: 0..1. Where the screen sits in the depth range. 0 puts everything in
    #: front of the screen, 1 behind it. Around 0.55 keeps most of the scene
    #: behind the screen with only the nearest objects reaching out.
    convergence: float = 0.55
    #: Extra multipliers on the parts of the scene in front of / behind the screen.
    near_scale: float = 1.0
    far_scale: float = 1.0
    #: Swap the eyes (cross-eyed viewing, or a player that expects right-first).
    swap_eyes: bool = False

    def max_disparity_px(self, width: int) -> float:
        return self.strength / 100.0 * 0.05 * width


# --- the kernel ------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _warp(src, depth, shift, out, hole, edge_thresh):
    """Forward-warp ``src`` by per-pixel ``shift`` (pixels), z-tested by ``depth``.

    ``out`` receives the image, ``hole`` marks pixels that had to be invented.
    """
    height, width, _ = src.shape
    for y in prange(height):
        zbuf = np.full(width, -1e9, np.float32)
        filled = np.zeros(width, np.uint8)

        for x in range(width):
            s0 = shift[y, x]
            z0 = depth[y, x]
            t0 = x + s0
            tx = int(np.floor(t0 + 0.5))
            if 0 <= tx < width and z0 >= zbuf[tx]:
                zbuf[tx] = z0
                filled[tx] = 1
                for c in range(3):
                    out[y, tx, c] = src[y, x, c]

            if x + 1 >= width:
                continue
            s1 = shift[y, x + 1]
            # A large jump in shift between neighbours is an object edge: do not
            # smear one across the other, leave the gap for the hole filler.
            if abs(s1 - s0) > edge_thresh:
                continue
            t1 = x + 1 + s1
            lo = int(np.ceil(min(t0, t1)))
            hi = int(np.floor(max(t0, t1)))
            span = t1 - t0
            if hi < lo or abs(span) < 1e-6:
                continue
            z1 = depth[y, x + 1]
            for tx in range(max(lo, 0), min(hi, width - 1) + 1):
                f = (tx - t0) / span
                if f < 0.0 or f > 1.0:
                    continue
                z = z0 + f * (z1 - z0)
                if z >= zbuf[tx]:
                    zbuf[tx] = z
                    filled[tx] = 1
                    for c in range(3):
                        out[y, tx, c] = src[y, x, c] + f * (src[y, x + 1, c] - src[y, x, c])

        # Fill each hole from its background side by mirroring the texture there,
        # which reads far more naturally than stretching a single edge pixel.
        x = 0
        while x < width:
            if filled[x]:
                x += 1
                continue
            a = x
            while x < width and not filled[x]:
                x += 1
            b = x - 1                       # hole is [a, b]
            have_left = a > 0
            have_right = b < width - 1
            use_left = have_left
            if have_left and have_right:
                use_left = zbuf[a - 1] <= zbuf[b + 1]      # the farther side is background
            elif not have_left and not have_right:
                for tx in range(a, b + 1):
                    hole[y, tx] = 1
                continue
            length = b - a + 1
            for i in range(length):
                if use_left:
                    sx = a - 1 - i
                    if sx < 0:
                        sx = 0
                    dx = a + i
                else:
                    sx = b + 1 + i
                    if sx > width - 1:
                        sx = width - 1
                    dx = b - i
                for c in range(3):
                    out[y, dx, c] = out[y, sx, c]
                hole[y, dx] = 1


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _disparity(depth, out, scale, convergence, near_scale, far_scale):
    height, width = depth.shape
    for y in prange(height):
        for x in range(width):
            v = depth[y, x] - convergence
            v = v * near_scale if v > 0.0 else v * far_scale
            out[y, x] = v * scale


def disparity_map(depth: np.ndarray, params: StereoParams, width: int) -> np.ndarray:
    """Signed full-range disparity in pixels: positive = in front of the screen."""
    out = np.empty(depth.shape, np.float32)
    _disparity(np.ascontiguousarray(depth, np.float32), out,
               np.float32(params.max_disparity_px(width)), np.float32(params.convergence),
               np.float32(params.near_scale), np.float32(params.far_scale))
    return out


@dataclass
class Pair:
    left: np.ndarray
    right: np.ndarray
    hole_left: np.ndarray       # uint8, 1 where the pixel was invented
    hole_right: np.ndarray
    disparity: np.ndarray       # the signed full-range disparity used, in pixels
    source: np.ndarray          # the unshifted picture the pair was made from


def synthesize(rgb: np.ndarray, depth: np.ndarray, params: StereoParams) -> Pair:
    """Left and right views of ``rgb`` (float32 HxWx3) given ``depth`` (HxW, 1 = near).

    Each eye is moved half the disparity, in opposite directions, so the centre
    of the depth range stays put and neither eye carries the whole distortion.
    """
    rgb = np.ascontiguousarray(rgb, np.float32)
    depth = np.ascontiguousarray(depth, np.float32)
    height, width = depth.shape
    edge = max(0.75, 0.0006 * width)   # a per-pixel disparity step this large is an object edge, not a slope
    outs = []
    with LOCK:
        disp = disparity_map(depth, params, width)
        for sign in (+0.5, -0.5):
            out = np.zeros_like(rgb)
            hole = np.zeros((height, width), np.uint8)
            _warp(rgb, depth, np.ascontiguousarray(disp * np.float32(sign)), out, hole, np.float32(edge))
            outs.append((out, hole))
    (left, hole_l), (right, hole_r) = outs
    if params.swap_eyes:
        left, right, hole_l, hole_r = right, left, hole_r, hole_l
    return Pair(left, right, hole_l, hole_r, disp, rgb)


# --- output layouts ----------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    key: str
    label: str
    tag: str                 # filename tag players use to detect the layout
    note: str
    stereo_mode: str = ""    # Matroska StereoMode value
    two_eyes: bool = True


LAYOUTS: tuple[Layout, ...] = (
    Layout("sbs_half", "Side-by-Side (Half)", "_SBS",
           "Both eyes squeezed into one frame's width. Works on nearly every 3D TV, VR player and headset.",
           "left_right"),
    Layout("sbs_full", "Side-by-Side (Full)", "_FSBS",
           "Full resolution per eye - twice the frame width. Best quality for VR headsets.", "left_right"),
    Layout("tb_half", "Top-and-Bottom (Half)", "_TB",
           "Both eyes stacked, each at half height. Common on 3D TVs and projectors.", "top_bottom"),
    Layout("tb_full", "Top-and-Bottom (Full)", "_FTB",
           "Full resolution per eye - twice the frame height.", "top_bottom"),
    Layout("anaglyph", "Anaglyph (red / cyan)", "_anaglyph",
           "Watch with cheap red/cyan glasses on any screen. Lower colour fidelity.", "anaglyph_cyan_red"),
    Layout("rgbd", "Colour + Depth (2D+Z)", "_RGBD",
           "Picture beside its depth map, for glasses-free displays and other 3D tools."),
    Layout("depth", "Depth map only", "_depth",
           "A grey depth video, near = white. For compositing and other software.", two_eyes=False),
)
LAYOUTS_BY_KEY = {layout.key: layout for layout in LAYOUTS}

# Dubois least-squares red/cyan matrices: rows are output R,G,B from left-eye
# RGB (first three columns) and right-eye RGB (last three).
_DUBOIS = np.array([
    [0.456, 0.500, 0.176, -0.043, -0.088, -0.002],
    [-0.040, -0.038, -0.016, 0.378, 0.734, -0.018],
    [-0.015, -0.021, -0.005, -0.072, -0.113, 1.226],
], np.float32)


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both = np.concatenate([left, right], axis=-1)
    return np.clip(both @ _DUBOIS.T, 0.0, 1.0)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def compose(layout_key: str, pair: Pair, depth: np.ndarray) -> np.ndarray:
    """Pack a stereo pair (or depth) into one output frame for ``layout_key``."""
    left, right = pair.left, pair.right
    height, width = left.shape[:2]
    if layout_key == "sbs_full":
        return np.concatenate([left, right], axis=1)
    if layout_key == "sbs_half":
        half = (width // 2, height)
        return np.concatenate([_resize(left, half), _resize(right, half)], axis=1)
    if layout_key == "tb_full":
        return np.concatenate([left, right], axis=0)
    if layout_key == "tb_half":
        half = (width, height // 2)
        return np.concatenate([_resize(left, half), _resize(right, half)], axis=0)
    if layout_key == "anaglyph":
        return anaglyph(left, right)
    grey = np.repeat(np.clip(depth, 0.0, 1.0)[..., None], 3, axis=-1)
    if layout_key == "rgbd":
        half = (width // 2, height)
        return np.concatenate([_resize(pair.source, half), _resize(grey, half)], axis=1)
    if layout_key == "depth":
        return grey
    raise ValueError(f"Unknown 3D layout {layout_key!r}")


def output_size(layout_key: str, width: int, height: int) -> tuple[int, int]:
    """Frame size the chosen layout produces from a ``width`` x ``height`` source."""
    return {
        "sbs_full": (width * 2, height), "tb_full": (width, height * 2),
    }.get(layout_key, (width, height))


# --- packing for the encoder --------------------------------------------------

_MODES = {"copy": 0, "sbs_half": 1, "sbs_full": 2, "tb_half": 3, "tb_full": 4, "anaglyph": 5,
          "rgbd": 6, "depth": 7}


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _pack(left, right, source, depth, mode, dubois, out, top):
    """Lay the eyes out for one output layout and quantise to integers, in one pass.

    ``out`` is uint8 or uint16 (the caller picks by bit depth); ``top`` is its
    full-scale value. Doing the layout, the half-size squeeze and the conversion
    together avoids three full-frame float temporaries per frame.
    """
    oh, ow, _ = out.shape
    h, w, _ = left.shape
    for y0 in prange(oh):
        y = np.int64(y0)                     # prange indices are unsigned; mixing them with ints makes floats
        for x in range(ow):
            r = g = b = 0.0
            if mode == 0:
                r, g, b = left[y, x, 0], left[y, x, 1], left[y, x, 2]
            elif mode == 1:                                    # sbs half
                eye = right if x >= ow // 2 else left
                sx = (x - ow // 2 if x >= ow // 2 else x) * 2
                for c in range(3):
                    v = 0.5 * (eye[y, sx, c] + eye[y, min(sx + 1, w - 1), c])
                    if c == 0:
                        r = v
                    elif c == 1:
                        g = v
                    else:
                        b = v
            elif mode == 2:                                    # sbs full
                eye = right if x >= w else left
                sx = x - w if x >= w else x
                r, g, b = eye[y, sx, 0], eye[y, sx, 1], eye[y, sx, 2]
            elif mode == 3:                                    # tb half
                eye = right if y >= oh // 2 else left
                sy = (y - oh // 2 if y >= oh // 2 else y) * 2
                for c in range(3):
                    v = 0.5 * (eye[sy, x, c] + eye[min(sy + 1, h - 1), x, c])
                    if c == 0:
                        r = v
                    elif c == 1:
                        g = v
                    else:
                        b = v
            elif mode == 4:                                    # tb full
                eye = right if y >= h else left
                sy = y - h if y >= h else y
                r, g, b = eye[sy, x, 0], eye[sy, x, 1], eye[sy, x, 2]
            elif mode == 5:                                    # anaglyph
                l0, l1, l2 = left[y, x, 0], left[y, x, 1], left[y, x, 2]
                r0, r1, r2 = right[y, x, 0], right[y, x, 1], right[y, x, 2]
                r = dubois[0, 0] * l0 + dubois[0, 1] * l1 + dubois[0, 2] * l2 + dubois[0, 3] * r0 + dubois[0, 4] * r1 + dubois[0, 5] * r2
                g = dubois[1, 0] * l0 + dubois[1, 1] * l1 + dubois[1, 2] * l2 + dubois[1, 3] * r0 + dubois[1, 4] * r1 + dubois[1, 5] * r2
                b = dubois[2, 0] * l0 + dubois[2, 1] * l1 + dubois[2, 2] * l2 + dubois[2, 3] * r0 + dubois[2, 4] * r1 + dubois[2, 5] * r2
            elif mode == 6:                                    # rgb + depth
                if x < ow // 2:
                    sx = x * 2
                    for c in range(3):
                        v = 0.5 * (source[y, sx, c] + source[y, min(sx + 1, w - 1), c])
                        if c == 0:
                            r = v
                        elif c == 1:
                            g = v
                        else:
                            b = v
                else:
                    sx = (x - ow // 2) * 2
                    r = g = b = 0.5 * (depth[y, sx] + depth[y, min(sx + 1, w - 1)])
            else:                                              # depth only
                r = g = b = depth[y, x]
            out[y, x, 0] = min(max(r, 0.0), 1.0) * top + 0.5
            out[y, x, 1] = min(max(g, 0.0), 1.0) * top + 0.5
            out[y, x, 2] = min(max(b, 0.0), 1.0) * top + 0.5


def pack_frame(layout_key: str, pair: Pair | None, rgb: np.ndarray, depth: np.ndarray | None,
               bits: int = 8) -> np.ndarray:
    """The frame the encoder wants: laid out for ``layout_key``, as uint8 (8) or uint16 (16 -> 10-bit).

    ``pair`` is None when 3D is off, in which case ``rgb`` is just quantised.
    """
    mode = _MODES["copy" if pair is None else layout_key]
    height, width = rgb.shape[:2]
    out_w, out_h = (width, height) if pair is None else output_size(layout_key, width, height)
    dtype, top = (np.uint8, 255.0) if bits <= 8 else (np.uint16, 65535.0)
    out = np.empty((out_h, out_w, 3), dtype)
    left = pair.left if pair is not None else rgb
    right = pair.right if pair is not None else rgb
    d = depth if depth is not None else np.zeros((height, width), np.float32)
    with LOCK:
        _pack(np.ascontiguousarray(left), np.ascontiguousarray(right), np.ascontiguousarray(rgb),
              np.ascontiguousarray(d, np.float32), mode, _DUBOIS, out, np.float32(top))
    return out
