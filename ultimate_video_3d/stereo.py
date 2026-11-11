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

SLIVER = 2          # a run of picture this short between two holes is edge fringe, not picture
REACH = 48          # how far into the background the hole filler looks for texture to reflect
#: Steep depth slopes narrower than this share of the picture width are object edges, made into a clean step before the
#: warp (``_steps``); and this many pixels of background next to an uncovered gap are replaced too (``_warp``'s ``erode``).
STEP_RADIUS = 1.0 / 320.0
STEP_MIN_PX = 4.0   # a slope must change the disparity by more than this (full, both eyes) inside the window to count as an edge
ERODE = 1.0 / 480.0


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _steps(depth, disp, depth_out, disp_out, radius, min_jump):
    """Turn steep, soft depth slopes into clean steps, row by row (disparity is horizontal).

    The depth model works at 518 px and the cleanup at 640 px, so at 1080p an object's depth edge is a slope ~12 px wide
    with a small stair every few rows, while the picture's own edge is sharp. Warped as it was, the slope stretched the
    object's rim across the uncovered background by a different amount on every row: a saw-tooth fringe along every edge
    on the side that opens up (reported on Discord on a bright sky behind a dark-rimmed disc). Here each pixel on a slope
    that changes the disparity by more than ``min_jump`` within ``radius`` takes the nearer of the local minimum and
    maximum, so the object moves as one piece and the gap is left whole for the hole filler. Gentle slopes (floors,
    walls) change far less than that inside the window and are left alone, and so is fine detail (see below).
    """
    height, width = depth.shape
    for y0 in prange(height):
        y = np.int64(y0)
        for x in range(width):
            lo_i = max(0, x - radius)
            hi_i = min(width - 1, x + radius)
            if abs(disp[y, hi_i] - disp[y, lo_i]) < 0.5 * min_jump:
                disp_out[y, x] = disp[y, x]                 # flat or gently sloping here: nearly every pixel leaves here
                depth_out[y, x] = depth[y, x]
                continue
            dmin = disp[y, x]
            dmax = disp[y, x]
            zmin = depth[y, x]
            zmax = depth[y, x]
            for k in range(lo_i, hi_i + 1):
                v = disp[y, k]
                if v < dmin:
                    dmin = v
                    zmin = depth[y, k]
                if v > dmax:
                    dmax = v
                    zmax = depth[y, k]
            v = disp[y, x]
            # Only a single smooth slope: one way all through the window (small wobbles allowed). Hair, leaves and other
            # fine detail zig-zag in depth; made into hard steps they broke up into dark speckles and stripes.
            steep = dmax - dmin > min_jump
            if steep:
                rising = disp[y, hi_i] >= disp[y, lo_i]
                wobble = 0.15 * (dmax - dmin)
                for k in range(lo_i, hi_i):
                    step = disp[y, k + 1] - disp[y, k]
                    if (rising and step < -wobble) or ((not rising) and step > wobble):
                        steep = False
                        break
            if steep:
                if v - dmin < dmax - v:
                    disp_out[y, x] = dmin
                    depth_out[y, x] = zmin
                else:
                    disp_out[y, x] = dmax
                    depth_out[y, x] = zmax
            else:
                disp_out[y, x] = v
                depth_out[y, x] = depth[y, x]


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _warp(src, depth, shift, out, hole, edge_thresh, erode):
    """Forward-warp ``src`` by per-pixel ``shift`` (pixels), z-tested by ``depth``.

    ``out`` receives the image, ``hole`` marks pixels that had to be invented. ``erode`` pixels of background next to
    an uncovered gap are replaced as well: they are usually the object's blended rim, left behind with background depth.
    """
    height, width, _ = src.shape
    for y in prange(height):
        zbuf = np.full(width, -1e9, np.float32)
        filled = np.zeros(width, np.uint8)
        hole[y, :] = 0                       # ``out`` and ``hole`` may be re-used from the last frame: every pixel is rewritten here

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

        # A sliver of picture (one or two pixels) squeezed between two holes is not picture, it is the
        # blended edge of an object that moved away: the filler used to take it for background and
        # repeat it as a stripe. Treat it as part of the hole.
        x = 0
        while x < width:
            if not filled[x]:
                x += 1
                continue
            a = x
            while x < width and filled[x]:
                x += 1
            if x - a <= SLIVER and a > 0 and x < width:        # [a, x) is a short run with a hole on each side
                for tx in range(a, x):
                    filled[tx] = 0

        # Fill each hole from its background side with the texture there, reflected back and forth
        # inside the run of real picture next to the hole. It never reads another hole (which is
        # still black at this point: copying from one was the black dots in the right eye), and it
        # reads far more naturally than stretching a single edge pixel.
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
                    for c in range(3):
                        out[y, tx, c] = 0.0
                continue
            if have_left and have_right and b - a + 1 >= erode and abs(zbuf[a - 1] - zbuf[b + 1]) > 0.02:
                # (only a real opening: the tiny gaps between strands of hair or leaves are left alone, or fine detail breaks up)
                # A real uncovered gap: its first background pixels are mostly the moved object's soft rim, which
                # kept background depth. Copied as background they become a dark ghost of the edge; replace them too.
                # (At least two pixels of background are kept to take texture from.)
                k = 0
                if use_left:
                    while k < erode and a - 3 >= 0 and filled[a - 2] and filled[a - 3]:
                        a -= 1
                        k += 1
                else:
                    while k < erode and b + 3 < width and filled[b + 2] and filled[b + 3]:
                        b += 1
                        k += 1
            length = b - a + 1
            # how many pixels of real picture lie beyond the hole on the background side
            run = 0
            if use_left:
                while a - 1 - run >= 0 and filled[a - 1 - run] and run < REACH:
                    run += 1
            else:
                while b + 1 + run < width and filled[b + 1 + run] and run < REACH:
                    run += 1
            for i in range(length):
                j = i % (2 * run)
                if j >= run:
                    j = 2 * run - 1 - j                    # bounce back instead of running into the next hole
                if use_left:
                    sx = a - 1 - j
                    dx = a + i
                else:
                    sx = b + 1 + j
                    dx = b - i
                for c in range(3):
                    out[y, dx, c] = out[y, sx, c]
                hole[y, dx] = 1
        # The reflected fill is smoothed a little across the hole, so a high-contrast edge that was
        # reflected does not show as a pattern; only invented pixels are touched.
        for x in range(1, width - 1):
            if hole[y, x]:
                for c in range(3):
                    out[y, x, c] = (out[y, x - 1, c] + 2.0 * out[y, x, c] + out[y, x + 1, c]) * 0.25


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


class WarpBuffers:
    """Output pictures for :func:`synthesize` that are kept from one frame to the next.

    A fresh 1080p float picture is 25 MB and the operating system has to hand over zeroed memory for each one: for
    the two eyes that was 15 ms a frame, more than the warp itself (measured), and it kept every core busy doing
    nothing useful. A conversion packs each pair at once and never keeps it, so it can use the same two pictures
    every time. The pair returned *is* these buffers: use it before the next call.
    """

    def __init__(self) -> None:
        self._shape: tuple | None = None
        self._pictures: list[np.ndarray] = []
        self._holes: list[np.ndarray] = []

    def get(self, shape: tuple) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._shape != shape:
            self._pictures = [np.empty(shape, np.float32) for _ in range(2)]
            self._holes = [np.empty(shape[:2], np.uint8) for _ in range(2)]
            self._shape = shape
        return self._pictures, self._holes


def synthesize(rgb: np.ndarray, depth: np.ndarray, params: StereoParams, buffers: WarpBuffers | None = None) -> Pair:
    """Left and right views of ``rgb`` (float32 HxWx3) given ``depth`` (HxW, 1 = near).

    Each eye is moved half the disparity, in opposite directions, so the centre
    of the depth range stays put and neither eye carries the whole distortion.
    """
    rgb = np.ascontiguousarray(rgb, np.float32)
    depth = np.ascontiguousarray(depth, np.float32)
    height, width = depth.shape
    edge = max(0.75, 0.0006 * width)   # a per-pixel disparity step this large is an object edge, not a slope
    outs = []
    pictures, holes = buffers.get(rgb.shape) if buffers is not None else (None, None)
    radius = max(2, int(round(width * STEP_RADIUS)))
    erode = max(1, int(round(width * ERODE)))
    with LOCK:
        disp = disparity_map(depth, params, width)
        stepped_depth, stepped = np.empty_like(depth), np.empty_like(disp)
        _steps(depth, disp, stepped_depth, stepped, radius, np.float32(STEP_MIN_PX))
        for i, sign in enumerate((+0.5, -0.5)):
            out = pictures[i] if pictures is not None else np.empty_like(rgb)
            hole = holes[i] if holes is not None else np.empty((height, width), np.uint8)
            _warp(rgb, stepped_depth, np.ascontiguousarray(stepped * np.float32(sign)), out, hole, np.float32(edge), erode)
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


# --- straight to the encoder's own pixel format ---------------------------------


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _rgb_to_yuv420(rgb, out, kr, kb):
    """8-bit full-range RGB -> planar 4:2:0 YCbCr in limited ("TV") range, all cores, one pass.

    ``out`` is the three planes back to back (Y, then U, then V); ``kr`` and ``kb`` pick the matrix (0.2126/0.0722
    for BT.709). Chroma is the mean of each 2x2 block.
    """
    h, w, _ = rgb.shape
    hh, hw = h // 2, w // 2
    u0 = h * w
    v0 = u0 + hh * hw
    kg = 1.0 - kr - kb
    cb_scale = 224.0 / (255.0 * 2.0 * (1.0 - kb))
    cr_scale = 224.0 / (255.0 * 2.0 * (1.0 - kr))
    y_scale = 219.0 / 255.0
    for by0 in prange(hh):
        by = np.int64(by0)
        for bx in range(hw):
            cb = 0.0
            cr = 0.0
            for dy in range(2):
                for dx in range(2):
                    y = by * 2 + dy
                    x = bx * 2 + dx
                    r = np.float32(rgb[y, x, 0])
                    g = np.float32(rgb[y, x, 1])
                    b = np.float32(rgb[y, x, 2])
                    luma = kr * r + kg * g + kb * b
                    out[y * w + x] = min(max(16.0 + y_scale * luma + 0.5, 0.0), 255.0)
                    cb += (b - luma) * cb_scale
                    cr += (r - luma) * cr_scale
            out[u0 + by * hw + bx] = min(max(128.0 + 0.25 * cb + 0.5, 0.0), 255.0)
            out[v0 + by * hw + bx] = min(max(128.0 + 0.25 * cr + 0.5, 0.0), 255.0)


def rgb_to_yuv420(rgb: np.ndarray, bt2020: bool = False) -> np.ndarray:
    """``rgb`` (uint8 HxWx3, even sides) as an I420 buffer shaped (H*3/2, W), ready for ``VideoFrame.from_ndarray``.

    This replaces swscale's single-threaded RGB -> YUV conversion (20 ms a 1080p frame, measured) with a kernel that
    uses every core (a few ms); the encoder then gets the planes it wants with no further conversion.
    """
    h, w, _ = rgb.shape
    out = np.empty(h * w * 3 // 2, np.uint8)
    kr, kb = (0.2627, 0.0593) if bt2020 else (0.2126, 0.0722)
    with LOCK:
        _rgb_to_yuv420(np.ascontiguousarray(rgb), out, np.float32(kr), np.float32(kb))
    return out.reshape(h * 3 // 2, w)
