"""Lay the back-wall logo onto the wall's baked textures, using the logo's real transparency.

    python scripts/blender/composite_logo.py <bake_dir>

Why: a Cycles bake cannot bake a see-through sticker (the transparent parts come out black and hide whatever is behind), so the
logo is baked as a fully opaque card in the chair atlas (bake_room.py, group chairs_details) and the wall is baked on its own.
This step reads ``logo_info.json`` (written by ``bake_room.py --phase logoinfo``), and for every texel of the wall atlas under
the logo blends the sticker's baked colour over the wall's baked colour by the PNG's alpha. The logo object is then left out of
room.glb. The glow the sticker casts on the wall is already in the wall bake. All three bakes are done (lit, dim, screen-only; the
screen-only ones each carry their own exposure, which is undone and redone here).

Safe to run twice: the first run keeps ``*.orig.jpg`` copies of the wall atlases and every run starts from them.
Needs numpy and OpenCV (the app's own environment has both).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main() -> int:
    out = Path(sys.argv[1])
    info = json.loads((out / "logo_info.json").read_text())
    corners = np.array(info["logo"]["corners"], np.float64)            # (4, 3), world
    bake_uv = np.array(info["logo"]["bake_uv"], np.float64)            # where the logo sits in the chair atlas
    png_uv = np.array(info["logo"]["png_uv"], np.float64)              # where it sits in the PNG
    tris = info["wall"]["tris"]

    # the wall is one flat island: UV is an affine function of (x, z). Fit and check it.
    pos = np.array([v for t in tris for v in t["pos"]], np.float64)
    uvs = np.array([v for t in tris for v in t["uv"]], np.float64)
    design = np.column_stack([pos[:, 0], pos[:, 2], np.ones(len(pos))])
    coeff, *_ = np.linalg.lstsq(design, uvs, rcond=None)               # (3, 2)
    residual = float(np.abs(design @ coeff - uvs).max())
    if residual > 1e-4:
        raise SystemExit(f"the wall is not one flat UV island (fit error {residual:.5f}); cannot composite")
    a_mat = coeff[:2].T                                                # uv = a_mat @ [x, z] + b
    b_vec = coeff[2]
    inverse = np.linalg.inv(a_mat)

    e1 = corners[1] - corners[0]
    e3 = corners[3] - corners[0]
    e1, e3 = e1[[0, 2]], e3[[0, 2]]
    origin = corners[0][[0, 2]]
    if abs(float(e1 @ e3)) > 1e-3 * np.linalg.norm(e1) * np.linalg.norm(e3):
        raise SystemExit("the logo card is not a rectangle in the wall's plane")

    png = cv2.imread(info["png"], cv2.IMREAD_UNCHANGED)
    if png is None or png.shape[2] < 4:
        raise SystemExit(f"{info['png']} has no alpha channel")
    alpha_full = png[:, :, 3].astype(np.float32) / (65535.0 if png.dtype == np.uint16 else 255.0)
    alpha_png = cv2.resize(alpha_full, (640, 640), interpolation=cv2.INTER_AREA)       # a light prefilter: the wall texels are coarser

    for state in ("lit", "dim", "screen"):
        wall_path, wall_orig = out / f"walls_plaster_{state}.jpg", out / f"walls_plaster_{state}.orig.jpg"
        chair_path = out / f"chairs_details_{state}.jpg"
        if not wall_orig.exists():
            shutil.copy2(wall_path, wall_orig)
        wall = read(wall_orig)
        chair = read(chair_path)
        wall_h, wall_w = wall.shape[:2]
        chair_h, chair_w = chair.shape[:2]
        ev_wall = ev_chair = 0.0
        if state == "screen":
            ev_wall, ev_chair = info["screen_ev"]["walls_plaster"], info["screen_ev"]["chairs_details"]

        # the texels of the wall atlas that lie under the logo
        corner_uv = (a_mat @ corners[:, [0, 2]].T).T + b_vec
        lo = np.floor((corner_uv.min(axis=0) - 0.004) * [wall_w, wall_h]).astype(int)
        hi = np.ceil((corner_uv.max(axis=0) + 0.004) * [wall_w, wall_h]).astype(int)
        cols = np.arange(max(lo[0], 0), min(hi[0], wall_w))
        rows_v = np.arange(max(wall_h - hi[1], 0), min(wall_h - lo[1], wall_h))      # image rows run top-down, v runs bottom-up
        if cols.size == 0 or rows_v.size == 0:
            raise SystemExit("the logo is outside the wall atlas")
        cc, rr = np.meshgrid(cols, rows_v)
        u = (cc + 0.5) / wall_w
        v = 1.0 - (rr + 0.5) / wall_h
        xz = (inverse @ (np.stack([u, v], axis=-1).reshape(-1, 2).T - b_vec[:, None])).T                # world (x, z)
        rel = xz - origin
        s = (rel @ e1) / (e1 @ e1)
        t = (rel @ e3) / (e3 @ e3)
        inside = ((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)).reshape(cc.shape)
        s, t = s.reshape(cc.shape), t.reshape(cc.shape)

        def bilinear(c, s_, t_):
            return ((1 - s_)[..., None] * (1 - t_)[..., None] * c[0] + s_[..., None] * (1 - t_)[..., None] * c[1]
                    + s_[..., None] * t_[..., None] * c[2] + (1 - s_)[..., None] * t_[..., None] * c[3])

        pu = bilinear(png_uv, s, t)                                    # (h, w, 2) positions in the PNG
        cu = bilinear(bake_uv, s, t)                                   # positions in the chair atlas
        alpha = cv2.remap(alpha_png, (pu[..., 0] * 640 - 0.5).astype(np.float32), ((1 - pu[..., 1]) * 640 - 0.5).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sticker = cv2.remap(chair, (cu[..., 0] * chair_w - 0.5).astype(np.float32),
                            ((1 - cu[..., 1]) * chair_h - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        alpha = np.where(inside, alpha, 0.0)[..., None]

        region = wall[rows_v[0]:rows_v[-1] + 1, cols[0]:cols[-1] + 1]
        wall_lin = to_linear(region) / (2.0 ** ev_wall)
        sticker_lin = to_linear(sticker) / (2.0 ** ev_chair)
        blended = wall_lin * (1 - alpha) + sticker_lin * alpha
        result = wall.copy()
        result[rows_v[0]:rows_v[-1] + 1, cols[0]:cols[-1] + 1] = to_srgb(blended * (2.0 ** ev_wall))
        image = cv2.cvtColor((np.clip(result, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(wall_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95 if state == "screen" else 93])
        print(f"{state}: logo composited over {int(inside.sum())} wall texels (mean alpha {float(alpha[inside[..., None].repeat(1, -1)].mean()):.2f}), "
              f"wall atlas {wall_w}px, sticker atlas {chair_w}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
