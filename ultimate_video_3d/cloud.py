"""The 3D orbit view: the frame rebuilt as a point cloud, with the camera swinging around it.

The clearest proof that the depth is real is to leave the stereo pair behind and
look at the geometry itself. Each pixel is placed in space at its depth and the
camera sways left and right, so near things swing past far ones with true
perspective parallax. If the depth map is wrong - inverted, flat, or torn at an
edge - it is obvious here in a way a heat-map hides.

Pure numpy, no Qt, so it can render offline for tuning and tests.
"""

from __future__ import annotations

import numpy as np

#: Points across the widest edge. Sparse on purpose: a dense wall reads as a flat
#: picture, a sparser cloud reads as structure you can see through.
POINTS_ACROSS = 210

_BG = np.asarray([4, 6, 10], np.float32)
_EXTRUDE = 1.7        # world-space depth of the whole scene
_CAM_DIST = 1.8
_FOCAL = 1.3
_YAW_AMP = 0.42       # radians of sway either side (~24 degrees)
_PITCH = -0.10


def render(colour_rgb: np.ndarray, depth: np.ndarray, out_w: int, out_h: int, phase: float) -> np.ndarray:
    """One frame as ``(out_h, out_w, 3)`` uint8. ``colour_rgb`` is HxWx3 uint8, ``depth`` HxW 0..1 (1 = near)."""
    out_w, out_h = max(1, int(out_w)), max(1, int(out_h))
    frame = np.empty((out_h, out_w, 3), np.float32)
    frame[:] = _BG
    step = max(1, depth.shape[1] // POINTS_ACROSS)
    col = colour_rgb[::step, ::step, :3].astype(np.float32)
    dz = np.clip(depth[::step, ::step].astype(np.float32), 0.0, 1.0)
    gh, gw = dz.shape
    gx = np.broadcast_to(np.linspace(0.0, 1.0, gw), (gh, gw))
    gy = np.broadcast_to(np.linspace(0.0, 1.0, gh)[:, None], (gh, gw))
    aspect = out_w / out_h

    wx, wy, wz = (gx - 0.5) * aspect, (0.5 - gy), (1.0 - dz) * _EXTRUDE
    cz = _EXTRUDE * 0.5
    yaw = _YAW_AMP * np.sin(phase)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(_PITCH), np.sin(_PITCH)
    zc = wz - cz
    xr = wx * cy_ + zc * sy_
    zr = -wx * sy_ + zc * cy_
    yr = wy * cp - zr * sp
    zr = wy * sp + zr * cp
    z_cam = zr + cz + _CAM_DIST

    valid = z_cam > 0.05
    inv_z = np.where(valid, 1.0 / np.maximum(z_cam, 0.05), 0.0)
    scale = out_h * ((cz + _CAM_DIST) / _FOCAL)
    px = (out_w - 1) * 0.5 + _FOCAL * xr * inv_z * scale
    py = (out_h - 1) * 0.5 - _FOCAL * yr * inv_z * scale
    ox = np.rint(px).astype(np.int32)
    oy = np.rint(py).astype(np.int32)

    keep = valid.ravel() & (ox.ravel() >= 0) & (ox.ravel() < out_w) & (oy.ravel() >= 0) & (oy.ravel() < out_h)
    if not keep.any():
        return frame.astype(np.uint8)
    rgb = (col * (0.95 + 0.45 * dz)[..., None]).reshape(-1, 3)[keep]
    ox_k, oy_k, z_k = ox.ravel()[keep], oy.ravel()[keep], z_cam.ravel()[keep]
    order = np.argsort(-z_k)                 # far first, so nearer points win a shared pixel
    # A 2x2 dot per point: single pixels vanish at preview size.
    for dy in (0, 1):
        for dx in (0, 1):
            xx = np.clip(ox_k[order] + dx, 0, out_w - 1)
            yy = np.clip(oy_k[order] + dy, 0, out_h - 1)
            frame[yy, xx] = rgb[order]
    return np.clip(frame, 0, 255).astype(np.uint8)
