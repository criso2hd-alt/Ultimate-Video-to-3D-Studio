"""Render the demo clip used for the README screenshots: a small ray-traced scene made for this project.

Screenshots of a converter need footage, and footage belongs to someone. This scene is ours: a checkered
floor, glossy spheres and columns receding into fog, with the camera trucking sideways so the depth (and
the parallax the 3D produces) is easy to see. Pure NumPy, about a minute for the whole clip:

    python scripts/make_demo_scene.py [out.mp4]

The clip is not part of the repository (see .gitignore: *.mp4); regenerate it with this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

W, H, FPS, SECONDS = 1280, 720, 24, 3
LIGHT = np.array([-0.55, 0.85, -0.45])
LIGHT /= np.linalg.norm(LIGHT)
SKY_TOP, SKY_HORIZON = np.array([0.16, 0.36, 0.78]), np.array([0.70, 0.80, 0.92])

# (centre x, y, z), radius, colour, gloss
SPHERES = [
    ((-1.7, 0.80, 4.6), 0.80, (0.86, 0.16, 0.14), 0.55),
    ((0.95, 0.60, 3.3), 0.60, (0.13, 0.36, 0.86), 0.45),
    ((0.15, 0.42, 2.1), 0.42, (0.98, 0.58, 0.10), 0.35),
    ((2.5, 1.00, 7.2), 1.00, (0.16, 0.66, 0.34), 0.40),
    ((-3.2, 1.25, 9.8), 1.25, (0.95, 0.80, 0.18), 0.35),
]
# (centre x, z), radius, height, colour
COLUMNS = [((4.4, 5.5), 0.42, 3.4, (0.92, 0.55, 0.30)), ((4.6, 8.5), 0.42, 3.4, (0.92, 0.55, 0.30)),
           ((4.8, 12.0), 0.42, 3.4, (0.92, 0.55, 0.30)), ((-5.0, 6.5), 0.42, 3.4, (0.35, 0.60, 0.85)),
           ((-5.2, 10.5), 0.42, 3.4, (0.35, 0.60, 0.85))]
WALL_Z = 18.0


def _sphere_hits(ro, rd, centre, radius):
    oc = ro - np.asarray(centre)
    b = (oc * rd).sum(-1)
    c = (oc * oc).sum(-1) - radius * radius
    disc = b * b - c
    ok = disc > 0
    t = -b - np.sqrt(np.where(ok, disc, 0))
    return np.where(ok & (t > 1e-3), t, np.inf)


def _column_hits(ro, rd, centre_xz, radius, height):
    ox, oz = ro[..., 0] - centre_xz[0], ro[..., 2] - centre_xz[1]
    dx, dz = rd[..., 0], rd[..., 2]
    a = dx * dx + dz * dz
    b = ox * dx + oz * dz
    c = ox * ox + oz * oz - radius * radius
    disc = b * b - a * c
    ok = (disc > 0) & (a > 1e-9)
    t = (-b - np.sqrt(np.where(ok, disc, 0))) / np.where(a > 1e-9, a, 1)
    y = ro[..., 1] + t * rd[..., 1]
    return np.where(ok & (t > 1e-3) & (y >= 0) & (y <= height), t, np.inf)


def trace(ro, rd, want_shadow=True):
    """Colour for each ray (arrays of shape (N, 3))."""
    n = rd.shape[0]
    best = np.full(n, np.inf)
    kind = np.zeros(n, np.int32)          # 0 sky, 1 floor, 2 wall, 3+ objects
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(rd[:, 1] < -1e-6, -ro[:, 1] / rd[:, 1], np.inf)
        t_wall = np.where(rd[:, 2] > 1e-6, (WALL_Z - ro[:, 2]) / rd[:, 2], np.inf)
    for t, k in ((t_floor, 1), (t_wall, 2)):
        m = t < best
        best, kind = np.where(m, t, best), np.where(m, k, kind)
    for i, (centre, radius, _, _) in enumerate(SPHERES):
        t = _sphere_hits(ro, rd, centre, radius)
        m = t < best
        best, kind = np.where(m, t, best), np.where(m, 3 + i, kind)
    for j, (cxz, radius, height, _) in enumerate(COLUMNS):
        t = _column_hits(ro, rd, cxz, radius, height)
        m = t < best
        best, kind = np.where(m, t, best), np.where(m, 3 + len(SPHERES) + j, kind)

    hit = ro + rd * np.where(np.isfinite(best), best, 0)[:, None]
    normal = np.zeros((n, 3))
    albedo = np.zeros((n, 3))
    gloss = np.zeros(n)
    sky_mix = np.clip(rd[:, 1] * 1.6, 0, 1)[:, None]
    colour = SKY_HORIZON * (1 - sky_mix) + SKY_TOP * sky_mix

    m = kind == 1
    if m.any():
        normal[m] = (0, 1, 0)
        checker = ((np.floor(hit[m, 0] * 0.8) + np.floor(hit[m, 2] * 0.8)) % 2)[:, None]
        albedo[m] = np.where(checker > 0, (0.86, 0.84, 0.78), (0.10, 0.12, 0.18))
        gloss[m] = 0.15
    m = kind == 2
    if m.any():
        normal[m] = (0, 0, -1)
        v = np.clip(hit[m, 1] / 9.0, 0, 1)[:, None]
        albedo[m] = np.array((0.92, 0.66, 0.40)) * (1 - v) + np.array((0.45, 0.38, 0.66)) * v
    for i, (centre, radius, col, g) in enumerate(SPHERES):
        m = kind == 3 + i
        if m.any():
            normal[m] = (hit[m] - np.asarray(centre)) / radius
            albedo[m], gloss[m] = col, g
    for j, (cxz, radius, height, col) in enumerate(COLUMNS):
        m = kind == 3 + len(SPHERES) + j
        if m.any():
            nx = (hit[m, 0] - cxz[0]) / radius
            nz = (hit[m, 2] - cxz[1]) / radius
            normal[m] = np.stack([nx, np.zeros_like(nx), nz], -1)
            albedo[m], gloss[m] = col, 0.25

    surface = kind > 0
    lam = np.clip((normal * LIGHT).sum(-1), 0, 1)
    if want_shadow and surface.any():
        origin = hit + normal * 2e-3
        ldir = np.broadcast_to(LIGHT, origin.shape)
        blocked = np.zeros(n, bool)
        for centre, radius, _, _ in SPHERES:
            blocked |= np.isfinite(_sphere_hits(origin, ldir, centre, radius))
        for cxz, radius, height, _ in COLUMNS:
            blocked |= np.isfinite(_column_hits(origin, ldir, cxz, radius, height))
        lam = np.where(blocked, 0.0, lam)
    half = LIGHT - rd
    half /= np.linalg.norm(half, axis=-1, keepdims=True) + 1e-9
    spec = (np.clip((normal * half).sum(-1), 0, 1) ** 48 * gloss * (lam > 0))[:, None]
    ambient = 0.30 + 0.12 * np.clip(normal[:, 1], 0, 1)
    lit = albedo * (ambient[:, None] + 0.85 * lam[:, None]) + spec
    fog = np.clip(1.0 - np.exp(-np.where(np.isfinite(best), best, 60) * 0.016), 0, 1)[:, None]
    lit = lit * (1 - fog) + SKY_HORIZON * fog
    return np.where(surface[:, None], lit, colour)


def render(frame: int, total: int) -> np.ndarray:
    t = frame / max(1, total - 1)
    eye = np.array([-1.0 + 2.0 * t, 1.25, -4.0 + 0.6 * t])                # trucking right, drifting forward
    target = np.array([0.0 + 0.3 * t, 1.0, 5.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array([0, 1, 0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    ys, xs = np.mgrid[0:H, 0:W]
    u = (xs + 0.5) / W * 2 - 1
    v = 1 - (ys + 0.5) / H * 2
    scale = np.tan(np.radians(38) / 2)
    d = fwd + (u[..., None] * (W / H) * scale) * right + (v[..., None] * scale) * up
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(eye, d.shape).copy()
    rgb = trace(o, d).reshape(H, W, 3)
    rgb = np.clip(rgb, 0, 1) ** (1 / 2.0)                                # gamma
    return (rgb * 255 + 0.5).astype(np.uint8)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_scene.mp4")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ultimate_video_3d import bootstrap

    bootstrap.activate_av()
    import av

    total = FPS * SECONDS
    container = av.open(str(out), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "14", "preset": "medium"}
    for i in range(total):
        frame = av.VideoFrame.from_ndarray(render(i, total), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        if i % 12 == 0:
            print(f"  frame {i}/{total}", flush=True)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
