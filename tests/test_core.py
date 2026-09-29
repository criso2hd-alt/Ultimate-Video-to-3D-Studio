"""Core maths and plumbing, on synthetic data: fast, deterministic, no GPU or model needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d import depth, encoders, grade, hdr, lut, stereo, sysinfo  # noqa: E402


# --- stereo ---------------------------------------------------------------


def _scene(w=160, h=90):
    """A far background with one near square in the middle."""
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., :] = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    d = np.full((h, w), 0.2, np.float32)
    a, b = w * 3 // 8, w * 5 // 8
    rgb[30:60, a:b] = (1.0, 0.2, 0.2)
    d[30:60, a:b] = 0.9
    return rgb, d


def _centroid_x(mask):
    return np.nonzero(mask)[1].mean()


def test_flat_depth_at_convergence_is_identity():
    rgb, _ = _scene()
    d = np.full(rgb.shape[:2], 0.55, np.float32)
    pair = stereo.synthesize(rgb, d, stereo.StereoParams(convergence=0.55))
    assert np.allclose(pair.left, rgb, atol=1e-5) and np.allclose(pair.right, rgb, atol=1e-5)
    assert pair.hole_left.sum() == 0


def test_near_object_moves_right_in_left_eye():
    rgb, d = _scene(w=480)
    pair = stereo.synthesize(rgb, d, stereo.StereoParams(strength=100, convergence=0.55))
    red = lambda img: (img[..., 0] > 0.9) & (img[..., 1] < 0.4)   # noqa: E731
    base = _centroid_x(red(rgb))
    assert _centroid_x(red(pair.left)) > base + 1.0     # in front of the screen: left eye sees it further right
    assert _centroid_x(red(pair.right)) < base - 1.0


def test_swap_eyes_exchanges_views():
    rgb, d = _scene()
    a = stereo.synthesize(rgb, d, stereo.StereoParams(strength=80))
    b = stereo.synthesize(rgb, d, stereo.StereoParams(strength=80, swap_eyes=True))
    assert np.allclose(a.left, b.right) and np.allclose(a.right, b.left)


def test_no_nan_and_holes_are_background_filled():
    rgb, d = _scene()
    pair = stereo.synthesize(rgb, d, stereo.StereoParams(strength=100))
    assert np.isfinite(pair.left).all() and np.isfinite(pair.right).all()
    # Nothing left black where the picture had no black: every gap was filled.
    assert pair.left.min() > 0.05 and pair.right.min() > 0.05


def test_no_cracks_on_smooth_slope():
    """Regression: a span holding exactly one pixel used to be skipped, leaving 1px cracks."""
    h, w = 40, 400
    rgb = np.random.default_rng(0).random((h, w, 3), dtype=np.float32)
    d = np.tile(np.linspace(0.3, 0.8, w, dtype=np.float32), (h, 1))
    pair = stereo.synthesize(rgb, d, stereo.StereoParams(strength=60))
    interior = (pair.hole_left | pair.hole_right)[:, 20:-20]
    assert interior.sum() == 0


@pytest.mark.parametrize("layout,shape", [
    ("sbs_half", (90, 160)), ("sbs_full", (90, 320)), ("tb_half", (90, 160)),
    ("tb_full", (180, 160)), ("anaglyph", (90, 160)), ("rgbd", (90, 160)), ("depth", (90, 160)),
])
def test_pack_frame_shapes_and_dtypes(layout, shape):
    rgb, d = _scene()
    pair = stereo.synthesize(rgb, d, stereo.StereoParams())
    for bits, dtype in ((8, np.uint8), (16, np.uint16)):
        out = stereo.pack_frame(layout, pair, rgb, d, bits)
        assert out.shape == (*shape, 3) and out.dtype == dtype
    assert stereo.output_size(layout, 160, 90) == (shape[1], shape[0])


def test_sbs_half_halves_hold_the_two_eyes():
    rgb, d = _scene()
    pair = stereo.synthesize(rgb, d, stereo.StereoParams(strength=100))
    out = stereo.pack_frame("sbs_half", pair, rgb, d, 8).astype(np.float32) / 255.0
    left_half, right_half = out[:, :80], out[:, 80:]
    assert np.abs(left_half - right_half).mean() > 0.001      # they differ: there is parallax
    ref_left = stereo._resize(pair.left, (80, 90))
    assert np.abs(left_half - ref_left).mean() < 0.01


# --- colour ---------------------------------------------------------------


def test_neutral_grade_returns_the_input_untouched():
    frame = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    engine = grade.GradeEngine()
    assert engine.apply(frame, grade.ColourSettings()) is frame
    assert engine.apply(frame, grade.ColourSettings(enabled=True)) is frame


def test_exposure_is_a_true_stop_of_light():
    mid = np.full((8, 8, 3), 0.5, np.float32)
    out = grade.GradeEngine().apply(mid, grade.ColourSettings(enabled=True, exposure=1.0))
    ratio = grade.decode(out[0, 0, 0], grade.SDR) / grade.decode(np.float32(0.5), grade.SDR)
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_grade_directions():
    base = np.full((8, 8, 3), 0.5, np.float32)
    eng = grade.GradeEngine()
    warm = eng.apply(base, grade.ColourSettings(enabled=True, temperature=60))
    assert warm[0, 0, 0] > warm[0, 0, 2]
    grey = eng.apply(np.random.default_rng(2).random((8, 8, 3), dtype=np.float32),
                     grade.ColourSettings(enabled=True, saturation=-100))
    assert np.allclose(grey[..., 0], grey[..., 1], atol=0.02)
    vig = eng.apply(np.full((64, 64, 3), 0.6, np.float32), grade.ColourSettings(enabled=True, vignette=100))
    assert vig[32, 32, 0] > vig[0, 0, 0]


def test_curve_is_monotone_and_hits_its_points():
    pts = [[0, 0], [0.25, 0.4], [0.75, 0.6], [1, 1]]
    curve = grade.monotone_curve(pts, 256)
    assert np.all(np.diff(curve) >= -1e-6)
    assert curve[0] == pytest.approx(0.0, abs=1e-4) and curve[-1] == pytest.approx(1.0, abs=1e-4)
    assert curve[64] == pytest.approx(0.4, abs=0.02)


def test_pq_and_hlg_round_trip():
    x = np.linspace(0.0, 1.0, 33)
    assert np.abs(grade.pq_encode(grade.pq_decode(x)) - x).max() < 1e-4
    assert np.abs(grade.hlg_encode(grade.hlg_decode(x)) - x).max() < 1e-4


def _cube(tmp_path, body, size=2):
    path = tmp_path / "t.cube"
    path.write_text(f"TITLE \"t\"\nLUT_3D_SIZE {size}\n{body}\n")
    return path


def test_identity_lut_changes_nothing(tmp_path):
    n = 5
    rows = [f"{r / (n - 1)} {g / (n - 1)} {b / (n - 1)}" for b in range(n) for g in range(n) for r in range(n)]
    path = _cube(tmp_path, "\n".join(rows), n)
    frame = np.random.default_rng(3).random((16, 16, 3), dtype=np.float32)
    out = grade.GradeEngine().apply(frame, grade.ColourSettings(lut_enabled=True, lut_path=str(path)))
    assert np.abs(out - frame).max() < 0.01


def test_lut_parsing_errors_are_plain(tmp_path):
    with pytest.raises(ValueError, match="entries"):
        lut.load_cube(_cube(tmp_path, "0 0 0\n1 1 1", 2))
    bad = tmp_path / "x.cube"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        lut.load_cube(bad)


def test_1d_lut_expands_exactly(tmp_path):
    path = tmp_path / "one.cube"
    path.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    loaded = lut.load_cube(path)
    assert loaded.table.shape[-1] == 3 and loaded.table[0, 0, -1, 0] == pytest.approx(1.0)


# --- HDR ------------------------------------------------------------------


def test_tonemap_keeps_diffuse_white_and_rolls_off_highlights():
    def sdr(nits):
        signal = float(grade.pq_encode(np.float32(nits / 10000.0)))
        return hdr.tonemap_to_sdr(np.full((2, 2, 3), signal, np.float32), "pq")[0, 0, 0]

    assert sdr(203) > 0.93                       # diffuse white stays near SDR white
    assert sdr(100) < sdr(203) < sdr(1000) + 1e-6
    assert sdr(4000) <= 1.0 + 1e-6


# --- depth stabiliser -----------------------------------------------------


def test_stabiliser_reduces_flicker_and_resets_on_cut():
    rng = np.random.default_rng(4)
    frame = np.full((72, 128, 3), 0.4, np.float32)
    raws = [np.clip(rng.normal(1.0, 0.25, (518, 518)), 0, None).astype(np.float32) for _ in range(12)]
    steady = depth.DepthStabilizer(0.8)
    loose = depth.DepthStabilizer(0.0)
    a = np.array([steady(r, frame) for r in raws])
    b = np.array([loose(r, frame) for r in raws])
    assert np.abs(np.diff(a[3:], axis=0)).mean() < np.abs(np.diff(b[3:], axis=0)).mean() * 0.6
    cut = np.full_like(frame, 0.95)
    steady(raws[0], cut)
    assert steady._prev_depth is not None and np.isfinite(steady._prev_depth).all()


def test_refine_keeps_range_and_shape():
    frame = np.random.default_rng(5).random((90, 160, 3), dtype=np.float32)
    d = np.random.default_rng(6).random((518, 518), dtype=np.float32)
    out = depth.refine(d, frame)
    assert out.shape == (90, 160) and out.min() >= 0.0 and out.max() <= 1.0


# --- encoders -------------------------------------------------------------


def test_ladder_prefers_the_detected_vendor_and_ends_in_software():
    nv = sysinfo.GpuInfo("RTX", sysinfo.VENDOR_NVIDIA, 8 << 30)
    amd = sysinfo.GpuInfo("RX", sysinfo.VENDOR_AMD, 8 << 30)
    h264 = encoders.FAMILIES_BY_KEY["h264"]
    assert encoders.ladder(h264, [nv])[0] == "h264_nvenc"
    assert encoders.ladder(h264, [amd])[0] == "h264_amf"
    assert encoders.ladder(h264, [amd, nv])[:2] == ["h264_amf", "h264_nvenc"]
    assert encoders.ladder(h264, [nv])[-1] == "libx264"
    assert encoders.ladder(encoders.FAMILIES_BY_KEY["prores"], [nv]) == ["prores_ks"]


def test_software_encoder_always_opens():
    assert encoders.can_open("libx264", encoders.FAMILIES_BY_KEY["h264"], (320, 180), 30.0)


def test_quality_maps_to_sensible_ranges():
    fam = encoders.FAMILIES_BY_KEY["h264"]
    lo = int(encoders.encoder_options("libx264", fam, 0, (1920, 1080), 30, None)["crf"])
    hi = int(encoders.encoder_options("libx264", fam, 100, (1920, 1080), 30, None)["crf"])
    assert lo > hi and 8 <= hi <= 16 and 28 <= lo <= 40
    assert encoders.bitrate_for((1920, 1080), 30, 100, "h264") > encoders.bitrate_for((1920, 1080), 30, 10, "h264")


def test_oversize_frame_falls_past_h264_hardware(monkeypatch):
    """A 7680-wide full-SBS frame exceeds NVENC H.264; pick() must not return an encoder that cannot open it."""
    fam = encoders.FAMILIES_BY_KEY["h264"]
    monkeypatch.setattr(encoders, "can_open", lambda name, *a, **k: name == "libx264")
    assert encoders.pick(fam, [sysinfo.GpuInfo("RTX", sysinfo.VENDOR_NVIDIA)], (7680, 2160), 24, False) == "libx264"
