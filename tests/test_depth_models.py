"""The two depth models (fast V2 / high-quality DA3), the optional NVIDIA add-on, and letterbox bars - no GPU or model needed."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ultimate_video_3d import accel, depth, models, paths
from ultimate_video_3d.settings import AppSettings


def _letterboxed(h=360, w=640, bar=44, seed=0):
    rgb = np.random.default_rng(seed).uniform(0.2, 0.9, (h, w, 3)).astype(np.float32)
    rgb[:bar] = 0.0
    rgb[h - bar:] = 0.01                                     # bars are rarely exactly 0 after compression
    return rgb


def test_bars_are_found_and_one_row_of_them_is_left_to_the_picture():
    assert depth.find_bars(_letterboxed()) == (43, 43, 0, 0)
    pillar = _letterboxed().transpose(1, 0, 2).copy()        # bars at the sides (a 4:3 film in a 16:9 frame, rotated here)
    assert depth.find_bars(pillar) == (0, 0, 43, 43)


@pytest.mark.parametrize("frame", [
    np.zeros((360, 640, 3), np.float32),                     # all black: says nothing
    np.random.default_rng(1).uniform(0.2, 0.9, (360, 640, 3)).astype(np.float32),   # no bars
    _letterboxed(bar=4),                                     # a hair of dark edge: not worth cutting
    _letterboxed(bar=140),                                   # 39% off each side: a dark picture, not bars
])
def test_frames_without_real_bars_cut_nothing(frame):
    assert depth.find_bars(frame) == (0, 0, 0, 0)


def test_bars_shrink_at_once_but_grow_only_when_they_hold():
    stab = depth.DepthStabilizer(0.5)
    assert stab.bars(_letterboxed()) == (43, 43, 0, 0)       # first frame: believed
    open_frame = _letterboxed(bar=0)
    assert stab.bars(open_frame) == (0, 0, 0, 0)             # picture reaches the edge: never cut it
    for _ in range(depth.BAR_SETTLE - 1):
        assert stab.bars(_letterboxed()) == (0, 0, 0, 0)     # a dark top and bottom for a moment is not a bar
    assert stab.bars(_letterboxed()) == (43, 43, 0, 0)       # held long enough: it is
    stab.reset()                                             # a scene cut keeps the bars (they belong to the film)
    assert stab.bars(_letterboxed()) == (43, 43, 0, 0)


class _Session:
    """Stands in for an ONNX session: records the input, returns a left-to-right ramp (and a sky map)."""

    def __init__(self, outputs=1, sky=None):
        self.outputs, self.sky, self.seen = outputs, sky, None

    def run(self, _names, feed):
        x = next(iter(feed.values()))
        self.seen = x
        h, w = x.shape[2:]
        ramp = np.tile(np.linspace(1.0, 2.0, w, dtype=np.float32), (h, 1))[None]
        if self.outputs == 1:
            return [ramp]
        sky = self.sky if self.sky is not None else np.zeros((h, w), np.float32)
        return [ramp, sky[None]]


def test_infer_runs_the_picture_only_and_gives_the_bars_the_far_value():
    engine = depth.DepthEngine("fast")
    engine.session, engine._input_name = _Session(), "x"
    raw = engine.infer(_letterboxed(), (43, 43, 0, 0))
    h, w = depth.INPUT, depth.INPUT
    pad = round(43 * h / (360 - 86))
    assert raw.shape == (h + 2 * pad, w)                     # the whole frame's shape again
    picture = raw[pad:-pad]
    far = np.percentile(picture[::2, ::2], 2.0)
    assert np.allclose(raw[:pad], far) and np.allclose(raw[-pad:], far)
    assert far >= picture.min() and far <= picture.max()     # inside the picture's own range: does not stretch it
    # the network saw (nearly) no black: its input is the picture between the bars, plus the one row of bar kept
    mean = depth._MEAN1.reshape(1, 3, 1, 1)
    std = depth._STD.reshape(1, 3, 1, 1)
    seen = (engine.session.seen * std + mean)[0].max(axis=0)
    black_rows = (seen.max(axis=1) < 0.05).mean()
    assert 0 < black_rows < 0.01


def test_the_quality_model_maps_depth_so_near_is_larger_and_puts_sky_far():
    engine = depth.DepthEngine("quality")
    h, w = depth.MODELS["quality"].size
    sky = np.zeros((h, w), np.float32)
    sky[:20] = 1.0
    engine.session, engine._input_name = _Session(outputs=2, sky=sky), "x"
    raw = engine.infer(np.full((360, 640, 3), 0.5, np.float32))
    assert raw.shape == (h, w)
    assert raw[100, -1] > raw[100, 0]                        # the export returns 1/depth: larger = nearer, kept that way
    assert np.allclose(raw[:20], raw[20:].min(), atol=0.02)  # sky at the far end of what is not sky
    # log mapping: equal steps of inverse depth become shrinking steps (between plain inverse and linear depth)
    row = raw[100]
    assert row[w // 4] - row[0] > row[-1] - row[3 * w // 4]


def test_model_files_are_looked_for_per_path(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "bundled_onnx_dir", lambda: tmp_path / "none")
    assert not depth.is_installed("quality")
    assert depth.missing_files("quality") == ["Depth-Anything-3-Mono-Large.onnx"]
    assert depth.missing_files("quality", nvidia=True) == ["Depth-Anything-3-Mono-Large.fp16.onnx"]
    (tmp_path / "Depth-Anything-3-Mono-Large.onnx").write_bytes(b"x")
    assert depth.is_installed("quality") and depth.missing_files("quality") == []
    # on NVIDIA the fp32 file would run, but slowly: the fp16 one is still wanted
    assert not depth.is_installed("quality", nvidia=True)
    assert depth.locate(False, "quality", True).name == "Depth-Anything-3-Mono-Large.onnx"
    (tmp_path / "Depth-Anything-3-Mono-Large.fp16.onnx").write_bytes(b"x")
    assert depth.is_installed("quality", nvidia=True)
    assert depth.locate(False, "quality", True).name == "Depth-Anything-3-Mono-Large.fp16.onnx"
    assert depth.locate(True, "quality", True).name == "Depth-Anything-3-Mono-Large.onnx"     # CPU: fp32
    assert depth.missing_files("fast", True) == []                                            # built in


def test_the_shared_engine_is_reloaded_only_when_model_cpu_or_nvidia_changes(monkeypatch):
    loads = []

    def fake_load(self, progress=None, force_cpu=False, nvidia=False):
        self._force_cpu, self._nvidia = force_cpu, nvidia
        self.session, self.device = object(), "test"
        loads.append((self.model.key, force_cpu, nvidia))
        return "test"

    monkeypatch.setattr(depth.DepthEngine, "load", fake_load)
    a = depth.shared_engine(False, None, "fast", False)
    assert depth.shared_engine(False, None, "fast", False) is a
    b = depth.shared_engine(False, None, "quality", False)
    assert b is not a and b.model.key == "quality"
    c = depth.shared_engine(False, None, "quality", True)
    assert c is not b
    assert depth.shared_engine(True, None, "quality", True).key == ("quality", True, False)   # CPU wins over NVIDIA
    depth.forget_shared()
    depth.shared_engine(False, None, "quality", True)
    assert loads == [("fast", False, False), ("quality", False, False), ("quality", False, True),
                     ("quality", True, True), ("quality", False, True)]


def test_without_the_nvidia_add_on_depth_runs_exactly_as_before(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(accel, "_registered", None)
    assert not accel.is_installed() and not accel.register()
    names = [p for p, _ in depth._providers(nvidia=True)]
    assert accel.EP_NAME not in names and names[-1] == "CPUExecutionProvider"
    assert "Not downloaded" in accel.status() or "Windows only" in accel.status()


def test_the_add_on_matches_the_driver(monkeypatch):
    assert accel._package(591.59).endswith("-cu13")
    assert accel._package(566.03).endswith("-cu12")
    assert accel._package(None).endswith("-cu13")
    from ultimate_video_3d.sysinfo import GpuInfo

    assert accel.has_nvidia([GpuInfo("NVIDIA GeForce RTX 4080", "NVIDIA")]) == (accel.sys.platform == "win32")
    assert not accel.has_nvidia([GpuInfo("AMD Radeon RX 7900", "AMD")])


def test_the_high_quality_model_downloads_from_the_project_release_and_is_checked(monkeypatch, tmp_path):
    from ultimate_video_3d import bootstrap

    payload = b"da3" * 1000
    name = "Depth-Anything-3-Mono-Large.fp16.onnx"
    monkeypatch.setattr(paths, "model_dir", lambda: tmp_path)
    monkeypatch.setitem(models.EXTRA, name, (hashlib.sha256(payload).hexdigest(), len(payload)))
    fetched = []

    def fake_download(url, destination, on_bytes):
        fetched.append(url)
        destination.write_bytes(payload)

    monkeypatch.setattr(bootstrap, "_download", fake_download)
    path = models.install(name=name)
    assert path == tmp_path / name and path.read_bytes() == payload
    assert fetched == [f"{models.RELEASE}/{name}"] and "/releases/download/models-v1/" in fetched[0]
    monkeypatch.setitem(models.EXTRA, name, ("0" * 64, len(payload)))
    path.unlink()
    with pytest.raises(OSError, match="integrity"):
        models.install(name=name)
    assert not path.exists()


def test_settings_remember_the_model_and_the_nvidia_switch(tmp_path):
    s = AppSettings()
    assert s.threed.depth_model == "fast" and s.nvidia_accel is False
    s.threed.depth_model, s.nvidia_accel = "quality", True
    s.save(tmp_path / "s.json")
    back = AppSettings.load(tmp_path / "s.json")
    assert back.threed.depth_model == "quality" and back.nvidia_accel is True
    (tmp_path / "s.json").write_text('{"threed": {"depth_model": "huge"}, "nvidia_accel": "yes"}', encoding="utf-8")
    odd = AppSettings.load(tmp_path / "s.json")
    assert odd.threed.depth_model == "fast" and odd.nvidia_accel is False


def test_a_look_with_the_other_model_is_a_different_preview_cache():
    from ultimate_video_3d.preview import PreviewParams

    a, b = PreviewParams(mode="anaglyph"), PreviewParams(mode="anaglyph", depth_model="quality")
    assert a != b and a.engine_key == ("fast", False, False) and b.engine_key == ("quality", False, False)
