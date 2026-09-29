"""Video I/O round trips on a tiny synthetic clip (needs PyAV, not a GPU)."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

av = pytest.importorskip("av")

from ultimate_video_3d import encoders, pipeline, settings, video  # noqa: E402


def _frame(i: int, w=320, h=180) -> np.ndarray:
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.concatenate([x + 0 * y, y + 0 * x, 0.5 + 0 * x + 0 * y], axis=-1)
    img[40:100, 60 + i * 2: 140 + i * 2] = (0.9, 0.3, 0.2)
    return img


def _make_clip(path: Path, frames=24, audio=True) -> None:
    fam = encoders.FAMILIES_BY_KEY["h264"]
    writer = video.VideoWriter(path.with_suffix(".v.mp4"), fam, "libx264", 24.0, (320, 180), 80)
    for i in range(frames):
        writer.write(_frame(i))
    writer.close()
    if not audio:
        path.write_bytes(path.with_suffix(".v.mp4").read_bytes())
        return
    # add a sine-wave soundtrack so the audio path has something to trim
    with av.open(str(path.with_suffix(".v.mp4"))) as vid, av.open(str(path), "w") as out:
        vs = out.add_stream_from_template(vid.streams.video[0])
        as_ = out.add_stream("aac", rate=44100)
        as_.layout = "mono"
        for packet in vid.demux(vid.streams.video[0]):
            if packet.dts is not None:
                packet.stream = vs
                out.mux(packet)
        t = np.arange(44100 * 1) / 44100.0
        pcm = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)[None, :]
        frame = av.AudioFrame.from_ndarray(pcm, format="fltp", layout="mono")
        frame.sample_rate = 44100
        frame.pts = 0
        frame.time_base = Fraction(1, 44100)
        for packet in as_.encode(frame):
            out.mux(packet)
        for packet in as_.encode():
            out.mux(packet)


def test_write_then_read_preserves_colour(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, audio=False)
    info = video.probe(clip)
    assert (info.width, info.height) == (320, 180) and info.frames == 24 and info.hdr == video.SDR
    original = _frame(0)
    decoded = next(video.frames(clip, 0, 1))[1]
    assert np.abs(decoded - original).mean() < 0.02      # lossy, but no matrix/range colour shift


def test_frames_seeks_to_a_range(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, audio=False)
    got = list(video.frames(clip, start=10, limit=5))
    assert [i for i, _ in got] == [10, 11, 12, 13, 14]


def test_hdr_is_tagged_and_kept_10_bit(tmp_path):
    fam = encoders.FAMILIES_BY_KEY["h265"]
    enc = "libx265"
    path = tmp_path / "pq.mp4"
    writer = video.VideoWriter(path, fam, enc, 24.0, (320, 180), 70, None, hdr="pq")
    for i in range(6):
        writer.write(_frame(i))
    writer.close()
    info = video.probe(path)
    assert info.hdr == "pq" and info.bit_depth == 10


@pytest.mark.parametrize("layout", ["sbs_half", "sbs_full", "anaglyph", "tb_half"])
def test_full_conversion_with_audio_and_range(tmp_path, layout, monkeypatch):
    """End to end on the CPU: decode, depth stand-in, warp, encode, trim the audio to the range."""
    from ultimate_video_3d import depth as depthmod

    # No model in CI: a fixed depth ramp stands in for the network.
    class _Ramp(depthmod.DepthEngine):
        def load(self, *a, **k):
            self.device = "test"
            self.session = object()
            return "test"

        def infer(self, rgb):
            return np.tile(np.linspace(0, 1, depthmod.INPUT, dtype=np.float32), (depthmod.INPUT, 1))

    monkeypatch.setattr(depthmod, "DepthEngine", _Ramp)
    monkeypatch.setattr(pipeline.depthmod, "DepthEngine", _Ramp)
    clip = tmp_path / "in.mp4"
    _make_clip(clip)
    s = settings.AppSettings()
    s.threed.layout = layout
    s.export.family = "h264"
    s.export.encoder = "libx264"
    dst = tmp_path / "out.mp4"
    last = None
    for update in pipeline.convert_video(clip, dst, s, start=6, limit=12, gpus=[]):
        last = update
    assert last.stage == "done" and dst.exists()
    info = video.probe(dst)
    assert info.frames == 12 and info.has_audio
    expect = {"sbs_half": (320, 180), "sbs_full": (640, 180), "anaglyph": (320, 180), "tb_half": (320, 180)}[layout]
    assert (info.width, info.height) == expect
    assert info.duration == pytest.approx(0.5, abs=0.1)         # 12 frames at 24 fps


def test_can_stop_midway(tmp_path, monkeypatch):
    from ultimate_video_3d import depth as depthmod

    class _Ramp(depthmod.DepthEngine):
        def load(self, *a, **k):
            self.session = object()
            return "test"

        def infer(self, rgb):
            return np.full((depthmod.INPUT, depthmod.INPUT), 0.5, np.float32)

    monkeypatch.setattr(pipeline.depthmod, "DepthEngine", _Ramp)
    clip = tmp_path / "in.mp4"
    _make_clip(clip, audio=False)
    s = settings.AppSettings()
    s.export.family, s.export.encoder = "h264", "libx264"
    count = {"n": 0}

    def stop():
        count["n"] += 1
        return count["n"] > 4

    stages = [u.stage for u in pipeline.convert_video(clip, tmp_path / "o.mp4", s, should_stop=stop, gpus=[])]
    assert stages[-1] == "stopped" and not (tmp_path / "o.mp4").exists()
