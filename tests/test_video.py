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


def test_wide_pixels_are_shown_at_their_true_width(tmp_path):
    """A DVD stores 720 x 480 with pixels wider than tall; everything downstream must see the real shape."""
    clip = tmp_path / "anamorphic.mkv"
    with av.open(str(clip), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 360, 240, "yuv420p"
        stream.codec_context.sample_aspect_ratio = Fraction(2, 1)
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(np.full((240, 360, 3), 40 * i, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    info = video.probe(clip)
    assert (info.width, info.height) == (720, 240)
    _, rgb = next(video.frames(clip, limit=1))
    assert rgb.shape[:2] == (240, 720)


def test_square_pixels_are_left_alone():
    assert video.display_width(1920, None) == 1920
    assert video.display_width(1920, Fraction(1, 1)) == 1920
    assert video.display_width(720, Fraction(32, 27)) == 854


# --- speed work: the same pictures, less copying -----------------------------------------------


def _planes_from_swscale(rgb8: np.ndarray) -> np.ndarray:
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb8), format="rgb24")
    frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG", src_color_range="JPEG")
    return frame.to_ndarray()


def test_the_fast_rgb_to_yuv_kernel_matches_swscale():
    from ultimate_video_3d import stereo

    rgb = np.clip(_frame(3, 320, 180) * 255 + 0.5, 0, 255).astype(np.uint8)
    rgb[:, :16] = 0
    rgb[:, 16:32] = 255                                          # the extremes land on 16 and 235, not clipped differently
    ours = stereo.rgb_to_yuv420(rgb).astype(np.int16)
    theirs = _planes_from_swscale(rgb).astype(np.int16)
    assert ours.shape == theirs.shape
    luma = 180
    assert np.abs(ours[:luma] - theirs[:luma]).mean() < 0.7 and np.abs(ours[:luma] - theirs[:luma]).max() <= 3
    assert np.abs(ours[luma:] - theirs[luma:]).mean() < 1.5       # chroma differs a little: the filters are not the same


def test_the_writer_gives_the_same_picture_with_or_without_the_fast_path(tmp_path, monkeypatch):
    fam = encoders.FAMILIES_BY_KEY["h264"]
    rgb = np.clip(_frame(2) * 255 + 0.5, 0, 255).astype(np.uint8)
    frames = {}
    for name, off in (("fast", False), ("slow", True)):
        monkeypatch.setattr(video, "_NO_DIRECT_YUV", off)
        path = tmp_path / f"{name}.mp4"
        writer = video.VideoWriter(path, fam, "libx264", 24.0, (320, 180), 95)
        for _ in range(6):
            writer.write(rgb)
        writer.close()
        frames[name] = next(f for _, f in video.frames(path, 0, 1))
    assert np.abs(frames["fast"] - frames["slow"]).mean() < 0.01


def test_a_ring_of_pictures_is_reused_and_holds_the_right_frames(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, frames=12, audio=False)
    plain = [f.copy() for _, f in video.frames(clip)]
    seen = []
    ids = set()
    for _, f in video.frames(clip, ring=3):
        ids.add(id(f))
        seen.append(f.copy())                                    # a frame is only valid until the ring wraps
    assert len(ids) == 3 and len(seen) == len(plain)
    assert all(np.array_equal(a, b) for a, b in zip(seen, plain))
