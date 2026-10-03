"""Quality, bitrate and file size as one linked choice: the maths, and real encodes that must land where they were aimed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d import encoders, ratecontrol  # noqa: E402
from ultimate_video_3d import settings as settingsmod  # noqa: E402

SIZE = (1920, 1080)
FPS = 24.0


def test_quality_and_bitrate_are_inverse_and_more_quality_always_costs_more():
    previous = 0
    for quality in range(0, 101, 10):
        bps = ratecontrol.bitrate_for_quality(SIZE, FPS, quality, "h265")
        assert bps > previous
        previous = bps
        assert abs(ratecontrol.quality_for_bitrate(bps, SIZE, FPS, "h265") - quality) <= 1
    assert ratecontrol.quality_for_bitrate(1_000, SIZE, FPS, "h265") == 0                          # far below the slider's reach
    assert ratecontrol.quality_for_bitrate(10_000_000_000, SIZE, FPS, "h265") == 100


def test_a_more_efficient_codec_needs_fewer_bits_for_the_same_quality():
    h264, h265, av1 = (ratecontrol.bitrate_for_quality(SIZE, FPS, 60, key) for key in ("h264", "h265", "av1"))
    assert h264 > h265 > av1
    assert encoders.bitrate_for(SIZE, FPS, 60, "h265") == h265                                      # the old VideoToolbox path agrees


def test_a_file_size_is_a_bitrate_times_the_running_time_plus_the_sound():
    seconds, audio = 5400.0, 384_000                                                                 # a 90 minute film with 5.1 AAC
    video_bps = ratecontrol.video_bps_for_size(4_700_000_000, audio, seconds)
    back = ratecontrol.size_bytes(video_bps, audio, seconds)
    assert back == pytest.approx(4_700_000_000, rel=0.001)
    assert ratecontrol.video_bps_for_size(1_000, audio, seconds) == ratecontrol.MIN_VIDEO_BPS      # an impossible size still encodes something
    assert ratecontrol.video_bps_for_size(1_000_000, audio, 0) == ratecontrol.MIN_VIDEO_BPS


def test_the_sound_is_counted_as_it_will_really_be_stored():
    assert ratecontrol.audio_bps("none") == 0
    assert ratecontrol.audio_bps("aac", channels=6) == 384_000                                      # AAC: 64 kb/s a channel
    assert ratecontrol.audio_bps("copy", "dts", 2_973_162, 6, mkv=True) == 2_973_162                # DTS-HD MA kept untouched in an MKV
    assert ratecontrol.audio_bps("copy", "dts", 2_973_162, 6, mkv=False) == 384_000                 # ... but turned into AAC in an MP4
    assert ratecontrol.audio_bps("copy", "ac3", 0, 6, mkv=False) == 640_000                         # AC-3 fits an MP4 as it is
    assert ratecontrol.audio_bps("copy", "weird", 0, 2, mkv=True) == 192_000                         # an unknown track in an MKV: a guess by channels


def test_what_the_encode_follows_is_whatever_was_chosen_last():
    export = settingsmod.ExportSettings()
    assert export.rate_mode == "quality" and ratecontrol.video_bitrate(export, True, 100.0, 0) == 0
    export.rate_mode, export.video_kbps = "bitrate", 5000
    assert ratecontrol.video_bitrate(export, True, 100.0, 0) == 5_000_000
    export.rate_mode, export.target_mb = "size", 100.0
    assert ratecontrol.video_bitrate(export, True, 100.0, 0) == pytest.approx(100e6 * 8 / 1.012 / 100.0, rel=0.001)
    assert ratecontrol.video_bitrate(export, False, 100.0, 0) == 0                                  # ProRes and the like have no bitrate to aim at
    export.target_mb = 0.0
    assert ratecontrol.video_bitrate(export, True, 100.0, 0) == 0                                   # a size of nothing means "not set"


def test_every_encoder_has_a_bitrate_mode_that_does_not_ask_for_a_quality():
    for name in ("libx264", "libx265", "libsvtav1", "libvpx-vp9", "h264_nvenc", "hevc_amf", "h264_qsv", "hevc_videotoolbox", "h264_mf"):
        options = encoders.bitrate_options(name, encoders.FAMILIES_BY_KEY["h264"], 4_000_000)
        assert not ({"crf", "cq", "qp_i", "qp_p", "global_quality"} & set(options)), name           # none of the constant-quality knobs
        assert options.get("b") != "0", name
    assert encoders.bitrate_options("libx264", encoders.FAMILIES_BY_KEY["h264"], 4_000_000)["maxrate"] == "6000000"


# --- real encodes -------------------------------------------------------------------------------------------------------


av = pytest.importorskip("av")


def noisy_clip(path: Path, seconds: float = 6.0, size=(640, 360)) -> Path:
    """Moving gradients plus noise: hard enough to encode that the bitrate setting, not the picture, decides the size."""
    w, h = size
    out = av.open(str(path), "w")
    stream = out.add_stream("libx264", rate=24)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.bit_rate = 30_000_000
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(int(24 * seconds)):
        base = (np.sin((xx + i * 3) / 37.0) * 60 + np.cos((yy - i * 2) / 23.0) * 60 + 128)[..., None]
        rgb = np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return path


def video_bitrate_of(path: Path) -> float:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.bit_rate or 0) or path.stat().st_size * 8 / (float(c.duration) / 1e6)


@pytest.mark.parametrize("target", [600_000, 2_000_000])
def test_an_encode_aimed_at_a_bitrate_lands_near_it(tmp_path, target):
    clip = noisy_clip(tmp_path / "src.mp4")
    family = encoders.FAMILIES_BY_KEY["h264"]
    from ultimate_video_3d import video

    reader = av.open(str(clip))
    writer = video.VideoWriter(tmp_path / "out.mp4", family, "libx264", 24.0, (640, 360), 60, None, bitrate_bps=target)
    for frame in reader.decode(video=0):
        writer.write(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
    writer.close()
    reader.close()
    actual = (tmp_path / "out.mp4").stat().st_size * 8 / 6.0
    assert target * 0.7 < actual < target * 1.3, (target, actual)


def test_converting_to_a_file_size_lands_near_that_size_and_a_quality_encode_does_not_pretend_to(tmp_path, monkeypatch):
    from ultimate_video_3d import depth as depthmod
    from ultimate_video_3d import pipeline, video

    class _Flat(depthmod.DepthEngine):
        def load(self, *a, **k):
            self.session = object()
            return "test"

        def infer(self, rgb):
            return np.full((depthmod.INPUT, depthmod.INPUT), 0.5, np.float32)

    monkeypatch.setattr(pipeline.depthmod, "DepthEngine", _Flat)
    monkeypatch.setattr(pipeline.paths, "scratch_dir", lambda: tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    clip = noisy_clip(tmp_path / "film.mp4", seconds=8.0, size=(320, 180))
    s = settingsmod.AppSettings()
    s.export.family, s.export.encoder, s.export.audio = "h264", "libx264", "none"
    s.export.rate_mode, s.export.target_mb = "size", 0.6
    out = pipeline.output_path_for(clip, s, tmp_path)
    last = None
    for update in pipeline.convert_video(clip, out, s, gpus=[]):
        last = update
    assert last.stage == "done" and out.exists()
    assert out.stat().st_size == pytest.approx(0.6e6, rel=0.3)                  # asked for 0.6 MB: within 30%
    info = video.probe(out)
    assert info.frames >= 190
    s.export.rate_mode = "bitrate"
    s.export.video_kbps = 2000
    out2 = tmp_path / "second.mp4"
    for update in pipeline.convert_video(clip, out2, s, gpus=[]):
        last = update
    assert out2.stat().st_size * 8 / 8.0 == pytest.approx(2_000_000, rel=0.3)   # 2 Mb/s for 8 seconds


# --- the linked controls in the export panel -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _panel(context=((1920, 1080), 24.0, 7200.0, 384_000)):
    from ultimate_video_3d.panels import ExportPanel

    panel = ExportPanel(settingsmod.ExportSettings())
    panel.family_box.setCurrentIndex(panel.family_box.findData("h265"))
    holder = {"value": context}
    panel.set_rate_context(lambda: holder["value"])
    panel._holder = holder
    return panel


def test_moving_the_quality_slider_moves_the_bitrate_and_the_size_estimates(qapp):
    panel = _panel()
    low = (panel.bitrate_box.value(), panel.size_box.value())
    panel.quality.set_value(90, emit=True)
    assert panel._s.rate_mode == "quality" and panel._s.quality == 90
    assert panel.bitrate_box.value() > low[0] and panel.size_box.value() > low[1]
    assert "estimate" in panel.rate_note.text()


def test_typing_a_bitrate_aims_at_it_and_the_slider_and_size_follow(qapp):
    panel = _panel()
    panel.bitrate_box.setValue(4000)
    assert panel._s.rate_mode == "bitrate" and panel._s.video_kbps == 4000
    expected_mb = ratecontrol.size_bytes(4_000_000, 384_000, 7200.0) / 1e6
    assert panel.size_box.value() == pytest.approx(expected_mb, abs=1.5)
    assert panel.quality.value() == ratecontrol.quality_for_bitrate(4_000_000, (1920, 1080), 24.0, "h265")
    assert panel._s.quality == 60                                              # the slider only SHOWS the equivalent; the last quality is kept
    assert "Aiming at 4.0 Mb/s" in panel.rate_note.text()


def test_typing_a_file_size_aims_at_it_and_the_bitrate_and_slider_follow(qapp):
    panel = _panel()
    panel.size_box.setValue(4000)                                              # a 2-hour film on a 4 GB budget
    assert panel._s.rate_mode == "size" and panel._s.target_mb == 4000
    bps = ratecontrol.video_bps_for_size(4000e6, 384_000, 7200.0)
    assert panel.bitrate_box.value() == pytest.approx(bps / 1000, abs=1)
    assert panel.size_box.value() == 4000                                      # what was typed stays exactly as typed
    assert "4.00 GB" in panel.rate_note.text() and "384 kb/s of sound" in panel.rate_note.text()
    panel.quality.set_value(70, emit=True)                                  # and moving the slider takes over again
    assert panel._s.rate_mode == "quality"


def test_a_size_stays_put_when_the_part_being_converted_changes_and_the_bitrate_adapts(qapp):
    panel = _panel()
    panel.size_box.setValue(1000)
    long = panel.bitrate_box.value()
    panel._holder["value"] = ((1920, 1080), 24.0, 3600.0, 384_000)            # the In/Out range was halved
    panel.refresh_rates()
    assert panel.size_box.value() == 1000 and panel.bitrate_box.value() > long * 1.8


def test_a_bitrate_stays_put_when_the_sound_changes_and_the_size_adapts(qapp):
    panel = _panel()
    panel.bitrate_box.setValue(4000)
    before = panel.size_box.value()
    panel._holder["value"] = ((1920, 1080), 24.0, 7200.0, 0)                   # no sound
    panel.refresh_rates()
    assert panel.bitrate_box.value() == 4000 and panel.size_box.value() < before


def test_without_a_video_the_boxes_wait_and_a_format_with_no_bitrate_hides_them(qapp):
    panel = _panel(context=None)
    assert not panel.bitrate_box.isEnabled() and not panel.size_box.isEnabled() and "Open a video" in panel.rate_note.text()
    panel._holder["value"] = ((1920, 1080), 24.0, 60.0, 0)
    panel.refresh_rates()
    assert panel.bitrate_box.isEnabled()
    panel.show()
    assert panel.rate_block.isVisible()
    panel.family_box.setCurrentIndex(panel.family_box.findData("prores"))
    assert not panel.rate_block.isVisible()                                    # an editing master has no bitrate to aim at
    panel.close()


def test_the_choice_is_remembered(tmp_path):
    s = settingsmod.AppSettings()
    s.export.rate_mode, s.export.video_kbps, s.export.target_mb = "size", 6500, 1500.0
    s.save(tmp_path / "s.json")
    got = settingsmod.AppSettings.load(tmp_path / "s.json").export
    assert (got.rate_mode, got.video_kbps, got.target_mb) == ("size", 6500, 1500.0)
