"""Fitting the 3D strength to the strongest moment of a whole film, and the gauge following a film as it plays."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

av = pytest.importorskip("av")

from ultimate_video_3d import comfort, depth as depthmod, zones  # noqa: E402
from ultimate_video_3d.settings import ThreeDSettings  # noqa: E402


class FakeEngine:
    """Stands in for the depth model: a near block whose size grows with the frame's brightness, so each sample has its own strength."""

    def infer(self, rgb):
        d = np.zeros(rgb.shape[:2], np.float32)
        level = float(rgb.mean())
        h, w = d.shape
        d[h // 4: h // 2, w // 4: w // 2] = 0.2 + level
        d += np.linspace(0, 0.1, w, dtype=np.float32)[None, :]
        return d


def test_sample_times_are_evenly_spread_and_stay_off_the_very_ends():
    times = comfort.sample_times(0, 100_000, 10)
    assert len(times) == 10 and times[0] == 5000 and times[-1] == 95_000
    assert comfort.sample_times(5, 5, 10) == [] and comfort.sample_times(0, 100, 0) == []


def test_the_strength_that_fits_scales_in_proportion_and_is_clamped():
    scan = comfort.ComfortScan(strength=50.0, samples=[comfort.Sample(1000, 0.8, 0.1), comfort.Sample(9000, 2.5, 0.4)])
    assert scan.worst.ms == 9000 and scan.worst_pct == 2.5
    assert scan.strength_for(1.0) == pytest.approx(20.0)                    # the strongest moment goes from 2.5% to 1.0%
    assert scan.strength_for(2.0) == pytest.approx(40.0)
    gentle = comfort.ComfortScan(strength=50.0, samples=[comfort.Sample(0, 0.1, 0.05)])
    assert gentle.strength_for(1.0) == 100.0                                # a very gentle film is raised, but never past the slider
    nothing = comfort.ComfortScan(strength=50.0, samples=[comfort.Sample(0, 0.0, 0.0)])
    assert nothing.strength_for(1.0) is None
    assert comfort.ComfortScan(strength=0.0, samples=[comfort.Sample(0, 1.0, 0.0)]).strength_for(1.0) is None


def test_the_fitted_strength_really_puts_the_strongest_frame_at_the_limit():
    """Measure a picture at one strength, refit, measure again: the worst frame lands on the limit (disparity is linear in strength)."""
    rng = np.random.default_rng(1)
    rgb = (0.3 + 0.4 * rng.random((90, 160, 3))).astype(np.float32)
    engine, stab = FakeEngine(), depthmod.DepthStabilizer(0.0)
    s = ThreeDSettings()
    s.strength = 60.0
    first = comfort.measure_picture(rgb, engine, stab, s.params())
    scan = comfort.ComfortScan(strength=60.0, samples=[comfort.Sample(0, first.out_pct, first.back_pct)])
    fitted = scan.strength_for(zones.COMFORT)
    s.strength = fitted
    again = comfort.measure_picture(rgb, engine, stab, s.params())
    assert max(again.out_pct, again.back_pct) == pytest.approx(zones.COMFORT, rel=0.03)


def _make_film(path: Path, frames: int = 48) -> None:
    from test_video import _frame
    from ultimate_video_3d import encoders, video

    writer = video.VideoWriter(path, encoders.FAMILIES_BY_KEY["h264"], "libx264", 24.0, (320, 180), 80)
    for i in range(frames):
        writer.write(_frame(i))
    writer.close()


def test_scanning_a_real_file_reports_progress_can_be_stopped_and_describes_the_result(tmp_path):
    film = tmp_path / "film.mp4"
    _make_film(film)
    seen = []
    result = comfort.scan(film, ThreeDSettings(), count=6, engine=FakeEngine(), progress=seen.append)
    assert 1 <= len(result.samples) <= 6 and seen[-1] == 1.0 and not result.stopped
    strength = result.strength_for(1.0)
    assert strength is not None
    text = comfort.describe(result, 1.0, strength)
    assert "strongest moment" in text and "everything else lower" in text
    calls = []
    stopped = comfort.scan(film, ThreeDSettings(), count=6, engine=FakeEngine(), should_stop=lambda: calls.append(1) or len(calls) > 2)
    assert stopped.stopped and len(stopped.samples) <= 2


def test_the_gauge_follows_the_film_while_it_plays(tmp_path):
    """The buffered renderer used to drop the depth numbers, so the gauge only moved when paused."""
    from ultimate_video_3d import buffered

    zones_now = zones.DepthStats(out_pct=1.4, back_pct=0.3, valid=True).to_dict()
    frame = buffered.CachedFrame([b"x"], "badge", False, 1.0, b"", zones=zones_now)
    cache = buffered.RenderCache(tmp_path)
    cache.put(7, frame)
    kept = cache.get(7)
    assert kept.files and kept.zones == zones_now                          # still there after the pictures were written to disk
