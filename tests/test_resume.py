"""Pause and resume: a stopped conversion keeps what it has encoded and carries on from there."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("av")

from test_video import _make_clip  # noqa: E402

from ultimate_video_3d import pipeline, resume, settings, video  # noqa: E402
from ultimate_video_3d import depth as depthmod  # noqa: E402


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    class _Flat(depthmod.DepthEngine):
        def load(self, *a, **k):
            self.session = object()
            return "test"

        def infer(self, rgb):
            return np.full((depthmod.INPUT, depthmod.INPUT), 0.5, np.float32)

    monkeypatch.setattr(pipeline.depthmod, "DepthEngine", _Flat)
    monkeypatch.setattr(pipeline.paths, "scratch_dir", lambda: tmp_path / "scratch")
    monkeypatch.setattr(resume.paths, "scratch_dir", lambda: tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(resume, "SEGMENT_FRAMES", 5)
    monkeypatch.setattr(resume, "PREROLL", 3)
    clip = tmp_path / "in.mp4"
    _make_clip(clip, frames=24)
    s = settings.AppSettings()
    s.export.family, s.export.encoder = "h264", "libx264"
    return clip, s, tmp_path


def _run(clip, dst, s, **kw):
    return [u for u in pipeline.convert_video(clip, dst, s, gpus=[], **kw)]


def _stop_after(n):
    count = {"n": 0}

    def stop():
        count["n"] += 1
        return count["n"] > n

    return stop


def _timestamps(path):
    import av

    with av.open(str(path)) as f:
        return [float(p.pts * p.time_base) for p in f.demux(f.streams.video[0]) if p.pts is not None]


def test_a_conversion_in_several_segments_joins_into_one_smooth_video(rig):
    clip, s, tmp = rig
    dst = tmp / "o.mp4"
    updates = _run(clip, dst, s)
    assert updates[-1].stage == "done"
    info = video.probe(dst)
    assert info.frames == 24 and info.has_audio
    times = sorted(_timestamps(dst))
    assert len(times) == 24 and all(abs((b - a) - 1 / 24) < 0.002 for a, b in zip(times, times[1:]))
    assert not list((tmp / "scratch" / "resume").glob("*"))                       # nothing left behind


def test_pausing_keeps_the_finished_part_and_resuming_completes_the_video(rig):
    clip, s, tmp = rig
    dst = tmp / "o.mp4"
    first = _run(clip, dst, s, should_stop=_stop_after(9))
    assert first[-1].stage == "stopped" and first[-1].message == "paused" and not dst.exists()
    kept = resume.find(clip, s, 0, None, 24)
    assert kept is not None and 0 < kept.frames_done < 24 and kept.percent > 0
    second = _run(clip, dst, s, resume=True)
    assert second[0].message.startswith("Resuming") and second[-1].stage == "done"
    info = video.probe(dst)
    assert info.frames == 24 and info.has_audio
    times = sorted(_timestamps(dst))
    assert all(abs((b - a) - 1 / 24) < 0.002 for a, b in zip(times, times[1:]))
    assert resume.find(clip, s, 0, None, 24) is None                             # finished: nothing to resume


def test_resume_within_an_in_out_range(rig):
    clip, s, tmp = rig
    dst = tmp / "r.mp4"
    _run(clip, dst, s, start=4, limit=16, should_stop=_stop_after(7))
    assert resume.find(clip, s, 4, 16, 16) is not None
    _run(clip, dst, s, start=4, limit=16, resume=True)
    assert video.probe(dst).frames == 16


def test_starting_over_discards_the_old_pieces(rig):
    clip, s, tmp = rig
    _run(clip, tmp / "o.mp4", s, should_stop=_stop_after(9))
    assert resume.find(clip, s, 0, None, 24) is not None
    updates = _run(clip, tmp / "o.mp4", s, resume=False)
    assert not updates[0].message.startswith("Resuming") and updates[-1].stage == "done"
    assert video.probe(tmp / "o.mp4").frames == 24


def test_other_settings_are_a_different_job(rig):
    clip, s, tmp = rig
    _run(clip, tmp / "o.mp4", s, should_stop=_stop_after(9))
    other = settings.AppSettings()
    other.export.family, other.export.encoder = "h264", "libx264"
    other.threed.strength = s.threed.strength + 10
    assert resume.find(clip, other, 0, None, 24) is None
    assert resume.find(clip, s, 0, None, 24) is not None


def test_stale_jobs_are_swept_but_recent_ones_stay(rig):
    import os
    import time

    clip, s, tmp = rig
    _run(clip, tmp / "o.mp4", s, should_stop=_stop_after(9))
    folder = next((tmp / "scratch" / "resume").iterdir())
    resume.cleanup_stale()
    assert folder.exists()
    old = time.time() - 20 * 86400
    os.utime(folder / "state.json", (old, old))
    resume.cleanup_stale()
    assert not folder.exists()


def test_window_says_where_it_paused_and_offers_to_continue(rig, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from ultimate_video_3d import app as appmod

    clip, s, tmp = rig
    monkeypatch.setattr(appmod.paths, "settings_path", lambda: tmp / "settings.json")
    appmod.QApplication.instance() or appmod.QApplication([])
    window = appmod.MainWindow()
    try:
        page = window.video_page
        window._open_path(clip)
        window._was_stopped(6, 24, True)
        assert "Paused at 25%" in page.info_label.text() and "continue" in page.info_label.text()
        window._was_stopped(6, 24, False)
        assert page.info_label.text() == "Stopped."
        assert window._ask_resume(page, s, 0, None) is False                 # nothing paused yet: just start
        _run(clip, tmp / "o.mp4", s, should_stop=_stop_after(9))
        answers = iter(["Continue", "Start over", "Cancel"])

        def fake_exec(box):
            want = next(answers)
            box._clicked = next(b for b in box.buttons() if b.text() == want)

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._clicked)
        assert window._ask_resume(page, s, 0, None) is True
        assert window._ask_resume(page, s, 0, None) is False
        assert resume.find(clip, s, 0, None, 24) is None                     # "Start over" removed the pieces
        _run(clip, tmp / "o.mp4", s, should_stop=_stop_after(9))
        assert window._ask_resume(page, s, 0, None) is None
    finally:
        window.close()
