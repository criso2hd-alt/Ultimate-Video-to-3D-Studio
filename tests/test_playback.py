"""Playback plumbing: the clock, the render cache, the wheel guard, and the app's defaults.

Runs Qt offscreen; needs no GPU, no model and no display.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QScrollArea, QSlider, QVBoxLayout, QWidget  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402
from ultimate_video_3d import buffered, player, preview  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


# --- clock ------------------------------------------------------------------


def test_clock_runs_stops_and_seeks():
    clock = player.Clock()
    clock.set(1000)
    assert clock.now() == pytest.approx(1000, abs=1)          # not running: does not advance
    clock.start(1000)
    time.sleep(0.12)
    assert 1090 < clock.now() < 1300
    clock.stop()
    frozen = clock.now()
    time.sleep(0.05)
    assert clock.now() == pytest.approx(frozen, abs=1)


# --- render cache -----------------------------------------------------------


def _frame() -> buffered.CachedFrame:
    return buffered.CachedFrame([b"x"], "", False, 0.0)


def test_cache_ahead_counts_the_covered_run():
    cache = buffered.RenderCache()
    for i in range(10, 20):
        cache.put(i, _frame())
    assert cache.ahead(10, 100) == 10
    assert cache.ahead(15, 100) == 5
    assert cache.ahead(5, 100) == 0                            # nothing at the playhead yet
    assert cache.ahead(10, 4) == 4                             # capped at the limit


def test_cache_tolerates_small_gaps_but_not_holes():
    cache = buffered.RenderCache()
    for i in (0, 1, 3, 4, 7):                                  # 2 is a 1-frame gap; 5-6 a 2-frame gap
        cache.put(i, _frame())
    assert cache.ahead(0, 100, max_gap=2) == 8
    cache2 = buffered.RenderCache()
    for i in (0, 1, 8):                                        # a 6-frame hole ends the run
        cache2.put(i, _frame())
    assert cache2.ahead(0, 100, max_gap=2) == 2


def test_cache_evicts_behind_the_playhead():
    cache = buffered.RenderCache()
    for i in range(20):
        cache.put(i, _frame())
    cache.evict_before(12)
    assert not cache.has(11) and cache.has(12) and len(cache) == 8
    cache.clear()
    assert len(cache) == 0


def test_disk_cache_stores_frames_as_files_and_cleans_up(tmp_path):
    cache = buffered.RenderCache(tmp_path / "c")
    for i in range(6):
        cache.put(i, buffered.CachedFrame([b"a" * 100, b"b" * 50], "badge", True, 1.5, b"s" * 10))
    assert len(list((tmp_path / "c").iterdir())) == 6 * 3           # two pictures + a scope thumbnail each
    frame = cache.get(2)
    assert frame.jpegs == [] and frame.load() == ([b"a" * 100, b"b" * 50], b"s" * 10)
    assert frame.badge == "badge" and frame.wiggle and frame.hole_pct == 1.5
    assert cache.nbytes == 6 * 160
    cache.evict_before(4)
    assert len(list((tmp_path / "c").iterdir())) == 2 * 3 and cache.nbytes == 2 * 160
    cache.destroy()
    assert not (tmp_path / "c").exists()


def test_disk_cache_frame_that_vanished_reads_as_missing(tmp_path):
    cache = buffered.RenderCache(tmp_path / "c")
    cache.put(0, buffered.CachedFrame([b"x"], "", False, 0.0))
    frame = cache.get(0)
    cache.clear()
    assert frame.load() is None                                     # cleared under the reader: no exception


def test_cache_over_its_limit_drops_the_frames_farthest_from_the_playhead():
    cache = buffered.RenderCache()
    for i in range(100):
        cache.put(i, buffered.CachedFrame([b"x" * 10], "", False, 0.0))
    cache.enforce_limit(50, 500)                                    # 1000 bytes held, 500 allowed
    kept = [i for i in range(100) if cache.has(i)]
    assert cache.nbytes <= 500 and 50 in kept and 0 not in kept and 99 not in kept
    assert max(kept) - 50 <= 50 and 50 - min(kept) <= 50


def test_stale_cache_folders_from_dead_processes_are_removed(tmp_path):
    import os

    (tmp_path / "99999999").mkdir()                                 # no such process
    (tmp_path / "99999999" / "0000001_0.jpg").write_bytes(b"x")
    mine = tmp_path / str(os.getpid())
    mine.mkdir()
    buffered.cleanup_stale_caches(tmp_path, keep=mine)
    assert not (tmp_path / "99999999").exists() and mine.exists()


def test_renderer_reuses_a_run_that_already_reaches_the_playhead(tmp_path):
    class _Info:
        fps = 24.0

    cache = buffered.RenderCache()
    renderer = buffered.BufferedRenderer(tmp_path / "x.mp4", _Info(), cache, None, lambda f, h: f)
    params = preview.PreviewParams(mode="anaglyph")
    renderer.params, renderer.origin_ms = params, 2000.0
    renderer.finished = True                                       # stands in for a live run
    assert renderer.covers(params, 2000.0)                         # started right there
    assert not renderer.covers(params, 9000.0)                     # elsewhere and nothing cached: restart
    cache.put(round(9.0 * 24), buffered.CachedFrame([b"x"], "", False, 0.0))
    assert renderer.covers(params, 9000.0)                         # its pictures already reach it
    assert not renderer.covers(preview.PreviewParams(mode="depth"), 2000.0)   # a different look: restart
    renderer.stop()
    assert not renderer.finished                                    # a stopped run is not a finished one


def test_jpeg_round_trip_is_close():
    rng = np.random.default_rng(0)
    base = np.repeat(np.linspace(0, 255, 96, dtype=np.uint8)[None, :, None], 3, axis=2)
    image = np.repeat(base, 64, axis=0) + rng.integers(0, 3, (64, 96, 3), dtype=np.uint8)
    out = buffered._decode(buffered._encode(image))
    assert out.shape == image.shape
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 4


# --- preview parameters ------------------------------------------------------


def test_preview_params_are_comparable_and_immutable():
    a, b = preview.PreviewParams(mode="anaglyph"), preview.PreviewParams(mode="anaglyph")
    assert a == b and a != preview.PreviewParams(mode="wiggle")
    with pytest.raises(Exception):
        a.mode = "depth"                                      # frozen: safe to share across threads
    assert not preview.PreviewParams().needs_depth             # the default is the plain 2D video
    assert preview.PreviewParams(mode="depth").needs_depth


def test_render_frame_matches_between_paths():
    """The buffered path and the live path use one render function, so a still is identical."""
    from ultimate_video_3d import grade

    rng = np.random.default_rng(1)
    frame = rng.random((72, 128, 3), dtype=np.float32)
    steady = rng.random((259, 259), dtype=np.float32)
    p = preview.PreviewParams(mode="anaglyph")
    a = preview.render_frame(p, frame, steady, grade.GradeEngine())
    b = preview.render_frame(p, frame, steady, grade.GradeEngine())
    assert np.array_equal(a.images[0], b.images[0]) and a.images[0].dtype == np.uint8


# --- the window --------------------------------------------------------------


def test_app_opens_on_the_plain_video_with_3d_preview_off(qapp):
    win = appmod.MainWindow()
    try:
        assert win._effective_mode() == "original"
        assert not hasattr(win.video_page, "preview_toggle")          # no separate switch: the view bar is it
        assert win.video_page.mode_buttons["original"].isChecked()
        win._set_mode("anaglyph")
        assert win._effective_mode() == "anaglyph" and win.video_page.mode_buttons["anaglyph"].isChecked()
        assert not win.video_page.mode_buttons["original"].isChecked()
        win._set_mode("original")
        assert win._effective_mode() == "original" and win.video_page.mode_buttons["original"].isChecked()
    finally:
        win.close()


def test_status_line_sits_above_the_banner(qapp):
    win = appmod.MainWindow()
    try:
        win.show()
        qapp.processEvents()
        banner = win.findChild(QWidget, "coffeeBanner")
        assert banner is not None
        assert win.status.geometry().bottom() <= banner.geometry().top() + 1     # banner is lowest
        assert banner.geometry().bottom() >= win.centralWidget().height() - 2
    finally:
        win.close()


def _spin(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_histogram_is_readable_for_a_dark_letterboxed_frame(qapp):
    """A dark picture with black bars piled a third of its pixels into bin 0; scaling by that flattened
    everything else to nothing (the reported 'histogram is very low')."""
    from ultimate_video_3d.controls import _scope_image

    rng = np.random.default_rng(3)
    frame = np.zeros((135, 240, 3), np.uint8)
    frame[20:115] = rng.normal(45, 22, (95, 240, 3)).clip(0, 255).astype(np.uint8)    # the picture
    image = _scope_image(frame, 0, 316, 120)
    lit = image.sum(axis=2) > 0
    top = int(lit.any(axis=1).nonzero()[0].min())                       # topmost lit row
    assert top <= 120 - 16 - 45                                         # the body of the histogram is tall enough to read
    assert lit[:, 10:140].any(axis=0).mean() > 0.9                      # no comb of empty columns across the range it spans


def test_click_on_the_picture_emits_once_and_only_for_the_left_button(qapp):
    from ultimate_video_3d.controls import PreviewView

    view = PreviewView()
    view.resize(600, 340)
    view.set_images([np.zeros((36, 64, 3), np.uint8)])
    view.show()
    qapp.processEvents()
    seen = []
    view.clicked.connect(lambda: seen.append(1))
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(200, 100))
    assert seen == [1]
    QTest.mouseClick(view, Qt.MouseButton.RightButton, pos=QPoint(200, 100))
    assert seen == [1]
    view.close()


def test_transport_overlay_shows_on_hover_and_fades_when_the_pointer_rests(qapp):
    from ultimate_video_3d.controls import PreviewView

    view = PreviewView()
    view._idle.setInterval(60)
    view.resize(400, 300)
    view.set_images([np.zeros((36, 64, 3), np.uint8)])
    view.show()
    qapp.processEvents()
    assert view._overlay == 0.0
    QApplication.sendEvent(view, QEvent(QEvent.Type.Enter))
    _spin(qapp, 0.4)
    assert view._overlay > 0.95                                         # visible while the pointer is over it
    view.set_playing(True)
    _spin(qapp, 0.15)
    _spin(qapp, 0.6)
    assert view._overlay < 0.05                                         # playing and resting: it fades away
    view.set_playing(False)
    _spin(qapp, 0.4)
    assert view._overlay > 0.95                                         # paused: the play button returns ...
    _spin(qapp, 0.3)
    assert view._overlay > 0.95                                         # ... and stays
    QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
    _spin(qapp, 0.4)
    assert view._overlay < 0.05
    view.close()


def test_view_bar_is_permanent_with_2d_first_and_sits_above_the_timeline(qapp):
    win = appmod.MainWindow()
    try:
        win.show()
        qapp.processEvents()
        page = win.video_page
        assert [b.text() for b in page.mode_buttons.values()] == [
            "2D", "Output", "Anaglyph", "Side by side", "Wiggle", "Depth", "Parallax", "Holes", "3D orbit"]
        assert list(page.mode_buttons)[0] == "original"
        assert page.mode_bar.isVisible()
        assert page.mode_bar.mapToGlobal(QPoint(0, 0)).y() < page.timeline.mapToGlobal(QPoint(0, 0)).y()
        assert all(b.isVisible() and b.width() > 0 for b in page.mode_buttons.values())
    finally:
        win.close()


def _wheel(target, delta):
    pos = QPointF(target.width() / 2, target.height() / 2)
    event = QWheelEvent(pos, target.mapToGlobal(pos), QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(target, event)
    QApplication.processEvents()


def test_wheel_over_a_control_scrolls_the_panel_not_the_control(qapp):
    guard = appmod.WheelGuard()
    qapp.installEventFilter(guard)
    try:
        holder = QWidget()
        col = QVBoxLayout(holder)
        slider, combo = QSlider(Qt.Orientation.Horizontal), QComboBox()
        slider.setRange(0, 100)
        slider.setValue(50)
        combo.addItems(["a", "b", "c"])
        col.addWidget(slider)
        col.addWidget(combo)
        col.addSpacing(2000)                                   # tall enough to scroll
        area = QScrollArea()
        area.setWidget(holder)
        area.setWidgetResizable(True)
        area.resize(300, 300)
        area.show()
        qapp.processEvents()
        slider.clearFocus()                                    # a shown window auto-focuses its first control
        combo.clearFocus()
        area.setFocus()
        qapp.processEvents()
        assert not slider.hasFocus() and not combo.hasFocus()
        for _ in range(3):
            _wheel(slider, -120)
            _wheel(combo, -120)
        assert slider.value() == 50 and combo.currentIndex() == 0        # controls untouched
        assert area.verticalScrollBar().value() > 0                       # panel scrolled instead
        # Even a control that was clicked (and so has focus) must not change as the wheel passes.
        area.verticalScrollBar().setValue(0)
        slider.setFocus()
        combo.setFocus()
        for _ in range(3):
            _wheel(slider, -120)
            _wheel(combo, -120)
        assert slider.value() == 50 and combo.currentIndex() == 0
        area.close()
    finally:
        qapp.removeEventFilter(guard)


def test_version_is_0_1_0_everywhere():
    import re

    import ultimate_video_3d

    root = Path(__file__).resolve().parent.parent
    assert ultimate_video_3d.__version__ == "0.1.0"
    assert re.search(r'^version = "0\.1\.0"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert 'CFBundleShortVersionString": "0.1.0"' in (root / "UltimateVideo3DStudio.spec").read_text(encoding="utf-8")


# --- automatic frame-rate tuning ---------------------------------------------


def _measure(tuning, fps_rate):
    """Feed the tuner frames arriving at ``fps_rate`` per second; return the capability it reports."""
    result = None
    for i in range(40):
        result = tuning.note_frame(i / fps_rate) or result
    return result


@pytest.mark.parametrize("rate,expected", [(80, 30.0), (52, 30.0), (40, 24.0), (34, 24.0), (20, 15.0), (10, 15.0)])
def test_auto_picks_the_rung_the_machine_can_sustain(rate, expected):
    tuning = buffered.Tuning()
    capability = _measure(tuning, rate)
    assert capability == pytest.approx(rate, rel=0.05)
    assert tuning.apply_capability(capability) == expected


def test_measurement_happens_once_and_skips_warmup():
    tuning = buffered.Tuning()
    # two slow warm-up frames, then a steady 50 fps: the warm-up must not drag the estimate down
    stamps = [0.0, 1.0] + [1.0 + i / 50 for i in range(1, 40)]
    result = None
    for t in stamps:
        r = tuning.note_frame(t)
        result = r or result
    assert result == pytest.approx(50, rel=0.05)
    assert tuning.note_frame(99.0) is None                      # measured once


def test_underrun_steps_down_the_ladder_and_stops_at_the_bottom():
    tuning = buffered.Tuning()
    tuning.fps = 30.0
    assert tuning.step_down() and tuning.fps == 24.0
    assert tuning.step_down() and tuning.fps == 15.0
    assert not tuning.step_down() and tuning.fps == 15.0


def test_a_fixed_choice_is_honoured_over_the_measurement():
    tuning = buffered.Tuning()
    tuning.set_choice(30)
    assert tuning.fps == 30.0
    assert tuning.apply_capability(12.0) == 30.0                 # slow machine, but the user asked for 30
    tuning.set_choice(0)                                        # back to auto
    assert tuning.choice == 0


def test_timeline_buffer_bar_state(qapp):
    from ultimate_video_3d.widgets import TimelineWidget

    timeline = TimelineWidget()
    timeline.set_duration(300, 30.0)
    timeline.set_buffered(40, 130)
    assert timeline._buffered == (40, 130) and not timeline._buffering
    timeline.set_buffered(40, 130, True)
    assert timeline._buffering
    timeline.set_buffered(50, 50)                                # nothing ahead: cleared
    assert timeline._buffered is None


def test_settings_cards_reflow_into_columns_without_overlap(qapp):
    win = appmod.MainWindow()
    try:
        win.tabs.setCurrentIndex(1)
        win.show()
        cols = win.settings_page.columns
        seen = {}
        for width in (2600, 1500, 1000, 1500, 1000):                # wide first, then narrower: it must shrink back
            win.resize(width, 860)
            for _ in range(4):
                qapp.processEvents()
            seen[width] = cols.column_count()
            rects = [c.mapTo(cols, c.rect().topLeft()) for c in cols._cards]
            boxes = [(r.x(), r.y(), r.x() + c.width(), r.y() + c.height()) for r, c in zip(rects, cols._cards)]
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    overlap = min(a[2], b[2]) - max(a[0], b[0]) > 1 and min(a[3], b[3]) - max(a[1], b[1]) > 1
                    assert not overlap, f"cards overlap at {width}px"
        assert seen[1000] == 2 and seen[1500] == 3 and seen[2600] == 3        # capped at three
    finally:
        win.close()


def test_card_columns_by_width(qapp):
    from ultimate_video_3d.widgets import CardColumns

    cards = [QWidget() for _ in range(5)]
    cols = CardColumns(cards, min_width=300, max_columns=4)
    cols.show()                                            # a hidden widget defers its resize events
    for width, expected in ((250, 1), (620, 2), (950, 3), (1300, 4), (3000, 4)):
        cols.resize(width, 400)
        qapp.processEvents()
        assert cols.column_count() == expected, (width, cols.column_count())
    assert all(card.parent() is not None for card in cards)                  # none orphaned by the reflow


def test_gauge_goes_compact_when_the_window_is_narrow(qapp):
    win = appmod.MainWindow()
    try:
        win.show()
        win.resize(1600, 800)
        qapp.processEvents()
        assert not win.gauge.compact
        win.resize(1000, 800)
        qapp.processEvents()
        assert win.gauge.compact
        win.resize(1600, 800)
        qapp.processEvents()
        assert not win.gauge.compact
    finally:
        win.close()


# --- caches are kept per look -------------------------------------------------


def _playback(tmp_path):
    from ultimate_video_3d.player import Clock

    return buffered.BufferedPlayback(Clock(), lambda ms: None, lambda: None, cache_dir=tmp_path / "cache" / "1")


def test_each_look_keeps_its_own_cache_and_switching_back_finds_it_complete(tmp_path):
    play = _playback(tmp_path)
    try:
        output = preview.PreviewParams(mode="output")
        anaglyph = preview.PreviewParams(mode="anaglyph")
        a = play._select(output)
        for i in range(10):
            a.put(i, buffered.CachedFrame([b"x" * 20], "", False, 0.0))
        b = play._select(anaglyph)
        assert b is not a and len(b) == 0 and len(a) == 10             # a new look starts empty, the old one is untouched
        b.put(0, buffered.CachedFrame([b"y" * 20], "", False, 0.0))
        again = play._select(output)
        assert again is a and len(again) == 10 and again.get(9).load()[0] == [b"x" * 20]   # nothing was rebuilt
        assert play.cache is a
        play.proxy_height = 360                                        # another proxy size is another picture
        assert play._select(output) is not a
    finally:
        play.shutdown()
    assert not (tmp_path / "cache" / "1").exists()                     # ... and it all goes when the app closes


def test_playing_does_not_evict_frames_behind_the_playhead():
    src = (Path(buffered.__file__)).read_text(encoding="utf-8")
    assert "evict_before(index" not in src                             # the display thread no longer drops the past


def test_looks_are_capped_and_the_oldest_unused_one_goes_first(tmp_path):
    play = _playback(tmp_path)
    try:
        first = play._select(preview.PreviewParams(mode="output", temporal=1.0))
        first.put(0, buffered.CachedFrame([b"x"], "", False, 0.0))
        for n in range(2, buffered.MAX_LOOKS + 3):
            play._select(preview.PreviewParams(mode="output", temporal=float(n)))
        assert len(play._caches) == buffered.MAX_LOOKS
        assert first not in play._caches.values()
    finally:
        play.shutdown()


def test_housekeeping_drops_other_looks_before_the_one_on_screen(tmp_path, monkeypatch):
    play = _playback(tmp_path)
    try:
        old = play._select(preview.PreviewParams(mode="output"))
        for i in range(50):
            old.put(i, buffered.CachedFrame([b"x" * 100], "", False, 0.0))
        current = play._select(preview.PreviewParams(mode="anaglyph"))
        for i in range(50):
            current.put(i, buffered.CachedFrame([b"y" * 100], "", False, 0.0))
        monkeypatch.setattr(buffered, "CACHE_LIMIT_BYTES", 6000)       # 10 000 held, 6 000 allowed
        play._housekeeping(25)
        assert old not in play._caches.values() and play.cache is current and len(current) == 50
    finally:
        play.shutdown()


def test_a_nearly_full_disk_stops_the_cache_growing(tmp_path, monkeypatch):
    play = _playback(tmp_path)
    try:
        class _Usage:
            free = 1 << 30                                             # under the 2 GB reserve
        monkeypatch.setattr(buffered.shutil, "disk_usage", lambda _p: _Usage())
        assert play._allowed_bytes(5000) == 5000                       # no room to grow: the current size is the ceiling
        _Usage.free = 100 << 30
        assert play._allowed_bytes(5000) == buffered.CACHE_LIMIT_BYTES
    finally:
        play.shutdown()


def test_a_keyframe_only_scrub_picture_does_not_move_the_playhead(qapp):
    from types import SimpleNamespace

    win = appmod.MainWindow()
    try:
        seen = []
        win.player.position_changed.connect(seen.append)
        win.player._feeder = SimpleNamespace(approximate=True)
        win.player._delivered(np.zeros((4, 4, 3), np.float32), 0, 0.0)
        assert seen == []                                              # frame 0 of a scrub must not yank the playhead back
        win.player._feeder.approximate = False
        win.player._delivered(np.zeros((4, 4, 3), np.float32), 72, 3000.0)
        assert seen == [3000]
        win.player._feeder = None
    finally:
        win.close()


def test_a_plain_click_on_the_timeline_seeks_precisely_without_a_keyframe_flash(qapp):
    win = appmod.MainWindow()
    try:
        calls = []
        win.player.seek_ms = lambda ms, precise=True: calls.append((ms, precise))
        win._scrub(72)                                                 # press
        win._scrub_done()                                              # release, straight away
        _spin(qapp, 0.2)
        assert calls == [(3000, True)] or all(c[1] for c in calls)      # only the exact seek, never a keyframe one
        calls.clear()
        win._scrub(48)                                                 # a drag that lasts
        _spin(qapp, 0.25)
        assert calls and calls[0][1] is False                          # now the keyframe preview kicks in
        win._scrub_done()
        assert calls[-1][1] is True
    finally:
        win.close()


# --- the support banner --------------------------------------------------------


def _shown_window(qapp):
    win = appmod.MainWindow()
    win.show()
    qapp.processEvents()
    return win


def test_banner_has_a_close_button_right_of_the_yellow_one(qapp):
    win = _shown_window(qapp)
    try:
        banner = win.banner
        yellow, close = banner._button, banner._close
        assert yellow.text().endswith("Buy me a coffee") and close.text() == "\u2715"
        assert close.geometry().left() >= yellow.geometry().right()            # the X is to its right
    finally:
        win.close()


def test_closing_the_banner_lasts_for_the_session_only(qapp):
    win = _shown_window(qapp)
    try:
        win.banner._close.click()
        qapp.processEvents()
        assert not win.banner.isVisible()
        assert win._guard_support() and not win.banner.isVisible()             # closing is not tampering
        assert win._support_strikes == 0
        # Nothing about it is written down: settings on disk know nothing of it ...
        assert "banner" not in repr(win.settings.__dict__).lower() and "coffee" not in repr(win.settings.__dict__).lower()
    finally:
        win.close()
    again = _shown_window(qapp)                                                # ... so the next launch shows it again
    try:
        assert again.banner.isVisible() and not again.banner.closed_for_session
    finally:
        again.close()


def test_the_button_opens_the_official_link_whatever_the_widget_says(qapp, monkeypatch):
    from ultimate_video_3d import support

    opened = []
    monkeypatch.setattr(support.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    win = _shown_window(qapp)
    try:
        win.banner._button.setProperty("url", "https://example.com/mine")
        win.banner._button.click()
        assert opened == [support.official_url()] and "buymeacoffee.com" in opened[0]
    finally:
        win.close()


def test_a_tampered_banner_is_put_back_and_a_repeat_offender_cannot_convert(qapp):
    win = _shown_window(qapp)
    try:
        assert win._guard_support()
        for tamper in (
            lambda b: b._text.setText("Buy ME a coffee instead"),
            lambda b: b.hide(),
            lambda b: b.setStyleSheet("QFrame { background: black; }"),
            lambda b: b._button.setText("Donate to someone else"),
            lambda b: b.setFixedHeight(2),
        ):
            win._support_strikes = 0
            tamper(win.banner)
            assert not win._guard_support(), "the change was not noticed"
            qapp.processEvents()
            assert win.banner.isVisible() and win.banner.verify()              # the official banner is back
        win._support_strikes = 0
        win.banner.hide()
        win._guard_support()
        win.banner.hide()
        win._guard_support()                                                   # second time: locked
        assert not win.video_page.start.isEnabled() and not win.tab_apply.isEnabled()
        assert "official version" in win.status.currentMessage()
    finally:
        win.close()


def test_a_removed_banner_is_rebuilt(qapp):
    win = _shown_window(qapp)
    try:
        win._root.removeWidget(win.banner)
        win.banner.deleteLater()
        qapp.processEvents()
        assert not win._guard_support()
        qapp.processEvents()
        assert win.banner.isVisible() and win._root.itemAt(win._root.count() - 1).widget() is win.banner
    finally:
        win.close()


def test_the_link_and_message_are_not_plain_strings_and_edits_to_the_blob_are_caught(monkeypatch):
    from ultimate_video_3d import support

    source = Path(support.__file__).read_text(encoding="utf-8")
    assert "buymeacoffee" not in source and "criso2hd" not in source           # nothing to find and swap
    assert support.intact() and support.official_url().startswith("https://buymeacoffee.com/")
    monkeypatch.setattr(support, "_URL", support._unmask.__globals__["base64"].b64encode(b"x" * 20).decode())
    assert not support.intact()                                                # a swapped link no longer matches its digest
    with pytest.raises(RuntimeError, match="modified"):
        support.require_intact()
    assert appmod._SUPPORT_PIN == support.DIGEST[:16]                          # app.py and support.py agree


def test_conversion_refuses_to_start_if_the_banner_data_was_edited(monkeypatch, tmp_path):
    from ultimate_video_3d import pipeline, settings as settingsmod, support

    monkeypatch.setattr(support, "DIGEST", "0" * 64)
    with pytest.raises(RuntimeError, match="modified"):
        next(pipeline.convert_video(tmp_path / "a.mp4", tmp_path / "b.mp4", settingsmod.AppSettings()))
