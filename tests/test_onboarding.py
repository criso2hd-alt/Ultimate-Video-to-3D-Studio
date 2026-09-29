"""The welcome dialog and the spotlight tutorial. Qt offscreen; no GPU, no video needed."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("UV3D_SKIP_TOUR", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402
from ultimate_video_3d import onboarding, paths  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spin(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture()
def win(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(appmod.paths, "settings_path", lambda: tmp_path / "settings.json")   # never touch the real one
    window = appmod.MainWindow()
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def test_welcome_dialog_lists_the_features_and_offers_tour_or_skip(qapp):
    dialog = onboarding.WelcomeDialog(paths.package_dir() / "assets" / "icon.png")
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "any GPU" in texts and "codecs" in texts and "LUT" in texts
    assert dialog.start_button.text() == "Show me around" and dialog.skip_button.text() == "Skip tutorial"
    dialog.start_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    other = onboarding.WelcomeDialog(None)
    other.skip_button.click()
    assert other.result() == QDialog.DialogCode.Rejected


def test_every_tour_step_points_at_a_real_visible_widget(win, qapp):
    steps = win._tour_steps()
    titles = [s.title for s in steps]
    assert titles[:3] == ["Your video", "Pick a view", "Timeline"] and "Export" in titles and "Convert" in titles
    assert "Professional colour correction" in titles and not any("Lumetri" in s.title + s.body for s in steps)
    assert len(steps) >= 11
    win.start_tutorial()
    overlay = win._tour_overlay
    for i in range(len(overlay._steps)):
        overlay._index = i
        overlay._show_step(animate=False)
        _spin(qapp, 0.15)
        target = overlay._steps[i].target
        assert target.isVisible(), f"step {i + 1} ({overlay._steps[i].title}) points at a hidden widget"
        assert not overlay._target.isEmpty() and overlay.rect().contains(overlay._target)
        assert overlay.card.geometry().right() <= overlay.width() and overlay.card.geometry().bottom() <= overlay.height()
        assert overlay.card.height() >= overlay.body.height() + 60            # the wrapped text fits: nothing is cut off
    overlay._skip()


def test_rail_steps_scroll_the_side_panel_to_reach_their_card(win, qapp):
    page = win.video_page
    win.resize(1400, 760)
    _spin(qapp, 0.1)
    win.start_tutorial()
    overlay = win._tour_overlay
    export_step = next(i for i, s in enumerate(overlay._steps) if s.title == "Export")
    assert page.rail_scroller.verticalScrollBar().value() == 0
    overlay._index = export_step
    overlay._show_step(animate=False)
    _spin(qapp, 0.15)
    assert page.rail_scroller.verticalScrollBar().value() > 0               # it scrolled down to the export card
    viewport = page.rail_scroller.viewport()
    top = page.export.mapTo(viewport, page.export.rect().topLeft()).y()
    assert top < viewport.height() and top + page.export.height() > 0      # (a tall card may overhang its top edge)
    overlay._skip()


def test_next_back_and_finish(win, qapp):
    win.start_tutorial()
    overlay = win._tour_overlay
    assert overlay.step_index == 0 and not overlay.back_button.isVisibleTo(overlay)
    overlay.next_button.click()
    overlay.next_button.click()
    assert overlay.step_index == 2 and overlay.back_button.isVisibleTo(overlay)
    overlay.back_button.click()
    assert overlay.step_index == 1
    last = len(overlay._steps) - 1
    overlay._index = last
    overlay._show_step(animate=False)
    assert overlay.next_button.text() == "Done" and not overlay.skip_button.isVisibleTo(overlay)
    overlay.next_button.click()
    _spin(qapp, 0.05)
    assert win._tour_overlay is None
    assert win.settings.onboarding_version == onboarding.ONBOARDING_VERSION
    assert win.tabs.currentWidget() is win.video_page                        # the Settings step is left behind


def test_escape_skips_and_still_counts_as_seen(win, qapp):
    win.start_tutorial()
    overlay = win._tour_overlay
    overlay.setFocus()
    QTest.keyClick(overlay, Qt.Key.Key_Escape)
    _spin(qapp, 0.05)
    assert win._tour_overlay is None and win.settings.onboarding_version == onboarding.ONBOARDING_VERSION


def test_overlay_follows_the_window_when_it_is_resized(win, qapp):
    win.start_tutorial()
    win.resize(1300, 800)
    _spin(qapp, 0.1)
    assert win._tour_overlay.geometry() == win.rect()
    win._tour_overlay._skip()


def test_the_tour_is_offered_once_on_a_fresh_install_only(win, qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(win, "show_welcome", lambda: shown.append(1))
    end = time.monotonic() + 40
    while win._probe_thread is not None and time.monotonic() < end:      # the first-launch encoder test must finish first:
        _spin(qapp, 0.1)                                                 # the welcome is deliberately held back until then
    assert win._probe_thread is None
    monkeypatch.delenv("UV3D_SKIP_TOUR", raising=False)
    win.settings.onboarding_version = 0
    win._offer_tour()
    _spin(qapp, 0.9)
    assert shown == [1]                                                     # new install: welcome appears
    shown.clear()
    win._welcome_open = False
    win.settings.onboarding_version = onboarding.ONBOARDING_VERSION
    win._offer_tour()
    _spin(qapp, 0.9)
    assert shown == []                                                      # already seen: never again
    monkeypatch.setenv("UV3D_SKIP_TOUR", "1")
    win.settings.onboarding_version = 0
    win._offer_tour()
    _spin(qapp, 0.9)
    assert shown == []                                                      # (tests and harnesses opt out)


def test_source_card_has_the_file_button_and_a_disabled_disc_placeholder(win):
    page = win.video_page
    assert page.pick_video.isEnabled() and "video file" in page.pick_video.text()
    assert not page.pick_disc.isEnabled() and "coming soon" in page.pick_disc.text()
    assert "0.2" in page.pick_disc.toolTip()


def test_timeline_is_drawn_empty_before_any_video_is_open(win, qapp):
    timeline = win.video_page.timeline
    assert timeline._total == 0
    image = timeline.grab().toImage()
    bg = image.pixelColor(2, image.height() - 2)
    track_y = int(timeline._track().center().y())
    row = {image.pixelColor(x, track_y).name() for x in range(10, image.width() - 10)}
    assert len(row) > 1 and any(c != bg.name() for c in row)              # a track with tick marks, not blank background
    x0 = int(timeline._track().left())
    assert image.pixelColor(x0, track_y).lightness() > 80                  # the playhead sits at the start
    seen = []
    timeline.seeked.connect(seen.append)
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton)
    assert seen == []                                                     # there is nothing to scrub yet


def test_replay_lives_in_settings(win, qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(win, "show_welcome", lambda: shown.append(1))
    button = win.settings_page.replay_tutorial
    assert isinstance(button, QPushButton) and "tutorial" in button.text().lower()
    button.click()
    assert shown == [1]


def test_skipping_the_welcome_marks_it_seen(win, qapp, monkeypatch):
    win.settings.onboarding_version = 0

    def fake_exec(self):
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(onboarding.WelcomeDialog, "exec", fake_exec)
    win.show_welcome()
    assert win.settings.onboarding_version == onboarding.ONBOARDING_VERSION and win._tour_overlay is None
    win.settings.onboarding_version = 0
    monkeypatch.setattr(onboarding.WelcomeDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    win.show_welcome()
    _spin(qapp, 0.4)
    assert win._tour_overlay is not None                                    # "Show me around" starts the tour
    win._tour_overlay._skip()
