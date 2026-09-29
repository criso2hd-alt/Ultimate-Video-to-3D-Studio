"""The update notice: version comparison and the GitHub check (against a local fake server)."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from ultimate_video_3d import updates  # noqa: E402


@pytest.mark.parametrize("candidate,current,expected", [
    ("0.1.1", "0.1.0", True), ("0.2.0", "0.1.9", True), ("1.0.0", "0.9.9", True),
    ("0.10.0", "0.9.0", True),                 # numeric, not alphabetical
    ("0.1.0", "0.1.0", False), ("0.1.0", "0.1.1", False), ("0.9.0", "0.10.0", False),
    ("v0.2.0", "0.1.0", True), ("V0.2.0", "0.1.0", True),
    ("0.1.0.0", "0.1.0", False),               # trailing zeros are not a new version
    ("0.1.0.1", "0.1.0", True),
    ("0.2.0-beta1", "0.1.0", True),
    ("0.2.0-beta1", "0.2.0", False),           # a pre-release is older than its own final
    ("0.2.0", "0.2.0-beta1", True),
    ("garbage", "0.1.0", False), ("", "0.1.0", False),
])
def test_is_newer(candidate, current, expected):
    assert updates.is_newer(candidate, current) is expected


def test_parse_rejects_non_versions():
    with pytest.raises(ValueError):
        updates.parse_version("latest")


class _Handler(BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802
        status, body = self.routes.get(self.path, (404, {"message": "Not Found"}))
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _Handler.routes = {}
    yield f"http://127.0.0.1:{httpd.server_port}", _Handler.routes
    httpd.shutdown()


def test_reports_a_newer_release(server):
    base, routes = server
    routes["/latest"] = (200, {"tag_name": "v0.2.0", "name": "Second", "body": "Faster.",
                               "html_url": "https://example.test/rel", "draft": False, "prerelease": False})
    info = updates.check_for_update("0.1.0", base + "/latest")
    assert info == updates.UpdateInfo("0.2.0", "https://example.test/rel", "Faster.", "Second")


def test_no_notice_when_up_to_date_or_older(server):
    base, routes = server
    routes["/latest"] = (200, {"tag_name": "v0.1.0", "html_url": "x"})
    assert updates.check_for_update("0.1.0", base + "/latest") is None
    assert updates.check_for_update("0.3.0", base + "/latest") is None


def test_drafts_and_prereleases_are_ignored(server):
    base, routes = server
    routes["/d"] = (200, {"tag_name": "v9.0.0", "draft": True})
    routes["/p"] = (200, {"tag_name": "v9.0.0", "prerelease": True})
    assert updates.check_for_update("0.1.0", base + "/d") is None
    assert updates.check_for_update("0.1.0", base + "/p") is None


@pytest.mark.parametrize("status,body", [
    (404, {"message": "Not Found"}),           # a private repository, or no releases yet
    (403, {"message": "API rate limit exceeded"}),
    (500, {"message": "boom"}),
    (200, b"this is not json"),
    (200, {"no_tag_here": True}),
    (200, [1, 2, 3]),
])
def test_every_failure_is_silent(server, status, body):
    base, routes = server
    routes["/latest"] = (status, body)
    assert updates.check_for_update("0.1.0", base + "/latest") is None


def test_unreachable_server_is_silent():
    assert updates.check_for_update("0.1.0", "http://127.0.0.1:9/nothing", timeout=1.0) is None


def test_window_shows_the_notice_and_links_to_the_page(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d import app as appmod

    qt = QApplication.instance() or QApplication(sys.argv)
    win = appmod.MainWindow()
    opened = []
    monkeypatch.setattr(appmod.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    try:
        assert not win.update_pill.isVisibleTo(win) and win.settings.check_updates
        win._update_checked(None)
        assert not win.update_pill.isVisibleTo(win)                     # nothing newer: no notice
        win._update_checked(updates.UpdateInfo("9.9.9", "https://example.test/download"))
        assert win.update_pill.isVisibleTo(win) and "9.9.9" in win.update_pill.text()
        win.update_pill.click()
        assert opened == ["https://example.test/download"]
    finally:
        win.close()
        qt.processEvents()


def test_settings_page_scrolls_instead_of_squashing_its_cards():
    """Regression: five cards overflowed a short window and were crushed on top of each other."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QScrollArea

    from ultimate_video_3d import app as appmod

    qt = QApplication.instance() or QApplication(sys.argv)
    win = appmod.MainWindow()
    try:
        assert win.settings_page.findChildren(QScrollArea)
    finally:
        win.close()
        qt.processEvents()
