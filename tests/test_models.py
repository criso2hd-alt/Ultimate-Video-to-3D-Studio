"""The depth model's download from the project's own GitHub release, against a local stand-in server."""

from __future__ import annotations

import ast
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("UV3D_SKIP_TOUR", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from ultimate_video_3d import models, paths  # noqa: E402

PAYLOAD = (b"onnx-model-bytes-" * 40000)[:600_000]


class _FakeResponse:
    """What ``urlopen`` returns, served from memory (real localhost sockets stall on some PCs with
    security software, and this tests the same logic without depending on that)."""

    def __init__(self, status, headers, body):
        self.status, self.headers, self._body, self._pos = status, headers, body, 0

    def read(self, n=-1):
        chunk = self._body[self._pos:] if n is None or n < 0 else self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payload: bytes, supports_range: bool):
    def urlopen(request, timeout=None):
        if isinstance(request, str):
            return _FakeResponse(200, {"Content-Length": str(len(payload))}, payload)
        start = 0
        rng = request.headers.get("Range")
        status = 200
        if supports_range and rng and rng.startswith("bytes="):
            start, status = int(rng[6:].split("-")[0]), 206
        body = b"" if request.get_method() == "HEAD" else payload[start:]
        length = len(payload) - start
        return _FakeResponse(status, {"Content-Length": str(length)}, body)

    return urlopen


@pytest.fixture()
def server(monkeypatch, tmp_path):
    """A stand-in 'GitHub release', and the model folder pointed at a temp directory."""
    from ultimate_video_3d import bootstrap

    monkeypatch.setattr(paths, "model_dir", lambda: tmp_path)

    def start(payload=PAYLOAD, supports_range=True, sha=None):
        monkeypatch.setattr(bootstrap, "urlopen", _fake_urlopen(payload, supports_range))
        monkeypatch.setattr(models, "URL", f"https://example.invalid/{models.NAME}")
        monkeypatch.setattr(models, "SHA256", sha or hashlib.sha256(payload).hexdigest())
        return tmp_path

    return start


def test_downloads_verifies_and_places_the_model(server):
    folder = server()
    seen, texts = [], []
    path = models.install(on_bytes=lambda d, t: seen.append((d, t)), on_text=texts.append)
    assert path == folder / models.NAME and path.read_bytes() == PAYLOAD
    assert not (folder / (models.NAME + ".part")).exists()               # no temp file left behind
    assert seen and seen[-1][0] == seen[-1][1] == len(PAYLOAD)           # progress reached 100%
    assert any("integrity" in t.lower() or "Checking" in t for t in texts)


def test_a_partial_download_resumes_instead_of_starting_over(server):
    folder = server()
    (folder / (models.NAME + ".part")).write_bytes(PAYLOAD[:250_000])    # an interrupted earlier attempt
    starts = []
    models.install(on_bytes=lambda d, t: starts.append(d))
    assert (folder / models.NAME).read_bytes() == PAYLOAD
    assert min(starts) >= 250_000                                        # it carried on from where it stopped


def test_a_download_that_fails_its_checksum_is_discarded(server):
    folder = server(sha="0" * 64)
    with pytest.raises(OSError, match="integrity"):
        models.install()
    assert not (folder / models.NAME).exists() and not (folder / (models.NAME + ".part")).exists()


def test_a_server_without_resume_support_still_works(server):
    folder = server(supports_range=False)
    (folder / (models.NAME + ".part")).write_bytes(b"stale-partial")
    models.install()
    assert (folder / models.NAME).read_bytes() == PAYLOAD


def test_an_unreachable_server_gives_a_plain_message(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "model_dir", lambda: tmp_path)
    from ultimate_video_3d import bootstrap

    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: None)

    def refuse(*args, **kwargs):
        raise ConnectionRefusedError("no route")

    monkeypatch.setattr(bootstrap, "urlopen", refuse)
    with pytest.raises(OSError):
        models.install()
    assert not (tmp_path / models.NAME).exists()


def test_the_source_setup_script_points_at_the_same_file():
    """scripts/get_model.py is standalone, so its copy of the address and checksum must not drift."""
    tree = ast.parse((Path(__file__).resolve().parent.parent / "scripts" / "get_model.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert values["NAME"] == models.NAME and values["SHA256"] == models.SHA256
    assert models.URL.endswith("/releases/download/models-v1/" + models.NAME)
    assert "criso2hd-alt/Ultimate-Video-to-3D-Studio" in models.URL


def test_the_bundled_model_matches_the_published_checksum():
    bundled = paths.bundled_onnx_dir() / models.NAME
    if not bundled.is_file():
        pytest.skip("model not present in this checkout")
    assert models._sha256_of(bundled) == models.SHA256                    # what we upload is what the app expects


def test_the_app_downloads_a_missing_model_and_then_carries_on(monkeypatch, server):
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d import app as appmod
    from ultimate_video_3d import depth

    qapp = QApplication.instance() or QApplication([])
    folder = server()
    monkeypatch.setattr(appmod.paths, "settings_path", lambda: folder / "settings.json")
    win = appmod.MainWindow()
    try:
        ran = []
        monkeypatch.setattr(depth, "is_installed", lambda: False)
        monkeypatch.setattr(win, "_download_depth_model", lambda then: ran.append(then))
        win._after_video_support()
        assert len(ran) == 1                                                # missing model -> download, no "go to GitHub" box
        # And the real worker, run to completion, reports success and leaves the file in the models folder.
        worker = appmod.ModelDownloadWorker()
        done, failed = [], []
        worker.finished.connect(lambda: done.append(1))
        worker.failed.connect(failed.append)
        worker.run()
        assert done == [1] and not failed and (folder / models.NAME).read_bytes() == PAYLOAD
    finally:
        win.close()


def test_a_dropped_connection_is_retried_and_resumes(monkeypatch, tmp_path):
    """One reset mid-download must not fail it: the second attempt carries on from the saved bytes."""
    from ultimate_video_3d import bootstrap

    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: None)
    real = bootstrap._download_once
    calls = []

    def flaky(url, destination, on_bytes):
        calls.append(destination.stat().st_size if destination.exists() else 0)
        if len(calls) == 1:
            destination.write_bytes(PAYLOAD[:100_000])                   # got part of it ...
            raise OSError("The download stalled after 0 MB.")             # ... then the connection was reset
        real(url, destination, on_bytes)

    monkeypatch.setattr(bootstrap, "_download_once", flaky)
    monkeypatch.setattr(bootstrap, "urlopen", _fake_urlopen(PAYLOAD, True))
    target = tmp_path / "f.part"
    bootstrap._download("https://example.invalid/x", target, None)
    assert calls == [0, 100_000] and target.read_bytes() == PAYLOAD


def test_the_download_gives_up_with_a_plain_message_after_a_few_tries(monkeypatch, tmp_path):
    from ultimate_video_3d import bootstrap

    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: None)
    attempts = []

    def always_fails(url, destination, on_bytes):
        attempts.append(1)
        raise OSError("The download stalled after 0 MB.")

    monkeypatch.setattr(bootstrap, "_download_once", always_fails)
    with pytest.raises(OSError, match="stalled"):
        bootstrap._download("http://127.0.0.1:9/x", tmp_path / "f.part", None)
    assert len(attempts) == bootstrap.DOWNLOAD_ATTEMPTS
