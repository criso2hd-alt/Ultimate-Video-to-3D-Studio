"""Reading a disc: titles from the DVD / Blu-ray tables, and copying one to a file. All on synthetic discs
built from small generated clips (no real disc, no drive, no copyrighted footage)."""

from __future__ import annotations

import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

av = pytest.importorskip("av")

from ultimate_video_3d import disc, video  # noqa: E402

SECTOR = disc.SECTOR


def _clip(path: Path, fmt: str, video_codec: str, audio_codec: str, seconds=2.0, fps=24, start_at=0.0) -> None:
    """A small colour-bar clip with a tone. ``start_at`` shifts the timestamps (a disc clip rarely starts at 0)."""
    w, h = 320, 176
    with av.open(str(path), "w", format=fmt) as out:
        v = out.add_stream(video_codec, rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.options = {"g": "12", "bf": "0"}
        a = out.add_stream(audio_codec, rate=48000)
        a.layout = "stereo"
        for i in range(int(seconds * fps)):
            img = np.zeros((h, w, 3), np.uint8)
            img[:, :, 0] = (np.arange(w) * 255 // w)[None, :]
            img[:, :, 1] = (i * 9) % 255
            img[20:60, 10 + i * 3 % 200:70 + i * 3 % 200] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i + int(start_at * fps)
            frame.time_base = Fraction(1, fps)
            for packet in v.encode(frame):
                out.mux(packet)
        for packet in v.encode():
            out.mux(packet)
        rate, block = 48000, 1536
        for n in range(int(seconds * rate) // block):
            tone = np.sin(2 * np.pi * 440 * (np.arange(block) + n * block) / rate).astype(np.float32) * 0.3
            frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = n * block + int(start_at * rate)
            frame.time_base = Fraction(1, rate)
            for packet in a.encode(frame):
                out.mux(packet)
        for packet in a.encode():
            out.mux(packet)


def _bcd(n: int) -> int:
    return (n // 10) << 4 | n % 10


def _time(seconds: int) -> bytes:
    return bytes([_bcd(seconds // 3600), _bcd(seconds // 60 % 60), _bcd(seconds % 60), 0xC0])   # 30 fps


def _ifo(pgcs: list[tuple[int, list[tuple[int, int]]]]) -> bytes:
    """A minimal VTS_nn_0.IFO: header magic, the PGCI pointer, and one chain per (seconds, [(first, last) sectors])."""
    data = bytearray(SECTOR * 8)
    data[:12] = b"DVDVIDEO-VTS"
    data[0xCC:0xD0] = struct.pack(">I", 1)                               # the table sits at sector 1
    base = SECTOR
    data[base:base + 2] = struct.pack(">H", len(pgcs))
    offset = 8 + 8 * len(pgcs)
    for n, (seconds, runs) in enumerate(pgcs):
        data[base + 8 + n * 8 + 4:base + 8 + n * 8 + 8] = struct.pack(">I", offset)
        pgc = base + offset
        data[pgc + 2], data[pgc + 3] = 1, len(runs)
        data[pgc + 4:pgc + 8] = _time(seconds)
        data[pgc + 0xE8:pgc + 0xEA] = struct.pack(">H", 0xEC)
        for c, (first, last) in enumerate(runs):
            cell = pgc + 0xEC + c * 24
            data[cell + 8:cell + 12] = struct.pack(">I", first)
            data[cell + 20:cell + 24] = struct.pack(">I", last)
        offset += 0xEC + 24 * len(runs) + 8
    return bytes(data)


def _pad(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % SECTOR)


@pytest.fixture(scope="module")
def dvd(tmp_path_factory):
    """VTS_01: one 4 s feature made of two cells whose clocks both restart, and a 10 s menu-length decoy."""
    root = tmp_path_factory.mktemp("disc")
    ts = root / "VIDEO_TS"
    ts.mkdir()
    a, b, decoy = root / "a.vob", root / "b.vob", root / "decoy.vob"
    _clip(a, "vob", "mpeg2video", "ac3", seconds=2.0, start_at=1.0)
    _clip(b, "vob", "mpeg2video", "ac3", seconds=2.0, start_at=1.0)      # same clock again: a restart
    _clip(decoy, "vob", "mpeg2video", "ac3", seconds=1.0)
    da, db, dd = _pad(a.read_bytes()), _pad(b.read_bytes()), _pad(decoy.read_bytes())
    (ts / "VTS_01_1.VOB").write_bytes(da + db + dd)
    sa, sb, sd = len(da) // SECTOR, len(db) // SECTOR, len(dd) // SECTOR
    ifo = _ifo([(120, [(0, sa - 1), (sa, sa + sb - 1)]), (10, [(sa + sb, sa + sb + sd - 1)])])
    (ts / "VTS_01_0.IFO").write_bytes(ifo)
    (ts / "VIDEO_TS.IFO").write_bytes(b"DVDVIDEO-VMG")
    return root


def test_dvd_is_recognised_from_the_drive_the_folder_or_video_ts(dvd):
    assert disc.disc_root(dvd) == (dvd, disc.DVD)
    assert disc.disc_root(dvd / "VIDEO_TS") == (dvd, disc.DVD)
    assert disc.disc_root(dvd.parent) is None or disc.disc_root(dvd.parent)[1] == disc.DVD


def test_dvd_titles_skip_the_short_decoy_and_keep_the_feature(dvd):
    titles = disc.list_titles(dvd, disc.DVD)
    assert len(titles) == 1
    assert titles[0].seconds == 120 and len(titles[0].cells) == 2


def test_dvd_copy_gives_one_continuous_file_with_picture_and_sound(dvd, tmp_path):
    title = disc.list_titles(dvd, disc.DVD)[0]
    seen = []
    out = disc.copy_title(title, tmp_path / "out.mkv", lambda d, t: seen.append((d, t)))
    assert out.exists() and not out.with_name(out.name + ".part").exists()
    assert seen[-1][0] == seen[-1][1]
    with av.open(str(out)) as f:
        assert f.streams.video and f.streams.audio
        video_dts, audio_dts, audio_end = [], [], 0.0
        for packet in f.demux():
            if packet.dts is None:
                continue
            t = float(packet.dts * packet.time_base)
            if packet.stream.type == "video":
                video_dts.append(t)
            else:
                audio_dts.append(t)
                audio_end = max(audio_end, t + float((packet.duration or 0) * packet.time_base))
        assert video_dts == sorted(video_dts) and audio_dts == sorted(audio_dts)   # never steps back at the boundary
        assert 3.6 < video_dts[-1] - video_dts[0] < 4.6                    # two 2 s cells end to end, not overlapped
        assert abs(audio_end - (video_dts[-1] + 1 / 24)) < 0.4             # and the sound ends where the picture does


def test_copy_can_be_cancelled_and_leaves_nothing_behind(dvd, tmp_path):
    title = disc.list_titles(dvd, disc.DVD)[0]
    with pytest.raises(disc.Stopped):
        disc.copy_title(title, tmp_path / "x.mkv", should_stop=lambda: True)
    assert list(tmp_path.iterdir()) == []


def test_a_scrambled_title_is_reported_not_turned_into_a_broken_file(tmp_path):
    root = tmp_path / "d"
    (root / "VIDEO_TS").mkdir(parents=True)
    rng = np.random.default_rng(3)
    (root / "VIDEO_TS" / "VTS_01_1.VOB").write_bytes(rng.integers(0, 256, SECTOR * 400, dtype=np.uint8).tobytes())
    title = disc.list_titles(root, disc.DVD)[0]
    with pytest.raises(disc.DiscError):
        disc.copy_title(title, tmp_path / "o.mkv")
    assert not (tmp_path / "o.mkv").exists()


def _mpls(items: list[tuple[str, float]]) -> bytes:
    body = bytearray()
    for clip, seconds in items:
        item = clip.encode() + b"M2TS" + b"\0\0" + b"\0" + struct.pack(">II", 0, int(seconds * 45000))
        body += struct.pack(">H", len(item)) + item
    header = b"MPLS0200" + struct.pack(">I", 40) + b"\0" * 28
    # PlayList(): length(4) reserved(2) items(2) subitems(2), then the play items
    return header + struct.pack(">I", 10 + len(body)) + b"\0\0" + struct.pack(">HH", len(items), 0) + bytes(body)


@pytest.fixture(scope="module")
def bluray(tmp_path_factory):
    root = tmp_path_factory.mktemp("bd")
    (root / "BDMV" / "STREAM").mkdir(parents=True)
    (root / "BDMV" / "PLAYLIST").mkdir()
    for name, start in (("00001", 10.0), ("00002", 10.0)):
        _clip(root / "BDMV" / "STREAM" / f"{name}.m2ts", "mpegts", "libx264", "ac3", seconds=2.0, start_at=start)
    (root / "BDMV" / "PLAYLIST" / "00000.mpls").write_bytes(_mpls([("00001", 90.0), ("00002", 90.0)]))
    (root / "BDMV" / "PLAYLIST" / "00001.mpls").write_bytes(_mpls([("00001", 5.0)]))          # a menu loop
    return root


def test_bluray_playlists_list_the_feature_and_skip_menu_loops(bluray):
    assert disc.disc_root(bluray) == (bluray, disc.BLURAY)
    titles = disc.list_titles(bluray, disc.BLURAY)
    assert len(titles) == 1 and titles[0].seconds == pytest.approx(180)
    assert [p.name for p in titles[0].parts] == ["00001.M2TS", "00002.M2TS"] or [p.name for p in titles[0].parts] == ["00001.m2ts", "00002.m2ts"]


def test_bluray_copy_joins_the_clips_end_to_end(bluray, tmp_path):
    title = disc.list_titles(bluray, disc.BLURAY)[0]
    out = disc.copy_title(title, tmp_path / "bd.mkv")
    with av.open(str(out)) as f:
        dts = [float(p.dts * p.time_base) for p in f.demux(f.streams.video[0]) if p.dts is not None]
    assert dts == sorted(dts) and 3.6 < dts[-1] - dts[0] < 4.6


def test_no_disc_means_no_titles(tmp_path):
    assert disc.disc_root(tmp_path) is None
    assert disc.find_drives() is not None                                   # never raises, whatever the machine has


def test_safe_name():
    assert disc.safe_name("MY_MOVIE: Disc 1?") == "MY MOVIE Disc 1"
    assert disc.safe_name("???") == "Disc"


# --- the dialog -------------------------------------------------------------------------------------


def _wait(app, condition, seconds=20.0):
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_dialog_lists_the_disc_copies_the_main_title_and_hands_back_the_file(dvd, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d.discdialog import DiscDialog

    monkeypatch.setattr(disc, "find_drives", lambda: [])                     # a real disc in this machine must not interfere
    app = QApplication.instance() or QApplication([])
    dialog = DiscDialog(folder=tmp_path)
    dialog._manual.append(disc.Drive(dvd, "MY MOVIE", disc.DVD))
    dialog.reload_drives()
    dialog.drives.setCurrentIndex(dialog.drives.findText("MY MOVIE", Qt.MatchFlag.MatchContains))
    assert _wait(app, lambda: dialog.titles.count() == 1)
    assert "0:02:00" in dialog.titles.item(0).text() and dialog.copy.isEnabled()
    assert "own" in dialog.NOTICE                                             # the reminder is on screen
    dialog.show()
    dialog.copy.click()
    assert _wait(app, lambda: dialog.copied is not None)
    assert dialog.copied.name == "MY MOVIE - Title 1.mkv" and dialog.copied.stat().st_size > 10_000
    dialog.close()


def test_dialog_with_no_disc_says_so(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d import discdialog

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(discdialog.disc, "find_drives", lambda: [])
    dialog = discdialog.DiscDialog(folder=tmp_path)
    assert _wait(app, lambda: "No disc found" in dialog.status.text())
    assert not dialog.copy.isEnabled()
    dialog.close()


def test_plain_discs_are_not_flagged_and_noise_is(dvd, bluray, tmp_path):
    assert not disc.is_scrambled(disc.list_titles(dvd, disc.DVD)[0])
    assert not disc.is_scrambled(disc.list_titles(bluray, disc.BLURAY)[0])
    root = tmp_path / "bd"
    (root / "BDMV" / "STREAM").mkdir(parents=True)
    (root / "BDMV" / "PLAYLIST").mkdir()
    noise = np.random.default_rng(1).integers(0, 256, 192 * 3000, dtype=np.uint8).tobytes()
    (root / "BDMV" / "STREAM" / "00001.m2ts").write_bytes(noise)
    (root / "BDMV" / "PLAYLIST" / "00000.mpls").write_bytes(_mpls([("00001", 120.0)]))
    title = disc.list_titles(root, disc.BLURAY)[0]
    assert disc.is_scrambled(title)
    with pytest.raises(disc.DiscError, match="copy-protected"):
        disc.copy_title(title, tmp_path / "o.mkv")


def test_a_title_can_be_converted_straight_from_the_disc(dvd, tmp_path):
    from ultimate_video_3d import encoders, pipeline, settings as settingsmod

    title = disc.list_titles(dvd, disc.DVD)[0]
    stream = disc.DiscStream(title, "MY MOVIE", tmp_path)
    assert stream.stem == "MY MOVIE - Title 1" and stream.info.frames > 0 and stream.info.width >= 320
    s = settingsmod.AppSettings()
    s.export.family = "h264"
    s.export.encoder = "libx264"
    s.threed.enabled = True
    dst = tmp_path / "out.mp4"
    stages = [u.stage for u in pipeline.convert_video(stream, dst, s, gpus=[])]
    assert stages[-1] == "done" and dst.exists()
    info = video.probe(dst)
    assert info.has_audio and 3.0 < info.duration < 5.0                     # two 2 s cells, picture and sound
    assert stream.audio_path is None and not list(tmp_path.glob("*.audio.mkv"))   # temporary sound file removed


def test_a_sample_from_part_way_in_is_short_and_starts_at_zero(dvd, tmp_path):
    title = disc.list_titles(dvd, disc.DVD)[0]
    out = disc.copy_title(title, tmp_path / "s.mkv", start_fraction=0.3, max_seconds=1.0)
    with av.open(str(out)) as f:
        dts = [float(p.dts * p.time_base) for p in f.demux(f.streams.video[0]) if p.dts is not None]
    assert dts and dts[0] < 0.5 and 0.5 < dts[-1] - dts[0] < 1.6


def test_dialog_direct_mode_hands_back_a_stream_and_a_short_sample(dvd, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ultimate_video_3d import discdialog, paths

    monkeypatch.setattr(disc, "find_drives", lambda: [])
    monkeypatch.setattr(paths, "scratch_dir", lambda: tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = discdialog.DiscDialog(folder=tmp_path)
    dialog._manual.append(disc.Drive(dvd, "MY MOVIE", disc.DVD))
    dialog.reload_drives()
    assert _wait(app, lambda: dialog.titles.count() == 1)
    dialog.mode_direct.setChecked(True)
    assert dialog.copy.text() == "Prepare preview"
    dialog.show()
    dialog.copy.click()
    assert _wait(app, lambda: dialog.stream is not None)
    assert dialog.copied is None and dialog.sample.exists()
    assert dialog.stream.info.frames > 0 and dialog.stream.stem == "MY MOVIE - Title 1"
    dialog.close()


def test_main_window_converts_a_disc_title_whole_and_names_the_file_after_the_disc(dvd, tmp_path, monkeypatch):
    from ultimate_video_3d import app as appmod

    monkeypatch.setattr(appmod.paths, "settings_path", lambda: tmp_path / "settings.json")
    app = appmod.QApplication.instance() or appmod.QApplication([])
    window = appmod.MainWindow()
    try:
        title = disc.list_titles(dvd, disc.DVD)[0]
        stream = disc.DiscStream(title, "MY MOVIE", tmp_path)
        sample = disc.copy_title(title, tmp_path / "s.mkv", start_fraction=0.0, max_seconds=1.5)
        window._open_disc_stream(stream, sample)
        page = window.video_page
        assert page.disc_stream is stream and not page.export.range_box.isEnabled()
        assert page.output_path.name.startswith("MY MOVIE - Title 1")
        window._open_path(sample)                                           # opening a plain file leaves disc mode
        assert page.disc_stream is None and page.export.range_box.isEnabled()
    finally:
        window.close()
