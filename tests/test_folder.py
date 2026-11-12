"""Opening a folder of videos (a ripped disc), choosing the audio and subtitle tracks, and converting with them."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("PySide6")
av = pytest.importorskip("av")

from PySide6.QtWidgets import QApplication  # noqa: E402
from test_tracks import make_rip  # noqa: E402

from ultimate_video_3d import app as appmod  # noqa: E402
from ultimate_video_3d import folderdialog, pipeline, video  # noqa: E402
from ultimate_video_3d import settings as settingsmod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(qapp, condition, seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture()
def disc_folder(tmp_path, monkeypatch):
    """Three files like a ripped disc: the film (longest), an extra, and a menu loop that is too small to be listed."""
    monkeypatch.setattr(folderdialog, "MIN_BYTES", 1)
    root = tmp_path / "My Movie"
    (root / "extras").mkdir(parents=True)
    make_rip(root / "title_t00.mkv", seconds=2.0)
    make_rip(root / "extras" / "title_t01.mkv", seconds=1.0)
    return root


# --- finding the files -------------------------------------------------------------------------------------------------


def test_videos_are_found_in_subfolders_but_not_too_deep_and_tiny_files_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(folderdialog, "MIN_BYTES", 100)
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (tmp_path / "one.mkv").write_bytes(b"x" * 200)
    (tmp_path / "a" / "two.m2ts").write_bytes(b"x" * 200)
    (tmp_path / "a" / "notes.txt").write_bytes(b"x" * 200)
    (tmp_path / "tiny.mkv").write_bytes(b"x" * 10)
    (deep / "too_deep.mkv").write_bytes(b"x" * 200)
    names = [p.name for p in folderdialog.find_videos(tmp_path)]
    assert sorted(names) == ["one.mkv", "two.m2ts"]
    assert [p.name for p in folderdialog.find_videos(tmp_path, recursive=False)] == ["one.mkv"]


def test_a_file_that_cannot_be_read_is_reported_without_stopping_the_scan(tmp_path):
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"this is not a video" * 100)
    entry = folderdialog.scan_file(broken)
    assert entry.error and entry.info is None and entry.audio == []


# --- the dialog -----------------------------------------------------------------------------------------------------------


def test_the_dialog_lists_each_file_longest_first_with_its_tracks(qapp, disc_folder):
    dialog = folderdialog.FolderDialog(None, disc_folder)
    assert _wait(qapp, lambda: dialog.tree.topLevelItemCount() == 2 and dialog._thread is None or "found" in dialog.status.text())
    names = [dialog.tree.topLevelItem(i).text(0) for i in range(dialog.tree.topLevelItemCount())]
    assert names == ["title_t00.mkv", "extras\\title_t01.mkv"] or names == ["title_t00.mkv", "extras/title_t01.mkv"]   # the film first
    film = dialog.tree.topLevelItem(0)
    assert film.text(4).startswith("3:") and "English" in film.text(4) and "French" in film.text(4)
    assert film.text(5).startswith("2:")
    assert "commentary" in film.toolTip(4) and "Dolby Digital 5.1" in film.toolTip(4)
    dialog.reject()


def test_the_best_audio_is_chosen_for_each_file_and_a_subtitle_switches_on_mkv(qapp, disc_folder):
    dialog = folderdialog.FolderDialog(None, disc_folder)
    assert _wait(qapp, lambda: "found" in dialog.status.text())
    film = disc_folder / "title_t00.mkv"
    audio, subs = video.list_tracks(film)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
    assert dialog.choice[film] == (audio[0].index, -1)                      # the main mix, not the commentary
    assert dialog.audio_box.count() == 4 and dialog.subtitle_box.count() == 3
    assert not dialog.mkv.isChecked()
    dialog.audio_box.setCurrentIndex(dialog.audio_box.findData(audio[1].index))        # the French track
    dialog.subtitle_box.setCurrentIndex(dialog.subtitle_box.findData(subs[1].index))   # French subtitles
    assert dialog.choice[film] == (audio[1].index, subs[1].index)
    assert dialog.mkv.isChecked()                                             # subtitles can only live in an MKV
    dialog.reject()


def test_opening_one_video_returns_it_with_its_tracks(qapp, disc_folder):
    dialog = folderdialog.FolderDialog(None, disc_folder)
    assert _wait(qapp, lambda: "found" in dialog.status.text())
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
    film = disc_folder / "title_t00.mkv"
    audio, _ = video.list_tracks(film)
    dialog.audio_box.setCurrentIndex(dialog.audio_box.findData(audio[2].index))
    dialog._finish("open")
    assert dialog.action == "open" and dialog.jobs == [(film, audio[2].index, -1)]


def test_converting_all_returns_every_readable_file_each_with_its_own_choice(qapp, disc_folder):
    dialog = folderdialog.FolderDialog(None, disc_folder)
    assert _wait(qapp, lambda: "found" in dialog.status.text())
    dialog._convert_all()
    assert dialog.action == "convert" and len(dialog.jobs) == 2
    assert {job[0].name for job in dialog.jobs} == {"title_t00.mkv", "title_t01.mkv"}
    assert all(job[2] == -1 for job in dialog.jobs)                         # no subtitles unless asked for
    assert dialog.all_button.text().startswith("Convert all (2)")


def test_an_empty_folder_says_so(qapp, tmp_path):
    dialog = folderdialog.FolderDialog(None, tmp_path)
    assert _wait(qapp, lambda: "No videos" in dialog.status.text())
    assert not dialog.open_button.isEnabled() and not dialog.all_button.isEnabled()
    dialog.reject()


# --- the export panel ------------------------------------------------------------------------------------------------------------


def _panel():
    from ultimate_video_3d.panels import ExportPanel

    return ExportPanel(settingsmod.ExportSettings())


def test_the_panel_offers_the_tracks_of_the_file_and_goes_back_to_defaults_for_the_next_one(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    audio, subs = video.list_tracks(rip)
    panel = _panel()
    panel.set_tracks(audio, subs)
    assert panel.audio_track_box.count() == 4 and panel.subtitle_box.count() == 3
    assert panel.audio_track_box.itemText(0).startswith("Best available")
    panel.audio_track_box.setCurrentIndex(2)
    assert panel._s.audio_track == audio[1].index
    panel.set_tracks([], [])                                                  # another file with none
    assert panel._s.audio_track == -1 and panel._s.subtitle_track == -1


def test_choosing_a_subtitle_switches_to_mkv_and_going_back_drops_the_subtitle(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    audio, subs = video.list_tracks(rip)
    panel = _panel()
    panel.family_box.setCurrentIndex(panel.family_box.findData("h265"))
    panel.set_tracks(audio, subs)
    assert panel._s.container == "auto" and not panel.subtitle_box.isEnabled()      # an MP4 cannot hold them
    panel.subtitle_box.setEnabled(True)
    panel.subtitle_box.setCurrentIndex(1)
    assert panel._s.subtitle_track == subs[0].index and panel._s.container == "mkv"
    panel.container_box.setCurrentIndex(panel.container_box.findData("auto"))
    assert panel._s.subtitle_track == -1 and panel.subtitle_box.currentIndex() == 0


def test_an_mkv_format_always_allows_subtitles_and_the_save_as_row_is_only_for_the_mp4_formats(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    panel = _panel()
    panel.show()
    panel.set_tracks(*video.list_tracks(rip))
    panel.family_box.setCurrentIndex(panel.family_box.findData("ffv1"))
    assert panel.subtitle_box.isEnabled() and not panel.container_box.isVisible()
    panel.family_box.setCurrentIndex(panel.family_box.findData("h264"))
    assert panel.container_box.isVisible()
    panel.close()


def test_a_sound_an_mp4_cannot_hold_is_flagged_and_mkv_clears_the_warning(qapp):
    panel = _panel()
    panel.show()
    truehd = video.Track("audio", 1, "truehd", language="eng", channels=8)
    aac = video.Track("audio", 2, "aac", language="fre", channels=2)
    panel.set_tracks([truehd, aac], [])
    panel.audio_track_box.setCurrentIndex(1)
    assert panel.audio_note.isVisible() and "Dolby TrueHD" in panel.audio_note.text()
    panel.set_container("mkv")
    assert not panel.audio_note.isVisible()
    panel.set_container("auto")
    panel.audio_track_box.setCurrentIndex(2)                                  # AAC: fine in an MP4
    assert not panel.audio_note.isVisible()
    panel.close()


def test_choosing_an_audio_track_tells_the_preview_which_one_to_play(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    audio, subs = video.list_tracks(rip)
    panel = _panel()
    heard = []
    panel.audio_track_chosen.connect(heard.append)
    panel.set_tracks(audio, subs)
    panel.audio_track_box.setCurrentIndex(3)                                  # the third track: position 2
    assert heard[-1] == 2
    panel.audio_track_box.setCurrentIndex(0)                                  # automatic: the main mix, position 0
    assert heard[-1] == 0


# --- the output file ---------------------------------------------------------------------------------------------------------------


def test_the_output_is_an_mkv_only_when_asked_and_only_for_the_mp4_formats(tmp_path):
    s = settingsmod.AppSettings()
    s.export.family = "h265"
    assert pipeline.output_path_for(tmp_path / "a.mkv", s, tmp_path).suffix == ".mp4"
    s.export.container = "mkv"
    assert pipeline.output_path_for(tmp_path / "a.mkv", s, tmp_path).suffix == ".mkv"
    s.export.family = "prores"
    assert pipeline.output_path_for(tmp_path / "a.mkv", s, tmp_path).suffix == ".mov"          # not offered there


# --- the batch converter -------------------------------------------------------------------------------------------------------------


def test_each_file_in_the_batch_is_converted_with_its_own_track_choice(qapp, tmp_path, monkeypatch):
    seen = []

    def fake_convert(source, output, settings, *a, **k):
        seen.append((Path(source).name, settings.export.audio_track, settings.export.subtitle_track))
        yield pipeline.VideoProgress("done", 1, 1)

    monkeypatch.setattr(pipeline, "convert_video", fake_convert)
    jobs = [(tmp_path / "a.mkv", tmp_path / "a_SBS.mp4", 3, 5), (tmp_path / "b.mkv", tmp_path / "b_SBS.mp4", -1, -1),
            (tmp_path / "c.mkv", tmp_path / "c_SBS.mp4")]                       # the old two-item form still works
    settings = settingsmod.AppSettings()
    worker = appmod.BatchWorker(jobs, settings, [], False)
    done = []
    worker.done.connect(lambda s, o: done.append(Path(s).name))
    worker.run()
    assert seen == [("a.mkv", 3, 5), ("b.mkv", -1, -1), ("c.mkv", -1, -1)] and done == ["a.mkv", "b.mkv", "c.mkv"]
    assert settings.export.audio_track == -1                                    # the shared settings were not touched


def test_the_batch_dialog_queues_a_job_with_a_tooltip_naming_its_tracks(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    audio, subs = video.list_tracks(rip)
    dialog = appmod.BatchDialog(None, settingsmod.AppSettings(), [], "")
    dialog.add_job(rip, audio[1].index, subs[0].index)
    item = dialog.list.item(0)
    assert item.data(appmod.Qt.ItemDataRole.UserRole) == (audio[1].index, subs[0].index)
    assert "French" in item.toolTip() and "English · Styled text" in item.toolTip()


# --- the whole thing, for real ---------------------------------------------------------------------------------------------------------


def test_converting_a_ripped_title_keeps_the_chosen_audio_and_subtitle_in_an_mkv(tmp_path, monkeypatch):
    from ultimate_video_3d import depth as depthmod

    class _Flat(depthmod.DepthEngine):
        def load(self, *a, **k):
            self.session = object()
            return "test"

        def infer(self, rgb, bars=None):
            return np.full((depthmod.INPUT, depthmod.INPUT), 0.5, np.float32)

    monkeypatch.setattr(pipeline.depthmod, "DepthEngine", _Flat)
    monkeypatch.setattr(pipeline.paths, "scratch_dir", lambda: tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    rip = make_rip(tmp_path / "title_t00.mkv", seconds=1.0, size=(320, 180))
    audio, subs = video.list_tracks(rip)
    s = settingsmod.AppSettings()
    s.export.family, s.export.encoder, s.export.container = "h264", "libx264", "mkv"
    s.export.audio_track, s.export.subtitle_track = audio[1].index, subs[1].index
    out = pipeline.output_path_for(rip, s, tmp_path)
    assert out.suffix == ".mkv"
    last = None
    for update in pipeline.convert_video(rip, out, s, gpus=[]):
        last = update
    assert last.stage == "done" and out.exists()
    got_audio, got_subs = video.list_tracks(out)
    assert [t.language for t in got_audio] == ["fre"] and got_audio[0].title == "Version française"
    assert [t.language for t in got_subs] == ["fre"] and got_subs[0].title == "Français"
    info = video.probe(out)
    assert (info.width, info.height) == (320, 180) and info.has_audio          # a real converted picture, with its sound


def test_save_as_says_mp4_and_goes_back_to_it_when_the_subtitle_that_asked_for_mkv_is_dropped_or_the_file_changes(qapp, tmp_path):
    rip = make_rip(tmp_path / "t.mkv")
    audio, subs = video.list_tracks(rip)
    panel = _panel()
    panel.family_box.setCurrentIndex(panel.family_box.findData("h265"))
    assert panel.container_box.itemText(0) == "MP4"                           # not "same as the format": say which
    panel.set_tracks(audio, subs)
    panel.subtitle_box.setEnabled(True)
    panel.subtitle_box.setCurrentIndex(1)
    assert panel._s.container == "mkv"                                         # a subtitle needs it ...
    panel.subtitle_box.setCurrentIndex(0)
    assert panel._s.container == "auto" and panel.container_box.currentData() == "auto"      # ... and no subtitle gives MP4 back
    panel.subtitle_box.setCurrentIndex(1)
    panel.set_tracks(audio, subs)                                              # opening another file
    assert panel._s.container == "auto"
    panel.container_box.setCurrentIndex(panel.container_box.findData("mkv"))   # chosen by the person: it stays
    panel.set_tracks(audio, subs)
    assert panel._s.container == "mkv"
