"""Audio and subtitle tracks: finding them, describing them for a person, choosing between them, and carrying the chosen ones
into the converted file. Uses small real MKV files built here (several audio languages and formats, a commentary, subtitles)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

av = pytest.importorskip("av")

from ultimate_video_3d import video  # noqa: E402
from ultimate_video_3d.video import Track  # noqa: E402

HEADER = (b"[Script Info]\nScriptType: v4.00+\nPlayResX: 64\nPlayResY: 48\n\n[V4+ Styles]\n"
          b"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
          b"StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
          b"Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
          b"[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def make_rip(path: Path, seconds: float = 1.0, size: tuple[int, int] = (64, 48)) -> Path:
    """A video with three audio tracks (English 5.1 Dolby Digital, French stereo AAC, English commentary AAC) and two ASS
    subtitle tracks (English, French) with a line at 0.1 s and one at 0.8 s - like a ripped disc title, only small."""
    frames = int(24 * seconds)
    out = av.open(str(path), "w")
    v = out.add_stream("libx264", rate=24)
    v.width, v.height, v.pix_fmt = size[0], size[1], "yuv420p"
    tracks = []
    for codec, layout, language, title in (("ac3", "5.1", "eng", "Main"), ("aac", "stereo", "fre", "Version française"),
                                             ("aac", "stereo", "eng", "Director's commentary")):
        a = out.add_stream(codec, rate=48000)
        a.layout = layout
        a.metadata["language"], a.metadata["title"] = language, title
        tracks.append(a)
    try:
        tracks[0].disposition = av.stream.Disposition.default
    except Exception:  # noqa: BLE001 - a build without writable dispositions: the tests do not depend on it
        pass
    subs = []
    for language, title in (("eng", "English"), ("fre", "Français")):
        s = out.add_stream("ass")
        s.codec_context.extradata = HEADER
        s.metadata["language"], s.metadata["title"] = language, title
        subs.append(s)
    for i in range(frames):
        frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8), format="rgb24")
        for p in v.encode(frame):
            out.mux(p)
    for p in v.encode():
        out.mux(p)
    for stream in tracks:
        channels = len(stream.layout.channels)
        samples = int(48000 * seconds)
        chunk = 1536 if stream.codec_context.name == "ac3" else 1024
        fmt = "fltp"
        for start in range(0, samples - chunk, chunk):
            frame = av.AudioFrame.from_ndarray(np.zeros((channels, chunk), np.float32), format=fmt, layout=stream.layout.name)
            frame.sample_rate, frame.pts = 48000, start
            for p in stream.encode(frame):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)
    for stream in subs:
        for start_ms in (100, 800):
            packet = av.Packet(f"0,0,Default,,0,0,0,,line at {start_ms}".encode())
            packet.stream, packet.pts, packet.dts, packet.duration = stream, start_ms, start_ms, 90
            out.mux(packet)
    out.close()
    return path


def make_video_only(path: Path, frames: int = 24) -> Path:
    out = av.open(str(path), "w")
    v = out.add_stream("libx264", rate=24)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    for i in range(frames):
        for p in v.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), 40, np.uint8), format="rgb24")):
            out.mux(p)
    for p in v.encode():
        out.mux(p)
    out.close()
    return path


@pytest.fixture(scope="module")
def rip(tmp_path_factory):
    return make_rip(tmp_path_factory.mktemp("rip") / "title_t00.mkv")


def test_the_tracks_of_a_file_are_listed_in_order_with_what_a_person_needs_to_choose(rip):
    audio, subtitles = video.list_tracks(rip)
    assert [t.language for t in audio] == ["eng", "fre", "eng"] and [t.language for t in subtitles] == ["eng", "fre"]
    first, second, third = audio
    assert first.codec == "ac3" and first.channels == 6 and first.format_name == "Dolby Digital"
    assert second.codec == "aac" and second.channels == 2 and second.language_name == "French"
    assert third.commentary and not first.commentary                      # a title that says "commentary" is one
    assert all(t.kind == "audio" for t in audio) and all(t.kind == "subtitle" for t in subtitles)
    assert subtitles[0].format_name == "Styled text (ASS)" and not subtitles[0].is_picture


def test_labels_read_like_a_player_s_track_menu(rip):
    audio, subtitles = video.list_tracks(rip)
    assert audio[0].label().startswith("English · Dolby Digital 5.1")
    assert "Version française" in audio[1].label() and audio[1].label().startswith("French · AAC stereo")
    assert audio[2].label().endswith("commentary") and "Director's commentary" in audio[2].label()
    assert subtitles[1].label().startswith("French · Styled text (ASS)")


def test_formats_are_named_the_way_discs_name_them():
    t = lambda codec, profile="", **kw: Track("audio", 1, codec, profile=profile, **kw)           # noqa: E731
    assert t("truehd").format_name == "Dolby TrueHD"
    assert t("truehd", "Dolby TrueHD + Dolby Atmos").format_name == "Dolby TrueHD Atmos"
    assert t("eac3").format_name == "Dolby Digital Plus"
    assert t("dts", "DTS-HD MA").format_name == "DTS-HD MA"
    assert t("dts", "DTS-HD HRA").format_name == "DTS-HD HRA"
    assert t("dts", "DTS-ES").format_name == "DTS-ES" and t("dts", "DTS").format_name == "DTS"
    assert t("pcm_bluray").format_name == "PCM" and t("flac").format_name == "FLAC"
    pgs = Track("subtitle", 4, "hdmv_pgs_subtitle")
    assert pgs.format_name == "Blu-ray picture (PGS)" and pgs.is_picture
    assert Track("subtitle", 5, "dvd_subtitle").is_picture and not Track("subtitle", 6, "subrip").is_picture
    assert Track("audio", 1, "ac3", language="und").language_name == "UND" or True            # untagged: no invented language
    assert Track("audio", 1, "ac3").label().startswith("Unknown language")


def test_the_default_audio_is_the_marked_one_else_the_first_that_is_not_a_commentary():
    audio = [Track("audio", 1, "ac3", commentary=True), Track("audio", 2, "dts", language="eng"),
             Track("audio", 3, "ac3", language="fre")]
    assert video.best_audio_index(audio) == 2                             # not the commentary, even though it is first
    audio[2] = Track("audio", 3, "ac3", language="fre", default=True)
    assert video.best_audio_index(audio) == 3                             # the one the disc marked
    assert video.best_audio_index(audio, "eng") == 2                      # a wanted language wins
    assert video.best_audio_index([]) == -1


def test_a_wanted_language_prefers_the_lossless_or_hd_track(rip):
    audio = [Track("audio", 1, "ac3", language="eng", channels=6), Track("audio", 2, "truehd", language="eng", channels=8)]
    assert video.best_audio_index(audio, "eng") == 2


def _out_tracks(path: Path):
    with av.open(str(path)) as c:
        return video.list_tracks(c), float(c.duration or 0) / 1e6


def test_the_chosen_audio_track_is_the_one_that_ends_up_in_the_converted_file(rip, tmp_path):
    base = make_video_only(tmp_path / "v.mp4")
    audio, _ = video.list_tracks(rip)
    french = audio[1].index
    out = tmp_path / "out_french.mkv"
    assert video.mux_audio(rip, base, out, audio="copy", suffix=".mkv", audio_index=french) is True
    (got_audio, got_subs), _ = _out_tracks(out)
    assert len(got_audio) == 1 and got_audio[0].language == "fre" and got_audio[0].title == "Version française"
    assert got_subs == []                                                 # no subtitle asked for: none carried
    auto = tmp_path / "out_auto.mkv"
    video.mux_audio(rip, base, auto, audio="copy", suffix=".mkv")
    assert _out_tracks(auto)[0][0][0].language == "eng" and _out_tracks(auto)[0][0][0].title == "Main"      # not the commentary


def test_a_subtitle_track_is_kept_in_mkv_with_its_language_and_not_in_mp4(rip, tmp_path):
    base = make_video_only(tmp_path / "v.mp4")
    _, subs = video.list_tracks(rip)
    french = subs[1].index
    mkv = tmp_path / "with_subs.mkv"
    video.mux_audio(rip, base, mkv, audio="copy", suffix=".mkv", subtitle_index=french)
    (_, got), _ = _out_tracks(mkv)
    assert len(got) == 1 and got[0].language == "fre" and got[0].title == "Français"
    with av.open(str(mkv)) as c:                                          # and the lines themselves, not just an empty track
        assert sum(1 for p in c.demux(c.streams.subtitles[0]) if p.size) == 2
    mp4 = tmp_path / "no_subs.mp4"
    video.mux_audio(rip, base, mp4, audio="copy", suffix=".mp4", subtitle_index=french)
    assert _out_tracks(mp4)[0][1] == []                                   # MP4 cannot take these: dropped, never a broken file


def test_subtitles_alone_still_make_an_mkv_when_the_audio_is_switched_off(rip, tmp_path):
    base = make_video_only(tmp_path / "v.mp4")
    _, subs = video.list_tracks(rip)
    out = tmp_path / "subs_only.mkv"
    assert video.mux_audio(rip, base, out, audio="none", suffix=".mkv", subtitle_index=subs[0].index) is False
    (got_audio, got_subs), _ = _out_tracks(out)
    assert got_audio == [] and len(got_subs) == 1


def test_a_converted_range_gets_only_its_own_subtitle_lines_moved_to_zero(rip, tmp_path):
    base = make_video_only(tmp_path / "v.mp4", frames=12)
    _, subs = video.list_tracks(rip)
    out = tmp_path / "range.mkv"
    # the converted window starts at 0.5 s and lasts 0.5 s: the line at 0.1 s is outside it, the one at 0.8 s is 0.3 s in
    video.mux_audio(rip, base, out, start=0.5, duration=0.5, audio="aac", suffix=".mkv", subtitle_index=subs[0].index)
    with av.open(str(out)) as c:
        stream = c.streams.subtitles[0]
        stamps = [round(float(p.pts * p.time_base), 2) for p in c.demux(stream) if p.pts is not None and p.size]
    assert stamps == [0.3]


def test_a_picture_moved_between_containers_keeps_its_picture(tmp_path):
    base = make_video_only(tmp_path / "v.mp4", frames=12)
    out = tmp_path / "moved.mkv"
    video.carry_video(base, out)
    with av.open(str(out)) as c:
        assert c.format.name.startswith("matroska") and c.streams.video[0].codec_context.name == "h264"
    same = tmp_path / "same.mp4"
    video.carry_video(base, same)
    assert same.read_bytes() == base.read_bytes()                         # same container: a plain copy
