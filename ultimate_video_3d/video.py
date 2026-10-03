"""Reading, writing and muxing video, with colour handled deliberately.

PyAV rather than OpenCV: OpenCV cannot read or write an audio stream, and its
encoder list is whatever its prebuilt FFmpeg happened to include. PyAV bundles a
full FFmpeg - NVENC, AMF, Quick Sync, VideoToolbox, x264/x265, SVT-AV1, ProRes,
DNxHR - and does the container work.

Two colour rules earn their keep here, each from a real failure mode:

- **The matrix is always stated.** swscale converts YUV to RGB with BT.601
  unless told otherwise, and most HD video carries no colour tags at all. Left
  alone, every untagged 1080p clip comes out with shifted greens. Untagged
  video is treated as BT.709 from 720p up (what every player does).
- **10-bit stays 10-bit.** HDR sources decode to 16-bit RGB and encode back to
  10-bit with PQ/HLG tags intact, instead of being squashed to 8-bit SDR.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import encoders

Progress = Callable[[str], None]

#: Extensions offered in the file dialog. PyAV reads far more; the list is the
#: common cases, kept short so the dialog is not a wall of formats.
INPUT_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".ts", ".mts", ".m2ts", ".wmv", ".flv"})

# FFmpeg's numeric colour codes (AVColorSpace / AVColorTransferCharacteristic).
_SPC_BT709, _SPC_BT470BG, _SPC_SMPTE170M, _SPC_BT2020 = 1, 5, 6, 9
_TRC_PQ, _TRC_HLG = 16, 18
_PRI_BT709, _PRI_BT2020 = 1, 9
_RANGE_LIMITED, _RANGE_FULL = 1, 2

SDR, PQ, HLG = "sdr", "pq", "hlg"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int            # 0 when the container does not report a count
    has_audio: bool
    duration: float        # seconds, 0.0 when unknown
    codec: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    hdr: str = SDR         # "sdr" | "pq" | "hlg"
    colorspace: int = 0
    color_range: int = 0
    rotation: int = 0
    audio_codec: str = ""

    @property
    def is_hdr(self) -> bool:
        return self.hdr != SDR

    def describe(self) -> str:
        count = f"{self.frames}" if self.frames else "?"
        audio = "with audio" if self.has_audio else "no audio"
        hdr = {"pq": " · HDR10 (PQ)", "hlg": " · HDR (HLG)"}.get(self.hdr, "")
        depth = f" · {self.bit_depth}-bit" if self.bit_depth > 8 else ""
        return (
            f"{self.width}×{self.height}, {self.fps:.3g} fps, {count} frames, "
            f"{self.codec}{depth}{hdr}, {audio}"
        )


def is_available() -> bool:
    """Whether PyAV can be imported. False means it needs downloading first."""
    from . import bootstrap

    bootstrap.activate_av()  # pick up a copy downloaded on a previous run
    try:
        import av  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def _bit_depth(pix_fmt: str) -> int:
    for depth in (16, 12, 10, 9):
        if str(depth) in pix_fmt:
            return depth
    return 8


def _pixel_aspect(sar) -> float:
    """Width of a pixel relative to its height; 1.0 for square pixels or when the file does not say."""
    try:
        value = float(sar) if sar else 1.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return value if 0.2 < value < 5.0 and abs(value - 1.0) > 0.01 else 1.0


def display_width(width: int, sar) -> int:
    """The picture's true width. A DVD stores 720 x 480 pixels that are wider than tall (32:27 for 16:9); shown as
    stored the whole film looks squeezed, so every stage works at the display width instead."""
    return max(2, round(width * _pixel_aspect(sar) / 2) * 2) if _pixel_aspect(sar) != 1.0 else width


# --- audio and subtitle tracks ----------------------------------------------------------------------------------

LANGUAGE_NAMES = {
    "eng": "English", "fre": "French", "fra": "French", "ger": "German", "deu": "German", "spa": "Spanish", "ita": "Italian",
    "por": "Portuguese", "rus": "Russian", "jpn": "Japanese", "kor": "Korean", "chi": "Chinese", "zho": "Chinese",
    "cmn": "Mandarin", "yue": "Cantonese", "dut": "Dutch", "nld": "Dutch", "swe": "Swedish", "nor": "Norwegian",
    "nob": "Norwegian", "dan": "Danish", "fin": "Finnish", "pol": "Polish", "tur": "Turkish", "ara": "Arabic",
    "heb": "Hebrew", "hin": "Hindi", "tha": "Thai", "cze": "Czech", "ces": "Czech", "hun": "Hungarian", "gre": "Greek",
    "ell": "Greek", "rum": "Romanian", "ron": "Romanian", "ukr": "Ukrainian", "vie": "Vietnamese", "ind": "Indonesian",
    "may": "Malay", "msa": "Malay", "cat": "Catalan", "hrv": "Croatian", "srp": "Serbian", "slo": "Slovak", "slk": "Slovak",
    "slv": "Slovenian", "bul": "Bulgarian", "ice": "Icelandic", "isl": "Icelandic", "per": "Persian", "fas": "Persian",
    "tam": "Tamil", "tel": "Telugu", "ben": "Bengali", "fil": "Filipino", "lat": "Latin", "und": "", "zxx": "No speech",
}
_AUDIO_FORMATS = {
    "truehd": "Dolby TrueHD", "mlp": "MLP lossless", "ac3": "Dolby Digital", "eac3": "Dolby Digital Plus", "flac": "FLAC",
    "aac": "AAC", "mp3": "MP3", "mp2": "MP2", "opus": "Opus", "vorbis": "Vorbis", "alac": "Apple Lossless",
    "wmav2": "Windows Media Audio", "wmapro": "WMA Pro", "dts": "DTS", "pcm_bluray": "PCM", "pcm_dvd": "PCM",
}
_SUBTITLE_FORMATS = {
    "subrip": "SubRip (text)", "srt": "SubRip (text)", "ass": "Styled text (ASS)", "ssa": "Styled text (ASS)",       # FFmpeg names both "ssa" when it reads them back
    "hdmv_pgs_subtitle": "Blu-ray picture (PGS)", "dvd_subtitle": "DVD picture", "dvb_subtitle": "DVB picture",
    "mov_text": "Text (MP4)", "webvtt": "WebVTT (text)", "text": "Text", "hdmv_text_subtitle": "Blu-ray text",
    "eia_608": "Closed captions", "xsub": "XSUB picture",
}
#: Pictures, not text: they cannot be turned into another subtitle format, only carried across in a Matroska (MKV) file.
PICTURE_SUBTITLES = frozenset({"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"})
FEW_LINES = 30           # a subtitle track with this few lines or fewer is the "forced" kind, not a full translation
_DISPOSITION_DEFAULT, _DISPOSITION_COMMENT, _DISPOSITION_FORCED, _DISPOSITION_HEARING = 1, 8, 64, 128


def _channel_text(channels: int) -> str:
    return {1: "mono", 2: "stereo", 3: "2.1", 4: "4.0", 5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1"}.get(channels, f"{channels} ch" if channels else "")


@dataclass(frozen=True)
class Track:
    """One audio or subtitle track of a file, described for a person to choose between."""

    kind: str                  # "audio" | "subtitle"
    index: int                 # the stream's index in its container: what the conversion is told to use
    codec: str                 # FFmpeg's name: "truehd", "dts", "ac3", "subrip", "hdmv_pgs_subtitle"...
    language: str = ""         # ISO 639-2 code as stored ("eng"), "" when not tagged
    title: str = ""
    channels: int = 0
    sample_rate: int = 0
    bit_rate: int = 0
    profile: str = ""          # e.g. "DTS-HD MA"
    default: bool = False
    forced: bool = False
    commentary: bool = False
    hearing_impaired: bool = False
    lines: int = 0             # subtitles only, when the ripper recorded it (MakeMKV): how many pictures / lines the track holds

    @property
    def language_name(self) -> str:
        code = self.language.lower()
        return LANGUAGE_NAMES.get(code, code.upper()) if code else ""

    @property
    def format_name(self) -> str:
        if self.kind == "subtitle":
            return _SUBTITLE_FORMATS.get(self.codec, self.codec)
        profile = self.profile.lower()
        if self.codec == "dts":
            if "ma" in profile.replace("-", " ").split() or "master" in profile:
                return "DTS-HD MA"
            if "x" in profile.replace(":", " ").split() and "dts" in profile:
                return "DTS:X"
            if "high res" in profile or "hra" in profile:
                return "DTS-HD HRA"
            if "es" in profile.replace("-", " ").split():
                return "DTS-ES"
            return "DTS"
        name = _AUDIO_FORMATS.get(self.codec, "PCM" if self.codec.startswith("pcm") else self.codec.upper())
        if "atmos" in profile:
            name += " Atmos"
        return name

    @property
    def is_picture(self) -> bool:
        return self.kind == "subtitle" and self.codec in PICTURE_SUBTITLES

    def _title_repeats_label(self, parts: list[str]) -> bool:
        """Rippers title tracks "Surround 5.1" / "Stereo" - which the label already says; showing it again is noise."""
        title = self.title.strip().lower()
        if title in {p.lower() for p in parts}:
            return True
        words = {"surround", "stereo", "mono", "dolby", "digital", "dts", "hd", "ma", "master", "audio", "lossless", "pcm", "truehd", "atmos"}
        rest = [w for w in title.replace("-", " ").replace(".", " ").split() if w not in words and not w.isdigit()]
        return not rest

    def label(self) -> str:
        """One line to pick it by, e.g.  English · DTS-HD MA 7.1 · default  or  French · Blu-ray picture (PGS) · forced."""
        parts = [self.language_name or "Unknown language", self.format_name]
        if self.kind == "audio":
            parts[1] = f"{self.format_name} {_channel_text(self.channels)}".strip()
        if self.title and not self._title_repeats_label(parts):
            parts.append(self.title)
        if self.kind == "subtitle" and 0 < self.lines <= FEW_LINES and not self.forced:
            parts.append(f"only {self.lines} lines - probably forced")          # discs keep the foreign-speech-only track beside the full one
        for flag, text in ((self.commentary, "commentary"), (self.forced, "forced"), (self.hearing_impaired, "for the hard of hearing"),
                           (self.default, "default")):
            if flag:
                parts.append(text)
        return " · ".join(parts)


def _track_of(stream) -> Track:
    ctx = stream.codec_context
    meta = dict(stream.metadata or {})
    try:
        disposition = int(stream.disposition)
    except Exception:  # noqa: BLE001 - an old or odd build: no flags
        disposition = 0
    title = str(meta.get("title") or meta.get("TITLE") or "")
    profile = ""
    try:
        profile = str(ctx.profile or "")
    except Exception:  # noqa: BLE001
        profile = ""
    commentary = bool(disposition & _DISPOSITION_COMMENT) or "commentary" in title.lower()
    # ctx.name is FFmpeg's *decoder* name ("dca", "pgssub"); the codec's own name ("dts", "hdmv_pgs_subtitle") is what everything else uses.
    try:
        codec = str(ctx.codec.canonical_name or ctx.name or "")
    except Exception:  # noqa: BLE001 - an older build: fall back to the decoder name
        codec = str(ctx.name or "")
    lines = 0
    if stream.type != "audio":
        for key, value in meta.items():
            if key.upper().startswith("NUMBER_OF_FRAMES"):         # MakeMKV writes this tag per track
                try:
                    lines = int(value)
                except ValueError:
                    pass
    return Track(
        kind="audio" if stream.type == "audio" else "subtitle", index=stream.index, codec=codec, lines=lines,
        language=str(meta.get("language") or meta.get("LANGUAGE") or ""), title=title,
        channels=int(getattr(ctx, "channels", 0) or 0), sample_rate=int(getattr(ctx, "sample_rate", 0) or 0),
        bit_rate=int(getattr(ctx, "bit_rate", 0) or 0), profile=profile,
        default=bool(disposition & _DISPOSITION_DEFAULT), forced=bool(disposition & _DISPOSITION_FORCED),
        commentary=commentary, hearing_impaired=bool(disposition & _DISPOSITION_HEARING),
    )


def list_tracks(path_or_container) -> tuple[list[Track], list[Track]]:
    """(audio tracks, subtitle tracks) of a file, in the order they are stored. Nothing is decoded."""
    import av

    def read(container):
        return ([_track_of(s) for s in container.streams.audio], [_track_of(s) for s in container.streams.subtitles])

    if hasattr(path_or_container, "streams"):
        return read(path_or_container)
    with av.open(str(path_or_container)) as container:
        return read(container)


_LOSSLESS_AUDIO = frozenset({"truehd", "mlp", "flac", "alac"})


def best_audio_index(audio: list[Track], language: str = "") -> int:
    """Which audio track to use when the person has not chosen: the one marked default, else the first that is not a commentary.

    With a ``language`` (an ISO 639-2 code) a track in that language wins, and among several the lossless or HD one.
    -1 when there is no audio at all.
    """
    if not audio:
        return -1
    pool = [t for t in audio if not t.commentary] or list(audio)
    if language:
        same = [t for t in pool if t.language.lower() == language.lower()]
        if same:
            pool = same

    def rank(t: Track) -> tuple:
        return (t.default, t.codec in _LOSSLESS_AUDIO or t.format_name.startswith("DTS-HD"), t.channels, -t.index)

    return max(pool, key=rank).index if language else (next((t for t in pool if t.default), pool[0])).index


def probe(path: str | Path) -> VideoInfo:
    """Read a video's shape without decoding it."""
    import av

    with av.open(str(path)) as container:
        return describe(container)


def describe(container) -> VideoInfo:
    """A video's shape from an open container (a file, or a disc title read as a stream)."""
    if not container.streams.video:
        raise ValueError("That file has no video stream.")
    stream = container.streams.video[0]
    ctx = stream.codec_context
    rate = stream.average_rate or stream.base_rate or Fraction(24, 1)
    duration = 0.0
    if stream.duration and stream.time_base:
        duration = float(stream.duration * stream.time_base)
    elif container.duration:
        duration = container.duration / 1_000_000
    trc = int(getattr(ctx, "color_trc", 0) or 0)
    rotation = 0
    try:
        rotation = int(round(stream.side_data.get("DISPLAYMATRIX", 0))) if hasattr(stream, "side_data") else 0
    except Exception:  # noqa: BLE001
        rotation = 0
    pix_fmt = ctx.pix_fmt or ""
    audio = container.streams.audio[0].codec_context.name if container.streams.audio else ""
    sar = stream.sample_aspect_ratio or ctx.sample_aspect_ratio
    return VideoInfo(
        width=display_width(ctx.width, sar), height=ctx.height, fps=float(rate),
        frames=stream.frames or (round(duration * float(rate)) if duration else 0),
        has_audio=bool(container.streams.audio), duration=duration,
        codec=ctx.name, pix_fmt=pix_fmt, bit_depth=_bit_depth(pix_fmt),
        hdr=PQ if trc == _TRC_PQ else HLG if trc == _TRC_HLG else SDR,
        colorspace=int(getattr(ctx, "colorspace", 0) or 0),
        color_range=int(getattr(ctx, "color_range", 0) or 0),
        rotation=rotation, audio_codec=audio,
    )


def _src_matrix(frame) -> str:
    """The YUV->RGB matrix to use for this frame, stated rather than assumed."""
    space = int(frame.colorspace or 0)
    if space == _SPC_BT2020:
        return "BT2020"
    if space in (_SPC_BT470BG, _SPC_SMPTE170M):
        return "ITU601"
    if space == _SPC_BT709:
        return "ITU709"
    return "ITU709" if frame.height >= 720 else "ITU601"


def frame_to_rgb(frame, high_bit_depth: bool, size: tuple[int, int] | None = None, sar=None) -> np.ndarray:
    """One decoded frame as float32 RGB in [0, 1], still in its transfer curve.

    The values are *encoded* (gamma / PQ / HLG), not linear: warping and
    encoding both want them that way, and linearising is left to the stages that
    need it (grading, tone mapping).

    ``size`` (width, height) scales inside FFmpeg while converting - far cheaper
    than converting a 4K frame and shrinking it afterwards, which is how the live
    preview gets a 720p proxy from a 4K source in a few milliseconds.

    ``sar`` is the stream's pixel aspect ratio (a decoded frame does not carry it): when it is not square and
    no ``size`` is given, the frame is scaled to its true width.
    """
    full = int(frame.color_range or 0) == _RANGE_FULL
    kwargs = {
        "src_colorspace": _src_matrix(frame),
        "src_color_range": "JPEG" if full else "MPEG",
        "dst_color_range": "JPEG",
    }
    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
    if size is None:
        shown = display_width(frame.width, sar)
        if shown != frame.width:
            size = (shown, frame.height)         # non-square pixels (DVD): scale to the real shape here
    if size is not None:
        width, height = size
        if rotation in (90, 270):
            width, height = height, width          # scale in the stored orientation, rotate after
        kwargs.update(width=width, height=height, interpolation="AREA")
    if high_bit_depth:
        rgb = frame.to_ndarray(format="rgb48le", **kwargs).astype(np.float32) * np.float32(1.0 / 65535.0)
    else:
        rgb = frame.to_ndarray(format="rgb24", **kwargs).astype(np.float32) * np.float32(1.0 / 255.0)
    if rotation:
        rgb = np.rot90(rgb, k=rotation // 90)
    return np.ascontiguousarray(rgb)


def frames(
    path: str | Path,
    start: int = 0,
    limit: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb)`` for a range of frames.

    Seeks to the keyframe at or before ``start`` rather than decoding from the top,
    so an In point at minute 40 does not cost forty minutes of discarded frames.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        high = _bit_depth(stream.codec_context.pix_fmt or "") > 8
        sar = stream.sample_aspect_ratio or stream.codec_context.sample_aspect_ratio
        fps = float(stream.average_rate or stream.base_rate or 24)
        offset = 0
        if start > 0 and stream.time_base:
            seconds = start / fps
            try:
                container.seek(int(seconds / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
            except Exception:  # noqa: BLE001 - decode from the top if seeking is refused
                pass
            offset = None  # resolved from the first decoded frame's timestamp
        emitted = 0
        index = 0
        for frame in container.decode(stream):
            if offset is None:
                pts = frame.pts if frame.pts is not None else 0
                index = int(round(float(pts * stream.time_base) * fps)) if stream.time_base else 0
                offset = index
            if should_stop is not None and should_stop():
                return
            if index < start:
                index += 1
                continue
            if limit is not None and emitted >= limit:
                return
            yield index, frame_to_rgb(frame, high, sar=sar)
            index += 1
            emitted += 1


def read_frame_at(path: str | Path, frame_index: int) -> np.ndarray | None:
    """One frame, decoded accurately - for a still preview when playback is paused."""
    for _, rgb in frames(path, start=frame_index, limit=1):
        return rgb
    return None


# --- writing ---------------------------------------------------------------


class VideoWriter:
    """An open output video, fed float RGB frames one at a time.

    Colour is tagged to match what was written - BT.709 for SDR, BT.2020 with a
    PQ or HLG curve for HDR - so a player does not have to guess.
    """

    def __init__(
        self,
        path: Path,
        family: encoders.Family,
        encoder: str,
        fps: float,
        size: tuple[int, int],
        quality: int = 60,
        profile: encoders.Profile | None = None,
        hdr: str = SDR,
        stereo_mode: str = "",
    ) -> None:
        import av

        self._av = av
        width, height = size
        self.size = (width - width % 2, height - height % 2)
        self.encoder_used = encoder
        self.hdr = hdr
        self._ten_bit = hdr != SDR or (profile is not None and "10" in (profile.pix_fmt or ""))
        if family.key in ("prores",) or (profile and "10" in profile.pix_fmt):
            self._ten_bit = True
        self._pix_fmt = encoders.pixel_format(family, encoder, self._ten_bit, profile)
        self._container = av.open(str(path), mode="w", options=_container_options(family))
        rate = Fraction(fps).limit_denominator(90000)
        self._stream = self._container.add_stream(encoder, rate=rate)
        self._stream.width, self._stream.height = self.size
        self._stream.pix_fmt = self._pix_fmt
        ctx = self._stream.codec_context
        ctx.options = encoders.encoder_options(encoder, family, quality, self.size, fps, profile)
        if encoder.endswith("_videotoolbox") or (encoder.endswith("_mf") and False):
            self._stream.bit_rate = encoders.bitrate_for(self.size, fps, quality, family.key)
        if not encoders.is_hardware(encoder):
            self._stream.thread_type = "AUTO"
        if family.key == "h265" and family.suffix == ".mp4":
            try:
                ctx.codec_tag = "hvc1"   # Apple/QuickTime refuse the default hev1 tag
            except Exception:  # noqa: BLE001
                pass
        self._tag_colour(ctx)
        if stereo_mode and family.suffix == ".mkv":
            self._stream.metadata["stereo_mode"] = stereo_mode

    def _tag_colour(self, ctx) -> None:
        try:
            if self.hdr == SDR:
                ctx.colorspace, ctx.color_primaries, ctx.color_trc = _SPC_BT709, _PRI_BT709, _SPC_BT709
            else:
                ctx.colorspace, ctx.color_primaries = _SPC_BT2020, _PRI_BT2020
                ctx.color_trc = _TRC_PQ if self.hdr == PQ else _TRC_HLG
            ctx.color_range = _RANGE_LIMITED
        except Exception:  # noqa: BLE001 - tags are advisory
            pass

    @property
    def ten_bit(self) -> bool:
        return self._ten_bit

    def write(self, rgb: np.ndarray) -> None:
        """Encode one frame: float32 0..1 RGB, or integers already quantised for this writer
        (uint8 for 8-bit output, uint16 for 10-bit), in the output's transfer curve."""
        width, height = self.size
        rgb = rgb[:height, :width]
        want = np.uint16 if self._ten_bit else np.uint8
        if rgb.dtype != want:
            if rgb.dtype.kind == "f":
                top = 65535.0 if self._ten_bit else 255.0
                rgb = (np.clip(rgb, 0.0, 1.0) * top + 0.5).astype(want)
            else:
                rgb = rgb.astype(want)
        fmt = "rgb48le" if self._ten_bit else "rgb24"
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format=fmt)
        matrix = "ITU709" if self.hdr == SDR else "BT2020"
        frame = frame.reformat(
            format=self._pix_fmt, dst_colorspace=matrix, dst_color_range="MPEG",
            src_color_range="JPEG",
        )
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class SegmentedWriter:
    """A :class:`VideoWriter` that writes the video as a series of small files (see ``resume.py``).

    A segment is closed and registered in the resume state every ``segment_frames`` frames, and again on
    :meth:`close`, however early that is, so a paused job keeps everything it has encoded. The next segment
    opens on the next frame, so nothing is created that would stay empty.
    """

    def __init__(self, state, family: encoders.Family, encoder: str, fps: float, size: tuple[int, int],
                 quality: int, profile, hdr: str, stereo_mode: str, segment_frames: int) -> None:
        self._state, self._family = state, family
        self._args = (encoder, fps, size, quality, profile, hdr, stereo_mode)
        self._limit = max(1, segment_frames)
        self._writer: VideoWriter | None = None
        self._path: Path | None = None
        self._count = 0
        self._open()                                 # the first one now, so ten_bit and the encoder are known
        self.encoder_used = self._writer.encoder_used
        self._ten_bit = self._writer.ten_bit

    def _open(self) -> None:
        self._path = self._state.next_segment_path(self._family.suffix)
        self._state.folder.mkdir(parents=True, exist_ok=True)
        encoder, fps, size, quality, profile, hdr, stereo_mode = self._args
        self._writer = VideoWriter(self._path, self._family, encoder, fps, size, quality, profile, hdr=hdr,
                                   stereo_mode=stereo_mode)
        self._count = 0

    @property
    def ten_bit(self) -> bool:
        return self._ten_bit

    def write(self, rgb: np.ndarray) -> None:
        if self._writer is None:
            self._open()
        self._writer.write(rgb)
        self._count += 1
        if self._count >= self._limit:
            self._finish()

    def _finish(self) -> None:
        writer, path, count = self._writer, self._path, self._count
        self._writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception:  # noqa: BLE001 - an encoder that never got a frame has nothing to flush
            if count:
                raise
        self._state.add(path, count)

    def close(self) -> None:
        self._finish()


def join_segments(files: list[Path], counts: list[int], fps: float, destination: Path,
                  family: encoders.Family) -> None:
    """Put the segments end to end in one file by copying their packets (no re-encoding)."""
    import av

    if len(files) == 1:
        shutil.copy2(files[0], destination)
        return
    with av.open(str(destination), mode="w", options=_container_options(family)) as out:
        out_stream = None
        before = 0
        last = None
        for path, count in zip(files, counts):
            with av.open(str(path)) as segment:
                stream = segment.streams.video[0]
                if out_stream is None:
                    out_stream = out.add_stream_from_template(stream)
                    for key, value in dict(stream.metadata).items():
                        out_stream.metadata[key] = value      # e.g. the Matroska stereo_mode tag
                for packet in segment.demux(stream):
                    if packet.dts is None:
                        continue
                    ticks = round(before / fps / float(packet.time_base))
                    packet.dts += ticks
                    if packet.pts is not None:
                        packet.pts += ticks
                    # Segments differ in how far decode order leads display order (a two-frame one has no
                    # B-frames), so a piece's first decode time can land at or before the last one's. Muxers
                    # need strictly rising decode times: nudge by a tick or two, never enough to see.
                    if last is not None and packet.dts <= last:
                        packet.dts = last + 1
                    last = packet.dts
                    packet.stream = out_stream
                    out.mux(packet)
            before += count


def _container_options(family: encoders.Family) -> dict[str, str]:
    if family.suffix in (".mp4", ".mov"):
        return {"movflags": "faststart"}
    return {}


# --- audio -----------------------------------------------------------------


#: Audio an MP4 / MOV can carry as it is. TrueHD, DTS and DTS-HD are not on the list: they are turned into AAC instead of
#: producing a file some players cannot open (an MKV keeps them untouched).
MP4_AUDIO = frozenset({"aac", "ac3", "eac3", "mp3", "mp2", "alac", "opus", "flac"})


def carry_video(video_only: Path, destination: Path) -> None:
    """Put the finished picture at its destination: a plain copy when the container is the same, otherwise the packets are
    moved into the new container (no re-encoding). This is what an MKV output needs when the picture was encoded as MP4."""
    import av

    if video_only.suffix.lower() == destination.suffix.lower():
        shutil.copy2(video_only, destination)
        return
    with av.open(str(video_only)) as vid, av.open(str(destination), mode="w") as out:
        stream = vid.streams.video[0]
        out_stream = out.add_stream_from_template(stream)
        for key, value in dict(stream.metadata).items():
            out_stream.metadata[key] = value
        for packet in vid.demux(stream):
            if packet.dts is None:
                continue
            packet.stream = out_stream
            out.mux(packet)


def mux_audio(
    source: Path,
    video_only: Path,
    destination: Path,
    start: float = 0.0,
    duration: float | None = None,
    audio: str = "copy",
    suffix: str = ".mp4",
    audio_index: int = -1,
    subtitle_index: int = -1,
) -> bool:
    """Put the source's audio (and, in an MKV, one subtitle track) onto the converted video, trimmed to match.

    ``audio_index`` / ``subtitle_index`` are stream indexes as listed by :func:`list_tracks`; -1 means the best audio track
    (:func:`best_audio_index`) and no subtitles. Subtitles are only carried into Matroska (``suffix == ".mkv"``), copied as they are
    - text or picture - and re-timed to the converted range; other containers cannot hold the picture kinds, and PyAV cannot
    convert one text format into another.

    Returns True if audio was carried across, False if there was none (or it was switched off). ``start`` and ``duration``
    (seconds) are the window the video covers: a converted In/Out range gets only its own slice of the soundtrack,
    re-timestamped to zero, rather than the whole track playing on over black.
    """
    import av

    with av.open(str(source)) as src:
        audio_in = None
        if audio != "none" and src.streams.audio:
            wanted = audio_index if audio_index >= 0 else best_audio_index(list_tracks(src)[0])
            audio_in = next((s for s in src.streams.audio if s.index == wanted), src.streams.audio[0])
        sub_in = None
        if suffix == ".mkv" and subtitle_index >= 0:
            sub_in = next((s for s in src.streams.subtitles if s.index == subtitle_index), None)
        if audio_in is None and sub_in is None:
            carry_video(video_only, destination)
            return False
        trimming = start > 0 or duration is not None
        end = None if duration is None else start + duration

        with av.open(str(video_only)) as vid, av.open(str(destination), mode="w") as out:
            video_in = vid.streams.video[0]
            video_out = out.add_stream_from_template(video_in)
            for key, value in dict(video_in.metadata).items():
                video_out.metadata[key] = value

            # Every output stream must exist before the first mux writes the
            # container header, so the audio stream is decided up front.
            remux = False
            audio_out = None
            if audio_in is not None:
                codec = "libopus" if suffix == ".webm" else "aac"
                codec_name = audio_in.codec_context.name
                holds = (suffix == ".mkv" or (suffix == ".mov" and (codec_name.startswith("pcm") or codec_name in MP4_AUDIO))
                         or (suffix == ".mp4" and codec_name in MP4_AUDIO))
                if not trimming and audio == "copy" and suffix != ".webm" and holds:
                    try:
                        audio_out = out.add_stream_from_template(audio_in)
                        remux = True
                    except Exception:  # noqa: BLE001 - container will not take it raw
                        audio_out = None
                if audio_out is None:
                    audio_out = out.add_stream(codec, rate=audio_in.rate)
                    audio_out.codec_context.time_base = Fraction(1, audio_in.rate)
                for key in ("language", "title"):                        # players show these in their track menus
                    if audio_in.metadata.get(key):
                        audio_out.metadata[key] = audio_in.metadata[key]
            sub_out = None
            if sub_in is not None:
                sub_out = out.add_stream_from_template(sub_in)
                for key in ("language", "title"):
                    if sub_in.metadata.get(key):
                        sub_out.metadata[key] = sub_in.metadata[key]

            for packet in vid.demux(video_in):
                if packet.dts is None:
                    continue
                packet.stream = video_out
                out.mux(packet)

            if audio_in is not None:
                if remux:
                    for packet in src.demux(audio_in):
                        if packet.dts is None:
                            continue
                        packet.stream = audio_out
                        out.mux(packet)
                else:
                    _reencode_audio_window(src, out, audio_in, audio_out, start if trimming else 0.0, end)

            if sub_in is not None:
                src.seek(0)                                              # the audio pass above read the file to its end
                shift = round(start / float(sub_in.time_base)) if trimming else 0
                for packet in src.demux(sub_in):
                    stamp = packet.pts if packet.pts is not None else packet.dts
                    if stamp is None:
                        continue
                    seconds = stamp * float(packet.time_base)
                    if seconds < start or (end is not None and seconds >= end):
                        continue
                    if shift:
                        if packet.pts is not None:
                            packet.pts -= shift
                        if packet.dts is not None:
                            packet.dts -= shift
                    packet.stream = sub_out
                    out.mux(packet)
    return audio_in is not None


def _reencode_audio_window(src, out, audio_in, audio_out, start: float, end: float | None) -> None:
    """Encode a time window of audio into an already-created audio stream.

    Each frame needs ``sample_rate`` as well as ``pts``/``time_base``, and a FIFO
    repacketises to the encoder's fixed frame size - decoded frames do not arrive
    that size and AAC will not take an odd one.
    """
    import av

    rate = audio_in.rate
    time_base = Fraction(1, rate)
    layout = audio_in.layout
    fmt = "fltp" if audio_out.codec_context.name == "aac" else "flt"
    resampler = av.AudioResampler(format=fmt, layout=layout, rate=rate)
    fifo = av.AudioFifo()
    frame_size = audio_out.codec_context.frame_size or 1024
    counter = 0

    if start > 0:
        try:
            src.seek(int(start / audio_in.time_base), stream=audio_in, backward=True)
        except Exception:  # noqa: BLE001 - decode from the top if seek is refused
            pass

    def emit(flush: bool) -> None:
        nonlocal counter
        while True:
            frame = fifo.read() if flush else fifo.read(frame_size)
            if frame is None:
                return
            frame.pts = counter
            frame.time_base = time_base
            frame.sample_rate = rate
            counter += frame.samples
            for packet in audio_out.encode(frame):
                out.mux(packet)
            if flush:
                return

    for frame in src.decode(audio_in):
        if frame.pts is None:
            continue
        t = float(frame.pts * audio_in.time_base)
        if t < start:
            continue
        if end is not None and t >= end:
            break
        for resampled in resampler.resample(frame):
            resampled.pts = None
            fifo.write(resampled)
        emit(flush=False)
    emit(flush=True)
    for packet in audio_out.encode():
        out.mux(packet)
