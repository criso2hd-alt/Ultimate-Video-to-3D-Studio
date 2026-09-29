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


def probe(path: str | Path) -> VideoInfo:
    """Read a video's shape without decoding it."""
    import av

    with av.open(str(path)) as container:
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
        return VideoInfo(
            width=ctx.width, height=ctx.height, fps=float(rate),
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


def frame_to_rgb(frame, high_bit_depth: bool, size: tuple[int, int] | None = None) -> np.ndarray:
    """One decoded frame as float32 RGB in [0, 1], still in its transfer curve.

    The values are *encoded* (gamma / PQ / HLG), not linear: warping and
    encoding both want them that way, and linearising is left to the stages that
    need it (grading, tone mapping).

    ``size`` (width, height) scales inside FFmpeg while converting - far cheaper
    than converting a 4K frame and shrinking it afterwards, which is how the live
    preview gets a 720p proxy from a 4K source in a few milliseconds.
    """
    full = int(frame.color_range or 0) == _RANGE_FULL
    kwargs = {
        "src_colorspace": _src_matrix(frame),
        "src_color_range": "JPEG" if full else "MPEG",
        "dst_color_range": "JPEG",
    }
    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
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
            yield index, frame_to_rgb(frame, high)
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


def _container_options(family: encoders.Family) -> dict[str, str]:
    if family.suffix in (".mp4", ".mov"):
        return {"movflags": "faststart"}
    return {}


# --- audio -----------------------------------------------------------------


def mux_audio(
    source: Path,
    video_only: Path,
    destination: Path,
    start: float = 0.0,
    duration: float | None = None,
    audio: str = "copy",
    suffix: str = ".mp4",
) -> bool:
    """Put the source's audio onto the converted video, trimmed to match.

    Returns True if audio was carried across, False if there was none (or it was
    switched off). ``start`` and ``duration`` (seconds) are the window the video
    covers: a converted In/Out range gets only its own slice of the soundtrack,
    re-timestamped to zero, rather than the whole track playing on over black.
    """
    import av

    with av.open(str(source)) as src:
        if audio == "none" or not src.streams.audio:
            shutil.copy2(video_only, destination)
            return False
        audio_in = src.streams.audio[0]
        trimming = start > 0 or duration is not None
        codec = "libopus" if suffix == ".webm" else "aac"

        with av.open(str(video_only)) as vid, av.open(str(destination), mode="w") as out:
            video_in = vid.streams.video[0]
            video_out = out.add_stream_from_template(video_in)

            # Every output stream must exist before the first mux writes the
            # container header, so the audio stream is decided up front.
            remux = False
            audio_out = None
            if not trimming and audio == "copy" and suffix != ".webm":
                try:
                    audio_out = out.add_stream_from_template(audio_in)
                    remux = True
                except Exception:  # noqa: BLE001 - container will not take it raw
                    audio_out = None
            if audio_out is None:
                audio_out = out.add_stream(codec, rate=audio_in.rate)
                audio_out.codec_context.time_base = Fraction(1, audio_in.rate)

            for packet in vid.demux(video_in):
                if packet.dts is None:
                    continue
                packet.stream = video_out
                out.mux(packet)

            if remux:
                for packet in src.demux(audio_in):
                    if packet.dts is None:
                        continue
                    packet.stream = audio_out
                    out.mux(packet)
            else:
                end = None if duration is None else start + duration
                _reencode_audio_window(src, out, audio_in, audio_out, start if trimming else 0.0, end)
    return True


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
