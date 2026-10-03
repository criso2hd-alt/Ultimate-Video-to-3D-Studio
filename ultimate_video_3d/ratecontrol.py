"""Quality, video bitrate and file size are three ways of saying one thing; this keeps them in step.

The Quality slider chooses a constant-quality encode (CRF / CQ / QP), whose bitrate depends on the picture, so for a slider
position the bitrate and the size can only be *estimated*: a bits-per-pixel model, scaled by how efficient the codec is. A
bitrate (or a file size, which is a bitrate times the running time, plus sound) is exact: the encoder is told to hit it.
Whichever of the three the person last changed is what the encode follows (``ExportSettings.rate_mode``); the other two are
shown as what that choice works out to, so changing one visibly moves the others.
"""

from __future__ import annotations

#: How many bits a codec needs, compared with H.264, for the same look.
EFFICIENCY = {"h264": 1.0, "h265": 0.62, "av1": 0.5, "vp9": 0.65}
MIN_VIDEO_BPS = 100_000
OVERHEAD = 1.012                  # the container's own bookkeeping, on top of the picture and the sound
#: Typical bitrates of audio that is copied as it is, when the file does not say (bits per second).
COPIED_AUDIO_BPS = {"ac3": 640_000, "eac3": 768_000, "dts": 1_509_000, "truehd": 4_000_000, "flac": 900_000,
                    "aac": 192_000, "mp3": 192_000, "opus": 128_000}
#: MP4 cannot hold these as they are: the app turns them into AAC there.
MP4_AUDIO = frozenset({"aac", "ac3", "eac3", "mp3", "mp2", "alac", "opus", "flac"})


def bits_per_pixel(quality: float) -> float:
    return 0.03 + 0.17 * (max(0.0, min(100.0, quality)) / 100.0) ** 2


def bitrate_for_quality(size: tuple[int, int], fps: float, quality: float, family_key: str) -> int:
    """Roughly what the Quality slider's position comes to, in bits per second (an estimate: it depends on the picture)."""
    width, height = size
    return int(width * height * fps * bits_per_pixel(quality) * EFFICIENCY.get(family_key, 1.0))


def quality_for_bitrate(bps: float, size: tuple[int, int], fps: float, family_key: str) -> int:
    """The slider position that comes closest to ``bps`` (0..100; a bitrate beyond the slider's reach gives its end)."""
    width, height = size
    pixels = max(1.0, width * height * max(fps, 1.0) * EFFICIENCY.get(family_key, 1.0))
    bpp = bps / pixels
    share = (bpp - 0.03) / 0.17
    if share <= 0:
        return 0
    return int(round(min(100.0, 100.0 * share ** 0.5)))


def audio_bps(mode: str, codec: str = "", bit_rate: int = 0, channels: int = 2, mkv: bool = False) -> int:
    """What the sound adds to the file: nothing, AAC at 64 kb/s a channel (what the encoder does), or the track as it is."""
    if mode == "none":
        return 0
    aac = 64_000 * max(1, channels)
    if mode == "aac":
        return aac
    if not mkv and codec and codec not in MP4_AUDIO and not codec.startswith("pcm"):
        return aac                                       # an MP4 cannot take it untouched: it becomes AAC
    if bit_rate > 0:
        return int(bit_rate)
    if codec.startswith("pcm"):
        return 48_000 * 16 * max(1, channels)
    return COPIED_AUDIO_BPS.get(codec, 96_000 * max(1, channels))


def size_bytes(video_bps: float, audio: float, seconds: float) -> int:
    return int((video_bps + audio) * max(0.0, seconds) / 8.0 * OVERHEAD)


def video_bps_for_size(target_bytes: float, audio: float, seconds: float) -> int:
    """The video bitrate that makes a file of ``target_bytes`` once the sound and the container are paid for."""
    if seconds <= 0:
        return MIN_VIDEO_BPS
    return int(max(MIN_VIDEO_BPS, target_bytes * 8.0 / OVERHEAD / seconds - audio))


def video_bitrate(export, uses_quality: bool, seconds: float, audio: float) -> int:
    """The bitrate to encode at, in bits per second, or 0 to follow the Quality slider (constant quality)."""
    if not uses_quality or export.rate_mode == "quality":
        return 0
    if export.rate_mode == "bitrate":
        return max(MIN_VIDEO_BPS, int(export.video_kbps) * 1000)
    if export.rate_mode == "size" and export.target_mb > 0:
        return video_bps_for_size(export.target_mb * 1_000_000, audio, seconds)
    return 0


def describe_size(megabytes: float) -> str:
    return f"{megabytes / 1000:.2f} GB" if megabytes >= 1000 else f"{megabytes:.0f} MB"
