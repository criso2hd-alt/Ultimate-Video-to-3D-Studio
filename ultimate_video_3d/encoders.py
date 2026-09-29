"""Which encoders this machine can actually run, and the best one for each format.

The rule the whole module is built on: **"present" is not "usable."** FFmpeg
lists ``h264_nvenc`` on every machine whose FFmpeg was built with it, whether or
not an NVIDIA card is installed. ``add_stream`` succeeds regardless, and the
failure only appears on the first frame as ``avcodec_open2 returned 22``. So a
candidate is only trusted after it has been opened and fed a frame for real.

That probe runs once at first launch (and again if the GPU changes) and is saved,
so the default encoder is chosen for *this* machine - NVENC on GeForce, AMF on
Radeon, QSV on Intel, VideoToolbox on a Mac - with the software encoder as the
floor that always works. Nothing needs CUDA installed: these are the fixed
function encoders built into every driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from . import sysinfo

# --- output formats -------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    label: str
    value: str
    pix_fmt: str = ""          # overrides the family's when set (ProRes 4444 needs 444)


@dataclass(frozen=True)
class Family:
    """One thing the user picks: a codec + container, independent of which encoder runs it."""

    key: str
    label: str
    suffix: str
    note: str
    #: Encoder names, hardware first per vendor, software last.
    hw: dict[str, tuple[str, ...]] = field(default_factory=dict)
    software: tuple[str, ...] = ()
    pix_fmt: str = "yuv420p"
    pix_fmt_10bit: str = "yuv420p10le"
    #: Whether the family can carry 10-bit HDR (PQ / HLG) at all.
    hdr: bool = True
    profiles: tuple[Profile, ...] = ()
    #: Editors: ProRes/DNxHR are intra-frame and ignore the quality slider.
    uses_quality: bool = True
    max_width: int = 0         # hard container/codec limit, 0 = none worth knowing


FAMILIES: tuple[Family, ...] = (
    Family(
        "h264", "H.264 / MP4", ".mp4",
        "Plays everywhere. Hardware-encoded. Limited to 4096 px wide, so full-width 3D from 4K needs H.265.",
        hw={
            sysinfo.VENDOR_NVIDIA: ("h264_nvenc",),
            sysinfo.VENDOR_AMD: ("h264_amf",),
            sysinfo.VENDOR_INTEL: ("h264_qsv",),
            sysinfo.VENDOR_APPLE: ("h264_videotoolbox",),
        },
        software=("libx264",), pix_fmt_10bit="yuv420p10le", hdr=False, max_width=4096,
    ),
    Family(
        "h265", "H.265 / HEVC / MP4", ".mp4",
        "Half the size of H.264 at the same quality. VR headsets and modern players. Carries 10-bit HDR.",
        hw={
            sysinfo.VENDOR_NVIDIA: ("hevc_nvenc",),
            sysinfo.VENDOR_AMD: ("hevc_amf",),
            sysinfo.VENDOR_INTEL: ("hevc_qsv",),
            sysinfo.VENDOR_APPLE: ("hevc_videotoolbox",),
        },
        software=("libx265",), max_width=8192,
    ),
    Family(
        "av1", "AV1 / MP4", ".mp4",
        "Smallest files. Needs an RTX 40 / RX 7000 / Arc GPU for hardware encoding; otherwise slow software.",
        hw={
            sysinfo.VENDOR_NVIDIA: ("av1_nvenc",),
            sysinfo.VENDOR_AMD: ("av1_amf",),
            sysinfo.VENDOR_INTEL: ("av1_qsv",),
        },
        software=("libsvtav1",),
    ),
    Family(
        "vp9", "VP9 / WebM", ".webm",
        "For web upload. Software only. Not for editors - WebM does not import cleanly.",
        software=("libvpx-vp9",),
    ),
    Family(
        "prores", "Apple ProRes / MOV", ".mov",
        "Editing master (Premiere, Resolve, Final Cut). Very large files, no quality slider.",
        software=("prores_ks",), pix_fmt="yuv422p10le", pix_fmt_10bit="yuv422p10le",
        uses_quality=False,
        profiles=(
            Profile("Proxy", "0"), Profile("LT", "1"), Profile("422", "2"),
            Profile("422 HQ", "3"), Profile("4444", "4", "yuv444p10le"),
        ),
    ),
    Family(
        "dnxhr", "Avid DNxHR / MOV", ".mov",
        "Editing master for Avid and Resolve. Intra-frame, no quality slider.",
        software=("dnxhd",), pix_fmt="yuv422p", pix_fmt_10bit="yuv422p10le",
        uses_quality=False, hdr=False,
        profiles=(
            Profile("LB", "dnxhr_lb"), Profile("SQ", "dnxhr_sq"), Profile("HQ", "dnxhr_hq"),
            Profile("HQX (10-bit)", "dnxhr_hqx", "yuv422p10le"),
            Profile("444 (10-bit)", "dnxhr_444", "yuv444p10le"),
        ),
    ),
    Family(
        "ffv1", "FFV1 lossless / MKV", ".mkv",
        "Mathematically lossless archive. Huge files.",
        software=("ffv1",), pix_fmt="yuv420p", pix_fmt_10bit="yuv420p10le", uses_quality=False,
    ),
)

FAMILIES_BY_KEY = {f.key: f for f in FAMILIES}
HARDWARE_SUFFIXES = ("_nvenc", "_amf", "_qsv", "_videotoolbox", "_mf")


def is_hardware(encoder: str) -> bool:
    return encoder.endswith(HARDWARE_SUFFIXES)


def encoder_label(encoder: str) -> str:
    for suffix, label in (
        ("_nvenc", "NVIDIA NVENC"), ("_amf", "AMD AMF"), ("_qsv", "Intel Quick Sync"),
        ("_videotoolbox", "Apple VideoToolbox"), ("_mf", "Windows Media Foundation"),
    ):
        if encoder.endswith(suffix):
            return label
    return f"{encoder} (CPU)"


# --- the ladder -----------------------------------------------------------


def _vendor_order(gpus: list[sysinfo.GpuInfo]) -> list[str]:
    order: list[str] = []
    for gpu in gpus:
        if gpu.vendor not in order:
            order.append(gpu.vendor)
    return order


def ladder(family: Family, gpus: list[sysinfo.GpuInfo]) -> list[str]:
    """Candidate encoders best-first: the primary GPU's, other GPUs', generic, software.

    Vendors are ordered by the detected GPUs (best card first), so a laptop with
    an Intel iGPU and an NVIDIA card tries NVENC before Quick Sync.
    """
    names: list[str] = []
    for vendor in _vendor_order(gpus):
        names.extend(family.hw.get(vendor, ()))
    # Encoders for GPUs we did not detect are still worth a probe: detection can
    # miss a card (a driver that hides from DXGI), and a probe is harmless.
    for vendor_names in family.hw.values():
        names.extend(vendor_names)
    if family.hw:
        base = family.hw.get(sysinfo.VENDOR_NVIDIA, ("",))[0].split("_")[0]
        if base:
            names.append(f"{base}_mf")            # Windows Media Foundation
    names.extend(family.software)
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


# --- probing --------------------------------------------------------------


def _pix_fmt(family: Family, ten_bit: bool, encoder: str, profile: Profile | None = None) -> str:
    if profile and profile.pix_fmt:
        return profile.pix_fmt
    fmt = family.pix_fmt_10bit if ten_bit else family.pix_fmt
    # Hardware encoders take semi-planar P010 rather than planar 10-bit.
    if ten_bit and is_hardware(encoder) and fmt == "yuv420p10le":
        return "p010le"
    return fmt


def pixel_format(family: Family, encoder: str, ten_bit: bool, profile: Profile | None = None) -> str:
    return _pix_fmt(family, ten_bit, encoder, profile)


def can_open(encoder: str, family: Family, size: tuple[int, int], fps: float,
             ten_bit: bool = False) -> bool:
    """Whether ``encoder`` opens and encodes one real frame at ``size`` right now."""
    import av

    width, height = size
    # Encoders want even dimensions; 4:2:0 subsampling demands it.
    width, height = width - width % 2, height - height % 2
    try:
        ctx = av.CodecContext.create(encoder, "w")
        ctx.width, ctx.height = width, height
        profile = family.profiles[0] if family.profiles else None
        if family.key == "dnxhr":
            profile = family.profiles[1]           # SQ: the smallest that is valid at any size
        ctx.pix_fmt = _pix_fmt(family, ten_bit, encoder, profile)
        ctx.framerate = Fraction(fps).limit_denominator(90000)
        ctx.time_base = Fraction(1, 90000)
        ctx.options = encoder_options(encoder, family, 60, (width, height), fps, profile)
        ctx.open()
        frame = av.VideoFrame(width, height, ctx.pix_fmt)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.pts = 0
        frame.time_base = Fraction(1, 90000)
        ctx.encode(frame)
        ctx.encode(None)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not on this machine"
        return False


@dataclass
class ProbeReport:
    """What the first-launch probe learned. Saved in settings; keyed by GPU."""

    gpu_signature: str = ""
    gpu_name: str = ""
    vendor: str = ""
    #: encoder name -> works
    results: dict[str, bool] = field(default_factory=dict)
    #: family key -> the default encoder for this machine
    defaults: dict[str, str] = field(default_factory=dict)

    def working(self, family: Family) -> list[str]:
        return [n for n in ladder(family, []) if self.results.get(n)]


def signature(gpus: list[sysinfo.GpuInfo]) -> str:
    return "|".join(g.signature() for g in gpus) or "none"


def probe_all(gpus: list[sysinfo.GpuInfo], progress=None) -> ProbeReport:
    """Try every candidate encoder once and record the winners. Takes a few seconds."""
    report = ProbeReport(
        gpu_signature=signature(gpus),
        gpu_name=gpus[0].name if gpus else "No GPU detected",
        vendor=gpus[0].vendor if gpus else "",
    )
    # 1280x720 is inside every encoder's limits, so a failure here is real.
    size, fps = (1280, 720), 30.0
    for family in FAMILIES:
        for name in ladder(family, gpus):
            if name in report.results:
                continue
            if progress:
                progress(f"Testing {name}…")
            report.results[name] = can_open(name, family, size, fps)
        for name in ladder(family, gpus):
            if report.results.get(name):
                report.defaults[family.key] = name
                break
    return report


def pick(family: Family, gpus: list[sysinfo.GpuInfo], size: tuple[int, int], fps: float,
         ten_bit: bool, report: ProbeReport | None = None, forced: str | None = None) -> str:
    """The encoder to use for this job: proven at this exact size, best first.

    The saved probe narrows the field so a machine with no NVENC does not retry
    it on every export, but the final check is always against the real frame size
    - a 7680x2160 full-width 3D frame is beyond H.264 hardware limits even on a
    card that passed the 720p probe, and the ladder falls through to HEVC/software.
    """
    candidates = [forced] if forced else ladder(family, gpus)
    for name in candidates:
        if report is not None and report.results and report.results.get(name) is False and not forced:
            continue
        if can_open(name, family, size, fps, ten_bit):
            return name
    raise RuntimeError(
        f"No encoder for {family.label} will open at {size[0]}x{size[1]}"
        f"{' in 10-bit' if ten_bit else ''}. Try H.265 or a smaller output size."
    )


# --- quality --------------------------------------------------------------


def _crf(base: float, slope: float, quality: int) -> str:
    return str(int(round(base - slope * quality)))


def bitrate_for(size: tuple[int, int], fps: float, quality: int, family_key: str) -> int:
    """A bitrate for encoders with no constant-quality mode (VideoToolbox), in bits/s.

    Bits-per-pixel is scaled by codec efficiency, and by the slider so 50 is a
    sensible streaming rate and 100 is near-transparent.
    """
    width, height = size
    efficiency = {"h264": 1.0, "h265": 0.62, "av1": 0.5}.get(family_key, 1.0)
    bpp = (0.03 + 0.17 * (quality / 100.0) ** 2) * efficiency
    return int(width * height * fps * bpp)


def encoder_options(encoder: str, family: Family, quality: int, size: tuple[int, int],
                    fps: float, profile: Profile | None) -> dict[str, str]:
    """Constant-quality settings for one encoder, from a single 0-100 slider.

    Each encoder spells "quality" differently (CRF, CQ, QP, global_quality) and on
    a different scale, so the slider is mapped per encoder to land at roughly
    matching visual quality: 50 is a good general-purpose default, 100 is
    near-lossless, 0 is small and rough.
    """
    q = int(max(0, min(100, quality)))
    if family.profiles and profile is not None:
        if family.key == "prores":
            return {"profile": profile.value, "vendor": "apl0"}
        if family.key == "dnxhr":
            return {"profile": profile.value}
    if encoder == "libx264":
        return {"crf": _crf(35, 0.24, q), "preset": "medium"}
    if encoder == "libx265":
        return {"crf": _crf(38, 0.26, q), "preset": "medium", "x265-params": "log-level=none"}
    if encoder == "libsvtav1":
        return {"crf": _crf(52, 0.32, q), "preset": "8"}
    if encoder == "libvpx-vp9":
        return {"crf": _crf(46, 0.3, q), "b": "0", "deadline": "good", "cpu-used": "4", "row-mt": "1"}
    if encoder.endswith("_nvenc"):
        return {"rc": "vbr", "cq": _crf(36, 0.22, q), "b": "0", "preset": "p5", "tune": "hq"}
    if encoder.endswith("_amf"):
        qp = _crf(38, 0.24, q)
        return {"rc": "cqp", "qp_i": qp, "qp_p": qp, "quality": "quality"}
    if encoder.endswith("_qsv"):
        return {"global_quality": _crf(38, 0.24, q), "preset": "medium"}
    if encoder.endswith("_videotoolbox"):
        return {"b": str(bitrate_for(size, fps, q, family.key)), "allow_sw": "0"}
    if encoder.endswith("_mf"):
        return {"rate_control": "quality", "quality": str(q), "hw_encoding": "1"}
    return {}
