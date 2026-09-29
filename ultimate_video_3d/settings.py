"""Everything the user can turn, in one serialisable place."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import stereo
from .grade import ColourSettings

DEFAULT_THEME = "Neural Cyan"


@dataclass
class ThreeDSettings:
    """The 2D -> 3D conversion."""

    enabled: bool = True
    layout: str = "sbs_half"
    strength: float = 50.0          # 0..100
    convergence: float = 55.0       # 0..100 (screen plane position)
    near_scale: float = 100.0       # % of the depth in front of the screen
    far_scale: float = 100.0        # % of the depth behind it
    swap_eyes: bool = False
    #: Depth steadiness over time, 0 (raw per frame) .. 100 (very heavy smoothing).
    temporal: float = 60.0
    edge_aware: bool = True         # snap depth edges to the picture's edges
    depth_blur: float = 0.0         # 0..100, softens depth to hide artefacts

    def params(self) -> stereo.StereoParams:
        return stereo.StereoParams(
            strength=self.strength, convergence=self.convergence / 100.0,
            near_scale=self.near_scale / 100.0, far_scale=self.far_scale / 100.0,
            swap_eyes=self.swap_eyes,
        )


@dataclass
class ExportSettings:
    family: str = "h265"
    #: "auto" picks the best working encoder for this machine; otherwise a name.
    encoder: str = "auto"
    quality: int = 60
    profile: int = 0                # index into the family's profiles
    audio: str = "copy"             # "copy" | "aac" | "none"
    #: Per-eye height cap. 0 keeps the source size.
    max_height: int = 0
    #: "keep" preserves HDR when the codec can carry it; "tonemap" writes SDR.
    hdr_mode: str = "keep"
    range_mode: str = "whole"


@dataclass
class AppSettings:
    threed: ThreeDSettings = field(default_factory=ThreeDSettings)
    colour: ColourSettings = field(default_factory=ColourSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    theme: str = DEFAULT_THEME
    density: str = "compact"
    preview_mode: str = "anaglyph"
    #: Whether the live preview shows the 3D effect. Deliberately not restored on
    #: launch: the app always opens on the plain video, and the effect is opt-in.
    preview_3d: bool = False
    #: Height of the preview proxy. Smaller is smoother; exports are never affected.
    preview_height: int = 540
    #: Seconds of 3D preview rendered ahead of the playhead (0 = off, render live).
    preview_buffer: int = 5
    #: Frame rate of the buffered preview: 0 = choose automatically for this computer.
    preview_fps: int = 0
    #: Look for a newer version on GitHub at launch (it only ever shows a notice and a link).
    check_updates: bool = True
    last_video_dir: str = ""
    last_lut_dir: str = ""
    #: The first-launch encoder probe (see encoders.ProbeReport), or {} if not run.
    encoder_probe: dict = field(default_factory=dict)
    #: 0 for a fresh install; the version of the tour last seen.
    onboarding_version: int = 0
    #: Run the depth model on the CPU even when a GPU is available (troubleshooting).
    force_cpu_depth: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        """Read settings, ignoring anything this version does not understand.

        A file written by a newer build must not stop an older one starting, and
        a key we removed must not raise: unknown keys are dropped, missing ones
        keep their defaults.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt file is not worth a crash
            return cls()
        if not isinstance(raw, dict):
            return cls()

        def build(target, payload):
            if not isinstance(payload, dict):
                return target()
            known = {f.name for f in fields(target)}
            return target(**{k: v for k, v in payload.items() if k in known})

        settings = cls(
            threed=build(ThreeDSettings, raw.get("threed")),
            colour=ColourSettings.from_dict(raw.get("colour")),
            export=build(ExportSettings, raw.get("export")),
        )
        for name in ("theme", "density", "preview_mode", "last_video_dir", "last_lut_dir"):
            if isinstance(raw.get(name), str):
                setattr(settings, name, raw[name])
        if isinstance(raw.get("encoder_probe"), dict):
            settings.encoder_probe = raw["encoder_probe"]
        for name in ("onboarding_version",):
            if isinstance(raw.get(name), int):
                setattr(settings, name, raw[name])
        if raw.get("preview_height") in (360, 480, 540, 720):
            settings.preview_height = raw["preview_height"]
        if raw.get("preview_buffer") in (0, 5, 10):
            settings.preview_buffer = raw["preview_buffer"]
        if raw.get("preview_fps") in (0, 24, 30):
            settings.preview_fps = raw["preview_fps"]
        if isinstance(raw.get("check_updates"), bool):
            settings.check_updates = raw["check_updates"]
        settings.force_cpu_depth = bool(raw.get("force_cpu_depth", False))
        return settings

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_json(), encoding="utf-8")
        except OSError:
            pass  # settings are a convenience; losing them must never interrupt work
