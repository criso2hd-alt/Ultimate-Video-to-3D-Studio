"""Find the strongest moment of a film and the 3D strength at which it is still comfortable.

The gauge only tells you about the frame on the screen, so judging a whole film meant scrubbing around looking for the scary bits.
This looks at the film instead: it samples frames across it (or across the In/Out range), measures how far each sticks out of or sinks
behind the screen with the person's *current* settings (the same depth model, cleanup and disparity as the preview, so the numbers are
the gauge's own), and takes the worst. Disparity is directly proportional to the 3D strength, so one number does it: the strength that
puts that worst moment exactly at the chosen limit. It is a single setting for the whole film - nothing changes from scene to scene,
nothing animates - so every other moment is simply scaled down (or up) by the same factor.

The worst frame is found among the samples, so a one-second flash between two samples can be missed; the sample count is high
enough that this is rare, and the figure it reports is honest about what it looked at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import depth as depthmod
from . import stereo, video, zones

#: How many places in the film are looked at (evenly spaced). About 20-60 s on a feature film.
SAMPLES = 240
#: The depth model runs on at most this long an edge (as in the preview), so the numbers match the gauge's.
LONGEST_EDGE = 640


@dataclass
class Sample:
    ms: float
    out_pct: float                         # strongest pop-out in this frame, % of picture width (at the strength it was measured with)
    back_pct: float                        # strongest push-back


@dataclass
class ComfortScan:
    strength: float                        # the 3D strength the numbers were measured at
    samples: list[Sample] = field(default_factory=list)
    stopped: bool = False

    @property
    def worst(self) -> Sample | None:
        return max(self.samples, key=lambda s: max(s.out_pct, s.back_pct), default=None)

    @property
    def worst_pct(self) -> float:
        w = self.worst
        return max(w.out_pct, w.back_pct) if w else 0.0

    def strength_for(self, limit_pct: float) -> float | None:
        """The 3D strength (0..100) that puts the strongest moment at ``limit_pct`` of the picture width; None if nothing sticks
        out or sinks at all (there is nothing to fit)."""
        worst = self.worst_pct
        if worst < 0.02 or self.strength <= 0:
            return None
        return float(min(100.0, max(0.0, self.strength * limit_pct / worst)))


def sample_times(start_ms: float, end_ms: float, count: int = SAMPLES) -> list[float]:
    """Evenly spaced moments, each in the middle of its own slice so the very first and last frames (titles, black) are not the sample."""
    span = max(0.0, end_ms - start_ms)
    if span <= 0 or count <= 0:
        return []
    return [start_ms + span * (i + 0.5) / count for i in range(count)]


def measure_picture(rgb: np.ndarray, engine, stabilizer: depthmod.DepthStabilizer, params: stereo.StereoParams,
                    edge_aware: bool = True, blur: float = 0.0, mode: str = depthmod.STANDARD) -> zones.DepthStats:
    """The gauge's numbers for one picture (float32 RGB 0..1), exactly as the preview would work them out."""
    width = rgb.shape[1]
    stabilizer.reset()                                       # a single frame stands alone: no history from the last sample
    steady = stabilizer.run(engine, rgb)
    d = depthmod.refine(steady, rgb, edge_aware, blur, mode)
    return zones.measure(stereo.disparity_map(d, params, width), d)


def scan(path: str | Path, threed, start_ms: float = 0.0, end_ms: float | None = None, count: int = SAMPLES,
         engine=None, progress: Callable[[float], None] | None = None,
         should_stop: Callable[[], bool] | None = None, nvidia: bool = False) -> ComfortScan:
    """Sample the film and measure it with ``threed`` (a settings.ThreeDSettings: strength, depth position, near/far, cleanup)."""
    import av

    info = video.probe(path)
    end = (info.duration * 1000.0) if end_ms is None else end_ms
    engine = engine or depthmod.shared_engine(False, None, getattr(threed, "depth_model", depthmod.FAST), nvidia)
    stab = depthmod.DepthStabilizer(0.0)
    params = threed.params()
    if params.strength <= 0:
        params.strength = 50.0                               # at zero nothing moves and there is nothing to measure: measure at a typical strength
    mode = depthmod.SMOOTH if threed.depth_quality != depthmod.STANDARD else depthmod.STANDARD
    result = ComfortScan(strength=float(params.strength))
    times = sample_times(start_ms, end, count)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        high = info.bit_depth > 8
        sar = stream.sample_aspect_ratio or stream.codec_context.sample_aspect_ratio
        tb = float(stream.time_base) if stream.time_base else 1.0 / 1000.0
        scale = LONGEST_EDGE / float(max(info.width, info.height))
        size = None if scale >= 1.0 else (max(2, round(info.width * scale)), max(2, round(info.height * scale)))
        for n, ms in enumerate(times):
            if should_stop is not None and should_stop():
                result.stopped = True
                break
            try:
                container.seek(int(ms / 1000.0 / tb), stream=stream, backward=True)
                frame = next(iter(container.decode(stream)), None)         # the first picture at or before the sample: a keyframe is fine
            except Exception:  # noqa: BLE001 - a bad spot in the file is skipped, not fatal
                frame = None
            if frame is not None:
                rgb = video.frame_to_rgb(frame, high, size, sar=sar if size is None else None)
                if np.ptp(rgb) > 0.02:                                       # a black or flat frame has no depth to speak of
                    stats = measure_picture(rgb, engine, stab, params, threed.edge_aware, threed.depth_blur / 100.0 * 6.0, mode)
                    if stats.valid:
                        result.samples.append(Sample(ms, stats.out_pct, stats.back_pct))
            if progress is not None:
                progress((n + 1) / len(times))
    return result


def describe(result: ComfortScan, limit_pct: float, new_strength: float | None) -> str:
    """A plain-language line for what was found and what was done."""
    worst = result.worst
    if worst is None:
        return "Could not find any depth to measure in this video."
    if new_strength is None:
        return f"Nothing in this video comes out of or sinks behind the screen at the current settings ({len(result.samples)} places checked)."
    kind = "pops out" if worst.out_pct >= worst.back_pct else "sinks back"
    return (f"The strongest moment is around {_clock(worst.ms / 1000.0)}, where the picture {kind} {result.worst_pct:.1f}% of its "
            f"width at strength {result.strength:.0f}. Strength is now {new_strength:.0f}, which puts that moment at "
            f"{limit_pct:.1f}% and everything else lower ({len(result.samples)} places checked).")


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"
