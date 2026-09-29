"""Monocular depth for video: Depth Anything V2 on ONNX Runtime, made stable.

Two separate problems live here.

**Running the model anywhere.** ONNX Runtime executes the network through
whichever accelerator the platform offers, with no vendor toolkit installed:

- Windows: DirectML - any DirectX 12 GPU (NVIDIA, AMD, Intel), through the
  driver the user already has. No CUDA, no cuDNN, no multi-gigabyte download.
- macOS: CoreML - the Apple GPU and Neural Engine.
- Everywhere else, and as the safety net if a GPU provider will not start: CPU.

**Making per-frame depth behave as video.** Depth Anything predicts each frame
independently, and its output has an arbitrary scale and offset every time. Fed
straight to a stereo renderer that shows up as depth that pumps and shimmers,
which in a headset is nauseating. ``DepthStabilizer`` fixes this three ways:
the normalisation range is smoothed over time, the depth itself is blended with
the previous frame *except where the picture actually moved*, and everything is
reset at a scene cut so one shot's depth never leaks into the next.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import os
import threading

import cv2
import numpy as np

from . import paths

if "NUMBA_CACHE_DIR" not in os.environ:
    try:
        os.environ["NUMBA_CACHE_DIR"] = str(paths.data_dir() / "numba_cache")
    except Exception:  # noqa: BLE001
        pass

from numba import njit

#: The half-precision export is what ships: 50 MB, ~25% faster on a GPU, and
#: 0.99999 correlated with the full-precision one. The full-precision file is
#: used instead on the CPU if it is present (fp16 is slower there), and as the
#: fallback if only it exists.
MODEL_FILE_FP16 = "Depth-Anything-V2-Small-hf.fp16.onnx"
MODEL_FILE = "Depth-Anything-V2-Small-hf.onnx"

#: The exported network is fixed-shape (see scripts/export_onnx.py).
INPUT = 518

#: Resolution the temporal stabiliser works at. Depth is smooth, so half of the
#: network's output loses nothing visible - and the refine step that follows
#: snaps its edges back to the picture - while the filter costs a quarter as much.
WORK = 259

_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def locate(cpu: bool = False) -> Path | None:
    """The model file to use: fp16 normally, fp32 first when running on the CPU."""
    names = (MODEL_FILE, MODEL_FILE_FP16) if cpu else (MODEL_FILE_FP16, MODEL_FILE)
    for name in names:
        for base in (paths.bundled_onnx_dir(), paths.model_dir()):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def is_installed() -> bool:
    return locate() is not None


def _providers() -> list[tuple[str, str]]:
    """(provider, friendly label) in preference order, filtered to what is installed."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if sys.platform == "darwin":
        wanted = [("CoreMLExecutionProvider", "Apple GPU / Neural Engine (CoreML)")]
    else:
        wanted = [
            ("CUDAExecutionProvider", "NVIDIA GPU (CUDA)"),
            ("DmlExecutionProvider", "GPU (DirectML)"),
        ]
    chosen = [(p, label) for p, label in wanted if p in available]
    chosen.append(("CPUExecutionProvider", "CPU"))
    return chosen


@njit(fastmath=True, cache=True, nogil=True)
def _to_nchw(img, out, mean, inv_std):
    """Normalise and transpose HWC -> NCHW.

    Deliberately *not* parallel: it is under a millisecond for a 518 px square, and a
    parallel kernel would have to queue for ``kernels.LOCK`` behind the (long) stereo
    warps - which measurably delayed every depth inference by ~10 ms. A serial kernel
    is thread-safe and needs no lock.
    """
    h, w, _ = img.shape
    for c in range(3):
        for y in range(h):
            for x in range(w):
                out[0, c, y, x] = (img[y, x, c] - mean[c]) * inv_std[c]


_MEAN1 = _MEAN.reshape(3).astype(np.float32)
_INV_STD1 = (1.0 / _STD.reshape(3)).astype(np.float32)


#: Every ONNX Runtime call - creating a session, its first run, and each inference -
#: goes through this lock. Two DirectML sessions on one GPU being created or run at the
#: same moment from different threads crashes inside the driver (an access violation,
#: seen when the live preview and the look-ahead renderer each owned a session). The
#: calls are ~20 ms and the GPU serialises them anyway, so nothing is lost.
_ORT_LOCK = threading.RLock()

_shared_lock = threading.Lock()
_shared: "DepthEngine | None" = None


def shared_engine(force_cpu: bool = False, progress=None) -> "DepthEngine":
    """The one depth engine the live preview and the buffered renderer both use."""
    global _shared
    with _shared_lock:
        if _shared is None or (_shared.device == "CPU") != force_cpu:
            engine = DepthEngine()
            engine.load(progress, force_cpu)
            _shared = engine
        return _shared


class DepthEngine:
    """Runs the model. Safe to share between threads (calls are serialised)."""

    def __init__(self) -> None:
        self.session = None
        self.device = "not loaded"
        self._input_name = ""

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def load(self, progress: Callable[[str], None] | None = None, force_cpu: bool = False) -> str:
        """Open the model on the best working accelerator. Returns its label."""
        with _ORT_LOCK:
            return self._load(progress, force_cpu)

    def _load(self, progress, force_cpu: bool) -> str:
        if self.session is not None:
            return self.device
        model = locate(cpu=force_cpu)
        if model is None:
            raise RuntimeError(
                f"The depth model ({MODEL_FILE_FP16}) is missing. The app downloads it automatically "
                "the next time it starts (an internet connection is needed once)."
            )
        import onnxruntime as ort

        if progress:
            progress("Loading the depth model…")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # DirectML is documented to want these two off; harmless on other providers.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        candidates = [("CPUExecutionProvider", "CPU")] if force_cpu else _providers()
        errors: list[str] = []
        for provider, label in candidates:
            try:
                path = locate(cpu=provider == "CPUExecutionProvider") or model
                opts = options
                if provider != "CPUExecutionProvider":
                    # The GPU does the work; ORT's CPU worker threads only feed it. By
                    # default they spin-wait between frames and eat most of the machine,
                    # starving the decoder, the warp and the UI for nothing.
                    opts = ort.SessionOptions()
                    opts.log_severity_level = 3
                    opts.graph_optimization_level = options.graph_optimization_level
                    opts.enable_mem_pattern = False
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    opts.intra_op_num_threads = 2
                    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                session = ort.InferenceSession(str(path), opts, providers=[provider])
                name = session.get_inputs()[0].name
                # A provider can open a session and still fail (or return garbage)
                # on the first real run, so prove it before committing.
                probe = np.random.default_rng(1).random((1, 3, INPUT, INPUT), dtype=np.float32)
                out = np.asarray(session.run(None, {name: probe})[0])
                if not np.isfinite(out).all():
                    raise RuntimeError("non-finite output")
            except Exception as error:  # noqa: BLE001 - fall to the next provider
                errors.append(f"{provider}: {error}")
                continue
            self.session, self.device, self._input_name = session, label, name
            return label
        raise RuntimeError("Could not start the depth model on any device:\n" + "\n".join(errors))

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Raw relative inverse depth (near = larger) at ``INPUT`` x ``INPUT``.

        ``rgb`` is float32 0..1 display-referred RGB at any size. The values are
        *unnormalised*: their scale and offset change every frame, which is why
        the stabiliser exists.
        """
        if self.session is None:
            raise RuntimeError("Load the depth model first.")
        small = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_AREA if max(rgb.shape[:2]) > INPUT else cv2.INTER_CUBIC)
        batch = np.empty((1, 3, INPUT, INPUT), np.float32)
        _to_nchw(np.ascontiguousarray(small, np.float32), batch, _MEAN1, _INV_STD1)
        with _ORT_LOCK:
            raw = self.session.run(None, {self._input_name: batch})[0]
        return np.asarray(raw, np.float32).reshape(INPUT, INPUT)


class DepthStabilizer:
    """Turns a stream of raw depth maps into a steady 0..1 depth video."""

    #: Mean per-pixel change between frames (0..1) above which we call it a cut.
    CUT_THRESHOLD = 0.16

    def __init__(self, smoothing: float = 0.6) -> None:
        #: 0 = trust each frame completely, 1 = lean heavily on history.
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.reset()

    def reset(self) -> None:
        self._lo: float | None = None
        self._hi: float | None = None
        self._prev_depth: np.ndarray | None = None
        self._prev_thumb: np.ndarray | None = None

    def _is_cut(self, thumb: np.ndarray) -> bool:
        if self._prev_thumb is None:
            return False
        return float(np.abs(thumb - self._prev_thumb).mean()) > self.CUT_THRESHOLD

    def __call__(self, raw: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
        """``raw`` from DepthEngine.infer, ``frame_rgb`` the matching frame. Returns 0..1, 1 = near."""
        thumb = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA).mean(axis=-1)
        if self._is_cut(thumb):
            self.reset()

        raw = cv2.resize(raw, (WORK, WORK), interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(raw[::2, ::2], (2.0, 98.0))
        keep = self.smoothing
        if self._lo is None:
            self._lo, self._hi = float(lo), float(hi)
        else:
            # Smooth the *range*, not just the picture: a bright object entering
            # the frame changes the percentiles, and re-normalising every frame
            # on them would re-scale all of depth for one frame's worth of change.
            self._lo = keep * self._lo + (1.0 - keep) * float(lo)
            self._hi = keep * self._hi + (1.0 - keep) * float(hi)
        span = max(self._hi - self._lo, 1e-4)
        depth = np.clip((raw - self._lo) / span, 0.0, 1.0)

        if self._prev_depth is not None and keep > 0.0:
            # Trust history where the picture is still, the new estimate where it
            # moved: static regions stop shimmering, moving ones stay responsive.
            delta = cv2.resize(np.abs(thumb - self._prev_thumb), (WORK, WORK), interpolation=cv2.INTER_LINEAR)
            moving = np.clip(delta / 0.08, 0.0, 1.0)
            history = keep * (1.0 - moving)
            depth = history * self._prev_depth + (1.0 - history) * depth
        self._prev_depth = depth
        self._prev_thumb = thumb
        return depth


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of ``src`` steered by ``guide`` (He et al.), via box filters."""
    size = (2 * radius + 1, 2 * radius + 1)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_REPLICATE)

    mean_i, mean_p = box(guide), box(src)
    cov = box(guide * src) - mean_i * mean_p
    var = box(guide * guide) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def refine(depth: np.ndarray, frame_rgb: np.ndarray, edge_aware: bool = True,
           blur: float = 0.0) -> np.ndarray:
    """Depth at the frame's own resolution, its edges snapped to the picture's.

    The network's output is a soft 518 px map. Stretched to 1080p and used as-is,
    the depth edge lands a few pixels off the picture edge, and every stereo
    warp then tears a halo around each near object. A guided filter pulls the
    depth boundary onto the luminance boundary, which is the single biggest
    visible-quality gain after the model itself.
    """
    height, width = frame_rgb.shape[:2]
    work_w = min(width, 640)
    work_h = max(1, round(height * work_w / width))
    guide = cv2.cvtColor(cv2.resize(frame_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY)
    small = cv2.resize(depth, (work_w, work_h), interpolation=cv2.INTER_CUBIC)
    if edge_aware:
        radius = max(2, work_w // 160)
        small = guided_filter(guide, small, radius, 2e-3)
    if blur > 0.0:
        small = cv2.GaussianBlur(small, (0, 0), blur * work_w / 480.0)
    small = np.clip(small, 0.0, 1.0)
    if (work_w, work_h) == (width, height):
        return small.astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
