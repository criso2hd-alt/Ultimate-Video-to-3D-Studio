"""Monocular depth for video: Depth Anything (V2 Small, or 3 Mono-Large) on ONNX Runtime, made stable.

Two models (``MODELS``), both Apache-2.0: *fast* is Depth Anything V2 Small (ships with the app, ~18 ms a frame on
DirectML) and *quality* is Depth Anything 3 Mono-Large (downloaded on request; clearly better layering and detail,
~43 ms on DirectML, ~8 ms with the NVIDIA TensorRT-RTX add-on, see ``accel.py``).

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
from dataclasses import dataclass
from pathlib import Path

import os
import threading
import time

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

FAST = "fast"
QUALITY = "quality"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    #: Files in order of preference for DirectML / CPU, and for the NVIDIA TensorRT-RTX add-on. DirectML runs DA3's
    #: fp16 export 6x *slower* than its fp32 one, TensorRT-RTX runs it 3.5x faster, so each gets its own file.
    files: tuple[str, ...]
    nvidia_files: tuple[str, ...]
    size: tuple[int, int]              # network input (height, width): the exports are fixed-shape
    #: "inverse": the network predicts relative inverse depth (V2). "depth": it predicts depth (DA3 Mono; the export
    #: returns 1/depth plus a sky mask), see ``DepthEngine._from_depth``.
    output: str = "inverse"


MODELS: dict[str, ModelSpec] = {
    FAST: ModelSpec(FAST, "Fast (Depth Anything V2 Small)",
                    (MODEL_FILE_FP16, MODEL_FILE), (MODEL_FILE_FP16, MODEL_FILE), (INPUT, INPUT)),
    # DA3 Mono-Large at 280 x 504 (20 x 36 patches): half the tokens of a 504 square, so 43 ms instead of 114 ms on
    # DirectML, and films are wide anyway. Export: scripts/export_da3.py.
    QUALITY: ModelSpec(QUALITY, "High quality (Depth Anything 3 Mono-Large)",
                       ("Depth-Anything-3-Mono-Large.onnx",),
                       ("Depth-Anything-3-Mono-Large.fp16.onnx", "Depth-Anything-3-Mono-Large.onnx"),
                       (280, 504), output="depth"),
}


def spec(model: str) -> ModelSpec:
    return MODELS.get(model, MODELS[FAST])

#: Resolution the temporal stabiliser works at. Depth is smooth, so half of the
#: network's output loses nothing visible - and the refine step that follows
#: snaps its edges back to the picture - while the filter costs a quarter as much.
WORK = 259

_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def locate(cpu: bool = False, model: str = FAST, nvidia: bool = False) -> Path | None:
    """The model file to use. V2: fp16 normally, fp32 first on the CPU. DA3: fp32 (DirectML / CPU), fp16 for TensorRT-RTX."""
    m = spec(model)
    if m.key == FAST:
        names = (MODEL_FILE, MODEL_FILE_FP16) if cpu else (MODEL_FILE_FP16, MODEL_FILE)
    else:
        names = m.nvidia_files if nvidia and not cpu else m.files
    for name in names:
        for base in (paths.bundled_onnx_dir(), paths.model_dir()):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def is_installed(model: str = FAST, nvidia: bool = False) -> bool:
    """Whether the file this model wants on this path is here. With ``nvidia``, DA3 wants its fp16 file: the fp32
    one would run, but at a third of the speed the add-on is there for."""
    found = locate(False, model, nvidia)
    if found is None:
        return False
    return not (nvidia and spec(model).key != FAST and found.name != spec(model).nvidia_files[0])


def missing_files(model: str = FAST, nvidia: bool = False) -> list[str]:
    """The file(s) to download so ``model`` runs at full speed on this path."""
    m = spec(model)
    if m.key == FAST or is_installed(model, nvidia):
        return []
    return [m.nvidia_files[0] if nvidia else m.files[0]]


def _providers(nvidia: bool = False) -> list[tuple[str, str]]:
    """(provider, friendly label) in preference order, filtered to what is installed.

    With ``nvidia`` and the TensorRT-RTX add-on present (``accel.py``) it comes first. DirectML stays behind it as
    the fallback, so a card or driver the add-on does not support simply runs as before."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen: list[tuple[str, str]] = []
    if nvidia:
        from . import accel

        if accel.register():
            chosen.append((accel.EP_NAME, "NVIDIA RTX (TensorRT)"))
    if sys.platform == "darwin":
        wanted = [("CoreMLExecutionProvider", "Apple GPU / Neural Engine (CoreML)")]
    else:
        wanted = [
            ("CUDAExecutionProvider", "NVIDIA GPU (CUDA)"),
            ("DmlExecutionProvider", "GPU (DirectML)"),
        ]
    chosen += [(p, label) for p, label in wanted if p in available]
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

#: Seconds a frame above which the NVIDIA path counts as not working (an RTX 4080 does DA3 in ~8 ms, V2 in ~2 ms;
#: DirectML does them in ~45 / ~20 ms). Something that slow is running on the CPU, or the card is not one it suits.
NV_TOO_SLOW = 0.25

_shared_lock = threading.Lock()
_shared: "DepthEngine | None" = None

#: GPU sessions are freed oldest first, never a newer one while an older one lives.
#: With onnxruntime-directml 1.24.4, freeing a DirectML session breaks every *older* session still alive: its next
#: ``run`` dies with an access violation - even in one thread, even under ``_ORT_LOCK``. Freeing the oldest while
#: newer ones live is fine, and so is freeing a CPU session. 0.2.0/0.2.1 crashed this way after every conversion
#: (the conversion made its own, newer session and dropped it while the preview's lived; the next preview frame
#: crashed). Repro: make A, make B, free B, run A. 0.2.2 shares one engine, but a model switch during a long
#: conversion, or a provider that fails its probe run, still freed a newer one.
_gpu_order: list[int] = []          # tokens of the GPU sessions not yet freed, oldest first
_parked: dict[int, object] = {}     # GPU sessions let go of while an older one was still alive
_next_token = 0


def _adopt_gpu_session() -> int:
    """A token for a GPU session just created (caller holds ``_ORT_LOCK``)."""
    global _next_token
    _next_token += 1
    _gpu_order.append(_next_token)
    return _next_token


def _free_parked() -> None:
    """Free parked sessions that have become the oldest (caller holds ``_ORT_LOCK``)."""
    while _gpu_order and _gpu_order[0] in _parked:
        _parked.pop(_gpu_order.pop(0))      # the dict held the last reference: the session is destroyed here


def engine_key(force_cpu: bool = False, model: str = FAST, nvidia: bool = False) -> tuple[str, bool, bool]:
    return (spec(model).key, bool(force_cpu), bool(nvidia) and not force_cpu)


def shared_engine(force_cpu: bool = False, progress=None, model: str = FAST, nvidia: bool = False) -> "DepthEngine":
    """The one depth engine the live preview, the buffered renderer and conversions all use.

    It stays resident (DA3 takes seconds to load, ~15 s the first time with TensorRT-RTX) and is replaced only when
    the model, the CPU switch or the NVIDIA switch changes. Holders of the old one finish with it; it is freed after.
    """
    global _shared
    want = engine_key(force_cpu, model, nvidia)
    with _shared_lock:
        if _shared is None or _shared.key != want:
            engine = DepthEngine(model)
            engine.load(progress, force_cpu, nvidia)
            _shared = engine
        return _shared


def forget_shared() -> None:
    """Drop the resident engine so the next caller loads afresh (a better model file has just arrived)."""
    global _shared
    with _shared_lock:
        _shared = None


def _open_nvidia(ort, path: Path, gpu_options):
    """A TensorRT-RTX session, from the saved *EP context* when there is one.

    Building the TensorRT engine from the ONNX file takes ~6 s (V2) / ~11 s (DA3) on an RTX 4080 at *every* launch, even
    with the kernel cache, and while it runs Windows makes every other thread that loads a DLL wait (the window froze for
    as long). The first build therefore also saves the engine as an EP-context model next to the kernel cache; later
    launches open that in 0.4 s / 2 s with identical output. Keyed by model file, add-on version and driver, so an update
    of any of them builds afresh; a saved one that will not open is deleted and rebuilt.
    """
    from . import accel

    def nvidia_options(extra: dict[str, str] | None = None):
        opts = gpu_options()
        # A plugin EP is attached to the options, not named in ``providers``. Its compiled kernels are cached on disk.
        opts.add_provider_for_devices(accel.devices(), {"nv_runtime_cache_path": str(accel.cache_dir())})
        # All of the model on the card, or not at all: a graph the card cannot take was otherwise run partly on the CPU,
        # and the fp16 file on the CPU crawls (Discord: "NVIDIA RTX acceleration is very slow, 0.2 fps"). Refused here,
        # the next provider (DirectML, with the fp32 file) takes over.
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        for key, value in (extra or {}).items():
            opts.add_session_config_entry(key, value)
        return opts

    saved = accel.context_path(path)
    if saved is not None and saved.exists():
        try:
            return ort.InferenceSession(str(saved), sess_options=nvidia_options())
        except Exception:  # noqa: BLE001 - stale or damaged: build from the model again
            saved.unlink(missing_ok=True)
    if saved is not None and accel.room_for_context(path):
        partial = saved.with_name(saved.stem + ".part.onnx")
        try:
            session = ort.InferenceSession(str(path), sess_options=nvidia_options({
                "ep.context_enable": "1", "ep.context_file_path": str(partial), "ep.context_embed_mode": "1"}))
            if partial.exists():
                os.replace(partial, saved)
            return session
        except Exception:  # noqa: BLE001 - saving is a bonus: fall back to a plain build below
            pass
        finally:
            partial.unlink(missing_ok=True)
    return ort.InferenceSession(str(path), sess_options=nvidia_options())


class DepthEngine:
    """Runs the model. Safe to share between threads (calls are serialised)."""

    def __init__(self, model: str = FAST) -> None:
        self.model = spec(model)
        self.session = None
        self.device = "not loaded"
        self._input_name = ""
        self._force_cpu = False
        self._nvidia = False
        self._token: int | None = None      # place in ``_gpu_order`` (None: no session, or a CPU one)

    @property
    def key(self) -> tuple[str, bool, bool]:
        """What was asked for, not what it ended up on - so a fallback (no add-on, say) is not retried on every call."""
        return engine_key(self._force_cpu, self.model.key, self._nvidia)

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def close(self) -> None:
        """Let go of the session. Never set ``session = None`` by hand: see ``_gpu_order`` for why the order matters.

        A GPU session is freed now if it is the oldest one alive, otherwise parked until the older ones are gone.
        """
        with _ORT_LOCK:
            session, self.session = self.session, None
            token, self._token = self._token, None
            if session is not None and token is not None:
                _parked[token] = session
            del session                     # a CPU session (the only reference) is destroyed here
            _free_parked()

    def __del__(self) -> None:
        # An engine replaced by ``shared_engine`` (model, CPU or NVIDIA switch) or dropped by ``forget_shared`` dies
        # in whichever thread lets go of it last: it must still go through ``close``.
        try:
            if self.session is not None:
                self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown: the lock or the module may already be gone
            pass

    def load(self, progress: Callable[[str], None] | None = None, force_cpu: bool = False, nvidia: bool = False) -> str:
        """Open the model on the best working accelerator. Returns its label."""
        self._force_cpu, self._nvidia = bool(force_cpu), bool(nvidia)
        with _ORT_LOCK:
            return self._load(progress, force_cpu, nvidia and not force_cpu)

    def _load(self, progress, force_cpu: bool, nvidia: bool = False) -> str:
        if self.session is not None:
            return self.device
        model = locate(force_cpu, self.model.key, nvidia)
        if model is None:
            if self.model.key != FAST:
                raise RuntimeError(f"The {self.model.label} depth model is not downloaded yet. Pick it again in the "
                                   "3D panel to download it, or switch back to Fast.")
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

        candidates = [("CPUExecutionProvider", "CPU")] if force_cpu else _providers(nvidia)
        errors: list[str] = []
        height, width = self.model.size
        for provider, label in candidates:
            session = token = None
            try:
                on_nvidia = provider == _NV_EP
                path = locate(provider == "CPUExecutionProvider", self.model.key, on_nvidia) or model
                if progress and on_nvidia:
                    progress("Preparing the depth model for your NVIDIA card (the first time takes a little while)…")
                def gpu_options():
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
                    return opts

                if on_nvidia:
                    session = _open_nvidia(ort, path, gpu_options)
                elif provider == "CPUExecutionProvider":
                    session = ort.InferenceSession(str(path), options, providers=[provider])
                else:
                    session = ort.InferenceSession(str(path), gpu_options(), providers=[provider])
                if provider != "CPUExecutionProvider":
                    token = _adopt_gpu_session()
                name = session.get_inputs()[0].name
                # A provider can open a session and still fail (or return garbage)
                # on the first real run, so prove it before committing.
                probe = np.random.default_rng(1).random((1, 3, height, width), dtype=np.float32)
                out = np.asarray(session.run(None, {name: probe})[0])
                if not np.isfinite(out).all():
                    raise RuntimeError("non-finite output")
                if on_nvidia:
                    # The probe above paid for the first-run compile; now it must be quick, or DirectML does better.
                    took = []
                    for _ in range(3):
                        began = time.perf_counter()
                        session.run(None, {name: probe})
                        took.append(time.perf_counter() - began)
                    per_frame = sorted(took)[1]
                    if per_frame > NV_TOO_SLOW:
                        raise RuntimeError(f"{per_frame * 1000:.0f} ms a frame")
            except Exception as error:  # noqa: BLE001 - fall to the next provider
                errors.append(f"{provider}: {error}")
                if provider == _NV_EP:
                    from . import accel

                    accel.reject(str(error).splitlines()[0][:120] if str(error) else type(error).__name__)
                if token is not None:
                    _parked[token] = session    # a GPU session that failed its probe is freed in order, like any other
                session = None
                continue
            if provider == _NV_EP:
                from . import accel

                accel.reject("")
            self.session, self.device, self._input_name, self._token = session, label, name, token
            _free_parked()
            return label
        _free_parked()
        raise RuntimeError("Could not start the depth model on any device:\n" + "\n".join(errors))

    def infer(self, rgb: np.ndarray, bars: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Raw relative inverse-depth-like values (near = larger), shaped like the network's input.

        ``rgb`` is float32 0..1 display-referred RGB at any size. The values are
        *unnormalised*: their scale and offset change every frame, which is why
        the stabiliser exists.

        ``bars`` = (top, bottom, left, right) rows/columns of ``rgb`` that are black bars burned into the video: only
        the picture between them goes through the network, and the bars get the picture's far value. Otherwise the
        model reads the bars as the nearest (or some random) thing in the scene and they eat the depth range
        (a quarter of it on a 2.39:1 film in a 16:9 frame), which flattens the real picture.
        """
        if self.session is None:
            raise RuntimeError("Load the depth model first.")
        full_h, full_w = rgb.shape[:2]
        cropped = bool(bars) and any(bars)
        if cropped:
            top, bottom, left, right = bars
            rgb = rgb[top:full_h - bottom, left:full_w - right]
        height, width = self.model.size
        big = rgb.shape[0] > height or rgb.shape[1] > width
        small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA if big else cv2.INTER_CUBIC)
        batch = np.empty((1, 3, height, width), np.float32)
        _to_nchw(np.ascontiguousarray(small, np.float32), batch, _MEAN1, _INV_STD1)
        with _ORT_LOCK:
            session = self.session
            if session is None:                 # closed by another thread since the check above
                raise RuntimeError("Load the depth model first.")
            outputs = session.run(None, {self._input_name: batch})
            del session
        raw = np.asarray(outputs[0], np.float32).reshape(height, width)
        if self.model.output == "depth":
            raw = _from_depth(raw, np.asarray(outputs[1], np.float32).reshape(height, width))
        if cropped:
            raw = _pad_bars(raw, bars, full_h, full_w)
        return raw


#: Name of NVIDIA's TensorRT-RTX plugin execution provider (``accel.EP_NAME``; here so ``depth`` need not import it).
_NV_EP = "NvTensorRTRTXExecutionProvider"


def _from_depth(inverse: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """DA3 Mono predicts *depth* (its export returns 1/depth). Used as plain inverse depth the nearest object takes
    nearly the whole range and everything behind it goes flat; as linear depth the distance swells. The log of depth
    sits between the two and gave the best 3D on real films (layers kept apart, distance kept). Sky is put at the far
    end of what is not sky, as DA3's own post-processing does (the sky head is a second output)."""
    non_sky = sky < 0.3
    n = int(non_sky.sum())
    if 10 < n < non_sky.size - 10:
        inverse = np.where(non_sky, inverse, np.quantile(inverse[non_sky], 0.01))
    return np.log(np.maximum(inverse, 1e-6)).astype(np.float32)


def _pad_bars(raw: np.ndarray, bars: tuple[int, int, int, int], full_h: int, full_w: int) -> np.ndarray:
    """The picture's depth back in a map shaped like the whole frame, the bars filled with the far value (the 2nd
    percentile: inside the picture's own range, so it neither stretches nor squeezes the normalisation)."""
    top, bottom, left, right = bars
    h, w = raw.shape
    sy = h / max(1, full_h - top - bottom)
    sx = w / max(1, full_w - left - right)
    far = float(np.percentile(raw[::2, ::2], 2.0))
    pad = ((int(round(top * sy)), int(round(bottom * sy))), (int(round(left * sx)), int(round(right * sx))))
    return np.pad(raw, pad, constant_values=far).astype(np.float32)


#: A row (or column) at the edge is black bar if fewer than ``BAR_SHARE`` of its pixels are brighter than ``BAR_BRIGHT``.
BAR_BRIGHT = 0.08
BAR_SHARE = 0.005
BAR_MIN = 0.02            # thinner than this share of the frame: not worth it (could be a dark edge of the picture)
BAR_MAX = 0.3             # more than this off one side: a dark picture, not bars
BAR_SETTLE = 24           # frames a *bigger* set of bars must hold before it is believed (a dark scene is not a bar)


def find_bars(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) black rows/columns at the edges of one frame. One row of each bar is left to the
    picture, so nothing that is picture is ever cut (a hair of black in the network's view is harmless)."""
    h, w = rgb.shape[:2]
    bright = rgb.max(axis=2) > BAR_BRIGHT
    rows = bright.mean(axis=1) > BAR_SHARE
    cols = bright.mean(axis=0) > BAR_SHARE
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)                                 # an all-dark frame says nothing about bars

    def keep(n: int, size: int) -> int:
        return max(0, n - 1) if BAR_MIN * size <= n <= BAR_MAX * size else 0

    r, c = np.flatnonzero(rows), np.flatnonzero(cols)
    return (keep(int(r[0]), h), keep(h - 1 - int(r[-1]), h), keep(int(c[0]), w), keep(w - 1 - int(c[-1]), w))


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

    def run(self, engine: "DepthEngine", frame_rgb: np.ndarray) -> np.ndarray:
        """Bars, model and stabiliser in one call: what every caller (preview, look-ahead, conversion) does per frame."""
        return self(engine.infer(frame_rgb, self.bars(frame_rgb)), frame_rgb)

    def bars(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int]:
        """This stream's burned-in black bars, steadied over frames, in this frame's pixels.

        Smaller bars are believed at once (a picture reaching into them must never be cut); bigger ones only after
        ``BAR_SETTLE`` frames agree, so a dark scene cannot make the crop flicker. Kept across scene cuts (``reset``
        leaves them): bars belong to the film, not the shot."""
        h, w = frame_rgb.shape[:2]
        seen = np.asarray(find_bars(frame_rgb), np.float64) / np.asarray((h, h, w, w), np.float64)
        held = getattr(self, "_bars", None)
        if held is None:
            self._bars, self._bars_next, self._bars_count = seen, None, 0
        elif np.all(seen <= held + 1e-9):
            self._bars, self._bars_next, self._bars_count = seen, None, 0
        elif self._bars_next is not None and np.allclose(seen, self._bars_next, atol=0.01):
            self._bars_count += 1
            if self._bars_count >= BAR_SETTLE:
                self._bars, self._bars_next, self._bars_count = seen, None, 0
        else:
            self._bars_next, self._bars_count = seen, 1
        top, bottom, left, right = self._bars
        return (int(top * h), int(bottom * h), int(left * w), int(right * w))

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


#: Working width of the refinement: the live preview and the standard mode use 640; "smooth_high" an export 960.
REFINE_FAST, REFINE_HIGH = 640, 960

#: ``refine`` modes (``ThreeDSettings.depth_quality``). "standard" is the original, and the default: it gave
#: excellent 3D in use. "smooth" and "smooth_high" flatten each object first; see ``refine``.
STANDARD, SMOOTH, SMOOTH_HIGH = "standard", "smooth", "smooth_high"


def refine_work(mode: str, preview: bool = False) -> int:
    """The refinement width for a mode: the live preview always uses the small one."""
    return REFINE_HIGH if mode == SMOOTH_HIGH and not preview else REFINE_FAST


def refine(depth: np.ndarray, frame_rgb: np.ndarray, edge_aware: bool = True,
           blur: float = 0.0, mode: str = STANDARD, work: int = REFINE_FAST) -> np.ndarray:
    """Depth at the frame's own resolution, its edges snapped to the picture's.

    The network's output is a soft 518 px map. Stretched to 1080p and used as-is,
    the depth edge lands a few pixels off the picture edge, and every stereo
    warp then tears a halo around each near object. A guided filter pulls the
    depth boundary onto the luminance boundary, which is the single biggest
    visible-quality gain after the model itself.

    ``mode`` "standard" does just that. "smooth" adds two steps before it, for footage where the result looks
    mottled: a median filter against speckle, then a filter steered by the *depth itself* that flattens each
    object (steered by the picture alone, the filter also drags in its texture: cloth, hair, carpet). The
    picture-steered snap then only has the outline left to do. It looks cleaner as a depth map and adds
    ~7 ms a frame at 640 px, ~25 ms at 960 ("smooth_high", exports only); the stereo result differs far less
    than the maps do, so it is an option, not the default.
    """
    height, width = frame_rgb.shape[:2]
    work_w = min(width, work)
    work_h = max(1, round(height * work_w / width))
    small = cv2.resize(depth, (work_w, work_h), interpolation=cv2.INTER_CUBIC)
    if edge_aware:
        guide = cv2.cvtColor(cv2.resize(frame_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2GRAY)
        if mode == STANDARD:
            small = guided_filter(guide, small, max(2, work_w // 160), 2e-3)
        else:
            small = cv2.medianBlur((np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
            small = guided_filter(small, small, max(2, work_w // 120), 2e-3)
            small = guided_filter(guide, small, max(2, work_w // 320), 4e-3)
    if blur > 0.0:
        small = cv2.GaussianBlur(small, (0, 0), blur * work_w / 480.0)
    small = np.clip(small, 0.0, 1.0)
    if (work_w, work_h) == (width, height):
        return small.astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
