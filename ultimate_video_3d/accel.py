"""Optional NVIDIA acceleration: the TensorRT-RTX execution provider as an ONNX Runtime *plugin*.

The app runs depth on DirectML, which works on every GPU. People with a GeForce RTX card can switch on (Settings >
Performance) NVIDIA's TensorRT-RTX plugin: about 110 MB, downloaded once into the data folder, and loaded into the very
same ONNX Runtime the app already ships (``onnxruntime.register_execution_provider_library``) - no CUDA toolkit, no
second runtime, no change for anyone who leaves it off. If it cannot start (no RTX card, an old driver), depth falls
back to DirectML on its own (``depth._providers`` lists it first, DirectML after it).

Measured on an RTX 4080 (driver 591.59), per frame, outputs within 0.5% of DirectML's:
Depth Anything V2 Small 18 ms -> ~2-8 ms; Depth Anything 3 Mono-Large 43 ms -> 7.5 ms (fp16).
The plain CUDA route (``onnxruntime-gpu``) was 16 ms for DA3 but needs a 2 GB download and a *different* onnxruntime
package that cannot sit beside the DirectML one; this plugin can.

The download is NVIDIA's own wheel from PyPI (``onnxruntime-ep-nv-tensorrt-rtx-cu13`` / ``-cu12``, Apache-2.0 plugin;
the TensorRT-RTX runtime inside it is under NVIDIA's licence), fetched the way ``bootstrap`` fetches PyAV: a pinned
version, a URL and an unzip into its own folder. Nothing of it is bundled with the app.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from . import bootstrap, paths

EP_NAME = "NvTensorRTRTXExecutionProvider"
VERSION = "0.5.0"
APPROX_BYTES = 115_000_000
DLL = "onnxruntime_providers_nv_tensorrt_rtx.dll"
#: CUDA 13 builds need an R580 or newer driver; older drivers get the CUDA 12 build.
CU13_MIN_DRIVER = 580.0

_lock = threading.Lock()
_registered: bool | None = None          # None = not tried this process; then True/False


def runtime_dir() -> Path:
    return paths.data_dir() / "nvidia-trt-rtx"


def cache_dir() -> Path:
    """Kernels TensorRT-RTX compiles for this card and driver (rebuilt by itself when the driver changes)."""
    path = runtime_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_path() -> Path:
    return runtime_dir() / "onnxruntime_ep_nv_tensorrt_rtx" / DLL


def is_installed() -> bool:
    return sys.platform == "win32" and library_path().is_file()


def driver_version() -> float | None:
    """The NVIDIA driver version (e.g. 591.59), or None if there is no NVIDIA driver / it cannot be read."""
    if sys.platform != "win32":
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, creationflags=flags).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no NVIDIA driver, or nvidia-smi missing: simply no add-on
        return None


def has_nvidia(gpus) -> bool:
    return sys.platform == "win32" and any("nvidia" in (g.vendor or g.name).lower() for g in gpus or [])


def _package(driver: float | None) -> str:
    cuda = "cu13" if driver is None or driver >= CU13_MIN_DRIVER else "cu12"
    return f"onnxruntime-ep-nv-tensorrt-rtx-{cuda}"


def _resolve_wheel(package: str) -> str:
    request = Request(f"https://pypi.org/pypi/{package}/{VERSION}/json", headers={"User-Agent": "ultimate-video-3d-studio"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    for entry in data.get("urls", []):
        name = entry.get("filename", "")
        if name.endswith(".whl") and "win_amd64" in name:
            return entry["url"]
    raise OSError(f"NVIDIA's TensorRT-RTX package {package} {VERSION} has no Windows download.")


def install(on_bytes: bootstrap.BytesProgress | None = None, on_text: bootstrap.TextProgress | None = None) -> Path:
    """Download and unpack the plugin (only its own folder of the wheel). Raises ``OSError`` with a plain message."""
    if sys.platform != "win32":
        raise OSError("NVIDIA TensorRT acceleration is only available on Windows.")
    target = runtime_dir()
    staging = target.with_name(target.name + ".partial")
    archive = target.with_name(target.name + ".whl.part")
    shutil.rmtree(staging, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if on_text:
            on_text("Finding NVIDIA's TensorRT-RTX package…")
        url = _resolve_wheel(_package(driver_version()))
        if on_text:
            on_text("Downloading NVIDIA TensorRT-RTX…")
        bootstrap._download(url, archive, on_bytes)
        if on_text:
            on_text("Unpacking…")
        bootstrap._extract(archive, staging, on_bytes)
        if not (staging / "onnxruntime_ep_nv_tensorrt_rtx" / DLL).is_file():
            raise OSError("The NVIDIA download is missing its execution provider library.")
        keep_cache = target / "cache"
        if keep_cache.is_dir():                          # compiled kernels survive a reinstall
            shutil.move(str(keep_cache), str(staging / "cache"))
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)
    return library_path()


def register() -> bool:
    """Load the plugin into this process's ONNX Runtime (once). False if it is not installed or will not load."""
    global _registered
    with _lock:
        if _registered is not None:
            return _registered
        _registered = False
        if not is_installed():
            _registered = None                           # may be installed later this session: try again then
            return False
        try:
            import onnxruntime as ort

            folder = str(library_path().parent)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(folder)             # cudart and the TensorRT-RTX DLLs sit beside the plugin
            ort.register_execution_provider_library(EP_NAME, str(library_path()))
            _registered = bool(devices())
        except Exception:  # noqa: BLE001 - an old driver or no RTX card: DirectML carries on
            _registered = False
        return _registered


def devices() -> list:
    import onnxruntime as ort

    return [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]


def status() -> str:
    """One line for Settings: what the switch would do on this machine."""
    if sys.platform != "win32":
        return "Windows only."
    if not is_installed():
        return "Not downloaded yet (about 110 MB, from NVIDIA's package on PyPI)."
    return "Ready." if register() else "Downloaded, but it did not start on this card or driver: DirectML is used."
