"""Where the app keeps data that must outlive an update, and where it finds its own files.

Layout rules, by platform:

- Windows frozen build: **portable** - luts/, models/, output/ sit beside the exe,
  so everything the app needs and produced is visible in one folder, and a
  rebuild that preserves those folders never re-downloads anything.
- macOS frozen build: a ``.app`` bundle is not a place to write, so data goes to
  ``~/Library/Application Support`` like any other Mac app.
- Running from source: the per-user data dir on every platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "UltimateVideo3DStudio"

#: Dropping this file beside the executable opts into keeping data with the app.
PORTABLE_MARKER = "portable.txt"

MODELS_DIR = "models"
OUTPUT_DIR = "output"
#: Where the user drops ``.cube`` LUTs. A plain folder they fill themselves; any
#: film-emulation pack works because .cube is the portable format.
LUTS_DIR = "luts"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """The folder holding the executable (frozen) or the repository (source)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """Where bundled read-only resources live (PyInstaller's ``_MEIPASS`` if frozen)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return Path(__file__).resolve().parent.parent


def package_dir() -> Path:
    return resource_dir() / "ultimate_video_3d"


def fonts_dir() -> Path:
    return package_dir() / "assets" / "fonts"


def bundled_onnx_dir() -> Path:
    """The Apache-2.0 Small depth model ships inside the app - no download."""
    return package_dir() / "assets" / "onnx"


def _user_data_dir() -> Path:
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
    except Exception:  # noqa: BLE001 - fall back rather than fail to start
        return Path.home() / f".{APP_NAME}"


def is_portable() -> bool:
    if (app_dir() / PORTABLE_MARKER).exists():
        return True
    return is_frozen() and sys.platform == "win32"


def data_dir() -> Path:
    path = app_dir() if is_portable() else _user_data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    path = data_dir() / MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    path = data_dir() / OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _known_pictures_dir() -> Path | None:
    """The folder Windows itself calls Pictures. It is not always ``~/Pictures``: with OneDrive backing up the Pictures folder
    it lives in ``~/OneDrive/Pictures``, and a plain ``~/Pictures`` beside it is a different folder the person never opens."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]

        folder_id = GUID(0x33E28130, 0x4E1E, 0x4676, (ctypes.c_ubyte * 8)(0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB))   # FOLDERID_Pictures
        result = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(result)) != 0:
            return None
        path = Path(result.value)
        ctypes.windll.ole32.CoTaskMemFree(result)
        return path if path.is_dir() else None
    except Exception:  # noqa: BLE001 - fall back to ~/Pictures
        return None


def screenshots_dir() -> Path:
    """Where headset screenshots are saved: the person's Pictures folder (they look there first), else beside the app's data."""
    pictures = _known_pictures_dir() or Path.home() / "Pictures"
    if pictures.is_dir():
        path = pictures / "Ultimate Video to 3D Studio"
    else:
        path = data_dir() / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def luts_dir() -> Path:
    path = data_dir() / LUTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return data_dir() / "settings.json"


def scratch_dir() -> Path:
    path = data_dir() / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def av_runtime_dir() -> Path:
    """Where the downloaded PyAV (FFmpeg) component lives."""
    return data_dir() / "pyav"
