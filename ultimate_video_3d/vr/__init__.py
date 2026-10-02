"""Virtual reality: watching converted video in an OpenXR headset (Windows first).

Everything here is optional. The app imports this package lazily and only when VR is used, and ``available()``
says whether it can work on this machine, so a missing runtime or a missing ``pyopenxr`` is never an error
anywhere else.
"""

from __future__ import annotations

import sys


def runtime_path() -> str:
    """The active OpenXR runtime's manifest (Windows registry), or "" if none is installed."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenXR\1") as key:
            value, _ = winreg.QueryValueEx(key, "ActiveRuntime")
            return str(value)
    except OSError:
        return ""


def available() -> bool:
    """True if a VR session could be attempted: Windows, the OpenXR Python bindings, and an installed runtime."""
    if sys.platform != "win32" or not runtime_path():
        return False
    try:
        import glfw  # noqa: F401
        import OpenGL  # noqa: F401
        import xr  # noqa: F401
    except Exception:  # noqa: BLE001 - any import trouble means "not here"
        return False
    return True
