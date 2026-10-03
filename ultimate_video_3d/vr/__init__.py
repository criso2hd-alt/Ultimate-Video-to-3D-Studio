"""Virtual reality: watching converted video in an OpenXR headset (Windows first).

Everything here is optional. The app imports this package lazily and only when VR is used, and ``available()``
says whether it can work on this machine, so a missing runtime or a missing ``pyopenxr`` is never an error
anywhere else.
"""

from __future__ import annotations

import sys


NO_HEADSET = ("No VR headset was found. Please connect your headset (on a Quest: start Link or Air Link and "
              "wake the headset), then try again.")


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


def unavailable_reason() -> str:
    """Why a VR session cannot be attempted here, in a sentence a person can act on; "" if it can."""
    if sys.platform != "win32":
        return "VR is only available on Windows for now."
    if not runtime_path():
        return "No OpenXR runtime is installed. Install Meta Quest Link, SteamVR or Virtual Desktop and make it the active OpenXR runtime."
    for module in ("xr", "glfw", "OpenGL"):
        try:
            __import__(module)
        except Exception as error:  # noqa: BLE001 - any import trouble means "not here"
            return f"The VR component '{module}' could not be loaded ({type(error).__name__}: {error})."
    return ""


def headset_present() -> bool:
    """Whether the active OpenXR runtime has a headset behind it right now (for Meta: Link is up and the headset is awake).

    Cheap enough to ask every few seconds: an OpenXR instance, one system query, and the instance is destroyed again. Without a
    headset the runtime answers "form factor unavailable", which is just False here. Call it from a worker thread, not the window.
    """
    if not available():
        return False
    try:
        import xr

        instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(application_name="Ultimate Video to 3D Studio", application_version=1,
                                                engine_name="none", engine_version=0, api_version=xr.XR_CURRENT_API_VERSION)))
        try:
            xr.get_system(instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
            return True
        finally:
            xr.destroy_instance(instance)
    except Exception:  # noqa: BLE001 - no runtime, no headset, a runtime that is still starting: all "not now"
        return False


def available() -> bool:
    """True if a VR session could be attempted: Windows, the OpenXR Python bindings, and an installed runtime."""
    return not unavailable_reason()
