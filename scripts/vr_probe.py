"""Ask the active OpenXR runtime what it offers: a read-only first look before any VR code is written.

    python scripts/vr_probe.py

Prints the runtime, the headset, the graphics APIs it can draw with (OpenGL / D3D11 / D3D12 / Vulkan) and the
per-eye picture size it recommends. It opens no session and draws nothing. Needs ``pip install pyopenxr``.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import xr
    except ImportError:
        print("pyopenxr is not installed: pip install pyopenxr")
        return 1

    print(f"pyopenxr {xr.__version__ if hasattr(xr, '__version__') else '?'}")
    try:
        extensions = sorted(e.extension_name.decode() if isinstance(e.extension_name, bytes) else e.extension_name
                            for e in xr.enumerate_instance_extension_properties())
    except Exception as error:  # noqa: BLE001 - this is a diagnostic
        print("Could not reach an OpenXR runtime:", error)
        return 2
    print(f"\n{len(extensions)} extensions offered. The ones that matter for drawing:")
    graphics = [e for e in extensions if any(k in e.lower() for k in ("opengl", "d3d11", "d3d12", "vulkan"))]
    for name in graphics:
        print("  ", name)
    interesting = [e for e in extensions if any(k in e for k in (
        "hand_tracking", "composition_layer", "passthrough", "depth", "eye_gaze", "performance", "display_refresh"))]
    print("Also available (a few of interest):")
    for name in interesting[:14]:
        print("  ", name)

    wanted = [e for e in ("XR_KHR_opengl_enable", "XR_KHR_D3D11_enable") if e in extensions]
    instance = xr.create_instance(xr.InstanceCreateInfo(
        application_info=xr.ApplicationInfo(application_name="UV3D probe", application_version=1,
                                            engine_name="none", engine_version=0, api_version=xr.XR_CURRENT_API_VERSION),
        enabled_extension_names=wanted))
    props = xr.get_instance_properties(instance)
    print(f"\nRuntime: {props.runtime_name.decode() if isinstance(props.runtime_name, bytes) else props.runtime_name}"
          f" {props.runtime_version}")
    try:
        system = xr.get_system(instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
    except Exception as error:  # noqa: BLE001
        print("No headset reported by the runtime:", error)
        print("Is the headset on and linked (Link / Air Link / Virtual Desktop / SteamVR)?")
        return 3
    sp = xr.get_system_properties(instance, system)
    name = sp.system_name.decode() if isinstance(sp.system_name, bytes) else sp.system_name
    print(f"Headset system: {name}")
    print(f"  max swapchain per eye: {sp.graphics_properties.max_swapchain_image_width} x "
          f"{sp.graphics_properties.max_swapchain_image_height}, layers: {sp.graphics_properties.max_layer_count}")
    print(f"  tracking: orientation={bool(sp.tracking_properties.orientation_tracking)} "
          f"position={bool(sp.tracking_properties.position_tracking)}")
    views = xr.enumerate_view_configuration_views(instance, system, xr.ViewConfigurationType.PRIMARY_STEREO)
    for i, v in enumerate(views):
        print(f"  eye {i}: recommended {v.recommended_image_rect_width} x {v.recommended_image_rect_height}"
              f" (max {v.max_image_rect_width} x {v.max_image_rect_height}), samples {v.recommended_swapchain_sample_count}")

    # What does it need from us to draw with each graphics API?
    if "XR_KHR_opengl_enable" in wanted:
        try:
            fn = xr.get_instance_proc_addr(instance, "xrGetOpenGLGraphicsRequirementsKHR")
            req = xr.GraphicsRequirementsOpenGLKHR()
            result = fn(instance, system, req)
            print(f"OpenGL: supported, needs {req.min_api_version_supported} .. {req.max_api_version_supported}")
        except Exception as error:  # noqa: BLE001
            print("OpenGL requirements query failed:", error)
    else:
        print("OpenGL: NOT offered by this runtime")
    print("D3D11:", "offered" if "XR_KHR_D3D11_enable" in wanted else "NOT offered by this runtime")
    xr.destroy_instance(instance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
