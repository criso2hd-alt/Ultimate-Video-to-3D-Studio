"""VR spike: open an OpenGL session on the connected headset and draw one colour per eye.

    python scripts/vr_spike.py [seconds]      (default 20)

Not part of the app. It answers three questions before any VR feature is built: does the runtime accept an
OpenGL session from Python, how do the session states move as the headset is worn and taken off, and does the
head pose arrive. Put the headset on during the run to see the states reach VISIBLE and FOCUSED.
Left eye is red, right eye is blue, so a swapped or mono picture is obvious at a glance.
"""

from __future__ import annotations

import math
import sys
import time


def main(seconds: float = 20.0) -> int:
    import xr
    from OpenGL import GL
    from xr.utils.gl import ContextObject
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider

    info = xr.InstanceCreateInfo(
        application_info=xr.ApplicationInfo(application_name="UV3D spike", application_version=1,
                                            engine_name="none", engine_version=0,
                                            api_version=xr.XR_CURRENT_API_VERSION),
        enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
    )
    started = time.monotonic()
    last_state = None
    frames = rendered = 0
    last_report = started
    asked_exit = False
    pose = None
    import os
    import threading

    with ContextObject(context_provider=GLFWOffscreenContextProvider(), instance_create_info=info) as ctx:
        print("session created; waiting for the headset (put it on, with Quest Link active)...", flush=True)

        def watch() -> None:
            """The frame loop only runs once the session is READY, so something else has to report while it waits."""
            seen = None
            while True:
                time.sleep(0.5)
                state = ctx.session_state
                if state != seen:
                    print(f"[{time.monotonic() - started:6.2f}s] session state -> {state.name}", flush=True)
                    seen = state
                if not ctx.session_is_running and time.monotonic() - started > seconds + 20:
                    print("the session never became READY: the runtime is not offering the headset to apps "
                          "(headset off or asleep, or Link not active).", flush=True)
                    os._exit(3)

        threading.Thread(target=watch, daemon=True).start()
        for frame_state in ctx.frame_loop():
            now = time.monotonic()
            last_state = ctx.session_state
            frames += 1
            for view in ctx.view_loop(frame_state):
                eye = ctx.swapchains[0].width and 0
                pulse = 0.35 + 0.25 * math.sin(now * 2.0)
                # view_loop yields left then right: remember which this is by counting within the frame
                rendered += 1
                colour = (pulse + 0.4, 0.05, 0.05) if rendered % 2 == 1 else (0.05, 0.05, pulse + 0.4)
                GL.glClearColor(*colour, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                pose = view.pose.position
            if now - last_report >= 2.0:
                print(f"[{now - started:6.2f}s] {ctx.session_state.name:12s} frames={frames} eyes drawn={rendered}"
                      + (f"  head=({pose.x:+.2f}, {pose.y:+.2f}, {pose.z:+.2f}) m" if pose else ""), flush=True)
                last_report = now
            if now - started > seconds and not asked_exit:
                print("asking the runtime to end the session...", flush=True)
                xr.request_exit_session(ctx.session)
                asked_exit = True
    print(f"done: {frames} frames, {rendered} eye images drawn, final state {last_state.name if last_state else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 20.0))
