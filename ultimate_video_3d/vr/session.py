"""The headset session: an OpenXR frame loop on its own thread, drawing the scene and reading the controllers.

Why a thread of its own: the loop must run at the headset's rate (72-120 Hz) and block in ``xrWaitFrame``; the
window's thread cannot do that, and the window must not be able to make the headset stutter. The session owns
the OpenGL context (a hidden GLFW window), the OpenXR instance and session, and the Renderer. It talks to the
rest of the app through :class:`~ultimate_video_3d.vr.media.VRMedia` (pictures in) and Qt signals (events out).

States, as the runtime reports them: IDLE -> READY -> SYNCHRONIZED -> VISIBLE -> FOCUSED. The headset counts as
*worn* from VISIBLE on; that is what starts a waiting preview.

Controls (any OpenXR controller, bound for Touch, Index, Vive, Windows MR and the generic "simple" profile):
  trigger on the screen / A or X ...... play and pause          B or Y ....... bring the screen back in front of me
  right stick up/down ................. screen size              left stick up/down ... screen distance
  left stick left/right ............... back / forward 10 s      grip ......... hold to drag the screen around
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from . import geometry as geo
from .media import VRMedia
from .renderer import ROOM, Renderer

APP_NAME = "Ultimate Video to 3D Studio"
SEEK_MS = 10_000
BEAM_IDLE, BEAM_HIT = (0.55, 0.6, 0.7, 0.8), (0.3, 0.85, 1.0, 1.0)


@dataclass
class Hand:
    """What one controller is doing this frame."""

    valid: bool = False
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)
    select: bool = False
    select_pressed: bool = False          # went down this frame
    squeeze: bool = False
    primary_pressed: bool = False         # A / X
    secondary_pressed: bool = False       # B / Y
    stick: tuple = (0.0, 0.0)

    def forward(self) -> np.ndarray:
        return geo.quat_to_matrix(self.orientation) @ np.array([0.0, 0.0, -1.0])


class _Inputs:
    """The OpenXR actions for the controllers, and reading them each frame."""

    PROFILES = {
        "/interaction_profiles/oculus/touch_controller": dict(
            select="trigger/value", squeeze="squeeze/value", stick="thumbstick", aim="aim/pose",
            primary={"left": "x/click", "right": "a/click"}, secondary={"left": "y/click", "right": "b/click"}),
        "/interaction_profiles/valve/index_controller": dict(
            select="trigger/click", squeeze="squeeze/value", stick="thumbstick", aim="aim/pose",
            primary={"left": "a/click", "right": "a/click"}, secondary={"left": "b/click", "right": "b/click"}),
        "/interaction_profiles/htc/vive_controller": dict(
            select="trigger/click", squeeze="squeeze/click", stick="trackpad", aim="aim/pose",
            primary={"left": "menu/click", "right": "menu/click"}, secondary={}),
        "/interaction_profiles/microsoft/motion_controller": dict(
            select="trigger/value", squeeze="squeeze/click", stick="thumbstick", aim="aim/pose",
            primary={"left": "menu/click", "right": "menu/click"}, secondary={}),
        "/interaction_profiles/khr/simple_controller": dict(
            select="select/click", squeeze=None, stick=None, aim="aim/pose",
            primary={"left": "menu/click", "right": "menu/click"}, secondary={}),
    }

    def __init__(self, ctx) -> None:
        import xr

        self.xr, self.ctx = xr, ctx
        instance = ctx.instance
        self.paths = {"left": xr.string_to_path(instance, "/user/hand/left"),
                      "right": xr.string_to_path(instance, "/user/hand/right")}
        subs = [self.paths["left"], self.paths["right"]]
        self.set = xr.create_action_set(instance, xr.ActionSetCreateInfo(
            action_set_name="uv3d", localized_action_set_name="Ultimate Video to 3D Studio", priority=0))

        def make(name: str, kind):
            return xr.create_action(self.set, xr.ActionCreateInfo(
                action_type=kind, action_name=name, localized_action_name=name,
                count_subaction_paths=len(subs), subaction_paths=subs))

        self.a_select = make("select", xr.ActionType.BOOLEAN_INPUT)
        self.a_squeeze = make("squeeze", xr.ActionType.BOOLEAN_INPUT)
        self.a_primary = make("primary", xr.ActionType.BOOLEAN_INPUT)
        self.a_secondary = make("secondary", xr.ActionType.BOOLEAN_INPUT)
        self.a_stick = make("stick", xr.ActionType.VECTOR2F_INPUT)
        self.a_aim = make("aim", xr.ActionType.POSE_INPUT)
        for profile, spec in self.PROFILES.items():
            try:
                self._suggest(profile, spec)
            except Exception:  # noqa: BLE001 - a profile this runtime does not know is simply skipped
                pass
        xr.attach_session_action_sets(ctx.session, xr.SessionActionSetsAttachInfo(
            count_action_sets=1, action_sets=(xr.ActionSet * 1)(self.set)))
        self.spaces = {hand: xr.create_action_space(ctx.session, xr.ActionSpaceCreateInfo(
            action=self.a_aim, subaction_path=path)) for hand, path in self.paths.items()}
        self._last_select = {"left": False, "right": False}

    def _suggest(self, profile: str, spec: dict) -> None:
        xr, instance = self.xr, self.ctx.instance

        def binding(action, hand: str, component: str | None):
            if not component:
                return []
            path = xr.string_to_path(instance, f"/user/hand/{hand}/input/{component}")
            return [xr.ActionSuggestedBinding(action, path)]

        found = []
        for hand in ("left", "right"):
            found += binding(self.a_select, hand, spec["select"])
            found += binding(self.a_squeeze, hand, spec["squeeze"])
            found += binding(self.a_stick, hand, spec["stick"])
            found += binding(self.a_aim, hand, spec["aim"])
            found += binding(self.a_primary, hand, spec["primary"].get(hand))
            found += binding(self.a_secondary, hand, spec["secondary"].get(hand))
        xr.suggest_interaction_profile_bindings(instance, xr.InteractionProfileSuggestedBinding(
            interaction_profile=xr.string_to_path(instance, profile),
            count_suggested_bindings=len(found), suggested_bindings=(xr.ActionSuggestedBinding * len(found))(*found)))

    def read(self, display_time) -> dict[str, Hand]:
        xr, ctx = self.xr, self.ctx
        active = xr.ActiveActionSet(action_set=self.set, subaction_path=xr.NULL_PATH)
        xr.sync_actions(ctx.session, xr.ActionsSyncInfo(count_active_action_sets=1,
                                                         active_action_sets=ctypes.pointer(active)))
        hands = {}
        for name, path in self.paths.items():
            hand = Hand()

            def boolean(action):
                state = xr.get_action_state_boolean(ctx.session, xr.ActionStateGetInfo(action=action, subaction_path=path))
                return bool(state.is_active and state.current_state), bool(state.changed_since_last_sync)

            now, _ = boolean(self.a_select)
            hand.select, hand.select_pressed = now, now and not self._last_select[name]
            self._last_select[name] = now
            hand.squeeze, _ = boolean(self.a_squeeze)
            prim, changed = boolean(self.a_primary)
            hand.primary_pressed = prim and changed
            sec, changed = boolean(self.a_secondary)
            hand.secondary_pressed = sec and changed
            stick = xr.get_action_state_vector2f(ctx.session, xr.ActionStateGetInfo(action=self.a_stick, subaction_path=path))
            if stick.is_active:
                hand.stick = (float(stick.current_state.x), float(stick.current_state.y))
            try:
                location = xr.locate_space(self.spaces[name], ctx.space, display_time)
                flags = location.location_flags
                if (flags & xr.SPACE_LOCATION_POSITION_VALID_BIT) and (flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT):
                    p, o = location.pose.position, location.pose.orientation
                    hand.valid = True
                    hand.position = np.array([p.x, p.y, p.z])
                    hand.orientation = (o.x, o.y, o.z, o.w)
            except Exception:  # noqa: BLE001 - controller not tracked this frame
                pass
            hands[name] = hand
        return hands


class VRSession(QObject):
    """Owns the headset thread. Create it, call :meth:`start`, and watch the signals."""

    state_changed = Signal(str)             # the OpenXR session state's name, e.g. "FOCUSED"
    worn_changed = Signal(bool)             # True once the headset is on (VISIBLE or FOCUSED)
    started = Signal()                      # the loop is running and drawing
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, media: VRMedia, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.media = media
        self.screen = geo.Screen()
        self.scene = ROOM
        self.fps = 0.0
        self.state = "IDLE"
        self.worn = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._recentre = threading.Event()
        self._toggle = threading.Event()
        self._placed = False
        self._head = (np.array([0.0, 1.6, 0.0]), (0.0, 0.0, 0.0, 1.0))

    # -- control from the UI thread ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._main, name="uv3d-vr", daemon=True)
        self._thread.start()

    def stop(self, wait: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and wait:
            thread.join(wait)

    def recentre(self) -> None:
        self._recentre.set()

    def set_scene(self, scene: str) -> None:
        self.scene = scene

    # -- the thread -------------------------------------------------------------------------------

    def _main(self) -> None:
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - reported, never allowed to take the app down
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.worn = False
            self.stopped.emit()

    def _run(self) -> None:
        import xr
        from OpenGL import GL
        from xr.utils.gl import ContextObject
        from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider

        info = xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(application_name=APP_NAME, application_version=1, engine_name="none",
                                                engine_version=0, api_version=xr.XR_CURRENT_API_VERSION),
            enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME])
        space = xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.STAGE)
        with ContextObject(context_provider=GLFWOffscreenContextProvider(), instance_create_info=info,
                           reference_space_create_info=space) as ctx:
            inputs = _Inputs(ctx)
            renderer = Renderer()
            renderer.init_gl()
            last_state = None
            last_time = time.monotonic()
            frames, window = 0, time.monotonic()
            exiting = False
            while True:
                ctx.poll_xr_events()
                if ctx.exit_render_loop:
                    break
                state = ctx.session_state.name
                if state != last_state:
                    last_state = self.state = state
                    self.state_changed.emit(state)
                    worn = state in ("VISIBLE", "FOCUSED")
                    if worn != self.worn:
                        self.worn = worn
                        self.worn_changed.emit(worn)
                    if state in ("READY", "SYNCHRONIZED"):
                        self.started.emit()
                if self._stop.is_set():
                    if ctx.session_is_running and not exiting:
                        xr.request_exit_session(ctx.session)
                        exiting = True
                    elif not ctx.session_is_running:
                        break
                if not ctx.session_is_running:
                    time.sleep(0.1)
                    continue
                if ctx.session_state not in (xr.SessionState.READY, xr.SessionState.SYNCHRONIZED,
                                             xr.SessionState.VISIBLE, xr.SessionState.FOCUSED):
                    time.sleep(0.02)
                    continue
                frame_state = xr.wait_frame(ctx.session)
                xr.begin_frame(ctx.session)
                ctx.render_layers = []
                ctx.graphics.make_current()
                now = time.monotonic()
                self._frame(ctx, inputs, renderer, frame_state, now - last_time)
                last_time = now
                xr.end_frame(ctx.session, frame_end_info=xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=ctx.environment_blend_mode, layers=ctx.render_layers))
                GL.glGetError()
                frames += 1
                if now - window >= 1.0:
                    self.fps, frames, window = frames / (now - window), 0, now

    # -- one frame --------------------------------------------------------------------------------

    def _frame(self, ctx, inputs: _Inputs, renderer: Renderer, frame_state, dt: float) -> None:
        media, screen = self.media, self.screen
        screen.aspect = media.aspect
        picture = media.take_frame()
        if picture is not None:
            renderer.upload(picture)
        elif not media.is_open and renderer.has_picture:
            renderer.clear_picture()

        hands = inputs.read(frame_state.predicted_display_time) if frame_state.should_render else {}
        beams = self._controls(hands, dt)

        eye = 0
        positions = []
        orientation = None
        for view in ctx.view_loop(frame_state):
            p, o, fov = view.pose.position, view.pose.orientation, view.fov
            position, quat = (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)
            positions.append(position)
            orientation = orientation or quat
            renderer.draw_eye(geo.view_matrix(position, quat),
                              geo.projection_matrix(fov.angle_left, fov.angle_right, fov.angle_up, fov.angle_down),
                              eye, position, screen, media.layout, self.scene, beams)
            eye += 1
        if positions:
            self._head = (np.mean(np.array(positions), axis=0), orientation)
            if not self._placed or self._recentre.is_set():
                self._recentre.clear()
                screen.recentre(*self._head)
                self._placed = True

    def _controls(self, hands: dict, dt: float) -> tuple:
        """Act on the controllers; returns the pointer beams to draw."""
        media, screen = self.media, self.screen
        beams = []
        right, left = hands.get("right"), hands.get("left")
        dt = min(dt, 0.1)
        for name, hand in hands.items():
            if not hand.valid:
                continue
            direction = hand.forward()
            hit = geo.ray_hits_screen(screen, hand.position, direction)
            length = hit[0] if hit else 2.5
            beams.append((hand.position, direction, length, BEAM_HIT if hit else BEAM_IDLE))
            if hand.select_pressed and hit:
                self._toggle.set()
            self._drag(name, hand)
            if hand.secondary_pressed:
                self._recentre.set()
        if any(h.primary_pressed for h in hands.values()):
            self._toggle.set()
        if self._toggle.is_set():
            self._toggle.clear()
            media.toggle()                                       # QObject methods are safe to call: Qt queues the work
        if right and abs(right.stick[1]) > 0.15:
            screen.resize(1.0 + right.stick[1] * 0.9 * dt)
        if left:
            if abs(left.stick[1]) > 0.15:
                screen.push(-left.stick[1] * 1.6 * dt)
            self._seek_with(left.stick[0])
        return tuple(beams)

    def _seek_with(self, x: float) -> None:
        """Left stick sideways: step 10 s back or forward, once per push."""
        if abs(x) < 0.4:
            self._seek_armed = True
        elif abs(x) > 0.8 and getattr(self, "_seek_armed", True):
            self._seek_armed = False
            self.media.seek_relative(SEEK_MS if x > 0 else -SEEK_MS)

    def _drag(self, name: str, hand: Hand) -> None:
        """While a grip is held, the screen follows where the controller points."""
        screen = self.screen
        forward = hand.forward()
        yaw, pitch = geo.yaw_of(hand.orientation), math.asin(max(-1.0, min(1.0, forward[1])))
        holding = getattr(self, "_holding", None)
        if hand.squeeze:
            if holding is None or holding[0] != name:
                self._holding = (name, yaw, pitch, screen.heading, screen.lift)
            else:
                _, y0, p0, h0, l0 = holding
                screen.heading = h0 + (yaw - y0)
                screen.lift = float(np.clip(l0 + (pitch - p0) * screen.distance, -1.5, 2.5))
        elif holding is not None and holding[0] == name:
            self._holding = None
