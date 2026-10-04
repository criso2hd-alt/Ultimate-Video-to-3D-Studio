"""The headset session: an OpenXR frame loop on its own thread, drawing the scene and reading the controllers.

Why a thread of its own: the loop must run at the headset's rate (72-120 Hz) and block in ``xrWaitFrame``; the
window's thread cannot do that, and the window must not be able to make the headset stutter. The session owns
the OpenGL context (a hidden GLFW window), the OpenXR instance and session, and the Renderer. It talks to the
rest of the app through :class:`~ultimate_video_3d.vr.media.VRMedia` (pictures in) and Qt signals (events out).

States, as the runtime reports them: IDLE -> READY -> SYNCHRONIZED -> VISIBLE -> FOCUSED. The headset counts as
*worn* from VISIBLE on; that is what starts a waiting preview.

Controls (any OpenXR controller, bound for Touch, Index, Vive, Windows MR and the generic "simple" profile):
  trigger on the screen / A or X ...... play and pause          left menu button / B / Y ... open or close the menu
  left stick left/right ............... back / forward 10 s      right stick, any way ....... go to the seat that way
  right stick PRESSED DOWN ............ reset the view: face the screen from the seat / screen back in front of me
In the plain room (no baked room): right stick up/down = screen size, left stick up/down = distance, grip = drag.
The menu is a floating panel (see menu.py): point with a controller, pull the trigger. It also shows, by itself, the
first time the headset goes on, so nobody has to guess what the buttons do.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from .. import bars
from . import geometry as geo
from . import room as roommod
from .media import VRMedia
from .menu import ASPECT as MENU_ASPECT
from .menu import Menu
from .renderer import BAKED, ROOM, Renderer
from .state import LightFader, SeatCycle, pick_seat

APP_NAME = "Ultimate Video to 3D Studio"
SEEK_MS = 10_000
BEAM_IDLE, BEAM_HIT = (0.55, 0.6, 0.7, 0.8), (0.3, 0.85, 1.0, 1.0)
MENU_WIDTH = 1.0                           # metres, the menu panel in the room
MENU_DISTANCE = 1.2                        # how far in front of the eyes it hangs (closer than the chair in front)
MENU_DROP = 0.2                            # how far below the eyes
SHOT_DELAY = 3.0                           # seconds between pressing Screenshot and the picture (time to look at the screen)


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
    menu_pressed: bool = False            # the left menu button
    reset_pressed: bool = False           # the right thumbstick pressed down: reset the view
    stick: tuple = (0.0, 0.0)

    def forward(self) -> np.ndarray:
        return geo.quat_to_matrix(self.orientation) @ np.array([0.0, 0.0, -1.0])


class _Inputs:
    """The OpenXR actions for the controllers, and reading them each frame."""

    PROFILES = {
        "/interaction_profiles/oculus/touch_controller": dict(
            select="trigger/value", squeeze="squeeze/value", stick="thumbstick", aim="aim/pose",
            primary={"left": "x/click", "right": "a/click"}, secondary={"left": "y/click", "right": "b/click"},
            menu={"left": "menu/click"}, reset={"right": "thumbstick/click"}),
        "/interaction_profiles/valve/index_controller": dict(
            select="trigger/click", squeeze="squeeze/value", stick="thumbstick", aim="aim/pose",
            primary={"left": "a/click", "right": "a/click"}, secondary={"left": "b/click", "right": "b/click"},
            reset={"right": "thumbstick/click"}),
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
        self.a_menu = make("menu", xr.ActionType.BOOLEAN_INPUT)
        self.a_reset = make("reset", xr.ActionType.BOOLEAN_INPUT)
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
            found += binding(self.a_menu, hand, spec.get("menu", {}).get(hand))
            found += binding(self.a_reset, hand, spec.get("reset", {}).get(hand))
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
            menu, changed = boolean(self.a_menu)
            hand.menu_pressed = menu and changed
            reset, changed = boolean(self.a_reset)
            hand.reset_pressed = reset and changed
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


def _prefer_srgb_swapchain() -> None:
    """Make pyopenxr pick an sRGB swapchain format when the runtime offers one.

    By default it prefers plain 8-bit UNORM, which the Meta runtime treats as linear light: display-encoded colours
    come out brightened and banded. An sRGB format is decoded by the compositor, so what we write is what is seen.
    """
    from OpenGL import GL
    from xr.utils.gl import OpenGLGraphics

    original = OpenGLGraphics.select_color_swapchain_format

    def select(runtime_formats):
        for wanted in (GL.GL_SRGB8_ALPHA8, GL.GL_SRGB8):
            if wanted in runtime_formats:
                return wanted
        return original(runtime_formats)

    OpenGLGraphics.select_color_swapchain_format = staticmethod(select)


from . import NO_HEADSET, XR_LOCK  # noqa: E402  (shared with the controller, which must not import this OpenGL module)
from .log import log as vrlog, log_error  # noqa: E402


#: Errors a start-up can answer with that are worth another go a moment later (the runtime still finishing with the last session).
_TRANSIENT = ("limitreached", "graphicsrequirementscallmissing")


def release_context(context, call_exit: bool = True) -> None:
    """Free everything a (possibly half-finished) pyopenxr ContextObject made.

    pyopenxr's own ``__exit__`` starts by touching ``default_action_set`` and ``space``, which do not exist if the start-up
    stopped before creating them; it dies on the first line and the instance (and session) stay alive. The Meta runtime then
    answers every later start with LimitReachedError. So its exit is tried, and then each piece is freed here on its own,
    ignoring the ones that were never made.
    """
    if call_exit:
        try:
            context.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - expected for a half-made context: the pieces are freed one by one below
            pass
    try:
        import xr
    except Exception:  # noqa: BLE001
        return
    for name, destroy in (("default_action_set", "destroy_action_set"), ("space", "destroy_space"), ("session", "destroy_session")):
        handle = getattr(context, name, None)
        if handle is not None:
            try:
                getattr(xr, destroy)(handle)
            except Exception:  # noqa: BLE001
                pass
            setattr(context, name, None)
    graphics = getattr(context, "graphics", None)
    if graphics is not None:
        try:
            graphics.destroy()
        except Exception:  # noqa: BLE001
            pass
        context.graphics = None
    instance = getattr(context, "instance", None)
    if instance is not None:
        try:
            xr.destroy_instance(instance)
        except Exception:  # noqa: BLE001
            pass
        context.instance = None


class GuardedContext:
    """pyopenxr's ContextObject, started under ``XR_LOCK``, cleaned up if the start-up fails half way, and tried again when the
    runtime says it is still busy.

    Without the lock a headset probe can create and destroy another OpenXR instance in the middle of this one's start-up;
    without the clean-up a start-up that stops half way leaves its instance behind, and the runtime then refuses every later
    try (LimitReachedError) until the app is restarted (see :func:`release_context`). The lock is let go as soon as the session
    exists. Give a *factory* (not an object) to allow ``attempts`` tries: every try needs a brand-new context.
    """

    def __init__(self, context, attempts: int = 1, wait: float = 2.5, stop=None) -> None:
        self._factory = (lambda: context) if hasattr(type(context), "__enter__") else context      # an object, or a factory for new ones
        self._context = None
        self._attempts, self._wait, self._stop = max(1, attempts), wait, stop

    def __enter__(self):
        for attempt in range(1, self._attempts + 1):
            context = self._context = self._factory()
            with XR_LOCK:
                try:
                    return context.__enter__()
                except BaseException as error:
                    release_context(context)                           # free whatever the failed start-up had made
                    text = f"{type(error).__name__} {error}".lower().replace("_", "")
                    retry = (attempt < self._attempts and any(t in text for t in _TRANSIENT)
                             and not (self._stop is not None and self._stop.is_set()))
                    if not retry:
                        raise
                    vrlog(f"start-up try {attempt} failed ({type(error).__name__}); trying again in {self._wait:.1f} s")
            time.sleep(self._wait)                                    # outside the lock: the probe and others may go meanwhile

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._context.__exit__(exc_type, exc_value, traceback)
        finally:
            release_context(self._context, call_exit=False)         # whatever its own exit missed


def friendly_error(error: BaseException) -> str:
    """A person-sized message for a session that could not start. A runtime with no headset behind it answers
    XR_ERROR_FORM_FACTOR_UNAVAILABLE when the system is requested: that is "plug it in", not a bug."""
    text = f"{type(error).__name__} {error}".lower().replace("_", "")
    if "formfactorunavailable" in text or "runtimeunavailable" in text:
        return NO_HEADSET
    if "limitreached" in text or "graphicsrequirementscallmissing" in text or "runtimefailure" in text:
        return (f"The VR runtime would not start a session ({type(error).__name__}). Close any other VR app, check that "
                "Meta Quest Link (or SteamVR) shows your headset as connected, then press Start VR again. If it keeps "
                "happening, close this app and open it again.")
    return f"{type(error).__name__}: {error}"


class VRSession(QObject):
    """Owns the headset thread. Create it, call :meth:`start`, and watch the signals."""

    state_changed = Signal(str)             # the OpenXR session state's name, e.g. "FOCUSED"
    worn_changed = Signal(bool)             # True once the headset is on (VISIBLE or FOCUSED)
    started = Signal()                      # the loop is running and drawing
    connected = Signal()                    # the runtime accepted us: an instance, a session and a graphics context exist
    failed = Signal(str)
    stopped = Signal()
    menu_action = Signal(object)            # something the menu asked for that the window must do: video, volume, exit

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
        # a baked room, once one is chosen
        self.room: roommod.Room | None = None
        self.seat: str | None = None                      # the seat the person is in, by name
        self._seats = SeatCycle([])
        self._pending_seat: str | None = None             # a seat to move to as soon as the head is tracked
        self._room_dirty = False
        self._rig = np.eye(4)                             # headset space -> room (see geometry.rig_to_seat)
        self._lights = LightFader()
        self._seat_armed = True
        self.menu = Menu()
        self.beam_on = True                               # the projector's light beam (its lens always glows); the menu and Settings change it
        self.beam_level = 1.0
        self._panel: np.ndarray | None = None             # where the menu hangs in the room, while it is open
        self._menu_uploaded = -1
        self._replace_panel = False                       # the person moved: hang the menu in front of them again
        self._welcomed = False                            # the first-time card has been shown
        self._shot_timer = 0.0                            # counting down to a screenshot; 0 = none asked for
        self._shot_now = False                            # this frame is the one to capture
        self._align_next = True                           # the next sit turns the person to face the seat (first time, and Reset)

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

    def set_room(self, room: roommod.Room | None) -> None:
        """Use a baked room (its own fixed screen, seats and lighting), or None for the plain stand-in room."""
        self.room = room
        self._align_next = True                           # entering a room: face the way the first seat faces
        if room is None:
            self.screen, self.scene, self.seat = geo.Screen(), ROOM, None
            self._seats = SeatCycle([])
            self._rig = np.eye(4)
            self._placed = False
        else:
            self.screen = geo.Screen.from_quad(room.screen.matrix, room.screen.width, room.screen.height)
            self.scene = BAKED
            self._seats = SeatCycle([s.name for s in room.seats])
            first = room.first_seat()
            self._pending_seat = first.name if first is not None else None
        self._room_dirty = True
        self.menu.set_beam(has=room is not None and room.projector_lens is not None)

    def set_beam(self, on: bool, level: float) -> None:
        """Show or hide the projector's beam and set how bright it is (any thread; takes effect on the next frame)."""
        self.beam_on, self.beam_level = bool(on), float(level)
        self.menu.set_beam(on=self.beam_on, level=self.beam_level)

    def seat_names(self) -> list[tuple[str, str]]:
        """(name, label) of every seat in the current room."""
        return [(s.name, s.label) for s in self.room.seats] if self.room else []

    def set_seat(self, name: str) -> None:
        """Move the person to a seat (takes effect on the next frame the head is tracked)."""
        if self.room is not None and self._seats.select(name):
            self._pending_seat = name

    def step_seat(self, direction: int) -> None:
        name = self._seats.step(direction)
        if name is not None:
            self._pending_seat = name

    # -- the thread -------------------------------------------------------------------------------

    def _main(self) -> None:
        vrlog("session thread started")
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - reported, never allowed to take the app down
            log_error("session failed", error)
            self.failed.emit(friendly_error(error))
        finally:
            self.worn = False
            vrlog("session thread ended")
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
        _prefer_srgb_swapchain()
        vrlog("connecting to the OpenXR runtime")
        # A new SessionCreateInfo for every try: pyopenxr's default argument is ONE shared object that the first session fills in
        # with its graphics binding. Every later session in the same process then skips creating its graphics, never asks the
        # runtime for the graphics requirements (GraphicsRequirementsCallMissingError) and leaks its instance (LimitReachedError).
        def make_context():
            return ContextObject(context_provider=GLFWOffscreenContextProvider(), instance_create_info=info,
                                 session_create_info=xr.SessionCreateInfo(), reference_space_create_info=space)

        with GuardedContext(make_context, attempts=4, stop=self._stop) as ctx:
            vrlog("connected: instance, session and graphics context exist")
            self.connected.emit()
            inputs = _Inputs(ctx)
            renderer = Renderer()
            renderer.init_gl()
            last_state = None
            last_time = time.monotonic()
            frames, window = 0, time.monotonic()
            last_stats = time.monotonic()
            exiting = False
            while True:
                ctx.poll_xr_events()
                if ctx.exit_render_loop:
                    break
                state = ctx.session_state.name
                if state != last_state:
                    vrlog(f"session state {state}")
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
                if now - last_stats >= 30.0:
                    last_stats = now
                    if media.playing:
                        vrlog(f"playing: the headset loop runs at {self.fps:.0f} fps; film {media.stats_text()}")

    # -- one frame --------------------------------------------------------------------------------

    def _frame(self, ctx, inputs: _Inputs, renderer: Renderer, frame_state, dt: float) -> None:
        media, screen = self.media, self.screen
        if self._room_dirty:
            renderer.set_room(self.room)
            self._room_dirty = False
        if screen.fixed is not None:
            screen.picture_aspect = media.aspect if media.is_open else None  # fitted inside the room's screen, never stretched
        else:
            screen.aspect = media.aspect
            screen.floor_clearance = 0.12 if self.scene == ROOM else None    # on the room's floor, or at eye height
        renderer.beam_on, renderer.beam_level = self.beam_on, self.beam_level
        renderer.crop = getattr(media, "crop", bars.NONE)
        if self.room is not None:
            renderer.room_light = self._lights.update(dt, media.playing, media.is_open, self.room.light_level_paused)
        self._sync_menu(renderer, dt)
        shooting = self._shot_due(dt)                         # this frame is the picture: no menu, no rays in it
        picture = media.take_frame()
        if picture is not None:
            renderer.upload(picture, media.layout)
        elif not media.is_open and renderer.has_picture:
            renderer.clear_picture()

        hands = {}
        if frame_state.should_render and ctx.session_state.name == "FOCUSED":
            # Controllers only report while this app has focus; a system overlay or the headset's own menu takes
            # it away, and asking then is an error (SessionNotFocused). Not having input for a while is fine.
            try:
                hands = inputs.read(frame_state.predicted_display_time)
            except Exception:  # noqa: BLE001 - focus was lost between the state check and the call
                hands = {}
        beams = self._controls(hands, dt)

        eye = 0
        positions = []
        orientation = None
        rig_inverse = np.linalg.inv(self._rig)
        for view in ctx.view_loop(frame_state):
            p, o, fov = view.pose.position, view.pose.orientation, view.fov
            position, quat = (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)
            positions.append(position)
            orientation = orientation or quat
            # the world is seen through the rig: the eye's place in the room, and the view into the room
            renderer.draw_eye(geo.view_matrix(position, quat) @ rig_inverse,
                              geo.projection_matrix(fov.angle_left, fov.angle_right, fov.angle_up, fov.angle_down),
                              eye, tuple(geo.through_rig(self._rig, position)), screen, media.layout, self.scene,
                              () if shooting else beams, self._panel if (self.menu.open and not shooting) else None)
            if shooting and eye == 0:
                self._save_eye_view()                         # the left eye, as drawn, straight from the swapchain image
            eye += 1
        if positions:
            self._head = (np.mean(np.array(positions), axis=0), orientation)
            if self.room is not None:
                self._sit(self._head)
            elif not self._placed or self._recentre.is_set():
                self._recentre.clear()
                screen.recentre(*self._head)
                self._placed = True
                self._replace_panel = True
            self._menu_after_move()

    def _sit(self, head: tuple) -> None:
        """Move the person onto the chosen seat (or back onto it, after the Recentre button)."""
        if self._recentre.is_set() and self.seat is not None:
            self._recentre.clear()
            self._pending_seat = self.seat
            self._align_next = True                       # Reset view: face the way the seat faces again
        name, self._pending_seat = self._pending_seat, None
        seat = self.room.seat(name) if name else None
        if seat is None:
            return
        # Moving to another seat only moves them: they keep facing whichever way they were turned in the room. Turning them
        # to the seat's direction every time spun the room round to wherever they happened to be looking.
        self._rig = geo.rig_to_seat(seat.position, seat.yaw, head[0], head[1], None if self._align_next else self._rig)
        self._align_next = False
        self.seat = seat.name
        self._replace_panel = True

    # -- the menu ---------------------------------------------------------------------------------

    def _sync_menu(self, renderer: Renderer, dt: float) -> None:
        """Keep the menu's words in step with the player, and put its picture on the GPU when it changed."""
        media, menu = self.media, self.menu
        seat = self.room.seat(self.seat) if self.room is not None and self.seat else None
        menu.sync(playing=media.playing, title=media.path.stem if media.is_open and media.path else "",
                  has_media=media.is_open, has_lights=self.room is not None, seat_label=seat.label if seat else "",
                  has_seats=self.room is not None and len(self.room.seats) > 1,
                  position_ms=media.position_ms if media.is_open else 0.0, duration_ms=media.duration_ms if media.is_open else 0.0)
        menu.tick(dt)
        if menu.open:
            image = menu.image()
            if menu.version != self._menu_uploaded:
                width, height = image.width(), image.height()
                pixels = np.frombuffer(bytes(image.constBits()), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
                renderer.set_panel_image(pixels[:, :width])
                self._menu_uploaded = menu.version

    def _place_panel(self) -> None:
        """Hang the menu in front of where the person is now looking, in the room's own space."""
        position, orientation = self._head
        forward = geo.quat_to_matrix(tuple(orientation)) @ np.array([0.0, 0.0, -1.0])
        self._panel = geo.panel_matrix(geo.through_rig(self._rig, position),
                                       geo.direction_through_rig(self._rig, forward), MENU_WIDTH, MENU_ASPECT,
                                       MENU_DISTANCE, MENU_DROP)
        self._replace_panel = False

    def _toggle_menu(self) -> None:
        if self.menu.open:
            self.menu.close()
        else:
            self._place_panel()
            self.menu.show("main")

    def _menu_after_move(self) -> None:
        """Once the head is tracked and the person is seated: the first-time card; and follow them to a new seat."""
        ready = self.seat is not None if self.room is not None else self._placed
        if self._replace_panel and self.menu.open and ready:
            self._place_panel()
        self._replace_panel = False
        if not self._welcomed and self.worn and ready:
            self._welcomed = True
            self._place_panel()
            self.menu.show("videos" if self.menu.tiles and not self.media.is_open else "help", auto_close=True)

    def set_audio_label(self, label: str | None) -> None:
        """Where the film's sound goes, shown on the menu's Sound row (None hides the row)."""
        self.menu.set_audio(label)

    def set_rooms(self, tiles: list, current: str | None = None) -> None:
        """The rooms on the menu's Rooms page (any thread; replaced as a whole)."""
        self.menu.set_rooms(tiles, current)

    def set_current_room(self, folder: str) -> None:
        self.menu.set_current_room(folder)

    def set_videos(self, tiles: list) -> None:
        """The Library, as shown on the menu's Videos page (any thread; replaced as a whole)."""
        self.menu.set_tiles(tiles)

    def _apply(self, action: tuple | None) -> None:
        """Do what a menu button asked: the headset's own things here, the rest through ``menu_action``."""
        if not action:
            return
        kind = action[0]
        if kind == "lights":
            self._lights.mode = action[1]
        elif kind == "seat":
            self.step_seat(action[1])
        elif kind == "recentre":
            self._recentre.set()
        elif kind == "play":
            self._toggle.set()
        elif kind == "beam":
            self.beam_on, self.beam_level = bool(action[1]), float(action[2])
            self.menu_action.emit(action)                       # the window saves the choice
        elif kind == "screenshot":
            self.menu.close()                             # out of the picture, then a few seconds to look at the screen
            self._shot_timer = SHOT_DELAY
        elif kind != "close":
            self.menu_action.emit(action)                       # "video", "volume", "exit": the window's business

    def _controls(self, hands: dict, dt: float) -> tuple:
        """Act on the controllers; returns the pointer beams to draw."""
        media, screen, menu = self.media, self.screen, self.menu
        beams = []
        right, left = hands.get("right"), hands.get("left")
        dt = min(dt, 0.1)
        hover, presses = None, []
        for name, hand in hands.items():
            if not hand.valid:
                continue
            origin = geo.through_rig(self._rig, hand.position)           # the controller as it is in the room
            direction = geo.direction_through_rig(self._rig, hand.forward())
            hit = geo.ray_hits_screen(screen, origin, direction)
            length = hit[0] if hit else 2.5
            on_menu = False
            if menu.open and self._panel is not None:
                where = geo.ray_hits_matrix(self._panel, origin, direction)
                if where is not None and (hit is None or where[0] < hit[0]):
                    on_menu, length = True, where[0]                      # the menu is nearer than the screen: it gets the pointer
                    button = menu.hit(where[1], where[2])
                    hover = button or hover
                    if hand.select_pressed and button:
                        presses.append((button, where[1]))
            landed = bool(hit or on_menu)
            beams.append((origin, direction, length, BEAM_HIT if landed else BEAM_IDLE, landed))     # the last: draw a dot at the tip
            if hand.select_pressed and hit and not on_menu:
                self._toggle.set()
            if screen.fixed is None:
                self._drag(name, hand)
        if menu.open:
            menu.set_hover(hover)
        for button, across in presses:
            self._apply(menu.press(button, across))
        if any(h.menu_pressed or h.secondary_pressed for h in hands.values()):
            self._toggle_menu()
        if any(h.reset_pressed for h in hands.values()):
            self._recentre.set()                                 # right thumbstick pressed: sit back / screen back in front
        if any(h.primary_pressed for h in hands.values()):
            self._toggle.set()
        if self._toggle.is_set():
            self._toggle.clear()
            media.toggle()                                       # QObject methods are safe to call: Qt queues the work
        if right and self.room is not None:
            self._aim_seat(right.stick)                              # right stick: go to the seat that way
        if right and self.room is None and abs(right.stick[1]) > 0.15:
            screen.resize(1.0 + right.stick[1] * 0.9 * dt)
        if left:
            if self.room is None and abs(left.stick[1]) > 0.15:
                screen.push(-left.stick[1] * 1.6 * dt)
            self._seek_with(left.stick[0])
        return tuple(beams)

    def _aim_seat(self, stick: tuple) -> None:
        """Right stick: push toward a seat to go there (forward = toward the screen, right = the seat on your right...).
        One move per push; the stick must come back to the middle before the next."""
        magnitude = math.hypot(*stick)
        if magnitude < 0.35:
            self._seat_armed = True
        elif magnitude > 0.75 and self._seat_armed and self.room is not None:
            self._seat_armed = False
            name = pick_seat([(s.name, s.position, s.yaw) for s in self.room.seats], self.seat, (stick[0], stick[1]))
            if name is not None and name != self.seat:
                self._seats.select(name)
                self._pending_seat = name

    # -- screenshots ------------------------------------------------------------------------------

    def _shot_due(self, dt: float) -> bool:
        """True on the one frame a requested screenshot is taken (SHOT_DELAY seconds after the button)."""
        if self._shot_timer <= 0:
            return False
        self._shot_timer -= max(0.0, dt)
        if self._shot_timer > 0:
            return False
        self._shot_timer = 0.0
        return True

    def _save_eye_view(self) -> None:
        """Read the eye image that was just drawn (the swapchain framebuffer is still bound) and save it as a PNG."""
        from OpenGL import GL

        from .. import paths

        x, y, w, h = (int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT))
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(x, y, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].copy()          # OpenGL rows run bottom to top
        try:
            path = paths.screenshots_dir() / time.strftime("VR screenshot %Y-%m-%d %H-%M-%S.png")
        except Exception as error:  # noqa: BLE001 - no place to save is reported, not fatal
            self.menu_action.emit(("screenshot-failed", f"{type(error).__name__}: {error}"))
            return
        threading.Thread(target=self._write_png, args=(path, image), name="uv3d-vr-shot", daemon=True).start()
        self._place_panel()
        self.menu.show("main")
        self.menu.set_note("Screenshot saved in your Pictures folder." if path.parent.parent.name == "Pictures"
                           else f"Screenshot saved: {path.name}")

    def _write_png(self, path, rgb: np.ndarray) -> None:
        try:
            import cv2

            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("could not write the file")
            self.menu_action.emit(("screenshot", str(path)))
        except Exception as error:  # noqa: BLE001
            self.menu_action.emit(("screenshot-failed", f"{type(error).__name__}: {error}"))

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
