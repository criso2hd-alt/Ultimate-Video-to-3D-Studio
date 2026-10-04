"""What the rest of the app talks to for VR: show a video in the headset, and react to the headset being put on.

The window and the Library never touch OpenXR. They call :meth:`VRController.show` with a file; the controller starts
the headset session on first use, loads the room if one is set, and plays the video when (and only when) the headset
is worn. A *preview* is shown muted and looped; the first press of the play button or Space then unmutes it and carries on.

Two conveniences so nobody has to reach for the mouse with a headset on their face:
* **Watching** (:meth:`watch`): while it is on, the controller looks every few seconds for a headset and, the moment there is one,
  starts the session by itself, so putting the headset on is all that is left to do (the room comes up with the Videos page).
  Failing to find one is silent. Leaving VR from the menu pauses watching until the person asks for VR again.
* **Sound** (:meth:`apply_audio`): the film's sound is sent to the headset's own audio device (Meta: "Oculus Virtual Audio Device")
  without the person visiting Windows' sound settings; the menu's Sound row can choose another device.

The session, media and the headset probe are passed in so the logic can be tested without a headset.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import NO_HEADSET, headset_present
from .rooms import PLAIN
from . import available as vr_available
from . import room as roommod
from .log import log as vrlog

#: Words in an audio device's name that mean "this is the headset's" (Meta Link, Rift, SteamVR headsets, Windows Mixed Reality).
AUDIO_HINTS = ("oculus", "meta quest", "quest", "rift", "valve index", "index hmd", "vive", "mixed reality", "hmd")
WATCH_SECONDS = 5
#: How long the runtime gets to accept a session (an instance, a session and a graphics context) before the start is given up on and
#: the person is told. Waiting for the *headset* (putting it on, waking it) is a different thing and never times out.
START_TIMEOUT = 40
#: How long a session that was asked to stop gets to do it before it is cut loose and a new one may be made.
STOP_PATIENCE = 12


def qt_audio_outputs() -> list:
    from PySide6.QtMultimedia import QMediaDevices

    return list(QMediaDevices.audioOutputs())


def headset_audio_index(devices: list) -> int | None:
    """Which of the output devices looks like the headset's, or None."""
    for i, device in enumerate(devices):
        name = device.description().lower()
        if any(hint in name for hint in AUDIO_HINTS):
            return i
    return None


class VRController(QObject):
    state_changed = Signal(str)             # "off" | "starting" (asking the runtime) | "waiting" (connected, headset not on) | "worn" | "stopping"
    notice = Signal(str)                    # a line for the status bar
    seats_changed = Signal(list)            # [(name, label)] once a room is loaded
    failed = Signal(str)
    audio_choice_changed = Signal(str)      # the person chose a sound device in the headset ("" = automatic): save it
    room_chosen = Signal(str)               # the person picked another room in the headset (its folder): remember it
    beam_changed = Signal(bool, float)      # the person switched the projector beam or changed its brightness in the headset: save it
    _probed = Signal(bool)                  # a headset probe finished (queued onto the UI thread)

    def __init__(self, parent: QObject | None = None, session_factory=None, media_factory=None, probe=None,
                 audio_outputs=None) -> None:
        super().__init__(parent)
        self._session_factory, self._media_factory = session_factory, media_factory
        self._probe_fn = probe or headset_present
        self._audio_outputs_fn = audio_outputs or qt_audio_outputs
        self.audio_choice = ""               # "" = automatic (the headset's device if there is one)
        self._room_tiles: list = []                     # the Rooms page's choices
        self._applied_room_key: str | None = None       # the room the session has been given (handing it over again would move the person back to the first seat)
        self.room_provider = None                       # () -> the folder of the room to sit in, or None (the app's choice); used when VR starts by itself
        self.trim_bars = True                           # leave a film's burned-in black bars off the screen (the app's settings fill this in)
        self.beam_on, self.beam_level = True, 1.0       # the projector's beam in the room (the app's settings fill these in)
        self._watching = self._paused = self._probing = False
        self._timer = QTimer(self)
        self._timer.setInterval(WATCH_SECONDS * 1000)
        self._timer.timeout.connect(self._poll)
        self._probed.connect(self._on_probed)
        self._start_timer = QTimer(self)               # the runtime must accept the session within START_TIMEOUT
        self._start_timer.setSingleShot(True)
        self._start_timer.setInterval(START_TIMEOUT * 1000)
        self._start_timer.timeout.connect(self._on_start_timeout)
        self._stop_timer = QTimer(self)                # polls a stopping session; stopping never blocks the window
        self._stop_timer.setInterval(150)
        self._stop_timer.timeout.connect(self._check_stopped)
        self._stop_clock = 0.0
        self.session = None
        self.media = None
        self.room: roommod.Room | None = None
        self._room_folder: Path | None = None
        self._worn = False
        self._loop = False
        self._muted_preview = False          # the first press of play unmutes a muted preview
        self._wants_play = False             # play as soon as the headset is on
        self._tiles: list = []               # the Library as the headset's Videos page shows it
        self._volume = 1.0
        self.state = "off"

    # -- facts ------------------------------------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return vr_available()

    @property
    def running(self) -> bool:
        return self.session is not None and self.session.running

    @property
    def worn(self) -> bool:
        return self._worn

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    # -- starting and stopping ----------------------------------------------------------------------

    def _make_parts(self) -> None:
        if self.session is not None:
            return
        if self._media_factory is None or self._session_factory is None:
            from .media import VRMedia
            from .session import VRSession

            self._media_factory = self._media_factory or VRMedia
            self._session_factory = self._session_factory or VRSession
        self.media = self._media_factory(self)
        self.session = self._session_factory(self.media, self)
        self.session.worn_changed.connect(self._on_worn)
        if hasattr(self.session, "connected"):
            self.session.connected.connect(self._on_connected)
        self.session.failed.connect(self._on_failed)
        self.session.stopped.connect(self._on_stopped)
        self.session.menu_action.connect(self._on_menu_action)
        self.session.set_videos(self._tiles)
        if hasattr(self.session, "set_rooms"):
            self.session.set_rooms(self._room_tiles, self.room_key())
        if hasattr(self.media, "set_volume"):
            self.media.set_volume(self._volume)
        if hasattr(self.media, "failed"):
            self.media.failed.connect(self.notice)
        self.apply_audio()
        self.apply_beam()
        self.apply_trim()

    def use_room(self, folder: str | Path | None) -> bool:
        """Choose the room to sit in (None for the plain stand-in). Returns False, with a notice, if it cannot be loaded."""
        wanted = Path(folder) if folder else None
        if wanted is not None and self.room is not None and self._room_folder == wanted and self._applied_room_key == str(wanted):
            return True                                      # already in it: nothing to load, and nobody is moved
        if wanted is None and self.room is None and self._room_folder is None and self._applied_room_key == PLAIN:
            return True
        loaded = None
        if wanted is not None:
            try:
                loaded = roommod.load(wanted)
            except roommod.RoomError as error:
                self.notice.emit(f"Could not load the VR room: {error}")
                if self.room is None:
                    self._room_folder = None                 # nothing was loaded before either: the plain room it is
                return False                                 # otherwise the person stays in the room they are in
        self._room_folder, self.room = wanted, loaded
        if self.session is not None:
            self.session.set_room(self.room)
            self._applied_room_key = self.room_key()
            self.seats_changed.emit(self.session.seat_names())
            if hasattr(self.session, "set_current_room"):
                self.session.set_current_room(self.room_key())
        return True

    def room_key(self) -> str:
        """Which room the person is in, as the Rooms page names it: its folder, or ``rooms.PLAIN`` for the plain built-in room."""
        return str(self._room_folder) if self._room_folder else PLAIN

    def set_rooms(self, tiles: list) -> None:
        """The rooms to offer on the headset's Rooms page (Tile list; UI thread)."""
        self._room_tiles = list(tiles)
        if self.session is not None and hasattr(self.session, "set_rooms"):
            self.session.set_rooms(self._room_tiles, self.room_key())

    def start_vr(self) -> bool:
        """The Start VR button: begin a session now, whatever automatic watching did or did not do.

        Forgets an earlier failure (which had paused watching), and says so if the last session is still closing.
        """
        vrlog("Start VR pressed")
        self._paused = False
        return self.start()

    def start(self) -> bool:
        """Start the headset session (once). False, with a notice, if VR is not available on this computer."""
        if self.state == "stopping":
            self.notice.emit("VR is still closing from last time - give it a few seconds, then start it again.")
            return False
        if not self.available() and self._session_factory is None:
            self.notice.emit("VR is not available: it needs Windows, an OpenXR runtime (Meta Link, SteamVR or "
                             "Virtual Desktop) and the VR components.")
            return False
        if self.room is None and self.room_provider is not None:       # a session that starts by itself (a headset appeared) still sits in the room
            folder = self.room_provider()
            if folder:
                self.use_room(folder)
        self._make_parts()
        if self._applied_room_key != self.room_key():       # not on every video: set_room puts the person back in the first seat
            self.session.set_room(self.room)
            self._applied_room_key = self.room_key()
            self.seats_changed.emit(self.session.seat_names())
        if not self.session.running:
            vrlog("starting a session")
            self._set_state("starting")
            self._start_timer.start()
            self.session.start()
        return True

    def stop(self, blocking: bool = False) -> None:
        """Leave VR. Watching pauses, so the app does not pull the person straight back in; asking for VR again resumes it.

        The window is never held up waiting for the headset: the session is asked to stop and polled until it has (state
        "stopping" meanwhile). ``blocking`` is for quitting the app, which waits a few seconds so the runtime is left clean.
        """
        vrlog("stop requested")
        self._paused = True
        self._start_timer.stop()
        if self.session is not None:
            self.session.stop(wait=5.0 if blocking else 0)
        if self.media is not None:
            self.media.close(wait=True)
        self._worn = False
        self._wants_play = False
        if blocking or self.session is None or not self.session.running:
            self._stop_timer.stop()
            self._set_state("off")
        else:
            self._stop_clock = time.monotonic()
            self._set_state("stopping")
            self._stop_timer.start()

    def _check_stopped(self) -> None:
        if self.session is None or not self.session.running:
            self._stop_timer.stop()
            self._set_state("off")
        elif time.monotonic() - self._stop_clock > STOP_PATIENCE:
            self._stop_timer.stop()
            self._abandon_session("did not close")

    def _on_start_timeout(self) -> None:
        """The runtime never accepted the session (it is hung, or the headset software is not running)."""
        if self.state != "starting":
            return
        vrlog(f"the runtime did not accept a session within {START_TIMEOUT} s")
        self._paused = True
        self.failed.emit(f"The VR runtime did not answer within {START_TIMEOUT} seconds. Check that Meta Quest Link (or SteamVR) is "
                         "running and shows your headset as connected, then press Start VR again.")
        self.notice.emit("VR did not start.")
        self.stop()

    def _abandon_session(self, why: str) -> None:
        """A session thread that will not end (stuck inside the runtime): let go of it so a fresh one can be made."""
        vrlog(f"abandoning the session: {why}")
        session = self.session
        if session is not None:
            for name in ("worn_changed", "connected", "failed", "stopped", "menu_action"):
                try:
                    getattr(session, name).disconnect()
                except Exception:  # noqa: BLE001 - a stand-in without the signal, or already disconnected
                    pass
        self.session = self.media = None
        self._applied_room_key = None
        self._set_state("off")
        self.notice.emit("The last VR session did not close. If VR will not start, close this app and open it again.")

    # -- showing a video ------------------------------------------------------------------------------

    def show(self, path: Path, layout: str, preview: bool = False) -> bool:
        """Put a video on the headset's screen. A preview loops muted; anything else plays through with sound.

        Nothing plays until the headset is on: if it already is, playback starts at once.
        """
        self._paused = False
        if not self.start():
            return False
        self._loop, self._muted_preview = preview, preview
        if not self.media.open(Path(path), layout, loop=preview, muted=preview):
            return False
        self._wants_play = True
        self.notice.emit("Put the headset on to watch." if not self._worn else "Playing in the headset.")
        if self._worn:
            self.media.play()
        return True

    def toggle(self) -> None:
        """Play/pause. On a muted preview the first press brings the sound in and keeps playing."""
        if self.media is None or not self.media.is_open:
            return
        if self._muted_preview:
            self._muted_preview = False
            self.media.set_muted(False)
            if not self.media.playing:
                self.media.play()
            return
        self.media.toggle()

    # -- the menu in the headset ---------------------------------------------------------------------

    def set_videos(self, tiles: list) -> None:
        """The Library for the menu's Videos page (a list of menu.Tile). Kept, and given to the headset when it starts."""
        self._tiles = list(tiles)
        if self.session is not None:
            self.session.set_videos(self._tiles)

    def _on_menu_action(self, action: tuple) -> None:
        """A button in the headset's menu that only this side can do (UI thread)."""
        kind = action[0]
        if kind == "video":
            tile = action[1]
            self.show(Path(tile.path), tile.layout, preview=False)
        elif kind == "skip":
            if self.media is not None and self.media.is_open:
                self.media.seek_relative(int(action[1]))
        elif kind == "seek":
            if self.media is not None and self.media.is_open and self.media.duration_ms:
                self.media.seek_ms(int(float(action[1]) * self.media.duration_ms))
        elif kind == "volume":
            self._volume = float(action[1])
            if self.media is not None and hasattr(self.media, "set_volume"):
                self.media.set_volume(self._volume)
        elif kind == "room":
            tile = action[1]
            if self.use_room(None if tile.path == PLAIN else Path(tile.path)):
                self.room_chosen.emit(str(tile.path))
        elif kind == "audio":
            self.cycle_audio(int(action[1]))
        elif kind == "beam":
            self.beam_on, self.beam_level = bool(action[1]), float(action[2])
            self.beam_changed.emit(self.beam_on, self.beam_level)
        elif kind == "exit":
            self.stop()
        elif kind == "screenshot":
            self.notice.emit(f"Headset screenshot saved: {action[1]}")
        elif kind == "screenshot-failed":
            self.notice.emit(f"Could not save the screenshot: {action[1]}")

    # -- watching for a headset --------------------------------------------------------------------------

    def watch(self, on: bool) -> None:
        """While on, look for a headset every few seconds and start the session when there is one (see the module doc)."""
        if on:
            self._watching, self._paused = True, False
            if not self._timer.isActive():
                self._timer.start()
            self._poll()
        else:
            self._watching = False
            self._timer.stop()

    def _poll(self) -> None:
        if not self._watching or self._paused or self._probing or self.running:
            return
        self._probing = True
        threading.Thread(target=self._probe_in_thread, name="uv3d-vr-probe", daemon=True).start()

    def _probe_in_thread(self) -> None:
        try:
            present = bool(self._probe_fn())
        except Exception:  # noqa: BLE001 - a probe that fails is "no headset"
            present = False
        self._probed.emit(present)

    def _on_probed(self, present: bool) -> None:
        self._probing = False
        if present and self._watching and not self._paused and not self.running:
            if self.start():
                self.notice.emit("VR headset found - put it on to pick a video.")

    # -- sound -------------------------------------------------------------------------------------------

    def _audio_devices(self) -> list:
        try:
            return list(self._audio_outputs_fn())
        except Exception:  # noqa: BLE001 - no audio stack: nothing to choose
            return []

    def audio_device(self):
        """The Qt device the film's sound should use now: the person's choice, else the headset's, else None (the default)."""
        devices = self._audio_devices()
        if self.audio_choice:
            for device in devices:
                if device.description() == self.audio_choice:
                    return device
        index = headset_audio_index(devices)
        return devices[index] if index is not None else None

    def audio_label(self) -> str:
        device = self.audio_device()
        if self.audio_choice and device is not None:
            return device.description()
        if device is not None:
            return f"Automatic: {device.description()}"
        return "Automatic: this computer's default"

    def apply_audio(self) -> None:
        """Send the sound to the right device and tell the menu which one it is."""
        device = self.audio_device()
        if device is not None and self.media is not None and hasattr(self.media, "set_audio_device"):
            self.media.set_audio_device(device)
        if self.session is not None and hasattr(self.session, "set_audio_label"):
            self.session.set_audio_label(self.audio_label())

    def apply_trim(self) -> None:
        if self.media is not None and hasattr(self.media, "trim_bars"):
            self.media.trim_bars = self.trim_bars

    def set_trim_bars(self, on: bool) -> None:
        self.trim_bars = bool(on)
        self.apply_trim()

    def apply_beam(self) -> None:
        if self.session is not None and hasattr(self.session, "set_beam"):
            self.session.set_beam(self.beam_on, self.beam_level)

    def set_beam(self, on: bool, level: float) -> None:
        """Switch the projector's beam on or off and set its brightness (the lens keeps glowing); the headset follows at once."""
        self.beam_on, self.beam_level = bool(on), float(level)
        self.apply_beam()

    def cycle_audio(self, direction: int) -> None:
        """The menu's Sound row: automatic, then each output device in turn."""
        options = [""] + [d.description() for d in self._audio_devices()]
        index = options.index(self.audio_choice) if self.audio_choice in options else 0
        self.audio_choice = options[(index + (1 if direction >= 0 else -1)) % len(options)]
        self.apply_audio()
        self.audio_choice_changed.emit(self.audio_choice)

    # -- seats ------------------------------------------------------------------------------------------

    def seat_names(self) -> list[tuple[str, str]]:
        return self.session.seat_names() if self.session is not None else []

    def set_seat(self, name: str) -> None:
        if self.session is not None:
            self.session.set_seat(name)

    # -- the session's events (arrive on the UI thread) -----------------------------------------------

    def _on_connected(self) -> None:
        """The runtime accepted the session. From here on it is the headset's turn (putting it on can take as long as it takes)."""
        self._start_timer.stop()
        vrlog("runtime connected")
        if self.state == "starting":
            self._set_state("waiting")
            self.notice.emit("VR is ready - put the headset on.")

    def _on_worn(self, worn: bool) -> None:
        self._start_timer.stop()
        vrlog(f"headset {'on' if worn else 'off'}")
        self._worn = worn
        self._set_state("worn" if worn else "waiting")
        if worn:
            self.apply_audio()                            # the headset's speakers may only exist once it is on
        if self.media is None or not self.media.is_open:
            return
        if worn and self._wants_play:
            self.media.play()
        elif not worn:
            self.media.pause()                            # taking the headset off pauses the film

    def _on_failed(self, message: str) -> None:
        self._start_timer.stop()
        vrlog(f"failed: {message}")
        if self._watching and message == NO_HEADSET:
            self._set_state("off")                        # quietly waiting for a headset: not an error, try again soon
            return
        self._paused = True                               # no automatic retry after a real failure: one box, then wait to be asked again
        self.failed.emit(message)
        self.notice.emit(f"VR stopped: {message}")
        self._set_state("off")

    def _on_stopped(self) -> None:
        self._start_timer.stop()
        self._stop_timer.stop()
        vrlog("session ended")
        self._worn = False
        self._set_state("off")
