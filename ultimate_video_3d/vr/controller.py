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
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import NO_HEADSET, headset_present
from . import available as vr_available
from . import room as roommod

#: Words in an audio device's name that mean "this is the headset's" (Meta Link, Rift, SteamVR headsets, Windows Mixed Reality).
AUDIO_HINTS = ("oculus", "meta quest", "quest", "rift", "valve index", "index hmd", "vive", "mixed reality", "hmd")
WATCH_SECONDS = 5


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
    state_changed = Signal(str)             # "off" | "starting" | "waiting" | "worn"
    notice = Signal(str)                    # a line for the status bar
    seats_changed = Signal(list)            # [(name, label)] once a room is loaded
    failed = Signal(str)
    audio_choice_changed = Signal(str)      # the person chose a sound device in the headset ("" = automatic): save it
    _probed = Signal(bool)                  # a headset probe finished (queued onto the UI thread)

    def __init__(self, parent: QObject | None = None, session_factory=None, media_factory=None, probe=None,
                 audio_outputs=None) -> None:
        super().__init__(parent)
        self._session_factory, self._media_factory = session_factory, media_factory
        self._probe_fn = probe or headset_present
        self._audio_outputs_fn = audio_outputs or qt_audio_outputs
        self.audio_choice = ""               # "" = automatic (the headset's device if there is one)
        self._watching = self._paused = self._probing = False
        self._timer = QTimer(self)
        self._timer.setInterval(WATCH_SECONDS * 1000)
        self._timer.timeout.connect(self._poll)
        self._probed.connect(self._on_probed)
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
        self.session.failed.connect(self._on_failed)
        self.session.stopped.connect(self._on_stopped)
        self.session.menu_action.connect(self._on_menu_action)
        self.session.set_videos(self._tiles)
        if hasattr(self.media, "set_volume"):
            self.media.set_volume(self._volume)
        if hasattr(self.media, "failed"):
            self.media.failed.connect(self.notice)
        self.apply_audio()

    def use_room(self, folder: str | Path | None) -> bool:
        """Choose the room to sit in (None for the plain stand-in). Returns False, with a notice, if it cannot be loaded."""
        self._room_folder = Path(folder) if folder else None
        if self._room_folder is None:
            self.room = None
        else:
            try:
                self.room = roommod.load(self._room_folder)
            except roommod.RoomError as error:
                self.room = None
                self.notice.emit(f"Could not load the VR room: {error}")
                return False
        if self.session is not None:
            self.session.set_room(self.room)
            self.seats_changed.emit(self.session.seat_names())
        return True

    def start(self) -> bool:
        """Start the headset session (once). False, with a notice, if VR is not available on this computer."""
        if not self.available() and self._session_factory is None:
            self.notice.emit("VR is not available: it needs Windows, an OpenXR runtime (Meta Link, SteamVR or "
                             "Virtual Desktop) and the VR components.")
            return False
        self._make_parts()
        if self.room is not None:
            self.session.set_room(self.room)
            self.seats_changed.emit(self.session.seat_names())
        if not self.session.running:
            self._set_state("starting")
            self.session.start()
        return True

    def stop(self) -> None:
        """Leave VR. Watching pauses, so the app does not pull the person straight back in; asking for VR again resumes it."""
        self._paused = True
        if self.session is not None:
            self.session.stop()
        if self.media is not None:
            self.media.close(wait=True)
        self._worn = False
        self._wants_play = False
        self._set_state("off")

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
        elif kind == "volume":
            self._volume = float(action[1])
            if self.media is not None and hasattr(self.media, "set_volume"):
                self.media.set_volume(self._volume)
        elif kind == "audio":
            self.cycle_audio(int(action[1]))
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

    def _on_worn(self, worn: bool) -> None:
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
        if self._watching and message == NO_HEADSET:
            self._set_state("off")                        # quietly waiting for a headset: not an error, try again soon
            return
        self.failed.emit(message)
        self.notice.emit(f"VR stopped: {message}")
        self._set_state("off")

    def _on_stopped(self) -> None:
        self._worn = False
        self._set_state("off")
