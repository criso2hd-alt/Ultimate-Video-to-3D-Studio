"""What the rest of the app talks to for VR: show a video in the headset, and react to the headset being put on.

The window and the Library never touch OpenXR. They call :meth:`VRController.show` with a file; the controller starts
the headset session on first use, loads the room if one is set, and plays the video when (and only when) the headset
is worn. A *preview* is shown muted and looped; the first press of the play button or Space then unmutes it and carries on.

The session and media classes are passed in so the logic can be tested without a headset.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from . import available as vr_available
from . import room as roommod


class VRController(QObject):
    state_changed = Signal(str)             # "off" | "starting" | "waiting" | "worn"
    notice = Signal(str)                    # a line for the status bar
    seats_changed = Signal(list)            # [(name, label)] once a room is loaded
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None, session_factory=None, media_factory=None) -> None:
        super().__init__(parent)
        self._session_factory, self._media_factory = session_factory, media_factory
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
        elif kind == "exit":
            self.stop()

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
        if self.media is None or not self.media.is_open:
            return
        if worn and self._wants_play:
            self.media.play()
        elif not worn:
            self.media.pause()                            # taking the headset off pauses the film

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)
        self.notice.emit(f"VR stopped: {message}")
        self._set_state("off")

    def _on_stopped(self) -> None:
        self._worn = False
        self._set_state("off")
