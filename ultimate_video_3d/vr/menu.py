"""The menu a person sees inside the headset: a floating panel they point at with a controller.

It is only a picture and a list of buttons. :class:`Menu` knows which page is showing, what each button says and
where it is, which one the pointer is on, and what pressing one means; it paints itself into a ``QImage`` that the
renderer puts on a quad in the room. No OpenXR and no OpenGL in here, so all of it is tested without a headset.

Pages: ``help`` (what the controllers do - shown on its own the first time the headset goes on), ``main`` (play,
lights, seat, volume, exit), ``videos`` (the Library, three by two thumbnails a page).

Pressing a button returns an *action* tuple for whoever owns the headset to carry out, or None when the press was only
about the menu itself (changing page, paging through videos):
    ("play",)  ("video", Tile)  ("lights", "auto"|"on"|"off")  ("volume", 0..1)  ("seat", +1|-1)  ("recentre",)
    ("exit",)  ("close",)  ("screenshot",)
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen

W, H = 1280, 800
ASPECT = W / H
PER_PAGE = 6
LIGHT_MODES = (("auto", "Auto"), ("on", "On"), ("off", "Off"))
HELP_SECONDS = 16.0                        # the first-time card goes away by itself after this

INK, DIM, FAINT = QColor("#f5f9ff"), QColor("#9bacc2"), QColor("#61748d")
PANEL, LINE = QColor(13, 22, 37, 238), QColor("#29405d")
SIGNAL, ON_SIGNAL = QColor("#42dcf5"), QColor("#041017")
RAISED, RAISED_HOT = QColor("#16243a"), QColor("#244061")
DANGER = QColor("#d9534f")


@dataclass
class Tile:
    """One video in the Videos page."""

    title: str
    path: str
    layout: str = ""
    detail: str = ""
    thumb: QImage | None = None


@dataclass
class _Button:
    id: str
    rect: QRectF


class Menu:
    def __init__(self) -> None:
        self.open = False
        self.page = "main"
        self.hover: str | None = None
        # what the main page shows (kept in step with the player by :meth:`sync`)
        self.playing = False
        self.title = ""
        self.has_media = False
        self.has_lights = False
        self.seat_label = ""
        self.has_seats = False
        self.lights = "auto"
        self.volume = 1.0
        self.tiles: list[Tile] = []
        self.video_page = 0
        self.help_left = 0.0                # seconds until the help card closes itself; 0 = it stays
        self.note = ""                      # a line under the main page's buttons ("Screenshot saved...")
        self.note_left = 0.0
        self._buttons: list[_Button] = []
        self._dirty = True
        self._image: QImage | None = None
        self.version = 0                    # bumps every time the picture changes (the renderer re-uploads it)

    # -- what is showing ---------------------------------------------------------------------------

    def show(self, page: str = "main", auto_close: bool = False) -> None:
        self.open, self.page, self.hover = True, page, None
        self.help_left = HELP_SECONDS if auto_close and page == "help" else 0.0
        self._dirty = True

    def close(self) -> None:
        if self.open:
            self.open, self.hover, self.help_left = False, None, 0.0
            self._dirty = True

    def toggle(self) -> None:
        self.close() if self.open else self.show("main")

    def set_note(self, text: str, seconds: float = 9.0) -> None:
        self.note, self.note_left, self._dirty = text, seconds, True

    def tick(self, dt: float) -> None:
        """Time passing: the first-time help card closes itself, and a note fades after a while."""
        if self.note and self.note_left > 0:
            self.note_left -= max(0.0, dt)
            if self.note_left <= 0:
                self.note, self._dirty = "", True
        if self.open and self.page == "help" and self.help_left > 0:
            self.help_left -= max(0.0, dt)
            if self.help_left <= 0:
                self.close()

    def sync(self, *, playing: bool, title: str, has_media: bool, has_lights: bool, seat_label: str, has_seats: bool) -> None:
        values = (playing, title, has_media, has_lights, seat_label, has_seats)
        if values != (self.playing, self.title, self.has_media, self.has_lights, self.seat_label, self.has_seats):
            (self.playing, self.title, self.has_media, self.has_lights, self.seat_label, self.has_seats) = values
            self._dirty = True

    def set_tiles(self, tiles: list[Tile]) -> None:
        self.tiles = list(tiles)
        self.video_page = min(self.video_page, self.page_count - 1)
        self._dirty = True

    @property
    def page_count(self) -> int:
        return max(1, -(-len(self.tiles) // PER_PAGE))

    # -- pointing ------------------------------------------------------------------------------------

    def hit(self, u: float, v: float) -> str | None:
        """The button under a point of the panel (u, v from its top-left corner, 0..1), or None."""
        self._ensure()
        x, y = u * W, v * H
        for button in self._buttons:
            if button.rect.contains(x, y):
                return button.id
        return None

    def set_hover(self, button: str | None) -> None:
        if button != self.hover:
            self.hover = button
            self._dirty = True

    # -- pressing ------------------------------------------------------------------------------------

    def press(self, button: str | None) -> tuple | None:
        """Carry out a press of ``button``; returns an action for the headset's owner, or None."""
        if button is None or not self.open:
            return None
        if button.startswith("page:"):
            self.show(button[5:])
            return None
        if button == "close":
            self.close()
            return ("close",)
        if button == "play":
            return ("play",)
        if button.startswith("lights:"):
            self.lights = button[7:]
            self._dirty = True
            return ("lights", self.lights)
        if button in ("seat:-1", "seat:1"):
            return ("seat", int(button[5:]))
        if button in ("vol:-", "vol:+"):
            self.volume = min(1.0, max(0.0, round(self.volume + (0.1 if button == "vol:+" else -0.1), 2)))
            self._dirty = True
            return ("volume", self.volume)
        if button in ("recentre", "exit", "screenshot"):
            return (button,)
        if button in ("vpage:-1", "vpage:1"):
            self.video_page = (self.video_page + int(button[6:])) % self.page_count
            self._dirty = True
            return None
        if button.startswith("video:"):
            index = int(button[6:])
            if 0 <= index < len(self.tiles):
                tile = self.tiles[index]
                self.close()
                return ("video", tile)
        return None

    # -- painting ------------------------------------------------------------------------------------

    def _ensure(self) -> None:
        if self._dirty or self._image is None:
            self._paint()

    def image(self) -> QImage:
        """The panel as straight-alpha RGBA, ``W`` x ``H``."""
        self._ensure()
        return self._image

    @staticmethod
    def _font(pixels: int, bold: bool = False) -> QFont:
        font = QFont("Segoe UI")
        font.setPixelSize(pixels)
        font.setBold(bold)
        return font

    def _paint(self) -> None:
        canvas = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(Qt.GlobalColor.transparent)
        p = QPainter(canvas)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                         | QPainter.RenderHint.SmoothPixmapTransform)
        self._buttons = []
        p.setPen(QPen(LINE, 3))
        p.setBrush(PANEL)
        p.drawRoundedRect(QRectF(3, 3, W - 6, H - 6), 36, 36)
        {"help": self._help_page, "videos": self._videos_page}.get(self.page, self._main_page)(p)
        p.end()
        self._image = canvas.convertToFormat(QImage.Format.Format_RGBA8888)
        self._dirty = False
        self.version += 1

    # drawing helpers -----------------------------------------------------------------------------

    def _text(self, p: QPainter, rect: QRectF, text: str, pixels: int = 30, colour: QColor = INK, bold: bool = False,
              align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, wrap: bool = False) -> None:
        p.setFont(self._font(pixels, bold))
        p.setPen(colour)
        if wrap:                                                    # a paragraph: lines break at the width, nothing is cut
            p.drawText(rect, int(align | Qt.TextFlag.TextWordWrap), text)
        else:                                                       # one line: too long ends in "..." rather than spilling
            p.drawText(rect, int(align), QFontMetricsF(p.font()).elidedText(text, Qt.TextElideMode.ElideRight, rect.width()))

    def _button(self, p: QPainter, id: str, rect: QRectF, label: str, *, style: str = "normal", pixels: int = 34) -> None:
        self._buttons.append(_Button(id, rect))
        hot = self.hover == id
        fill = {"primary": SIGNAL.lighter(115) if hot else SIGNAL, "selected": SIGNAL.darker(130) if not hot else SIGNAL,
                "danger": DANGER.lighter(115) if hot else DANGER.darker(115)}.get(style, RAISED_HOT if hot else RAISED)
        ink = ON_SIGNAL if style in ("primary", "selected") else INK
        p.setPen(QPen(SIGNAL if hot else LINE, 4 if hot else 2))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 22, 22)
        self._text(p, rect, label, pixels, ink, bold=True, align=Qt.AlignmentFlag.AlignCenter)

    def _header(self, p: QPainter, title: str, subtitle: str = "", back: bool = False) -> None:
        self._text(p, QRectF(48, 28, 700, 50), title, 40, INK, bold=True)
        if subtitle:
            self._text(p, QRectF(48, 78, 760, 36), subtitle, 26, DIM)
        if back:
            self._button(p, "page:main", QRectF(W - 400, 34, 170, 64), "Back", pixels=28)
        self._button(p, "close", QRectF(W - 215, 34, 170, 64), "Close", pixels=28)

    # pages ---------------------------------------------------------------------------------------

    def _main_page(self, p: QPainter) -> None:
        self._header(p, "Ultimate Video to 3D Studio",
                     (f"{'Playing' if self.playing else 'Paused'}: {self.title}" if self.has_media else "Nothing is playing yet."))
        y = 150
        self._button(p, "play", QRectF(48, y, 380, 120), "Pause" if self.playing else "Play", style="primary", pixels=44)
        self._button(p, "page:videos", QRectF(450, y, 380, 120), "Videos", pixels=44)
        self._button(p, "page:help", QRectF(852, y, 380, 120), "Help", pixels=44)
        y = 310
        if self.has_lights:
            self._text(p, QRectF(48, y, 200, 80), "Lights", 36, INK, bold=True)
            for i, (mode, label) in enumerate(LIGHT_MODES):
                self._button(p, f"lights:{mode}", QRectF(260 + i * 200, y, 184, 80), label,
                             style="selected" if self.lights == mode else "normal", pixels=34)
            self._text(p, QRectF(880, y, 350, 80), "Auto: down while a film plays.", 22, FAINT)
            y += 110
        if self.has_seats:
            self._text(p, QRectF(48, y, 200, 80), "Seat", 36, INK, bold=True)
            self._button(p, "seat:-1", QRectF(260, y, 90, 80), "<", pixels=40)
            self._text(p, QRectF(360, y, 330, 80), self.seat_label, 30, INK, align=Qt.AlignmentFlag.AlignCenter)
            self._button(p, "seat:1", QRectF(700, y, 90, 80), ">", pixels=40)
            y += 110
        self._text(p, QRectF(48, y, 200, 80), "Volume", 36, INK, bold=True)
        self._button(p, "vol:-", QRectF(260, y, 90, 80), "-", pixels=44)
        self._text(p, QRectF(360, y, 330, 80), f"{round(self.volume * 100)}%", 34, INK, align=Qt.AlignmentFlag.AlignCenter)
        self._button(p, "vol:+", QRectF(700, y, 90, 80), "+", pixels=44)
        if self.note:
            self._text(p, QRectF(48, 616, W - 96, 44), self.note, 24, SIGNAL)
        self._button(p, "recentre", QRectF(48, 668, 380, 80), "Reset view", pixels=30)
        self._button(p, "screenshot", QRectF(450, 668, 380, 80), "Screenshot", pixels=30)
        self._button(p, "exit", QRectF(852, 668, 380, 80), "Leave VR", style="danger", pixels=30)

    def _videos_page(self, p: QPainter) -> None:
        self._header(p, "Your videos", f"Page {self.video_page + 1} of {self.page_count}", back=True)
        if not self.tiles:
            self._text(p, QRectF(48, 250, W - 96, 200), "No videos yet. Convert one, or add videos in the app's Library tab.",
                       34, DIM, align=Qt.AlignmentFlag.AlignCenter, wrap=True)
            return
        first = self.video_page * PER_PAGE
        for slot, tile in enumerate(self.tiles[first:first + PER_PAGE]):
            col, row = slot % 3, slot // 3
            x, y = 40 + col * 410, 130 + row * 265
            rect = QRectF(x, y, 380, 245)
            self._buttons.append(_Button(f"video:{first + slot}", rect))
            hot = self.hover == f"video:{first + slot}"
            p.setPen(QPen(SIGNAL if hot else LINE, 5 if hot else 2))
            p.setBrush(RAISED_HOT if hot else RAISED)
            p.drawRoundedRect(rect, 18, 18)
            picture = QRectF(x + 10, y + 10, 360, 160)
            if tile.thumb is not None and not tile.thumb.isNull():
                scaled = tile.thumb.scaled(360, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                p.drawImage(QRectF(picture.center().x() - scaled.width() / 2, picture.center().y() - scaled.height() / 2,
                                   scaled.width(), scaled.height()), scaled)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#0a1220"))
                p.drawRoundedRect(picture, 10, 10)
            self._text(p, QRectF(x + 14, y + 172, 352, 38), tile.title, 27, INK, bold=True)
            self._text(p, QRectF(x + 14, y + 208, 352, 30), tile.detail, 21, DIM)
        if self.page_count > 1:
            self._button(p, "vpage:-1", QRectF(48, 690, 260, 70), "< Previous", pixels=28)
            self._button(p, "vpage:1", QRectF(W - 308, 690, 260, 70), "Next >", pixels=28)

    def _help_page(self, p: QPainter) -> None:
        self._header(p, "How to use the headset", "Point a controller at a button and pull the trigger.")
        rows = (
            ("Trigger on the screen,  or  A / X", "Play and pause"),
            ("Left menu button,  or  B / Y", "Open this menu: videos, lights, seat, volume"),
            ("Left stick, left or right", "Back or forward 10 seconds"),
            ("Right stick, any direction", "Go to the seat that way"),
            ("Press the right stick down", "Reset the view: face the screen"),
        )
        y = 150
        for keys, what in rows:
            self._text(p, QRectF(48, y, 600, 56), keys, 29, SIGNAL, bold=True)
            self._text(p, QRectF(660, y, 580, 56), what, 28, INK)
            y += 66
        self._text(p, QRectF(48, y + 14, W - 96, 120),
                   "The lights go down by themselves while a film plays and come up when you pause. "
                   "To choose them yourself: Menu > Lights > On or Off. Taking the headset off pauses the film.", 28, DIM,
                   align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, wrap=True)
        self._button(p, "close", QRectF(48, 668, 380, 80), "Got it", style="primary", pixels=34)
        self._button(p, "page:videos", QRectF(450, 668, 380, 80), "Pick a video", pixels=34)
        self._button(p, "page:main", QRectF(852, 668, 380, 80), "Open the menu", pixels=34)
