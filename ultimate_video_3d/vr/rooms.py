"""Finding the rooms a person can sit in: every folder under the data folder's ``rooms`` that has a ``room.glb``.

A room's name comes from its ``room.json`` (``"name"``), else the folder's name. Its picture in the headset's menu is a
``thumbnail.jpg`` (or ``.png``, or ``preview.jpg``) in the room's own folder: drop one there and it shows up, nothing to configure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: What stands for "the plain built-in room" (a dark room with a mirror floor, no baked room at all) in settings and on the Rooms page.
PLAIN = "plain"
PLAIN_NAME = "Plain room"
PLAIN_NOTE = "Mirror floor"

THUMBNAILS = ("thumbnail.jpg", "thumbnail.jpeg", "thumbnail.png", "preview.jpg", "preview.png")


@dataclass(frozen=True)
class RoomEntry:
    name: str
    folder: Path
    thumbnail: Path | None


def plain_thumbnail():
    """A small painted picture of the plain room (a screen over a dark floor that reflects it), for its card on the Rooms page."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter

    w, h = 640, 360
    image = QImage(w, h, QImage.Format.Format_RGB32)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    back = QLinearGradient(0, 0, 0, h)
    back.setColorAt(0.0, QColor("#07090f"))
    back.setColorAt(0.62, QColor("#0c1018"))
    back.setColorAt(1.0, QColor("#05070b"))
    p.fillRect(0, 0, w, h, QBrush(back))
    floor_y = h * 0.62
    screen = QRectF(w * 0.2, h * 0.13, w * 0.6, (w * 0.6) * 9 / 16)
    if screen.bottom() > floor_y - 8:
        screen = QRectF(w * 0.25, h * 0.1, w * 0.5, (w * 0.5) * 9 / 16)
    glow = QLinearGradient(screen.left(), 0, screen.right(), 0)
    glow.setColorAt(0.0, QColor("#4a5a78"))
    glow.setColorAt(1.0, QColor("#2c3a56"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(glow))
    p.drawRoundedRect(screen, 6, 6)
    mirror = QLinearGradient(0, floor_y, 0, h)                    # the floor: the screen again, upside down, fading away
    mirror.setColorAt(0.0, QColor(74, 90, 120, 120))
    mirror.setColorAt(1.0, QColor(44, 58, 86, 0))
    p.setBrush(QBrush(mirror))
    p.drawRect(QRectF(screen.left(), floor_y, screen.width(), min(h - floor_y, screen.height() * 0.8)))
    p.setBrush(QColor(255, 255, 255, 18))
    p.drawRect(QRectF(0, floor_y - 1, w, 2))
    p.end()
    return image


def find_rooms(root: Path) -> list[RoomEntry]:
    """The rooms under ``root``, in name order. A folder without a ``room.glb`` is not a room."""
    found: list[RoomEntry] = []
    try:
        folders = sorted(p for p in Path(root).iterdir() if p.is_dir())
    except OSError:
        return []
    for folder in folders:
        if not (folder / "room.glb").is_file():
            continue
        name = folder.name.replace("_", " ")
        try:
            data = json.loads((folder / "room.json").read_text(encoding="utf8"))
            if isinstance(data.get("name"), str) and data["name"].strip():
                name = data["name"].strip()
        except (OSError, ValueError, AttributeError):
            pass
        thumbnail = next((folder / t for t in THUMBNAILS if (folder / t).is_file()), None)
        found.append(RoomEntry(name, folder, thumbnail))
    return sorted(found, key=lambda r: r.name.lower())
