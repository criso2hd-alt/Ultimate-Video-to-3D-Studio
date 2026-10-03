"""Finding the rooms a person can sit in: every folder under the data folder's ``rooms`` that has a ``room.glb``.

A room's name comes from its ``room.json`` (``"name"``), else the folder's name. Its picture in the headset's menu is a
``thumbnail.jpg`` (or ``.png``, or ``preview.jpg``) in the room's own folder: drop one there and it shows up, nothing to configure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

THUMBNAILS = ("thumbnail.jpg", "thumbnail.jpeg", "thumbnail.png", "preview.jpg", "preview.png")


@dataclass(frozen=True)
class RoomEntry:
    name: str
    folder: Path
    thumbnail: Path | None


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
