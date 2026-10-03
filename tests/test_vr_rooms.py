"""Choosing the room from inside the headset: finding the rooms, the menu's Rooms page, and carrying the choice out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d.vr import menu as menumod  # noqa: E402
from ultimate_video_3d.vr import rooms as roomsmod  # noqa: E402
from ultimate_video_3d.vr.menu import Menu, Tile  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def make_rooms(root: Path) -> Path:
    for folder, name, thumb in (("home_theatre", "Home theatre", "thumbnail.jpg"), ("grand_cinema", None, "preview.png"), ("bare", "Bare", None)):
        (root / folder).mkdir(parents=True)
        (root / folder / "room.glb").write_bytes(b"glTF")
        if name:
            (root / folder / "room.json").write_text(json.dumps({"name": name}))
        if thumb:
            (root / folder / thumb).write_bytes(b"x")
    (root / "not_a_room").mkdir()                                              # no room.glb
    (root / "notes.txt").write_text("hello")
    return root


def test_rooms_are_the_folders_with_a_room_glb_named_from_their_json_and_shown_with_their_picture(tmp_path):
    found = roomsmod.find_rooms(make_rooms(tmp_path))
    assert [r.name for r in found] == ["Bare", "grand cinema", "Home theatre"]                # name order; "grand_cinema" read as a name
    by_name = {r.name: r for r in found}
    assert by_name["Home theatre"].thumbnail.name == "thumbnail.jpg"
    assert by_name["grand cinema"].thumbnail.name == "preview.png"
    assert by_name["Bare"].thumbnail is None
    assert roomsmod.find_rooms(tmp_path / "missing") == []


def tiles(n=3, pictures=True):
    picture = QImage(64, 36, QImage.Format.Format_RGB32)
    picture.fill(QColor("#336699"))
    return [Tile(f"Room {i}", f"C:/rooms/r{i}", "", "", picture if pictures else None) for i in range(n)]


def ready(rooms, current="C:/rooms/r0") -> Menu:
    menu = Menu()
    menu.sync(playing=False, title="Film", has_media=True, has_lights=True, seat_label="Row 2", has_seats=True)
    menu.set_rooms(rooms, current)
    menu.show("main")
    return menu


def click(menu: Menu, button: str):
    menu.image()
    rect = next(b.rect for b in menu._buttons if b.id == button)
    u, v = rect.center().x() / menumod.W, rect.center().y() / menumod.H
    assert menu.hit(u, v) == button
    return menu.press(button)


def test_the_rooms_button_appears_only_when_there_is_a_choice_of_rooms(qapp):
    one = ready(tiles(1))
    one.image()
    assert "page:rooms" not in {b.id for b in one._buttons}
    several = ready(tiles(3))
    several.image()
    assert {"play", "page:videos", "page:rooms", "page:help"} <= {b.id for b in several._buttons}


def test_every_button_fits_on_the_main_page_with_four_top_buttons_and_on_the_rooms_page(qapp):
    for page in ("main", "rooms"):
        menu = ready(tiles(6))
        menu.show(page)
        menu.image()
        rects = [(b.id, b.rect) for b in menu._buttons]
        for i, (a_id, a) in enumerate(rects):
            assert 0 <= a.left() and a.right() <= menumod.W and 0 <= a.top() and a.bottom() <= menumod.H, (page, a_id)
            for b_id, b in rects[i + 1:]:
                assert not a.intersects(b), (page, a_id, b_id)


def test_picking_another_room_hands_it_over_and_closes_the_menu_and_the_room_you_are_in_just_closes_it(qapp):
    menu = ready(tiles(3), current="C:/rooms/r0")
    click(menu, "page:rooms")
    assert menu.page == "rooms"
    menu.image()
    assert {"room:0", "room:1", "room:2", "page:main"} <= {b.id for b in menu._buttons}
    action = click(menu, "room:2")
    assert action[0] == "room" and action[1].path == "C:/rooms/r2" and not menu.open
    menu.show("rooms")
    assert click(menu, "room:0") is None and not menu.open                      # already there: nothing to switch


def test_the_page_paints_with_and_without_pictures_and_marks_the_current_room(qapp):
    for with_pictures in (True, False):
        menu = ready(tiles(4, with_pictures), current="C:/rooms/r1")
        menu.show("rooms")
        image = menu.image()
        assert image.width() == menumod.W and not image.isNull()
    menu.set_current_room("C:/rooms/r3")
    assert menu.current_room == "C:/rooms/r3"
