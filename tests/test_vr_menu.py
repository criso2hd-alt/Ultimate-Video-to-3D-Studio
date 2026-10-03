"""The headset's menu: its pages, its buttons, what pressing them means, and how the session drives it. No headset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimate_video_3d.vr import geometry as geo  # noqa: E402
from ultimate_video_3d.vr import menu as menumod  # noqa: E402
from ultimate_video_3d.vr.menu import Menu, Tile  # noqa: E402
from ultimate_video_3d.vr.session import Hand, VRSession  # noqa: E402
from ultimate_video_3d.vr.state import LightFader  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def centre_of(menu: Menu, button: str) -> tuple[float, float]:
    """The (u, v) middle of a named button, found the way a pointer would."""
    menu.image()
    for b in menu._buttons:
        if b.id == button:
            return b.rect.center().x() / menumod.W, b.rect.center().y() / menumod.H
    raise AssertionError(f"no button {button!r} on page {menu.page}: {[b.id for b in menu._buttons]}")


def click(menu: Menu, button: str):
    u, v = centre_of(menu, button)
    assert menu.hit(u, v) == button
    return menu.press(menu.hit(u, v))


def ready(*, lights=True, seats=True, playing=False, media=True) -> Menu:
    menu = Menu()
    menu.sync(playing=playing, title="Home movie" if media else "", has_media=media, has_lights=lights,
              seat_label="Row 2, centre", has_seats=seats)
    menu.show("main")
    return menu


def tiles(n: int) -> list[Tile]:
    return [Tile(f"Film {i}", f"film{i}.mp4", "sbs_half", "SBS - 1:30") for i in range(n)]


# --- pages and painting -------------------------------------------------------------------------------


def test_the_panel_is_painted_with_a_transparent_corner_and_an_opaque_body(qapp):
    image = ready().image()
    assert (image.width(), image.height()) == (menumod.W, menumod.H)
    assert image.format() == QImage.Format.Format_RGBA8888
    assert image.pixelColor(2, 2).alpha() == 0                              # outside the rounded corner
    assert image.pixelColor(menumod.W // 2, menumod.H - 40).alpha() > 200   # the body of the panel


def test_the_picture_changes_only_when_something_shown_changes(qapp):
    menu = ready()
    menu.image()
    version = menu.version
    menu.sync(playing=False, title="Home movie", has_media=True, has_lights=True, seat_label="Row 2, centre", has_seats=True)
    menu.image()
    assert menu.version == version                                              # nothing changed: no repaint, no GPU upload
    menu.set_hover("play")
    menu.image()
    assert menu.version == version + 1
    menu.set_hover("play")
    menu.image()
    assert menu.version == version + 1


def test_lights_and_seat_rows_appear_only_in_a_room_with_something_to_choose(qapp):
    plain = ready(lights=False, seats=False)
    plain.image()
    ids = {b.id for b in plain._buttons}
    assert "play" in ids and "exit" in ids and not any(i.startswith(("lights:", "seat:")) for i in ids)
    room = ready()
    room.image()
    ids = {b.id for b in room._buttons}
    assert {"lights:auto", "lights:on", "lights:off", "seat:-1", "seat:1"} <= ids


def test_every_button_can_be_reached_and_none_overlap(qapp):
    for page in ("main", "videos", "help"):
        menu = ready()
        menu.set_tiles(tiles(14))
        menu.show(page)
        menu.image()
        rects = [(b.id, b.rect) for b in menu._buttons if b.id != "close" or page != "help"]
        for i, (a_id, a) in enumerate(rects):
            assert 0 <= a.left() and a.right() <= menumod.W and 0 <= a.top() and a.bottom() <= menumod.H, (page, a_id)
            for b_id, b in rects[i + 1:]:
                assert not a.intersects(b), (page, a_id, b_id)


# --- pressing buttons ------------------------------------------------------------------------------------


def test_main_page_buttons_return_actions_and_pages_switch(qapp):
    menu = ready()
    assert click(menu, "play") == ("play",)
    assert click(menu, "lights:on") == ("lights", "on") and menu.lights == "on"
    assert click(menu, "seat:1") == ("seat", 1)
    assert click(menu, "seat:-1") == ("seat", -1)
    assert click(menu, "recentre") == ("recentre",)
    assert click(menu, "exit") == ("exit",)
    assert click(menu, "page:help") is None and menu.page == "help"
    assert click(menu, "page:main") is None and menu.page == "main"
    assert click(menu, "close") == ("close",) and not menu.open


def test_volume_steps_by_tenths_and_stops_at_the_ends(qapp):
    menu = ready()
    assert click(menu, "vol:-") == ("volume", 0.9)
    for _ in range(12):
        action = click(menu, "vol:-")
    assert action == ("volume", 0.0) and menu.volume == 0.0
    for _ in range(12):
        action = click(menu, "vol:+")
    assert action == ("volume", 1.0)


def test_a_closed_menu_ignores_presses_and_empty_space_does_nothing(qapp):
    menu = ready()
    menu.close()
    assert menu.press("play") is None
    menu.show("main")
    assert menu.hit(0.5, 0.995) is None and menu.press(None) is None


# --- the videos page ---------------------------------------------------------------------------------------


def test_videos_are_three_by_two_a_page_and_paging_wraps(qapp):
    menu = ready()
    menu.set_tiles(tiles(14))
    assert menu.page_count == 3
    click(menu, "page:videos")
    menu.image()
    shown = [b.id for b in menu._buttons if b.id.startswith("video:")]
    assert shown == [f"video:{i}" for i in range(6)]
    click(menu, "vpage:1")
    menu.image()
    assert [b.id for b in menu._buttons if b.id.startswith("video:")][0] == "video:6"
    click(menu, "vpage:1")
    click(menu, "vpage:1")
    assert menu.video_page == 0                                                  # wrapped round
    click(menu, "vpage:-1")
    assert menu.video_page == 2


def test_choosing_a_video_hands_it_over_and_closes_the_menu(qapp):
    menu = ready()
    menu.set_tiles(tiles(8))
    click(menu, "page:videos")
    click(menu, "vpage:1")
    action = click(menu, "video:7")
    assert action[0] == "video" and action[1].path == "film7.mp4" and action[1].layout == "sbs_half"
    assert not menu.open


def test_an_empty_library_says_so_and_a_thumbnail_is_drawn_when_there_is_one(qapp):
    menu = ready()
    click(menu, "page:videos")
    menu.image()
    assert not [b for b in menu._buttons if b.id.startswith("video:")] and menu.page_count == 1
    thumb = QImage(320, 180, QImage.Format.Format_RGB32)
    thumb.fill(QColor(250, 30, 30))
    menu.set_tiles([Tile("Red", "r.mp4", "", "x", thumb)])
    image = menu.image()
    assert image.pixelColor(40 + 190, 130 + 90).red() > 200                  # the tile's picture is the thumbnail


def test_the_tile_list_shrinking_keeps_the_page_in_range(qapp):
    menu = ready()
    menu.set_tiles(tiles(14))
    menu.video_page = 2
    menu.set_tiles(tiles(3))
    assert menu.video_page == 0


# --- help and its timer --------------------------------------------------------------------------------------


def test_the_first_time_card_closes_itself_but_a_card_opened_from_the_menu_stays(qapp):
    menu = ready()
    menu.show("help", auto_close=True)
    menu.tick(menumod.HELP_SECONDS - 1)
    assert menu.open
    menu.tick(2)
    assert not menu.open
    menu.show("help")
    menu.tick(100)
    assert menu.open                                                              # asked for: it stays until dismissed
    assert click(menu, "close") == ("close",) and not menu.open


# --- the lights --------------------------------------------------------------------------------------------


def test_the_lights_follow_the_film_unless_the_person_chooses(qapp):
    fader = LightFader(level=1.0)
    for _ in range(100):
        level = fader.update(0.05, playing=True, has_media=True)
    assert level == 0.0                                                          # auto: dark while it plays
    fader.mode = "on"
    for _ in range(100):
        level = fader.update(0.05, playing=True, has_media=True)
    assert level == 1.0                                                          # their choice holds through the film
    fader.mode = "off"
    for _ in range(100):
        level = fader.update(0.05, playing=False, has_media=True, paused_level=0.6)
    assert level == 0.0                                                          # and through a pause
    fader.mode = "auto"
    for _ in range(100):
        level = fader.update(0.05, playing=False, has_media=True, paused_level=0.6)
    assert level == pytest.approx(0.6)


# --- where the panel hangs -------------------------------------------------------------------------------------


def test_the_panel_hangs_in_front_of_the_viewer_facing_them_and_a_pointer_can_hit_it(qapp):
    head, forward = np.array([1.0, 1.2, -2.0]), np.array([0.0, 0.0, -1.0])
    matrix = geo.panel_matrix(head, forward, 1.3, menumod.ASPECT, distance=1.5, drop=0.2)
    centre = matrix[:3, 3]
    assert centre == pytest.approx([1.0, 1.0, -3.5], abs=1e-6)                 # 1.5 m ahead, 20 cm below the eyes
    normal = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    assert normal[2] > 0.98 and normal[1] > 0                                    # faces the viewer, tilted up toward the eyes
    hit = geo.ray_hits_matrix(matrix, head, centre - head)
    assert hit is not None and hit[1] == pytest.approx(0.5, abs=1e-6) and hit[2] == pytest.approx(0.5, abs=1e-6)
    assert hit[0] == pytest.approx(np.linalg.norm(centre - head), abs=1e-6)
    assert geo.ray_hits_matrix(matrix, head, np.array([0.0, 0.0, 1.0])) is None   # pointing away
    left = geo.ray_hits_matrix(matrix, head, centre - head + np.array([-0.5, 0.0, 0.0]))
    assert left is not None and left[1] < 0.5                                    # u grows to the right: left of centre is < 0.5
    up = geo.ray_hits_matrix(matrix, head, centre - head + np.array([0.0, 0.25, 0.0]))
    assert up is not None and up[2] < 0.5                                        # v grows downwards: above centre is < 0.5
    turned = geo.panel_matrix(head, np.array([1.0, 0.0, 0.0]), 1.3, menumod.ASPECT)   # looking along +x
    assert turned[0, 3] > head[0] + 1.4 and abs(turned[2, 3] - head[2]) < 0.01


# --- the session drives it ---------------------------------------------------------------------------------------


class _Media:
    is_open = True
    playing = False
    path = Path("home movie.mp4")
    layout = "sbs_half"
    aspect = 16 / 9

    def __init__(self):
        self.toggled = 0

    def toggle(self):
        self.toggled += 1

    def seek_relative(self, ms):
        pass


def aimed_at(point, origin=(0.0, 1.2, 0.0)) -> Hand:
    """A controller at ``origin`` pointing at ``point``."""
    hand = Hand(valid=True, position=np.array(origin, float))
    direction = np.asarray(point, float) - hand.position
    direction /= np.linalg.norm(direction)
    # a rotation that turns the controller's -z onto the wanted direction
    z = -direction
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.column_stack([x, y, z])
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    hand.orientation = ((m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w)
    return hand


@pytest.fixture()
def session(qapp):
    s = VRSession(_Media())
    s._head = (np.array([0.0, 1.2, 0.0]), (0.0, 0.0, 0.0, 1.0))
    s.menu.sync(playing=False, title="Home movie", has_media=True, has_lights=False, seat_label="", has_seats=False)
    return s


def world_point(session: VRSession, button: str) -> np.ndarray:
    u, v = centre_of(session.menu, button)
    return (session._panel @ np.array([u - 0.5, 0.5 - v, 0.0, 1.0]))[:3]


def test_the_menu_button_opens_and_closes_the_menu_in_front_of_the_person(session):
    session._controls({"left": Hand(valid=True, menu_pressed=True)}, 0.01)
    assert session.menu.open and session._panel is not None
    assert session._panel[:3, 3] == pytest.approx([0.0, 1.0, -1.2], abs=0.01)
    session._controls({"right": Hand(valid=True, secondary_pressed=True)}, 0.01)      # B / Y closes it again
    assert not session.menu.open


def test_pointing_at_a_button_highlights_it_and_the_trigger_presses_it(session):
    session._controls({"left": Hand(valid=True, menu_pressed=True)}, 0.01)
    target = world_point(session, "page:videos")
    hand = aimed_at(target)
    session._controls({"right": hand}, 0.01)
    assert session.menu.hover == "page:videos"
    hand.select_pressed = True
    session._controls({"right": hand}, 0.01)
    assert session.menu.page == "videos"
    assert session.media.toggled == 0                                                 # the click was the menu's, not the screen's


def test_menu_buttons_reach_the_player_and_the_window(session):
    seen = []
    session.menu_action.connect(seen.append)
    session._controls({"left": Hand(valid=True, menu_pressed=True)}, 0.01)
    for button in ("play",):
        hand = aimed_at(world_point(session, button))
        hand.select_pressed = True
        session._controls({"right": hand}, 0.01)
    assert session.media.toggled == 1                                                 # Play: the player was toggled
    for button in ("vol:-", "exit"):
        hand = aimed_at(world_point(session, button))
        hand.select_pressed = True
        session._controls({"right": hand}, 0.01)
    assert seen == [("volume", 0.9), ("exit",)]                                       # the window's business goes to the window


def test_pressing_the_right_stick_down_resets_the_view(session):
    session._controls({"right": Hand(valid=True, reset_pressed=True)}, 0.01)
    assert session._recentre.is_set()
    assert not session.menu.open                                                      # it is a reset, not the menu


def test_the_trigger_still_plays_and_pauses_when_the_menu_is_closed(session):
    session.screen = geo.Screen.from_quad(geo.translation([0.0, 1.5, -4.0]) @ np.diag([4.0, 2.25, 1.0, 1.0]), 4.0, 2.25)
    hand = aimed_at([0.0, 1.5, -4.0])
    hand.select_pressed = True
    session._controls({"right": hand}, 0.01)
    assert session.media.toggled == 1
    for name in ("primary_pressed",):
        pressed = Hand(valid=True)
        setattr(pressed, name, True)
        session._controls({"left": pressed}, 0.01)
    assert session.media.toggled == 2                                                 # A / X too


def test_putting_the_headset_on_shows_help_by_itself_once(session):
    session.worn = True
    session._placed = True
    session._menu_after_move()
    assert session.menu.open and session.menu.page == "help" and session.menu.help_left > 0
    session.menu.close()
    session._menu_after_move()
    assert not session.menu.open                                                     # once per session, not every frame


def test_with_nothing_playing_the_videos_are_offered_first(session):
    session.media.is_open = False
    session.menu.set_tiles(tiles(3))
    session.worn = True
    session._placed = True
    session._menu_after_move()
    assert session.menu.open and session.menu.page == "videos"
    session.media.is_open = True


def test_the_lights_choice_reaches_the_fader_and_a_seat_change_follows_the_person(session):
    session._apply(("lights", "off"))
    assert session._lights.mode == "off"
    session._toggle_menu()
    session._rig = geo.translation([2.0, 0.0, 0.0])                                    # they moved to another seat
    session._replace_panel = True
    session._placed = True
    session._menu_after_move()
    assert session._panel[0, 3] == pytest.approx(2.0, abs=0.01)
