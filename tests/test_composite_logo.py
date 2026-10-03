"""scripts/blender/composite_logo.py: the sticker is laid onto the wall's atlas by its own alpha, and nowhere else."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "blender" / "composite_logo.py"


def _write(path: Path, rgb: tuple[int, int, int], size: int = 64) -> None:
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = rgb[::-1]                                  # OpenCV writes BGR
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])


def _setup(tmp_path: Path):
    for state in ("lit", "dim", "screen"):
        _write(tmp_path / f"walls_plaster_{state}.jpg", (90, 90, 90))
        _write(tmp_path / f"chairs_details_{state}.jpg", (230, 40, 40))        # the baked sticker is red
    # a PNG whose left half is opaque and right half fully transparent
    png = np.zeros((32, 32, 4), np.uint8)
    png[:, :16] = (255, 255, 255, 255)
    cv2.imwrite(str(tmp_path / "logo.png"), png)
    # the wall: a 4 x 4 m quad in the x/z plane, its UV the plain 0..1 square; the logo: a 2 x 2 m card in its middle
    wall = {"tris": [{"pos": [[-2, -9.5, -2], [2, -9.5, -2], [2, -9.5, 2]], "uv": [[0, 0], [1, 0], [1, 1]]},
                     {"pos": [[-2, -9.5, -2], [2, -9.5, 2], [-2, -9.5, 2]], "uv": [[0, 0], [1, 1], [0, 1]]}]}
    logo = {"corners": [[-1, -9.46, -1], [1, -9.46, -1], [1, -9.46, 1], [-1, -9.46, 1]],
            "bake_uv": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "png_uv": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    info = {"png": str(tmp_path / "logo.png"), "wall": wall, "logo": logo, "screen_ev": {"walls_plaster": 0.0, "chairs_details": 0.0}}
    (tmp_path / "logo_info.json").write_text(json.dumps(info))


def _run(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def _pixel(path: Path, x: float, z: float) -> np.ndarray:
    """The RGB of the wall atlas at world (x, z) of the 4 m wall (u = (x + 2) / 4, v = (z + 2) / 4, rows run top-down)."""
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    return image[int((1 - (z + 2) / 4) * h), int((x + 2) / 4 * w)].astype(int)


def test_the_sticker_lands_only_where_it_is_opaque_and_the_wall_shows_elsewhere(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    wall = tmp_path / "walls_plaster_lit.jpg"
    opaque_side = _pixel(wall, -0.5, 0.0)                 # the logo's left half: opaque, so the red sticker
    clear_side = _pixel(wall, 0.5, 0.0)                   # its right half: transparent, so the wall
    outside = _pixel(wall, 1.7, 1.7)                      # beyond the card: the wall, untouched
    assert opaque_side[0] > 180 and opaque_side[1] < 90   # red
    assert abs(int(clear_side[0]) - 90) < 8 and abs(int(clear_side[1]) - 90) < 8
    assert abs(int(outside[0]) - 90) < 8


def test_it_can_be_run_again_without_the_sticker_building_up(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    first = cv2.imread(str(tmp_path / "walls_plaster_dim.jpg")).astype(int)
    _run(tmp_path)
    second = cv2.imread(str(tmp_path / "walls_plaster_dim.jpg")).astype(int)
    assert np.abs(first - second).max() <= 2              # starts from the saved original every time
    assert (tmp_path / "walls_plaster_dim.orig.jpg").is_file()


def test_a_wall_that_is_not_one_flat_island_is_refused(tmp_path):
    _setup(tmp_path)
    info = json.loads((tmp_path / "logo_info.json").read_text())
    info["wall"]["tris"][1]["uv"] = [[0, 0], [0.3, 0.9], [0.7, 0.1]]            # a second island that does not match the first
    (tmp_path / "logo_info.json").write_text(json.dumps(info))
    done = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode != 0 and "flat UV island" in (done.stdout + done.stderr)
