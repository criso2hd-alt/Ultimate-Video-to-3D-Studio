"""Parse ``.cube`` LUTs (Adobe / DaVinci / Resolve format) for the grading kernel.

3D LUTs are used as they are. 1D LUTs - shaper curves, some camera log
transforms - are expanded into an equivalent 3D table so the kernel has a single
path; a 1D LUT is per-channel, so the expansion is exact.

A malformed file raises ``ValueError`` with a plain reason. The UI shows it,
because a LUT that silently does nothing is worse than one that says why.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Grid size used when a 1D LUT is expanded to 3D.
_EXPAND = 33


class CubeLUT:
    """A 3D table indexed ``[blue, green, red, channel]`` (``.cube`` runs red fastest)."""

    def __init__(self, size: int, table: np.ndarray, domain_min, domain_max, title: str = "") -> None:
        self.size = size
        self.table = np.ascontiguousarray(table, np.float32)
        self.domain_min = np.asarray(domain_min, np.float32)
        self.domain_max = np.asarray(domain_max, np.float32)
        self.title = title


def load_cube(path: Path) -> CubeLUT:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ValueError(f"Could not read the LUT: {error}") from error

    size_3d = size_1d = None
    title = ""
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows: list[tuple[float, float, float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        try:
            if upper.startswith("TITLE"):
                title = line.split(None, 1)[1].strip().strip('"') if " " in line else ""
            elif upper.startswith("LUT_3D_SIZE"):
                size_3d = int(line.split()[-1])
            elif upper.startswith("LUT_1D_SIZE"):
                size_1d = int(line.split()[-1])
            elif upper.startswith("DOMAIN_MIN"):
                domain_min = [float(v) for v in line.split()[1:4]]
            elif upper.startswith("DOMAIN_MAX"):
                domain_max = [float(v) for v in line.split()[1:4]]
            elif upper.startswith(("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE")):
                lo, hi = (float(v) for v in line.split()[1:3])
                domain_min, domain_max = [lo] * 3, [hi] * 3
            else:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except (ValueError, IndexError) as error:
            raise ValueError(f"Cannot read line {line!r}: {error}") from error

    data = np.asarray(rows, np.float32)
    if size_3d:
        expected = size_3d ** 3
        if len(data) != expected:
            raise ValueError(f"Expected {expected} entries for a {size_3d}³ LUT but found {len(data)}.")
        table = data.reshape(size_3d, size_3d, size_3d, 3)
        return CubeLUT(size_3d, table, domain_min, domain_max, title)
    if size_1d:
        if len(data) != size_1d:
            raise ValueError(f"Expected {size_1d} entries for a 1D LUT but found {len(data)}.")
        # A per-channel curve becomes a 3D table by evaluating it at each grid point.
        grid = np.linspace(0.0, 1.0, _EXPAND, dtype=np.float32)
        positions = np.linspace(0.0, 1.0, size_1d, dtype=np.float32)
        curves = [np.interp(grid, positions, data[:, c]) for c in range(3)]
        table = np.empty((_EXPAND, _EXPAND, _EXPAND, 3), np.float32)
        table[..., 0] = curves[0][None, None, :]
        table[..., 1] = curves[1][None, :, None]
        table[..., 2] = curves[2][:, None, None]
        return CubeLUT(_EXPAND, table, domain_min, domain_max, title)
    raise ValueError("Not a .cube LUT: no LUT_3D_SIZE or LUT_1D_SIZE line.")


def available_luts(folder: Path) -> list[Path]:
    """The ``.cube`` files in ``folder``, sorted by name, for the picker."""
    try:
        return sorted(Path(folder).glob("*.cube"), key=lambda p: p.name.lower())
    except OSError:
        return []
