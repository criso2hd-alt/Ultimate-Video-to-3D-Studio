"""One-off edit of the owner's room file: lower the screen, refit the BiasGlow, and put a Seat Empty on every chair.

    blender --background --factory-startup <room.blend> --python scripts/blender/adjust_room_screen_and_seats.py

Run it with the SAME Blender version that saved the file (a file from a newer Blender loses data when an older one
saves it). It edits the file in place; Blender keeps the previous save as ``.blend1``. Numbers come from the chairs'
real geometry (read from the file on 2026-10-02): three rows of three armchairs, rows on 0, 0.43 and 0.861 m risers.
"""

from __future__ import annotations

import bpy

SCREEN_BOTTOM = 0.65                       # the picture's lower edge above the floor (was 1.03)
GLOW_BOTTOM, GLOW_TOP = 0.10, 4.25         # the BiasGlow halo, fitted around the lowered screen (stays off the floor)
SEAT_X = (-1.575, 0.0, 1.575)              # the three armchairs across, as the viewer sees them (+x is their right)
SIDES = ("Left", "Centre", "Right")
#: (row label, riser height under the chairs, backrest front face y). Eyes sit 0.15 m in front of the backrest face.
ROWS = (("R1", 0.0, -4.43), ("R2", 0.43, -6.48), ("R3", 0.861, -8.59))
EYE_ABOVE_RISER = 1.22                     # headrest top (1.32) minus about 0.1 m
EYE_FORWARD = 0.15


def remap_world_z(obj, fn) -> tuple[float, float]:
    """Move an object's vertices by changing their WORLD z with ``fn``; works whatever the object's own scale is."""
    world = obj.matrix_world
    back = world.inverted()
    zs = []
    for vertex in obj.data.vertices:
        point = world @ vertex.co
        point.z = fn(point.z)
        zs.append(point.z)
        vertex.co = back @ point
    obj.data.update()
    return min(zs), max(zs)


def world_z_range(obj) -> tuple[float, float]:
    zs = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
    return min(zs), max(zs)


def main() -> None:
    screen, glow = bpy.data.objects["Screen"], bpy.data.objects["BiasGlow"]

    # --- the screen: same size, lowered ---------------------------------------------------------------
    lo, hi = world_z_range(screen)
    shift = SCREEN_BOTTOM - lo
    new_lo, new_hi = remap_world_z(screen, lambda z: z + shift)
    print(f"Screen z {lo:.3f}..{hi:.3f} -> {new_lo:.3f}..{new_hi:.3f}  (height {new_hi - new_lo:.3f})")

    # --- the glow: stretched to sit around the new screen, clear of the floor and ceiling ----------------
    glo, ghi = world_z_range(glow)
    span = (GLOW_TOP - GLOW_BOTTOM) / (ghi - glo)
    new_glo, new_ghi = remap_world_z(glow, lambda z: GLOW_BOTTOM + (z - glo) * span)
    print(f"BiasGlow z {glo:.3f}..{ghi:.3f} -> {new_glo:.3f}..{new_ghi:.3f}")

    # --- seats: one Empty per chair, eyes at the headrest, zero rotation = looking at the screen ---------
    seats = bpy.data.collections["02_SEATS"]
    for obj in [o for o in bpy.data.objects if o.name.startswith("Seat_")]:
        bpy.data.objects.remove(obj, do_unlink=True)
    for row, riser, back_y in ROWS:
        for x, side in zip(SEAT_X, SIDES):
            empty = bpy.data.objects.new(f"Seat_{row}_{side}", None)
            empty.empty_display_type = "ARROWS"
            empty.empty_display_size = 0.3
            empty.location = (x, back_y + EYE_FORWARD, riser + EYE_ABOVE_RISER)
            empty["label"] = f"Row {row[1]} - {side}"
            seats.objects.link(empty)
    spawn = bpy.data.objects.get("Spawn")
    if spawn is not None:
        spawn.location = (0.0, -2.4, 0.0)                    # standing in front of the first row, facing the screen
        print("Spawn ->", tuple(spawn.location))

    # --- a sight-line check: does each seat clear the chairs in front to see the picture's lowest edge? ---
    print("Sight lines to the bottom edge of the picture (clearance over the chair in front, metres):")
    blockers = {"R1": (-4.60, 1.32), "R2": (-6.65, 1.75)}        # backrest y and top z of the rows that can be in the way
    y_screen = -0.04
    for row, riser, back_y in ROWS:
        eye_y, eye_z = back_y + EYE_FORWARD, riser + EYE_ABOVE_RISER
        notes = []
        for name, (by, top) in blockers.items():
            if by > eye_y + 0.3:                                 # that row is in front of this viewer
                t = (by - eye_y) / (y_screen - eye_y)
                notes.append(f"over {name} {eye_z + (SCREEN_BOTTOM - eye_z) * t - top:+.3f}")
        print(f"  {row}: eye=({eye_y:.2f}, {eye_z:.2f})  " + ("; ".join(notes) or "nothing in front"))

    bpy.ops.wm.save_mainfile()
    print("saved", bpy.data.filepath)


main()
