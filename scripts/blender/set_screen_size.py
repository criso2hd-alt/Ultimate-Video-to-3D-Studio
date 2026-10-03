"""Change ONLY the size of the room's Screen quad and save the result as a NEW .blend (the file it was opened from is never touched).

    blender -b room_template_projector.blend --python set_screen_size.py -- <out.blend> <width_m> <height_m> [bottom_m]

The quad keeps its place on the wall (centred on x = 0, y and its bottom edge unchanged unless ``bottom_m`` is given) and its UVs,
so nothing needs to be re-baked: the app reads the screen's shape from the exported room.glb, and the baked lit / dim / screen-light
pictures are separate files. Then run ``bake_room.py --phase export`` on a copy of the bake folder to write the new room.glb.
"""
import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out_path, width, height = args[0], float(args[1]), float(args[2])
bottom = float(args[3]) if len(args) > 3 else None
if os.path.exists(out_path):
    raise SystemExit(f"{out_path} already exists; not replacing it")

screen = bpy.data.objects["Screen"]
mesh = screen.data
world = [screen.matrix_world @ v.co for v in mesh.vertices]
old_w, old_h = max(p.x for p in world) - min(p.x for p in world), max(p.z for p in world) - min(p.z for p in world)
z0 = min(p.z for p in world) if bottom is None else bottom
cx = (max(p.x for p in world) + min(p.x for p in world)) / 2
print(f"SCREEN | was {old_w:.3f} x {old_h:.3f} m (bottom z {min(p.z for p in world):.3f}); now {width:.3f} x {height:.3f} m, bottom {z0:.3f}")
inverse = screen.matrix_world.inverted()
for vertex, p in zip(mesh.vertices, world):
    x = cx + (width / 2 if p.x > cx else -width / 2)
    z = z0 + (height if p.z > (min(q.z for q in world) + old_h / 2) else 0.0)
    vertex.co = inverse @ type(p)((x, p.y, z))
mesh.update()
bpy.ops.wm.save_as_mainfile(filepath=out_path)
print("SCREEN | saved", out_path)
