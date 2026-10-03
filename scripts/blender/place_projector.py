"""Place a stand-in projector (box + ceiling pole) and the Projector_Lens empty where nothing is in the way.

Reads the room, searches positions, checks (a) the pole and the body do not touch any geometry and (b) every ray from the lens
to a grid over the whole screen reaches it unobstructed, then builds the objects and saves a NEW file (the owner's file is
never overwritten). Run: blender -b room_template.blend --python place_projector.py -- <out.blend>
"""
import sys
import itertools
import math

import bpy
import mathutils
from mathutils.bvhtree import BVHTree

out_path = sys.argv[sys.argv.index("--") + 1]
dg = bpy.context.evaluated_depsgraph_get()

BODY = (0.47, 0.47, 0.19)            # W, D, H of a typical home-theatre projector (metres)
POLE = 0.05
CEILING_Z = 5.0
SKIP_PREFIX = ("BiasGlow", "Projector", "BeamGuide")
SKIP = {"Screen", "REF_Person_1.75m", "Ceiling.001"}


def room_tree():
    verts, polys = [], []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name in SKIP or o.name.startswith(SKIP_PREFIX) or o.hide_render:
            continue
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = o.matrix_world
        base = len(verts)
        verts += [tuple(mw @ v.co) for v in me.vertices]
        flip = mw.to_3x3().determinant() < 0
        for p in me.polygons:
            ids = [base + i for i in p.vertices]
            polys.append(ids[::-1] if flip else ids)
        ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys, all_triangles=False)


def box_tree(lo, hi, grow=0.0):
    x0, y0, z0 = (lo[0] - grow, lo[1] - grow, lo[2] - grow)
    x1, y1, z1 = (hi[0] + grow, hi[1] + grow, hi[2] + grow)
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return BVHTree.FromPolygons(v, quads, all_triangles=False)


tree = room_tree()
screen = bpy.data.objects["Screen"]
pts = [screen.matrix_world @ v.co for v in screen.data.vertices]
sx0, sx1 = min(p.x for p in pts), max(p.x for p in pts)
sz0, sz1 = min(p.z for p in pts), max(p.z for p in pts)
sy = sum(p.y for p in pts) / len(pts)
screen_width = sx1 - sx0
targets = [mathutils.Vector((sx0 + (sx1 - sx0) * i / 20, sy, sz0 + (sz1 - sz0) * j / 10)) for i in range(21) for j in range(11)]
print(f"PLACE | screen x {sx0:.2f}..{sx1:.2f} z {sz0:.2f}..{sz1:.2f} at y {sy:.2f}; {len(targets)} beam test rays")


def blocked_rays(lens):
    n = 0
    for t in targets:
        d = t - lens
        dist = d.length
        hit = tree.ray_cast(lens, d.normalized(), dist - 0.03)
        if hit[0] is not None:
            n += 1
    return n


candidates = []
for x, pole_y, frac, z_top in itertools.product((0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3), (-7.10, -7.03, -6.96),
                                                (0.2, 0.5, 0.8), (4.25, 4.20, 4.15)):
    # the body hangs below the panels; the pole is at `frac` of its depth measured from the back, so the lens (front face) is ahead of it
    back = pole_y - frac * BODY[1]
    front = back + BODY[1]
    lo = (x - BODY[0] / 2, back, z_top - BODY[2])
    hi = (x + BODY[0] / 2, front, z_top)
    if tree.overlap(box_tree(lo, hi, 0.03)):
        continue
    pole_lo = (x - POLE / 2, pole_y - POLE / 2, z_top)
    pole_hi = (x + POLE / 2, pole_y + POLE / 2, CEILING_Z - 0.04)       # it is meant to touch the ceiling: test just short of it
    if tree.overlap(box_tree(pole_lo, pole_hi, 0.015)):
        continue
    lens = mathutils.Vector((x, front + 0.005, z_top - BODY[2] / 2))
    bad = blocked_rays(lens)
    throw = abs(lens.y - sy)
    candidates.append((bad, abs(x), abs(throw / screen_width - 1.5), lens, lo, hi, pole_y, throw))

candidates.sort(key=lambda c: c[:3])
print(f"PLACE | {len(candidates)} placements where the body and the pole touch nothing")
for c in candidates[:6]:
    bad, ax, tr, lens, lo, hi, pole_y, throw = c
    print(f"PLACE |   blocked rays {bad:3d}  lens ({lens.x:+.2f}, {lens.y:+.2f}, {lens.z:.2f})  throw {throw:.2f} m = ratio {throw / screen_width:.2f}  pole y {pole_y:+.2f}")
clear = [c for c in candidates if c[0] == 0]
if not clear:
    print("PLACE | NO placement keeps the whole beam clear")
    sys.exit(0)
bad, ax, tr, lens, lo, hi, pole_y, throw = clear[0]
print(f"PLACE | CHOSEN lens {tuple(round(v, 3) for v in lens)} body {tuple(round(v, 3) for v in lo)}..{tuple(round(v, 3) for v in hi)} pole y {pole_y}")

# ---- build the stand-ins -------------------------------------------------------------------------------------------------
coll = bpy.data.collections.get("06_PROJECTOR") or bpy.data.collections.new("06_PROJECTOR")
if coll.name not in bpy.context.scene.collection.children:
    bpy.context.scene.collection.children.link(coll)
black = bpy.data.materials.get("Projector_Temp") or bpy.data.materials.new("Projector_Temp")
black.use_nodes = True
bsdf = black.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5


def add_box(name, lo, hi):
    verts = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    me.materials.append(black)
    coll.objects.link(ob)
    # outward normals: recalc from the box's centre
    bpy.context.view_layer.objects.active = ob
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return ob


body = add_box("Projector", lo, hi)
mount = add_box("Projector_Mount", (lo[0] + BODY[0] / 2 - POLE / 2, pole_y - POLE / 2, hi[2]),
                (lo[0] + BODY[0] / 2 + POLE / 2, pole_y + POLE / 2, CEILING_Z))
target = mathutils.Vector((0.0, sy, (sz0 + sz1) / 2))
empty = bpy.data.objects.new("Projector_Lens", None)
empty.empty_display_type = "SINGLE_ARROW"
empty.empty_display_size = 0.35
empty.location = lens
empty.rotation_euler = (target - lens).to_track_quat("Z", "Y").to_euler()      # the arrow (local +Z) points at the screen
coll.objects.link(empty)

# a wireframe of the beam (lens -> the screen's four corners): look at it in the viewport, then delete it
corners = [mathutils.Vector((x, sy, z)) for x in (sx0, sx1) for z in (sz0, sz1)]
verts = [tuple(lens)] + [tuple(c) for c in corners]
edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 4), (4, 3), (3, 1)]
gm = bpy.data.meshes.new("BeamGuide_TEMP")
gm.from_pydata(verts, edges, [])
guide = bpy.data.objects.new("BeamGuide_TEMP", gm)
guide.display_type = "WIRE"
guide.hide_render = True
coll.objects.link(guide)

bpy.ops.wm.save_as_mainfile(filepath=out_path, copy=False)
print("PLACE | saved", out_path)
