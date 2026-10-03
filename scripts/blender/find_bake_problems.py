"""Find the faces of the room that cannot bake properly, and select them so they can be seen and fixed.

Run it from inside Blender (Scripting workspace > Open > this file > Run Script) or headless:

    blender -b room_template.blend --python scripts/blender/find_bake_problems.py

A face bakes black when (a) its normal points into solid geometry (it is inside-out, or the object is mirrored and its
normals were never fixed), or (b) it is buried a few centimetres inside another surface (a trim piece sunk into a wall, the
inside of a console bin that is closed over, two coplanar faces). Cycles lights from the face's outward side, so in both
cases there is nothing for the light to reach and the result is black, whatever the material looks like.

What it does: for every face it casts a ray straight out along the normal.
  INSIDE   the ray leaves through the BACK of another face: the face points into solid. Fix: Edit Mode > select the face(s)
           > Mesh > Normals > Flip (or Recalculate Outside for a closed object).
  BURIED   the ray hits another face within 3 cm: the face is sunk into or pressed against something. Fix: move it out,
           delete the hidden duplicate, or open the closed-over cavity.
It prints a table per object and material, and SELECTS those faces (Edit Mode on the object shows them) without changing
the geometry. Large single planes (walls, floor, ceiling) are skipped: they are one-sided shells by design.
Nothing is saved; close the file without saving, or use the selection and fix as you like.
"""

from __future__ import annotations

import bpy
import mathutils
from mathutils.bvhtree import BVHTree

SKIP_OBJECTS = {"REF_Person_1.75m", "BiasGlow", "Screen", "icon_transparent"}
SKIP_FACES_LARGER_THAN = 3.0         # square metres: the room shell's own planes
#: The chair's leather, metal and the ceiling fabric are thick, overlapping shells full of hidden internal faces; those are
#: harmless (never seen, never lit) and would bury the real findings. Empty this set to see everything.
IGNORE_MATERIALS = {"Leather", "Metal", "soundbuffer", "Carpet"}
BURIED_DISTANCE = 0.03               # metres
PROBE_RANGE = 3.0


def world_mesh(obj, depsgraph):
    """World-space vertices and faces of the evaluated mesh, with the winding fixed for a mirrored (negative-scale) object."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    matrix = obj.matrix_world
    mirrored = matrix.to_3x3().determinant() < 0
    verts = [matrix @ v.co for v in mesh.vertices]
    faces = []
    for poly in mesh.polygons:
        ids = list(poly.vertices)
        faces.append((ids[::-1] if mirrored else ids, poly.material_index, poly.index))
    slots = [s.material.name if s.material else None for s in obj.material_slots]
    evaluated.to_mesh_clear()
    return verts, faces, slots


def newell(points):
    """Centre, unit normal and area of a polygon given its corner points."""
    centre = sum(points, mathutils.Vector()) / len(points)
    normal = mathutils.Vector()
    for i in range(len(points)):
        normal += points[i].cross(points[(i + 1) % len(points)])
    length = normal.length
    return centre, (normal / length if length else normal), length / 2


def main() -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in SKIP_OBJECTS
               and not o.name.startswith("BiasGlow") and not o.hide_render]       # glow planes are not part of the room
    all_verts, all_faces, owner = [], [], []
    per_object = {}
    for obj in objects:
        verts, faces, slots = world_mesh(obj, depsgraph)
        base = len(all_verts)
        all_verts += verts
        rows = []
        for ids, material_index, original_index in faces:
            all_faces.append([base + i for i in ids])
            rows.append((len(all_faces) - 1, material_index, original_index))
        per_object[obj.name] = (rows, slots)
    tree = BVHTree.FromPolygons([tuple(v) for v in all_verts], all_faces, all_triangles=False)

    def geometry(face_id):
        return newell([all_verts[i] for i in all_faces[face_id]])

    flagged = {}
    for name, (rows, slots) in per_object.items():
        obj = bpy.data.objects[name]
        for face_id, material_index, original_index in rows:
            centre, normal, area = geometry(face_id)
            if area < 1e-6 or area > SKIP_FACES_LARGER_THAN:
                continue
            hit = tree.ray_cast(centre + normal * 0.002, normal, PROBE_RANGE)
            if hit[0] is None:
                continue
            material = slots[material_index] if material_index < len(slots) else None
            if material in IGNORE_MATERIALS:
                continue
            kind = None
            if geometry(hit[2])[1].dot(normal) > 0:
                kind = "INSIDE"
            elif hit[3] < BURIED_DISTANCE:
                kind = "BURIED"
            if kind:
                entry = flagged.setdefault((name, material, kind), {"count": 0, "area": 0.0, "lo": [1e9] * 3, "hi": [-1e9] * 3})
                entry["count"] += 1
                entry["area"] += area
                for k in range(3):
                    entry["lo"][k], entry["hi"][k] = min(entry["lo"][k], centre[k]), max(entry["hi"][k], centre[k])
                obj.data.polygons[original_index].select = True         # shown when the object is put in Edit Mode

    print("\n=== faces that cannot bake properly ===")
    if not flagged:
        print("none found")
    for (name, material, kind), e in sorted(flagged.items(), key=lambda kv: -kv[1]["area"]):
        lo, hi = [round(v, 2) for v in e["lo"]], [round(v, 2) for v in e["hi"]]
        print(f"{kind:6} {name!r} / {material}: {e['count']} faces, {e['area']:.2f} m2, between {lo} and {hi}")
    print("Select the object, press Tab (Edit Mode): the problem faces are already selected.\n")


main()
