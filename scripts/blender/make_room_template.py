"""Builds the VR room TEMPLATE in Blender: a bare shell at the right size with the named objects the app looks for.

Run it with Blender (any version from 3.3 up):

    blender --background --factory-startup --python scripts/blender/make_room_template.py -- <out_dir>

It writes ``room_template.blend`` and ``room_template.glb`` (the GLB is what the app reads; the .blend is for you to
build on). Nothing here is the final room: copy the shapes, keep the NAMES, replace the flat panels with real meshes.

Axes (Blender, Z up):
  * The room runs from the screen wall at y = 0 backwards to y = -9.5. The viewers sit at negative y and LOOK TOWARD +Y.
    (Blender's glTF exporter turns Blender +Y into glTF -Z, which is the direction the app looks.)
  * An Empty with ZERO rotation already looks at the screen: its green Y arrow is its "forward".
  * x = 0 is the middle of the room; +x is the viewer's right.
Sizes are metres. Screen: 120 in, 16:9.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# --- the numbers (change them here, run again) ---------------------------------------------------------
ROOM_W, ROOM_D, ROOM_H = 7.0, 9.5, 3.6                 # interior: x -3.5..3.5, y -9.5..0, z 0..3.6
DIAGONAL_IN = 120.0
SCREEN_W = DIAGONAL_IN * 0.0254 * 16 / math.hypot(16, 9)    # 2.657 m
SCREEN_H = SCREEN_W * 9 / 16                                # 1.494 m
SCREEN_BOTTOM = 0.70                                        # floor to the picture's lower edge
SEAT_EYE = 1.15                                             # a seated person's eyes above the floor
EPS = 0.02                                                  # the screen floats this far off the wall

README = """VR ROOM TEMPLATE  -  Ultimate Video to 3D Studio

THE ONE RULE: the app finds things by NAME. Keep these names; replace the shapes however you like.

AXES  The screen wall is at y = 0. Viewers sit at negative y and look toward +Y. An Empty with zero rotation
already looks at the screen (its green Y arrow). x = 0 is the middle; +x is the viewer's right. Metres.

COLLECTIONS
  00_SHELL      Floor, Ceiling, Wall_*: flat single quads at the real room size. REPLACE with your real room mesh.
                (Keep it ONE material and ONE texture atlas. See docs/vr-room-spec.md.)
  01_SCREEN     Screen (120 in, 16:9, normal toward the seats, UV 0..1), BiasGlow (halo quad on the wall behind
                the screen), FloorMirror (the floor patch that reflects the screen).
  02_SEATS      Seat_* Empties at EYE height, forward (green Y) at the screen. Copy one, rename, move. Spawn = standing start.
  03_LIGHTS_LIT   room lights ON  (bake 1: Lit)
  04_LIGHTS_DIM   movie lighting  (bake 2: Dim)
  05_REFERENCE  scale people, a centre camera. NOT exported.

THE THREE BAKES (same UVs, same size)
  1 Lit    : collection 03 on, 04 off, Screen_Emit strength 0.
  2 Dim    : collection 04 on, 03 off, Screen_Emit strength 0.
  3 Screen : 03 and 04 OFF, Screen_Emit strength = the custom property bake3_strength on the Screen (about 8).

EXPORT  Select everything in 00, 01 and 02 (NOT lights, NOT the reference collection) and use File > Export > glTF 2.0:
  Format GLB, +Y Up, Include: Selected Objects + Custom Properties, no Cameras, no Lights, Apply Modifiers.
"""


def collection(name: str, parent=None, hide_render: bool = False):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    col.hide_render = hide_render
    return col


def material(name: str, colour: tuple[float, float, float, float]):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    if node is not None:
        node.inputs["Base Color"].default_value = colour
    mat.diffuse_color = colour
    return mat


def emission_material(name: str):
    """White emitter with a Strength of 0: raised to bake 3 (the screen-only bake)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emit.inputs["Strength"].default_value = 0.0
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.02, 0.02, 0.02, 1.0)
    return mat


def quad(name: str, corners: list[Vector], normal: Vector, col, mat=None, uv=None):
    """One flat panel. ``corners`` in any order that goes round it; the winding is fixed so the normal points ``normal``."""
    verts = [tuple(c) for c in corners]
    face = (0, 1, 2, 3)
    a, b, c = (Vector(verts[i]) for i in (0, 1, 2))
    if (b - a).cross(c - b).dot(normal) < 0:                 # wound the wrong way round: reverse
        verts = verts[::-1]
        uv = None if uv is None else uv[::-1]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [face])
    mesh.update()
    layer = mesh.uv_layers.new(name="UVMap")
    coords = uv or [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for loop_index, vertex_index in enumerate(mesh.polygons[0].vertices):
        layer.data[loop_index].uv = coords[vertex_index]
    if mat is not None:
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    assert mesh.polygons[0].normal.dot(normal) > 0.99, f"{name}: normal points the wrong way"
    return obj


def empty(name: str, location, col, display="ARROWS", size=0.25, label: str | None = None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    obj.location = location
    if label is not None:
        obj["label"] = label                                  # exported to glTF "extras"
    col.objects.link(obj)
    return obj


def light(name: str, kind: str, location, col, energy: float, colour=(1.0, 0.85, 0.65), **kw):
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = colour
    if kind == "SPOT":
        data.spot_size = math.radians(kw.get("cone", 70))
        data.spot_blend = 0.6
    if kind == "AREA":
        data.shape = "RECTANGLE"
        data.size, data.size_y = kw.get("size", (0.3, 0.3))
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    obj.rotation_euler = kw.get("rotation", (0.0, 0.0, 0.0))
    col.objects.link(obj)
    return obj


def build() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.name = "VR Room Template"

    shell = collection("00_SHELL")
    screen_col = collection("01_SCREEN")
    seats = collection("02_SEATS")
    lit = collection("03_LIGHTS_LIT")
    dim = collection("04_LIGHTS_DIM")
    ref = collection("05_REFERENCE", hide_render=True)

    w2, d, h = ROOM_W / 2, ROOM_D, ROOM_H
    atlas = material("Room_Baked", (0.55, 0.55, 0.58, 1.0))          # ONE material for the whole baked room

    # --- the shell: six flat panels, normals pointing INTO the room ------------------------------------
    quad("Floor", [Vector(p) for p in ((-w2, -d, 0), (w2, -d, 0), (w2, 0, 0), (-w2, 0, 0))], Vector((0, 0, 1)), shell, atlas)
    quad("Ceiling", [Vector(p) for p in ((-w2, -d, h), (w2, -d, h), (w2, 0, h), (-w2, 0, h))], Vector((0, 0, -1)), shell, atlas)
    quad("Wall_Front", [Vector(p) for p in ((-w2, 0, 0), (w2, 0, 0), (w2, 0, h), (-w2, 0, h))], Vector((0, -1, 0)), shell, atlas)
    quad("Wall_Back", [Vector(p) for p in ((-w2, -d, 0), (w2, -d, 0), (w2, -d, h), (-w2, -d, h))], Vector((0, 1, 0)), shell, atlas)
    quad("Wall_Left", [Vector(p) for p in ((-w2, -d, 0), (-w2, 0, 0), (-w2, 0, h), (-w2, -d, h))], Vector((1, 0, 0)), shell, atlas)
    quad("Wall_Right", [Vector(p) for p in ((w2, -d, 0), (w2, 0, 0), (w2, 0, h), (w2, -d, h))], Vector((-1, 0, 0)), shell, atlas)

    # --- the screen and the things that belong to it ---------------------------------------------------
    zc = SCREEN_BOTTOM + SCREEN_H / 2
    sw = SCREEN_W / 2
    y = -EPS
    screen = quad(
        "Screen",
        [Vector(p) for p in ((-sw, y, SCREEN_BOTTOM + SCREEN_H), (sw, y, SCREEN_BOTTOM + SCREEN_H),
                             (sw, y, SCREEN_BOTTOM), (-sw, y, SCREEN_BOTTOM))],      # top-left, top-right, bottom-right, bottom-left
        Vector((0, -1, 0)), screen_col, emission_material("Screen_Emit"),
        uv=[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
    screen["bake3_strength"] = 8.0
    screen["diagonal_in"] = DIAGONAL_IN
    glow_w, glow_h = SCREEN_W + 0.9, SCREEN_H + 0.7
    quad("BiasGlow",
         [Vector(p) for p in ((-glow_w / 2, -0.005, zc + glow_h / 2), (glow_w / 2, -0.005, zc + glow_h / 2),
                              (glow_w / 2, -0.005, zc - glow_h / 2), (-glow_w / 2, -0.005, zc - glow_h / 2))],
         Vector((0, -1, 0)), screen_col, material("BiasGlow_Mat", (0.1, 0.1, 0.1, 1.0)),
         uv=[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
    quad("FloorMirror", [Vector(p) for p in ((-3.0, -3.4, 0.003), (3.0, -3.4, 0.003), (3.0, -0.02, 0.003), (-3.0, -0.02, 0.003))],
         Vector((0, 0, 1)), screen_col, material("FloorMirror_Mat", (0.05, 0.05, 0.06, 1.0)))

    # --- seats: Empties at EYE height, zero rotation = looking at the screen ---------------------------
    for name, x, yy, z, label in (
        ("Seat_Centre", 0.0, -4.2, SEAT_EYE, "Centre"),
        ("Seat_Left", -1.35, -4.2, SEAT_EYE, "Left"),
        ("Seat_Right", 1.35, -4.2, SEAT_EYE, "Right"),
        ("Seat_Close", 0.0, -3.1, SEAT_EYE, "Close (immersive)"),
        ("Seat_Back_Riser", 0.0, -6.0, SEAT_EYE + 0.35, "Back row (raised)"),
    ):
        empty(name, (x, yy, z), seats, "ARROWS", 0.3, label)
    empty("Spawn", (0.0, -4.2, 0.0), seats, "PLAIN_AXES", 0.3, "Standing start")

    # --- lights: what to bake. Lit = room on; Dim = film time. They are NOT exported. ----------------
    for i, (lx, ly) in enumerate(((-1.8, -2.2), (1.8, -2.2), (-1.8, -5.0), (1.8, -5.0), (-1.8, -7.8), (1.8, -7.8)), 1):
        light(f"Light_Down_{i:02d}", "SPOT", (lx, ly, h - 0.05), lit, 500.0, cone=70)
    light("Light_Wash_L", "AREA", (-w2 + 0.1, -4.5, 2.0), lit, 120.0, size=(0.2, 2.0), rotation=(0.0, math.radians(-90), 0.0))
    light("Light_Wash_R", "AREA", (w2 - 0.1, -4.5, 2.0), lit, 120.0, size=(0.2, 2.0), rotation=(0.0, math.radians(90), 0.0))
    for i, (lx, ly) in enumerate(((-3.35, -3.0), (-3.35, -5.5), (3.35, -3.0), (3.35, -5.5)), 1):
        light(f"Light_Step_{i:02d}", "AREA", (lx, ly, 0.15), dim, 6.0, colour=(1.0, 0.7, 0.45), size=(0.4, 0.05),
              rotation=(0.0, math.radians(-90 if lx < 0 else 90), 0.0))
    light("Light_Cove_Dim", "AREA", (0.0, -0.4, h - 0.1), dim, 15.0, colour=(1.0, 0.75, 0.5), size=(5.0, 0.08),
          rotation=(math.radians(180), 0.0, 0.0))

    # --- reference (not exported) ----------------------------------------------------------------------
    person = bpy.data.objects.new("REF_Person_1.75m", bpy.data.meshes.new("REF_Person"))
    person.data.from_pydata(
        [(-0.25, -0.15, 0.0), (0.25, -0.15, 0.0), (0.25, 0.15, 0.0), (-0.25, 0.15, 0.0),
         (-0.25, -0.15, 1.75), (0.25, -0.15, 1.75), (0.25, 0.15, 1.75), (-0.25, 0.15, 1.75)],
        [], [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
    person.data.update()
    person.location = (2.4, -2.0, 0.0)
    ref.objects.link(person)
    cam_data = bpy.data.cameras.new("Cam_Centre")
    cam_data.lens = 18.0
    cam = bpy.data.objects.new("Cam_Centre", cam_data)
    cam.location = (0.0, -4.2, SEAT_EYE)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)           # a camera looks along -Z; turned 90 degrees about X it looks +Y
    ref.objects.link(cam)
    scene.camera = cam

    text = bpy.data.texts.new("README_ROOM_TEMPLATE")
    text.write(README)


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "room_template.blend"))
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    chosen = [o for name in ("00_SHELL", "01_SCREEN", "02_SEATS") for o in bpy.data.collections[name].objects]
    for obj in chosen:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = chosen[0]
    kwargs = dict(filepath=str(out_dir / "room_template.glb"), export_format="GLB", use_selection=True, export_yup=True,
                  export_apply=True)
    for extra in ({"export_extras": True, "export_lights": False, "export_cameras": False},
                  {"export_extras": True}, {}):
        try:
            bpy.ops.export_scene.gltf(**kwargs, **extra)
            break
        except TypeError:
            continue


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build()
    export(Path(argv[0]) if argv else Path.cwd())
    print("room template written")
