"""Rough sketch ("blockout") of a new VR room, in the same layout as the home theatre, for the owner to model on top of.

    blender -b --python make_blockout.py -- imax   docs/room-template/room_imax_blockout.blend
    blender -b --python make_blockout.py -- drivein docs/room-template/room_drivein_blockout.blend
(an existing file is never replaced unless --force follows the file name)

It starts from an empty scene and writes a NEW file; nothing existing is opened or overwritten. What it makes follows
docs/vr-room-spec.md and the home theatre's file: metres, Z up, the screen wall at y = 0 and the room reaching toward -Y,
the viewers looking toward +Y, the same numbered collections, and the names the app finds (Screen, Seat_*, BiasGlow*, Spawn,
Projector_Lens). The pieces are plain boxes and prisms so the proportions can be judged; every number that matters is a
constant at the top of its room so it can be changed and the file made again.
"""
import math
import sys

import bmesh
import bpy
from mathutils import Euler, Vector

# --------------------------------------------------------------------------------------------------------------------
# IMAX-style auditorium
# --------------------------------------------------------------------------------------------------------------------
IMAX = dict(
    # about the size of the largest US IMAX houses (roughly 90+ ft wide - adjust to taste): 1.9:1, the modern IMAX shape; the
    # picture is fitted inside the screen, never stretched
    screen_w=28.0, screen_h=14.7, screen_z0=2.2,
    rows=20, tread=1.2, riser=0.30, first_edge=16.0,      # stadium seating: 20 rows, the first tread starts 16 m from the screen
    half_width=24.0, ceiling_front=28.0, ceiling_back=15.0,
    eye_above_tread=1.15,                                  # eyes of a seated person, from the floor they sit on (spec: about 1.15 m)
)

# --------------------------------------------------------------------------------------------------------------------
# Open-air drive-in
# --------------------------------------------------------------------------------------------------------------------
DRIVEIN = dict(
    screen_w=15.0, screen_h=8.44, screen_z0=3.2,           # 16:9 on a freestanding tower
    board_pad_x=1.5, board_pad_z=0.8,                      # white board margin around the picture
    bands=(-22.0, -30.0),                                  # where the two parking ramps (berms) are centred, in y
    band_depth=8.0, band_rise=0.9, band_half_width=16.0,   # each ramp rises toward the screen, so a car looks up at the picture
    projector=(0.0, -60.5, 4.2),                           # the booth window at the back of the lot: where the beam starts
    sky_radius=500.0,
)


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.002, 0.002, 0.004, 1.0)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    return scene


def collections(scene, names):
    made = {}
    for name in names:
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
        made[name] = col
    return made


def material(name, colour, rough=0.8, emission_strength=None, emission_colour=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    mat.diffuse_color = (*colour, 1.0)                 # what the viewport and the quick previews show
    return mat


def screen_material():
    """Black; an Emission node (strength 0) that the bake script raises to light the room in the screen-only bake."""
    mat = bpy.data.materials.new("Screen")
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 0.0
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.01, 0.01, 0.012, 1.0)
    return mat


def mesh_object(name, verts, faces, col, mat=None, uvs=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    if uvs is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            for i, loop in enumerate(poly.loop_indices):
                layer.data[loop].uv = uvs[poly.vertices[i]]
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def fix_normals(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def box(name, centre, size, col, mat=None, rotation=(0.0, 0.0, 0.0)):
    """A box centred at ``centre`` (x, y, z), ``size`` = (x, y, z) in metres, turned by ``rotation`` (radians)."""
    sx, sy, sz = (s / 2 for s in size)
    local = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    turn = Euler(rotation).to_matrix()
    verts = [turn @ v + Vector(centre) for v in local]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    obj = mesh_object(name, verts, faces, col, mat)
    fix_normals(obj)
    return obj


def boxes(name, specs, col, mat=None):
    """Many boxes as ONE object. ``specs`` = [(centre, size, rotation)]."""
    verts, faces = [], []
    for centre, size, rotation in specs:
        sx, sy, sz = (s / 2 for s in size)
        turn = Euler(rotation).to_matrix()
        base = len(verts)
        verts += [turn @ Vector((x * sx, y * sy, z * sz)) + Vector(centre) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        faces += [tuple(base + i for i in f) for f in [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]]
    obj = mesh_object(name, verts, faces, col, mat)
    fix_normals(obj)
    return obj


def cylinder(name, centre, radius, length, col, mat=None, axis="X", segments=20):
    """A wheel-like cylinder lying along ``axis``."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=length)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.rotation_euler = {"X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0), "Z": (0, 0, 0)}[axis]
    obj.location = centre
    col.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def prism(name, profile_yz, x0, x1, col, mat=None):
    """A closed outline in (y, z) pushed along x from ``x0`` to ``x1``: stairs, wedges, sloped walls."""
    n = len(profile_yz)
    verts = [(x0, y, z) for y, z in profile_yz] + [(x1, y, z) for y, z in profile_yz]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    obj = mesh_object(name, verts, faces, col, mat)
    fix_normals(obj)
    return obj


def quad_xz(name, x0, x1, z0, z1, y, col, mat=None, uv=False):
    """A flat quad in the XZ plane at ``y``, facing -Y (toward the viewers), like the home theatre's Screen."""
    verts = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)] if uv else None
    return mesh_object(name, verts, [(0, 1, 2, 3)], col, mat, uvs)


def empty(name, location, col, label=None, kind="ARROWS", size=0.4):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type, obj.empty_display_size = kind, size
    obj.location = location
    if label:
        obj["label"] = label
    col.objects.link(obj)
    return obj


def light(name, kind, location, power, col, size=None, colour=(1.0, 0.93, 0.8)):
    data = bpy.data.lights.new(name, kind)
    data.energy, data.color = power, colour
    if kind == "AREA" and size:
        data.shape, data.size = "SQUARE", size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    col.objects.link(obj)
    return obj


def note(title, text):
    block = bpy.data.texts.new(title)
    block.write(text)


# --------------------------------------------------------------------------------------------------------------------
# the IMAX room
# --------------------------------------------------------------------------------------------------------------------
def build_imax(scene):
    p = IMAX
    cols = collections(scene, ["00_SHELL", "01_SCREEN", "02_SEATS", "03_LIGHTS_LIT", "04_LIGHTS_DIM", "05_REFERENCE"])
    wall = material("Blockout_Wall", (0.55, 0.55, 0.58))
    dark = material("Blockout_Floor", (0.08, 0.08, 0.09))
    hw, rows = p["half_width"], p["rows"]
    last_edge = p["first_edge"] + rows * p["tread"]
    back_y = -(last_edge + 1.5)
    ceil_at = lambda y: p["ceiling_front"] + (p["ceiling_back"] - p["ceiling_front"]) * (y / back_y)        # the ceiling's height at depth y

    # the floor: a flat pit in front of the screen, then one tread per row, each higher than the last
    profile = [(0.0, 0.0), (-p["first_edge"], 0.0)]
    for k in range(1, rows + 1):
        h = p["riser"] * k
        profile += [(-(p["first_edge"] + (k - 1) * p["tread"]), h), (-(p["first_edge"] + k * p["tread"]), h)]
    profile += [(back_y, p["riser"] * rows), (back_y, -0.3), (0.0, -0.3)]
    prism("Floor", profile, -hw, hw, cols["00_SHELL"], dark)

    # walls and ceiling: the front wall the screen hangs on, two side walls and the ceiling that slopes down toward the back
    box("Wall_Front", (0, 0.15, p["ceiling_front"] / 2), (2 * hw + 0.6, 0.3, p["ceiling_front"]), cols["00_SHELL"], wall)
    box("Wall_Back", (0, back_y - 0.15, p["ceiling_back"] / 2), (2 * hw + 0.6, 0.3, p["ceiling_back"]), cols["00_SHELL"], wall)
    side = [(0.0, 0.0), (back_y, 0.0), (back_y, p["ceiling_back"]), (0.0, p["ceiling_front"])]
    prism("Wall_Left", side, -hw - 0.3, -hw, cols["00_SHELL"], wall)
    prism("Wall_Right", side, hw, hw + 0.3, cols["00_SHELL"], wall)
    prism("Ceiling", [(0.0, p["ceiling_front"]), (back_y, p["ceiling_back"]), (back_y, p["ceiling_back"] + 0.3),
                      (0.0, p["ceiling_front"] + 0.3)], -hw - 0.3, hw + 0.3, cols["00_SHELL"], wall)
    ribs = []                                                          # modern wall: tall ribs along both side walls
    y = -2.0
    while y > back_y + 1.5:
        top = ceil_at(y) - 0.05
        for sign in (-1, 1):
            ribs.append(((sign * (hw - 0.3), y, top / 2), (0.6, 0.9, top), (0, 0, 0)))
        y -= 3.0
    boxes("Wall_Ribs", ribs, cols["00_SHELL"], wall)

    # the screen: 1.9:1, flat on the front wall, facing the seats; a glow plane behind it and along the floor, ceiling and sides
    w, h, z0 = p["screen_w"], p["screen_h"], p["screen_z0"]
    quad_xz("Screen", -w / 2, w / 2, z0, z0 + h, -0.04, cols["01_SCREEN"], screen_material(), uv=True)
    gx, gtop = w / 2 + 4.0, min(z0 + h + 4.0, p["ceiling_front"] - 0.3)
    quad_xz("BiasGlow", -gx, gx, 0.2, gtop, -0.01, cols["01_SCREEN"])
    pit = p["first_edge"] - 0.05
    mesh_object("BiasGlow.001", [(-gx, -pit, 0.01), (gx, -pit, 0.01), (gx, 0.0, 0.01), (-gx, 0.0, 0.01)], [(0, 1, 2, 3)], cols["01_SCREEN"])
    cy = -10.0
    mesh_object("BiasGlow.002", [(gx, cy, ceil_at(cy) - 0.01), (-gx, cy, ceil_at(cy) - 0.01),
                                 (-gx, 0.0, ceil_at(0) - 0.01), (gx, 0.0, ceil_at(0) - 0.01)], [(0, 1, 2, 3)], cols["01_SCREEN"])
    for name, sx in (("BiasGlow.003", -hw + 0.01), ("BiasGlow.004", hw - 0.01)):
        mesh_object(name, [(sx, cy, 0.01), (sx, 0.0, 0.01), (sx, 0.0, ceil_at(0) - 0.01), (sx, cy, ceil_at(cy) - 0.01)],
                    [(0, 1, 2, 3)], cols["01_SCREEN"])

    # the seats: ONE column down the middle, eyes of a seated person on each tread; the owner fills in the sides
    for k in range(1, rows + 1):
        y = -(p["first_edge"] + (k - 0.5) * p["tread"])
        top = p["riser"] * k
        empty(f"Seat_R{k}_Centre", (0.0, y, top + p["eye_above_tread"]), cols["02_SEATS"], label=f"Row {k}")
        box(f"REF_SeatSlot_R{k}", (0.0, y, top + 0.45), (0.55, 0.9, 0.9), cols["05_REFERENCE"]).display_type = "WIRE"
        bpy.data.objects[f"REF_SeatSlot_R{k}"].hide_render = True
    empty("Spawn", (0.0, -3.5, 0.0), cols["02_SEATS"], kind="PLAIN_AXES")
    person = box("REF_Person_1.75m", (-hw + 1.2, -(p["first_edge"] + 0.5 * p["tread"]), p["riser"] + 0.875), (0.5, 0.3, 1.75),
                 cols["05_REFERENCE"])
    person.hide_render = True

    # lights: ceiling panels for "lights up"; small step lights along the side walls for the film
    for i, y in enumerate((-8.0, -18.0, -28.0, -38.0)):
        for sx in (-12.0, 0.0, 12.0):
            light(f"Lit_Ceiling_{i}_{int(sx)}", "AREA", (sx, y, ceil_at(y) - 0.3), 14000, cols["03_LIGHTS_LIT"], size=6.0)
    for k in range(1, rows + 1):
        y = -(p["first_edge"] + (k - 0.5) * p["tread"])
        for sx in (-hw + 0.5, hw - 0.5):
            light(f"Dim_Step_{k}_{'L' if sx < 0 else 'R'}", "POINT", (sx, y, p["riser"] * k + 0.15), 20, cols["04_LIGHTS_DIM"],
                  colour=(1.0, 0.7, 0.4))
    note("README_BLOCKOUT", (
        "IMAX-style auditorium blockout (rough), the size of the biggest US houses. Screen wall at y=0, viewers toward -Y looking +Y.\n"
        f"Screen {p['screen_w']} x {p['screen_h']} m (1.9:1), bottom edge {p['screen_z0']} m. {rows} rows, tread {p['tread']} m, riser {p['riser']} m.\n"
        "Row 10 is the sweet spot (about 53 degrees of picture); row 1 is far too close, as in a real IMAX.\n"
        "Only the CENTRE seat of each row exists (Seat_R<k>_Centre); REF_SeatSlot_R<k> shows the space one seat takes. Copy left and right.\n"
        "BiasGlow..BiasGlow.004 are the five glow planes (front wall, floor, ceiling, two sides). Keep their names.\n"
        "A screen this wide limits how much BEHIND-the-screen depth the eyes can take (about 6 cm / width): the app needs its depth limit for it.\n"))


# --------------------------------------------------------------------------------------------------------------------
# the drive-in
# --------------------------------------------------------------------------------------------------------------------
CAR_COLOURS = [(0.55, 0.08, 0.06), (0.1, 0.25, 0.45), (0.75, 0.7, 0.55), (0.15, 0.35, 0.2), (0.6, 0.45, 0.1), (0.3, 0.3, 0.32)]


def ground_height(p, y):
    """How high the ground is at depth y: flat, except on the parking ramps, which rise toward the screen."""
    for centre in p["bands"]:
        lo, hi = centre - p["band_depth"] / 2, centre + p["band_depth"] / 2
        if lo <= y <= hi:
            return p["band_rise"] * (y - lo) / p["band_depth"]
    return 0.0


def chevy_truck(col, mats, origin, label_prefix, seat_col, seat_z):
    """A rough 1950s pickup (about 5.1 x 1.9 m) nose toward the screen. Returns its parts' names are all prefixed."""
    ox, oy, oz = origin
    n = label_prefix
    body, glass = mats["body"], mats["dark"]
    parts = [
        ((ox, oy + 1.2, oz + 0.75), (1.9, 1.5, 0.1), (0, 0, 0)),           # cab floor (the cab is open inside: model the interior)
        ((ox - 0.93, oy + 1.2, oz + 1.15), (0.08, 1.5, 0.8), (0, 0, 0)),   # door slabs
        ((ox + 0.93, oy + 1.2, oz + 1.15), (0.08, 1.5, 0.8), (0, 0, 0)),
        ((ox, oy + 0.45, oz + 1.35), (1.9, 0.08, 1.2), (0, 0, 0)),         # back panel behind the seat
        ((ox, oy + 1.2, oz + 1.9), (1.9, 1.7, 0.08), (0, 0, 0)),           # roof
        ((ox - 0.9, oy + 1.95, oz + 1.4), (0.1, 0.1, 1.0), (0, 0, 0)),     # windscreen pillars and header (no glass)
        ((ox + 0.9, oy + 1.95, oz + 1.4), (0.1, 0.1, 1.0), (0, 0, 0)),
        ((ox, oy + 1.95, oz + 1.9), (1.9, 0.1, 0.1), (0, 0, 0)),
        ((ox, oy + 1.9, oz + 1.05), (1.9, 0.3, 0.6), (0, 0, 0)),           # dashboard
        ((ox, oy + 0.8, oz + 1.0), (1.8, 0.45, 0.4), (0, 0, 0)),           # bench seat: back ...
        ((ox, oy + 1.25, oz + 0.9), (1.8, 0.7, 0.2), (0, 0, 0)),           # ... and cushion
        ((ox, oy + 2.75, oz + 0.9), (1.85, 1.3, 0.6), (0, 0, 0)),          # hood
        ((ox, oy - 0.9, oz + 0.75), (1.9, 2.0, 0.15), (0, 0, 0)),          # bed floor
        ((ox - 0.9, oy - 0.9, oz + 1.05), (0.1, 2.0, 0.55), (0, 0, 0)),    # bed sides
        ((ox + 0.9, oy - 0.9, oz + 1.05), (0.1, 2.0, 0.55), (0, 0, 0)),
        ((ox, oy - 1.9, oz + 1.05), (1.9, 0.1, 0.55), (0, 0, 0)),          # tailgate
    ]
    boxes(f"{n}_Body", parts, col, body)
    for i, (wx, wy) in enumerate(((-0.85, 2.4), (0.85, 2.4), (-0.85, -1.2), (0.85, -1.2))):
        cylinder(f"{n}_Wheel{i}", (ox + wx, oy + wy, oz + 0.38), 0.38, 0.25, col, glass, axis="X")
    # the seats: driver on the left, passenger on the right, eyes about 1.55 m above the ground
    empty(f"Seat_{n}_Driver", (ox - 0.4, oy + 1.2, seat_z), seat_col, label="Chevy truck - driver")
    empty(f"Seat_{n}_Passenger", (ox + 0.4, oy + 1.2, seat_z), seat_col, label="Chevy truck - passenger")


def cadillac(col, mats, origin, label_prefix, seat_col, seat_z):
    """A rough 1950s Cadillac convertible (about 5.7 x 2.0 m), top down, nose toward the screen."""
    ox, oy, oz = origin
    n = label_prefix
    parts = [
        ((ox, oy + 1.85, oz + 0.7), (2.0, 2.0, 0.7), (0, 0, 0)),           # hood
        ((ox, oy - 2.0, oz + 0.7), (2.0, 1.7, 0.7), (0, 0, 0)),            # trunk
        ((ox - 0.95, oy - 0.15, oz + 0.7), (0.1, 2.0, 0.7), (0, 0, 0)),    # doors (the cabin between them is open: model the interior)
        ((ox + 0.95, oy - 0.15, oz + 0.7), (0.1, 2.0, 0.7), (0, 0, 0)),
        ((ox, oy - 0.15, oz + 0.45), (1.9, 2.0, 0.1), (0, 0, 0)),          # floor
        ((ox, oy + 0.3, oz + 0.62), (1.8, 0.8, 0.3), (0, 0, 0)),           # front bench seat: cushion ...
        ((ox, oy - 0.35, oz + 1.0), (1.8, 0.18, 0.55), (0, 0, 0)),         # ... and back
        ((ox, oy + 0.85, oz + 1.05), (1.9, 0.3, 0.4), (0, 0, 0)),          # dashboard
        ((ox - 0.9, oy + 0.95, oz + 1.3), (0.06, 0.06, 0.5), (0, 0, 0)),   # windscreen pillars (top down, no header)
        ((ox + 0.9, oy + 0.95, oz + 1.3), (0.06, 0.06, 0.5), (0, 0, 0)),
        ((ox - 0.9, oy - 2.6, oz + 1.2), (0.12, 0.6, 0.6), (0, 0, 0)),     # the tail fins
        ((ox + 0.9, oy - 2.6, oz + 1.2), (0.12, 0.6, 0.6), (0, 0, 0)),
    ]
    boxes(f"{n}_Body", parts, col, mats["body2"])
    for i, (wx, wy) in enumerate(((-0.9, 1.8), (0.9, 1.8), (-0.9, -1.8), (0.9, -1.8))):
        cylinder(f"{n}_Wheel{i}", (ox + wx, oy + wy, oz + 0.36), 0.36, 0.25, col, mats["dark"], axis="X")
    empty(f"Seat_{n}_Driver", (ox - 0.45, oy + 0.2, seat_z), seat_col, label="Cadillac - driver")
    empty(f"Seat_{n}_Passenger", (ox + 0.45, oy + 0.2, seat_z), seat_col, label="Cadillac - passenger")


def build_drivein(scene):
    p = DRIVEIN
    cols = collections(scene, ["00_SHELL", "01_SCREEN", "02_SEATS", "03_LIGHTS_LIT", "04_LIGHTS_DIM", "05_REFERENCE", "06_PROJECTOR",
                               "07_CARS"])
    gravel = material("Blockout_Ground", (0.12, 0.11, 0.09), rough=1.0)
    white = material("Blockout_ScreenBoard", (0.8, 0.8, 0.78))
    steel = material("Blockout_Steel", (0.2, 0.2, 0.22))
    mats = {"body": material("Car_Truck", (0.1, 0.35, 0.4)), "body2": material("Car_Cadillac", (0.75, 0.3, 0.35)),
            "dark": material("Car_Dark", (0.02, 0.02, 0.02))}
    w, h, z0 = p["screen_w"], p["screen_h"], p["screen_z0"]

    # ground and the two parking ramps
    quad = [(-300, -300, 0), (300, -300, 0), (300, 300, 0), (-300, 300, 0)]
    mesh_object("Ground", [(x, y, z) for x, y, z in quad], [(0, 1, 2, 3)], cols["00_SHELL"], gravel)
    bpy.data.objects["Ground"].data.polygons[0].flip()
    for i, centre in enumerate(p["bands"]):
        lo, hi = centre - p["band_depth"] / 2, centre + p["band_depth"] / 2
        prism(f"Ramp_{i}", [(lo, 0.0), (hi, 0.0), (hi, p["band_rise"])], -p["band_half_width"], p["band_half_width"], cols["00_SHELL"], gravel)

    # the screen tower: a white board a little bigger than the picture, on two steel legs with braces
    bw, bh = w + 2 * p["board_pad_x"], h + 2 * p["board_pad_z"]
    bz0 = z0 - p["board_pad_z"]
    box("Screen_Board", (0, 0.3, bz0 + bh / 2), (bw, 0.6, bh), cols["00_SHELL"], white)
    legs = [((sx * (bw / 2 - 1.2), 1.0, bz0 / 2), (0.7, 0.7, bz0), (0, 0, 0)) for sx in (-1, 1)]
    for sx in (-1, 1):
        for sy in (1.0, 6.0):
            legs.append(((sx * (bw / 2 - 1.2), 1.0 + sy / 2 + 0.3, bz0 / 2 + bh * 0.2),
                         (0.4, 0.4, bh * 0.55), (math.atan2(sy, bh * 0.55) * (1 if sy > 0 else -1), 0, 0)))
    boxes("Screen_Legs", legs, cols["00_SHELL"], steel)

    # the screen and its glow planes (the picture area in front of the board, the ground before it)
    quad_xz("Screen", -w / 2, w / 2, z0, z0 + h, -0.04, cols["01_SCREEN"], screen_material(), uv=True)
    quad_xz("BiasGlow", -(bw / 2 - 0.1), bw / 2 - 0.1, bz0 + 0.1, bz0 + bh - 0.1, -0.01, cols["01_SCREEN"])
    gx = bw / 2 + 2.0
    mesh_object("BiasGlow.001", [(-gx, -10.0, 0.02), (gx, -10.0, 0.02), (gx, 0.0, 0.02), (-gx, 0.0, 0.02)], [(0, 1, 2, 3)], cols["01_SCREEN"])

    # the concession stand and projection booth at the back of the lot, and the beam's start
    px, py, pz = p["projector"]
    box("Concession", (0, py - 6.0, 2.0), (14.0, 12.0, 4.0), cols["00_SHELL"], white)
    box("Booth_Window_Frame", (px, py + 0.0, pz), (1.4, 0.3, 0.9), cols["06_PROJECTOR"], steel)
    empty("Projector_Lens", (px, py + 0.2, pz), cols["06_PROJECTOR"], kind="SINGLE_ARROW", size=1.0)

    # two locations: a Chevy pickup and a Cadillac, each on its own ramp, with two seats (the people's eyes)
    truck_y, cad_y = p["bands"][0] - 1.0, p["bands"][1] - 1.0
    tg, cg = ground_height(p, truck_y), ground_height(p, cad_y)
    chevy_truck(cols["07_CARS"], mats, (-3.5, truck_y, tg), "Truck", cols["02_SEATS"], tg + 1.55)
    cadillac(cols["07_CARS"], mats, (3.8, cad_y, cg), "Cadillac", cols["02_SEATS"], cg + 1.38)
    # the drive-in speakers: a pole and a box on the window of each car, and an empty where the sound would come from (later)
    for name, car_x, car_y, ground, side, z in (("Truck", -3.5, truck_y, tg, 1, 1.3), ("Cadillac", 3.8, cad_y, cg, -1, 1.1)):
        x = car_x + side * 1.35
        boxes(f"Speaker_{name}", [((x, car_y + 0.6, ground + 0.9), (0.08, 0.08, 1.8), (0, 0, 0)),
                                  ((x - side * 0.1, car_y + 0.6, ground + z), (0.22, 0.12, 0.28), (0, 0, 0))], cols["07_CARS"], steel)
        empty(f"AudioSource_Speaker_{name}", (x - side * 0.1, car_y + 0.6, ground + z), cols["05_REFERENCE"], kind="SPHERE", size=0.15)

    # filler cars on the ramps (just boxes: model or replace)
    fillers = [(-9.0, p["bands"][0] - 0.5, 0.05), (9.5, p["bands"][0] - 1.5, -0.04), (-12.0, p["bands"][1], 0.0),
               (-4.0, p["bands"][1] - 1.5, 0.06), (11.0, p["bands"][1] + 0.5, -0.05), (1.5, p["bands"][0] + 0.3, 0.02)]
    for i, (x, y, yaw) in enumerate(fillers):
        g = ground_height(p, y)
        colour = material(f"Car_Filler_{i}", CAR_COLOURS[i % len(CAR_COLOURS)])
        boxes(f"Car_Filler_{i}", [((x, y, g + 0.75), (1.9, 4.8, 0.8), (0, 0, yaw)), ((x, y - 0.3, g + 1.35), (1.7, 2.4, 0.5), (0, 0, yaw))],
              cols["07_CARS"], colour)
        for j, (wx, wy) in enumerate(((-0.85, 1.5), (0.85, 1.5), (-0.85, -1.5), (0.85, -1.5))):
            cylinder(f"Car_Filler_{i}_Wheel{j}", (x + wx, y + wy, g + 0.35), 0.35, 0.25, cols["07_CARS"], mats["dark"], axis="X")
    empty("Spawn", (0.0, -14.0, 0.0), cols["02_SEATS"], kind="PLAIN_AXES")

    # the sky: a big dome with a material ready for the HDRI (assign the image in the node), seen from inside
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=p["sky_radius"])
    for face in bm.faces:
        face.normal_flip()
    mesh = bpy.data.meshes.new("Sky_Dome")
    bm.to_mesh(mesh)
    bm.free()
    sky = bpy.data.objects.new("Sky_Dome", mesh)
    cols["00_SHELL"].objects.link(sky)
    sky_mat = bpy.data.materials.new("Sky_HDRI")
    sky_mat.diffuse_color = (0.02, 0.03, 0.08, 1.0)
    sky_mat.use_nodes = True
    tree = sky_mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    image = tree.nodes.new("ShaderNodeTexImage")
    image.label = "Assign the night HDRI here (equirectangular)"
    emission = tree.nodes.new("ShaderNodeEmission")
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(image.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    sky.data.materials.append(sky_mat)

    # lights: lamp poles on for "lights up"; a couple of small ones for the film
    for i, (x, y) in enumerate(((-16, -12), (16, -12), (-16, -40), (16, -40), (-16, -55), (16, -55))):
        box(f"LampPole_{i}", (x, y, 3.0), (0.2, 0.2, 6.0), cols["00_SHELL"], steel)
        light(f"Lit_Lamp_{i}", "POINT", (x, y, 6.1), 6000, cols["03_LIGHTS_LIT"], colour=(1.0, 0.75, 0.45))
    for i, x in enumerate((-4.0, 4.0)):
        light(f"Dim_Concession_{i}", "POINT", (x, p["projector"][1] - 0.2, 3.2), 40, cols["04_LIGHTS_DIM"], colour=(1.0, 0.8, 0.5))
    person = box("REF_Person_1.75m", (6.0, -16.0, 0.875), (0.5, 0.3, 1.75), cols["05_REFERENCE"])
    person.hide_render = True
    note("README_BLOCKOUT", (
        "Open-air drive-in blockout (rough). Screen wall/tower at y=0, viewers toward -Y looking +Y.\n"
        f"Screen {p['screen_w']} x {p['screen_h']} m (16:9) on a tower; projector booth at the back (Projector_Lens, beam 60 m).\n"
        "Two locations: Seat_Truck_Driver/Passenger (Chevy pickup, ramp 1) and Seat_Cadillac_Driver/Passenger (ramp 2).\n"
        "Sky_Dome: assign the equirectangular night HDRI in the Sky_HDRI material's image node.\n"
        "AudioSource_Speaker_* mark the window speakers (for the tinny-speaker audio, later).\n"
        "Keep BiasGlow and BiasGlow.001 names (screen-board glow and the ground before the screen).\n"))


# --------------------------------------------------------------------------------------------------------------------
def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    room, out = argv[0], argv[1]
    import os
    if os.path.exists(out) and "--force" not in argv:             # the owner edits these files: never replace one by accident
        raise SystemExit(f"{out} already exists; it may hold your work. Pass --force after the file name to replace it.")
    scene = reset()
    {"imax": build_imax, "drivein": build_drivein}[room](scene)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"BLOCKOUT | wrote {out}: {len(bpy.data.objects)} objects")


main()
