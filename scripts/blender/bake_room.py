"""Bakes the owner's finished room into texture atlases and exports the GLB the VR player loads.

    blender --background --factory-startup <room.blend> --python scripts/blender/bake_room.py -- <out_dir> [options]

Options:  --phase bake|export   "bake" (default): unwrap and bake the groups named by --only; "export": build and write room.glb
          --samples N      Cycles samples per bake (default 1024)
          --only a,b       bake only these groups
          --states lit,dim,screen    which of the three bakes to do
          --scale F        multiply every atlas size by F (0.25 for a quick test)

The work is split so each run is short: a group's UV layout is written to ``uv_<group>.npz`` the first time it is made
and reused by every later run (the unwrap is not repeatable run to run, and the three bakes plus the export must all
share one layout). A full job is therefore one ``bake`` run per group (or per state for the heavy chair group), then one
``export`` run. ``bake_report.json`` accumulates the results of all runs.

It reads the .blend and NEVER saves it. For each group of objects it makes a fresh UV layout ("BakeUV", equal texel
density across the whole group), bakes the room three ways with Cycles and writes JPEGs:
    <group>_lit.jpg     the room lights ON                      (this one is embedded in the GLB)
    <group>_dim.jpg     the lights as during a film
    <group>_screen.jpg  ONLY the screen lighting the room, exposure-normalised (see room.json: screen_ev)
then replaces every baked object's materials with one baked material per group and exports room.glb with the app's
named objects (Screen, BiasGlow, Seat_*, Spawn). Run with the same Blender version that saved the file.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np

# --- what to bake together ----------------------------------------------------------------------------
# size: edge of the square atlas for the lit and dim bakes. The screen-only bake is smoother and uses a quarter of
# the pixels (half the edge). Group choice balances texel density (pixels per metre) so no atlas is stretched thin.
GROUPS: dict[str, dict] = {
    "wall_left": {"objects": ["Wall_Right_Left"], "size": 4096},       # the side walls + stone; renamed from Wall_Left in the owner's file
    "walls_plaster": {"objects": ["Wall_Back", "Wall_Front", "Door2"], "size": 4096},
    "floor_carpet": {"objects": ["Floor"], "size": 4096},
    # Cylinder: the black rods that hold the panels up. Projector + Projector_Mount: the ceiling-hung projector (the
    # Projector_Lens empty, where the beam starts, is not geometry and is not baked)
    "ceiling_panels": {"objects": ["Ceiling", "AcusticPanels", "Cylinder", "Projector", "Projector_Mount"], "size": 4096},
    "chairs_details": {"objects": ["Home Theater Armchair N220224.008", "Text.001", "icon_transparent"], "size": 4096},
    # The hallway seen through the doors: its floor and the long wood wall at the end. It was 84 px per metre (a 2048 atlas
    # for a 24 m strip) against ~340 in the room; a 6144 atlas of just these two objects brings it level with the room.
    "hall": {"objects": ["Floor.001", "Wall_Back.002"], "size": 6144},
}
EXPORT_SKIP = ("icon_transparent",)          # baked, then composited onto the wall (composite_logo.py); not exported as geometry
NOT_BAKED_HIDDEN = ("BiasGlow",)            # drawn live by the app: must not paint the wall behind the screen
GLOW_PREFIX = "BiasGlow"                    # ...and so is every object whose name starts with it (BiasGlow.001, BiasGlow_Floor...)
LIGHT_COLLECTIONS = ("03_LIGHTS_LIT", "04_LIGHTS_DIM")
SCREEN_STRENGTH = 8.0                       # the Screen's emission for the screen-only bake (matches its bake3_strength)
MARGIN_PX = 16


#: Meshes that are deliberately not baked: the glow planes and the screen are drawn live, the reference figure is hidden, and
#: the second ceiling is never seen.
NOT_BAKED_OK = ("Screen", "REF_Person_1.75m", "Ceiling.001", "ProjectorLensGlow")
#: Hidden during every bake: ProjectorLensGlow is a bright emissive quad that only belongs while a film runs (the app draws
#: the beam and the lens glow live), so it must neither be baked nor light the room in the bake.
HIDDEN_IN_BAKE = ("ProjectorLensGlow", "REF_Person_1.75m")


def uncovered_meshes() -> list[str]:
    """Visible meshes that no group bakes. They would be exported with their raw material and show up black (the owner's
    first bake lost things this way), so a bake refuses to start until each is in a group or on the not-baked list."""
    grouped = {name for spec in GROUPS.values() for name in spec["objects"]}
    return sorted(o.name for o in bpy.data.objects
                  if o.type == "MESH" and not o.hide_render and o.name not in grouped
                  and o.name not in NOT_BAKED_OK and not o.name.startswith(GLOW_PREFIX))


def missing_objects() -> list[str]:
    """Names listed in GROUPS that the file does not have (an object renamed or deleted since the groups were written)."""
    return sorted({name for spec in GROUPS.values() for name in spec["objects"] if name not in bpy.data.objects})


def deselect_all() -> None:
    """Clear the selection object by object (the operator needs a 3D-view context that a background run lacks)."""
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)


def ensure_object_mode() -> None:
    """Bring every object out of Edit Mode. The owner's file was saved with AcusticPanels still in it, and in that mode
    Blender's transform-apply (and much else) refuses to run ("context is incorrect")."""
    for obj in bpy.data.objects:
        if obj.mode != "OBJECT":
            deselect_all()
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="OBJECT")
            log(f"{obj.name}: was in {obj.mode if obj.mode != 'OBJECT' else 'edit'} mode, switched to object mode")
    deselect_all()


def log(*parts) -> None:
    print("BAKE |", *parts, flush=True)


def parse(argv: list[str]) -> dict:
    # absolute: Blender resolves a relative image path from the .blend's folder, not from where it was started, and the export
    # then cannot find the baked images that the bake runs wrote relative to the working folder
    out = {"dir": Path(argv[0]).resolve(), "samples": 1024, "only": None, "states": ["lit", "dim", "screen"], "scale": 1.0,
           "phase": "bake"}
    i = 1
    while i < len(argv):
        key = argv[i]
        if key == "--allow-unbaked":                  # a flag, not a pair
            i += 1
            continue
        if key == "--samples":
            out["samples"] = int(argv[i + 1])
        elif key == "--only":
            out["only"] = set(argv[i + 1].split(","))
        elif key == "--states":
            out["states"] = argv[i + 1].split(",")
        elif key == "--scale":
            out["scale"] = float(argv[i + 1])
        elif key == "--phase":
            out["phase"] = argv[i + 1]
        i += 2
    return out


# --- scene switches ------------------------------------------------------------------------------------


class Switches:
    """Turns the lights and glowing materials into each of the three bake states, and puts everything back."""

    def __init__(self) -> None:
        self.light_state = {o.name: o.hide_render for o in bpy.data.objects if o.type == "LIGHT"}
        self.emission: list[tuple[object, str, float]] = []        # (socket-owner, key, original strength)
        for mat in bpy.data.materials:
            if not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "EMISSION":
                    self.emission.append((node.inputs["Strength"], "default_value", node.inputs["Strength"].default_value))
                elif node.type == "BSDF_PRINCIPLED" and "Emission Strength" in node.inputs:
                    socket = node.inputs["Emission Strength"]
                    if socket.default_value > 0 and not socket.is_linked:
                        self.emission.append((socket, "default_value", socket.default_value))
        self.screen_socket = None
        screen = bpy.data.objects.get("Screen")
        if screen is not None and screen.data.materials:
            for node in screen.data.materials[0].node_tree.nodes:
                if node.type == "EMISSION":
                    self.screen_socket = node.inputs["Strength"]

    def apply(self, state: str) -> None:
        for obj in bpy.data.objects:
            if obj.type != "LIGHT":
                continue
            cols = {c.name for c in obj.users_collection}
            if state == "screen":
                obj.hide_render = True
            elif "04_LIGHTS_DIM" in cols:
                obj.hide_render = state != "dim"
            else:
                obj.hide_render = state != "lit"                    # the lit lights, and any light not in the dim set
        for socket, key, original in self.emission:
            setattr(socket, key, 0.0 if state == "screen" else original)
        if self.screen_socket is not None:
            self.screen_socket.default_value = SCREEN_STRENGTH if state == "screen" else 0.0

    def restore(self) -> None:
        for name, hidden in self.light_state.items():
            if name in bpy.data.objects:
                bpy.data.objects[name].hide_render = hidden
        for socket, key, original in self.emission:
            setattr(socket, key, original)


# --- UVs and bake targets -----------------------------------------------------------------------------


def make_visible(objs) -> None:
    for obj in objs:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        for col in obj.users_collection:
            col.hide_viewport = False


def apply_transforms(objs) -> None:
    """Bake each object's location, rotation and scale into its mesh. The unwrap works on the mesh as stored, ignoring the
    object's scale, so a door squashed to 5% thickness or a chair stored at 0.001 scale would get a wrong texel density.
    (Blender flips the faces itself when the scale is negative; the check below confirms the result still faces outward.)"""
    for obj in objs:
        flipped = obj.scale.x * obj.scale.y * obj.scale.z < 0
        deselect_all()
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        mesh = obj.data
        centre = sum((v.co for v in mesh.vertices), type(mesh.vertices[0].co)()) / max(1, len(mesh.vertices))
        outward = sum(p.normal.dot(p.center - centre) * p.area for p in mesh.polygons)
        log(f"{obj.name}: transforms applied" + (" (had a negative scale)" if flipped else "")
            + f"; outward-facing score {'+' if outward >= 0 else '-'}")


_TREE = None


def scene_tree():
    """One BVH of everything that is solid in the room (world space), built once per run; used to tell which way a face looks."""
    global _TREE
    if _TREE is not None:
        return _TREE
    import bmesh  # noqa: F401  (imported here so a missing module fails early)
    from mathutils.bvhtree import BVHTree

    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for o in bpy.data.objects:
        if (o.type != "MESH" or o.name in NOT_BAKED_OK or o.name.startswith((GLOW_PREFIX, "BeamGuide")) or o.hide_render):
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
    _TREE = BVHTree.FromPolygons(verts, polys, all_triangles=False)
    return _TREE


def _hemisphere(count: int = 24):
    """Fixed, repeatable directions on the +z hemisphere (cosine-weighted): the same on every run, so every run flips the same faces."""
    from mathutils import Vector

    golden = math.pi * (3 - math.sqrt(5))
    return [Vector((math.sqrt((i + 0.5) / count) * math.cos(i * golden), math.sqrt((i + 0.5) / count) * math.sin(i * golden),
                    math.sqrt(max(0.0, 1 - (i + 0.5) / count)))) for i in range(count)]


def _free_fraction(tree, origin, normal, reach: float, dirs) -> float:
    q = normal.to_track_quat("Z", "Y")
    return sum(1 for d in dirs if tree.ray_cast(origin, q @ d, reach)[0] is None) / len(dirs)


def fix_inside_out(objs) -> None:
    """Flip, in memory, every connected piece of a mesh whose faces point into solid matter instead of out into the room.

    Why: a bake lights a face from the side its normal points to. A piece that is inside-out (the imported chair is stored
    mirrored, and its wood console, trim and metal parts came out that way) has nothing in front of it but the chair's
    own body, so it bakes black - while an ordinary render hides the problem by shading both sides. The owner's first bake
    lost the console wood and the wall-base framing exactly like this.

    How it decides: from a few faces of each piece it sends rays out of both sides. The real outside is open room; the back of
    a hollow skin looks into an enclosed cavity. A piece whose back is far more open than its front is flipped. Pieces that
    are open on both sides (a wall plane) and pieces that are clearly right are left alone. The .blend is never saved.
    """
    import random

    import bmesh

    tree = scene_tree()
    dirs = _hemisphere()
    for obj in objs:
        rng = random.Random(7)                                        # a fixed seed: the same faces are sampled every run
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        seen: set[int] = set()
        flipped_pieces = flipped_faces = 0
        for face in bm.faces:
            if face.index in seen:
                continue
            stack, piece = [face], []
            seen.add(face.index)
            while stack:
                f = stack.pop()
                piece.append(f)
                for e in f.edges:
                    for g in e.link_faces:
                        if g.index not in seen:
                            seen.add(g.index)
                            stack.append(g)
            picks = piece if len(piece) <= 48 else rng.sample(piece, 48)
            votes_in = votes_out = 0
            for f in picks:
                n = f.normal
                if n.length < 0.5:
                    continue
                c = f.calc_center_median()
                front = _free_fraction(tree, c + n * 0.004, n, 1.0, dirs)
                back = _free_fraction(tree, c - n * 0.004, -n, 1.0, dirs)
                if abs(front - back) < 0.25:
                    continue
                if front > back:
                    votes_out += 1
                else:
                    votes_in += 1
            total = votes_in + votes_out
            if total and total >= min(6, len(picks)) and votes_in > 0.7 * total:
                bmesh.ops.reverse_faces(bm, faces=piece)
                flipped_pieces += 1
                flipped_faces += len(piece)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        if flipped_pieces:
            log(f"{obj.name}: {flipped_pieces} inside-out pieces flipped ({flipped_faces} faces)")


def prepare(objs) -> None:
    """Everything done to a group's objects before they are unwrapped, baked or exported - the same in every run."""
    apply_transforms(objs)
    fix_inside_out(objs)


def make_logo_opaque() -> None:
    """Bake the back-wall logo as a fully opaque card. A Cycles bake does not composite a see-through surface over what is
    behind it (it comes out black), so for the chair-group bake only, the logo's alpha is forced to 1; its see-through parts are
    near-black and are masked out afterwards. scripts/blender/composite_logo.py then lays the logo onto the WALL's own atlas
    using the PNG's real alpha, and the logo object is left out of the exported room, so the sticker keeps its outline and the
    wall (with the glow the logo casts on it) shows around it - exactly as in Blender's own render."""
    mat = bpy.data.materials.get("icon_transparent")
    if mat is None or not mat.node_tree:
        return
    tree = mat.node_tree
    logo = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if logo is None:
        log("logo: material not in the expected shape; left as it is")
        return
    for link in list(tree.links):
        if link.to_node == logo and link.to_socket.name == "Alpha":
            tree.links.remove(link)
    logo.inputs["Alpha"].default_value = 1.0
    log("logo: alpha forced to 1 for the bake (composited onto the wall afterwards)")


def write_logo_info(out: Path, report: dict) -> None:
    """Facts the compositor needs: where the logo sits on the wall, how both are mapped into their atlases."""
    info = {"png": bpy.path.abspath(bpy.data.images["icon_transparent.png"].filepath), "wall": None, "logo": None, "screen_ev": {}}
    for group in ("walls_plaster", "chairs_details"):
        info["screen_ev"][group] = report["groups"][group]["bakes"]["screen"]["screen_ev"]
    wall_objs = [bpy.data.objects[n] for n in GROUPS["walls_plaster"]["objects"]]
    chair_objs = [bpy.data.objects[n] for n in GROUPS["chairs_details"]["objects"]]
    for group, objs in (("walls_plaster", wall_objs), ("chairs_details", chair_objs)):
        make_visible(objs)
        prepare(objs)
        if not restore_layout(group, objs, Path(out)):
            raise SystemExit(f"no saved UV layout for {group}")
    logo_obj = bpy.data.objects["icon_transparent"]
    slots = [sl.material.name if sl.material else None for sl in logo_obj.material_slots]
    face = next(p for p in logo_obj.data.polygons if slots[p.material_index] == "icon_transparent")
    bake_uv = logo_obj.data.uv_layers["BakeUV"].data
    png_uv = logo_obj.data.uv_layers["UVMap"].data
    info["logo"] = {
        "corners": [list(logo_obj.data.vertices[i].co) for i in face.vertices],
        "bake_uv": [list(bake_uv[i].uv) for i in face.loop_indices],
        "png_uv": [list(png_uv[i].uv) for i in face.loop_indices],
    }
    wall = bpy.data.objects["Wall_Back"]
    wuv = wall.data.uv_layers["BakeUV"].data
    info["wall"] = {"tris": [{"pos": [list(wall.data.vertices[i].co) for i in p.vertices],
                              "uv": [list(wuv[i].uv) for i in p.loop_indices]} for p in wall.data.polygons]}
    (Path(out) / "logo_info.json").write_text(json.dumps(info, indent=1))
    log("wrote logo_info.json")


def unwrap(objs) -> None:
    """A fresh UV layout for the whole group: one shared atlas, equal pixels per metre on every surface."""
    deselect_all()
    for obj in objs:
        mesh = obj.data
        if "BakeUV" in mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers["BakeUV"])
        layer = mesh.uv_layers.new(name="BakeUV")
        mesh.uv_layers.active = layer                              # the bake writes here...
        for other in mesh.uv_layers:
            other.active_render = other.name != "BakeUV"           # ...while the shading keeps using the original UVs
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # correct_aspect=False: by default it uses the aspect of the material's texture image, which in a background run
    # (external textures not loaded: size 0 x 0) turned a 9.5 x 5.2 m wall into a 14:1 sliver and wasted 80% of the atlas.
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0015, scale_to_bounds=False,
                             correct_aspect=False)
    bpy.ops.uv.select_all(action="SELECT")
    # No average_islands_scale here: measured on the carpet floor it left 23% of the atlas used, against 87% without it.
    # smart_project already sizes every island by its real size, and pack_islands scales them all alike to fill the square.
    try:
        bpy.ops.uv.pack_islands(margin=0.0015, rotate=True, scale=True)
    except TypeError:
        bpy.ops.uv.pack_islands(margin=0.0015)
    bpy.ops.object.mode_set(mode="OBJECT")


def save_layout(group: str, objs, out: Path) -> None:
    arrays = {}
    for obj in objs:
        layer = obj.data.uv_layers["BakeUV"].data
        uv = np.empty(len(layer) * 2, np.float32)
        layer.foreach_get("uv", uv)
        arrays[obj.name] = uv
    np.savez(out / f"uv_{group}.npz", **arrays)


def restore_layout(group: str, objs, out: Path) -> bool:
    """Put a group's saved UV layout back (as BakeUV). False if there is none yet."""
    path = out / f"uv_{group}.npz"
    if not path.is_file():
        return False
    saved = np.load(path)
    for obj in objs:
        mesh = obj.data
        if "BakeUV" in mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers["BakeUV"])
        layer = mesh.uv_layers.new(name="BakeUV")
        data = saved[obj.name]
        if len(data) != len(layer.data) * 2:
            raise RuntimeError(f"saved layout for {obj.name} does not match its mesh (was the model changed?)")
        layer.data.foreach_set("uv", data)
        mesh.uv_layers.active = layer
        for other in mesh.uv_layers:
            other.active_render = other.name != "BakeUV"
    return True


def uv_density(objs, size: int) -> tuple[float, float]:
    """(UV coverage 0..1, pixels per metre) of the BakeUV layout, for the report."""
    uv_area = 0.0
    world_area = 0.0
    for obj in objs:
        mesh = obj.data
        layer = mesh.uv_layers["BakeUV"].data
        for poly in mesh.polygons:
            pts = [layer[i].uv for i in poly.loop_indices]
            uv_area += abs(sum(pts[k][0] * pts[(k + 1) % len(pts)][1] - pts[(k + 1) % len(pts)][0] * pts[k][1]
                               for k in range(len(pts)))) / 2
            world_area += poly.area
    density = math.sqrt(uv_area * size * size / max(world_area, 1e-9))
    return uv_area, density


def bake_nodes(objs, image) -> list:
    """Give every material the group uses an Image Texture node, selected and active, aimed at ``image``."""
    made = []
    seen = set()
    for obj in objs:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            if not mat.node_tree:
                mat.use_nodes = True
            tree = mat.node_tree
            node = tree.nodes.get("BAKE_TARGET") or tree.nodes.new("ShaderNodeTexImage")
            node.name = "BAKE_TARGET"
            node.image = image
            node.select = True
            tree.nodes.active = node
            made.append((mat, node))
    return made


def new_image(name: str, size: int):
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
    try:
        image.colorspace_settings.name = "Linear Rec.709"
    except TypeError:
        image.colorspace_settings.name = "Linear"
    return image


def save_jpeg(image, path: Path, quality: int = 93) -> None:
    scene = bpy.context.scene
    settings = scene.render.image_settings
    settings.file_format, settings.quality, settings.color_mode = "JPEG", quality, "RGB"
    image.save_render(str(path), scene=scene)


# --- the bakes ------------------------------------------------------------------------------------------


def bake_one(group: str, objs, state: str, size: int, out: Path, samples: int, switches: Switches, view) -> dict:
    began = time.monotonic()
    switches.apply(state)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    image = new_image(f"{group}_{state}", size)
    nodes = bake_nodes(objs, image)
    deselect_all()
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    log(f"{group} {state}: baking {size}x{size} at {samples} samples ...")
    bpy.ops.object.bake(type="COMBINED", margin=MARGIN_PX, margin_type="EXTEND", use_clear=True, uv_layer="BakeUV")
    info = {"seconds": round(time.monotonic() - began, 1)}
    path = out / f"{group}_{state}.jpg"
    if state == "screen":
        pixels = np.empty(size * size * 4, np.float32)
        image.pixels.foreach_get(pixels)
        lum = pixels.reshape(-1, 4)[:, :3].mean(axis=1)
        covered = lum[lum > 1e-6]
        p99 = float(np.percentile(covered, 99.0)) if covered.size else 1.0
        ev = math.log2(0.9 / max(p99, 1e-6))                       # bring the bright end to 0.9 so nothing clips
        scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = "Standard", "None", ev
        save_jpeg(image, path, 95)
        info.update(screen_ev=ev, p99=p99)
    else:
        scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = view
        save_jpeg(image, path)
    pixels = np.empty(size * size * 4, np.float32)
    image.pixels.foreach_get(pixels)
    info["mean"] = float(pixels.reshape(-1, 4)[:, :3].mean())
    for mat, node in nodes:                                         # leave no bake nodes behind in the shading
        mat.node_tree.nodes.remove(node)
    bpy.data.images.remove(image)
    scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = view
    log(f"{group} {state}: done in {info['seconds']} s -> {path.name}  {info}")
    return info


def configure_cycles(samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices() if not hasattr(prefs, "refresh_devices") else prefs.refresh_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"                        # the GPU once (it is also listed as CUDA)
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.004
    scene.cycles.use_denoising = False                             # would smear the texture detail being baked
    scene.cycles.sample_clamp_indirect = 10.0                      # tames fireflies from small bright lights
    log("devices:", [(d.name, d.type, d.use) for d in prefs.devices])


# --- the final scene and the export ------------------------------------------------------------------------


def finish_and_export(groups_done: dict, out: Path) -> None:
    """Swap each baked object to one baked material, one UV map, then write room.glb."""
    import itertools

    keep = []
    for group, info in groups_done.items():
        lit = out / f"{group}_lit.jpg"
        image = bpy.data.images.load(str(lit))
        image.colorspace_settings.name = "sRGB"
        image.name = f"{group}_lit"
        mat = bpy.data.materials.new(f"Bake_{group}")
        mat.use_nodes = True
        tree = mat.node_tree
        for node in list(tree.nodes):
            tree.nodes.remove(node)
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 1.0
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        for name in info["objects"]:
            obj = bpy.data.objects[name]
            mesh = obj.data
            for layer in [l for l in mesh.uv_layers if l.name != "BakeUV"]:
                mesh.uv_layers.remove(layer)
            mesh.uv_layers["BakeUV"].name = "UVMap"
            mesh.materials.clear()
            mesh.materials.append(mat)
            for poly in mesh.polygons:
                poly.material_index = 0
            deselect_all()
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            keep.append(obj)
    for name in ("Screen", "BiasGlow", "Spawn"):
        if name in bpy.data.objects:
            keep.append(bpy.data.objects[name])
    keep += [o for o in bpy.data.objects if o.name.startswith("Seat_")]
    deselect_all()
    for obj in keep:
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = keep[0]
    kwargs = dict(filepath=str(out / "room.glb"), export_format="GLB", use_selection=True, export_yup=True,
                  export_apply=True, export_extras=True, export_image_format="AUTO", export_cameras=False,
                  export_lights=False)
    for drop in ((), ("export_lights",), ("export_lights", "export_cameras")):
        try:
            bpy.ops.export_scene.gltf(**{k: v for k, v in kwargs.items() if k not in drop})
            break
        except TypeError:
            continue
    log("exported", out / "room.glb", round((out / "room.glb").stat().st_size / 1e6, 1), "MB")


def load_report(out: Path) -> dict:
    path = out / "bake_report.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"groups": {}}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    opts = parse(argv)
    out: Path = opts["dir"]
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    view = (scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure)
    log("view transform:", view, "| file saved by Blender", bpy.data.version, "| phase", opts["phase"])
    ensure_object_mode()
    problems = []
    if missing_objects():
        problems.append("listed in GROUPS but not in the file (renamed or deleted?): " + ", ".join(missing_objects()))
    if uncovered_meshes():
        problems.append("in the file but in no group, so they would come out black: " + ", ".join(uncovered_meshes()))
    if problems and "--allow-unbaked" not in argv:
        for line in problems:
            log("STOP |", line)
        raise SystemExit("fix GROUPS in scripts/blender/bake_room.py (or pass --allow-unbaked after --): " + " | ".join(problems))
    report = load_report(out)
    report.update({"samples": opts["samples"], "view": view[0]})

    if opts["phase"] == "logoinfo":
        write_logo_info(out, report)
        return

    if opts["phase"] == "export":
        done = {}
        for group, spec in GROUPS.items():
            objs = [bpy.data.objects[n] for n in spec["objects"]]
            make_visible(objs)
            prepare(objs)
            if not restore_layout(group, objs, out):
                raise SystemExit(f"no saved UV layout for {group}: bake it first")
            done[group] = {**report["groups"].get(group, {}), "objects": [n for n in spec["objects"] if n not in EXPORT_SKIP]}
        finish_and_export(done, out)
        write_manifest(out, report)
        log("ALL DONE (export)")
        return

    configure_cycles(opts["samples"])
    for obj in bpy.data.objects:
        if obj.name in NOT_BAKED_HIDDEN or obj.name.startswith(GLOW_PREFIX) or obj.name in HIDDEN_IN_BAKE:
            obj.hide_render = True
    switches = Switches()

    for group, spec in GROUPS.items():
        if opts["only"] and group not in opts["only"]:
            continue
        objs = [bpy.data.objects[n] for n in spec["objects"]]
        make_visible(objs)
        prepare(objs)
        size = max(256, int(spec["size"] * opts["scale"]) // 64 * 64)
        if restore_layout(group, objs, out):
            coverage, density = uv_density(objs, size)
            log(f"{group}: saved UV layout reused, coverage {coverage:.0%}, {density:.0f} px per metre at {size}px")
        else:
            for attempt in range(3):
                # The very first unwrap after Blender loads a file packs badly (23% of the atlas used, measured; every
                # later one uses 87%), whatever the settings. A low coverage is therefore simply done again.
                unwrap(objs)
                coverage, density = uv_density(objs, size)
                if coverage >= 0.6:
                    break
                log(f"{group}: UV coverage only {coverage:.0%}; unwrapping again (attempt {attempt + 2})")
            save_layout(group, objs, out)
            log(f"{group}: {len(objs)} objects, UV coverage {coverage:.0%}, {density:.0f} px per metre at {size}px (layout saved)")
        if group == "chairs_details":
            make_logo_opaque()                                     # only for this group: the wall must be baked with the real, see-through logo
        entry = report["groups"].setdefault(group, {"bakes": {}})
        entry.update({"objects": spec["objects"], "size": size, "uv_coverage": round(coverage, 3), "px_per_m": round(density)})
        for state in opts["states"]:
            edge = size if state != "screen" else max(256, size // 2)
            entry["bakes"][state] = bake_one(group, objs, state, edge, out, opts["samples"], switches, view)
            (out / "bake_report.json").write_text(json.dumps(report, indent=1, default=str))
    switches.restore()
    log("ALL DONE (bake)")


def write_manifest(out: Path, report: dict) -> None:
    manifest = {
        "name": "Home theatre", "units": "metres", "default_seat": "Seat_R2_Centre",
        "screen_emission": SCREEN_STRENGTH, "light_level_paused": 0.6,
        "groups": {g: {"material": f"Bake_{g}", "lit": f"{g}_lit.jpg", "dim": f"{g}_dim.jpg", "screen": f"{g}_screen.jpg",
                       "screen_ev": e.get("bakes", {}).get("screen", {}).get("screen_ev")}
                   for g, e in report["groups"].items()},
    }
    (out / "room.json").write_text(json.dumps(manifest, indent=1))


main()
