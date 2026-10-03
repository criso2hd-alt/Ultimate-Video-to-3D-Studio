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
    "wall_left": {"objects": ["Wall_Left"], "size": 4096},
    "walls_plaster": {"objects": ["Wall_Back", "Wall_Front", "Wall_Back.002", "Door2"], "size": 4096},
    "floor_carpet": {"objects": ["Floor"], "size": 4096},
    "ceiling_panels": {"objects": ["Ceiling", "AcusticPanels"], "size": 4096},        # Ceiling.001 is never seen: not baked
    "chairs_details": {"objects": ["Home Theater Armchair N220224.008", "Text.001", "icon_transparent"], "size": 4096},
    "hall_floor": {"objects": ["Floor.001"], "size": 2048},        # only seen through the doors
}
NOT_BAKED_HIDDEN = ("BiasGlow",)            # drawn live by the app: must not paint the wall behind the screen
LIGHT_COLLECTIONS = ("03_LIGHTS_LIT", "04_LIGHTS_DIM")
SCREEN_STRENGTH = 8.0                       # the Screen's emission for the screen-only bake (matches its bake3_strength)
MARGIN_PX = 16


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
    out = {"dir": Path(argv[0]), "samples": 1024, "only": None, "states": ["lit", "dim", "screen"], "scale": 1.0,
           "phase": "bake"}
    i = 1
    while i < len(argv):
        key = argv[i]
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
    report = load_report(out)
    report.update({"samples": opts["samples"], "view": view[0]})

    if opts["phase"] == "export":
        done = {}
        for group, spec in GROUPS.items():
            objs = [bpy.data.objects[n] for n in spec["objects"]]
            make_visible(objs)
            apply_transforms(objs)
            if not restore_layout(group, objs, out):
                raise SystemExit(f"no saved UV layout for {group}: bake it first")
            done[group] = {"objects": spec["objects"], **report["groups"].get(group, {})}
        finish_and_export(done, out)
        write_manifest(out, report)
        log("ALL DONE (export)")
        return

    configure_cycles(opts["samples"])
    switches = Switches()
    for name in NOT_BAKED_HIDDEN:
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = True
    if "REF_Person_1.75m" in bpy.data.objects:
        bpy.data.objects["REF_Person_1.75m"].hide_render = True

    for group, spec in GROUPS.items():
        if opts["only"] and group not in opts["only"]:
            continue
        objs = [bpy.data.objects[n] for n in spec["objects"]]
        make_visible(objs)
        apply_transforms(objs)
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
