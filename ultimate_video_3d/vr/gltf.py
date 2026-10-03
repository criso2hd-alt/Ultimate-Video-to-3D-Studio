"""A small glTF 2.0 binary (.glb) reader: just what the VR room needs, with no dependencies beyond NumPy and OpenCV.

It reads the node tree (with each node's world matrix and ``extras``), the mesh data (positions, normals, the first UV
set, indices), the materials' base-colour textures, and decodes the embedded images. It does not do skins, animation,
cameras, lights, sparse accessors or any extension: the room is baked, static geometry, and Blender's exporter writes
plain float accessors for it. Anything it does not understand is ignored rather than guessed at.

The axes are glTF's: +Y up, the front of a model faces +Z. The room is exported by Blender with "+Y up", which puts
the screen wall at z = 0 and the seats at positive z looking toward -Z, the same world the renderer draws in.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GltfError(Exception):
    """The file is not a .glb this reader understands."""


@dataclass
class Primitive:
    positions: np.ndarray                    # (N, 3) float32
    normals: np.ndarray | None               # (N, 3) float32
    uv: np.ndarray | None                    # (N, 2) float32, glTF convention: (0, 0) is the picture's top-left
    indices: np.ndarray                      # (M,) uint32, triangles
    material: int | None


@dataclass
class Mesh:
    name: str
    primitives: list[Primitive] = field(default_factory=list)


@dataclass
class Node:
    name: str
    mesh: int | None
    matrix: np.ndarray                       # 4x4 world matrix (row-major maths: column vectors)
    extras: dict
    parent: int | None = None

    @property
    def position(self) -> np.ndarray:
        return self.matrix[:3, 3]


@dataclass
class Material:
    name: str
    base_colour_image: int | None            # index into Gltf.images
    base_colour_factor: tuple = (1.0, 1.0, 1.0, 1.0)


@dataclass
class Gltf:
    nodes: list[Node]
    meshes: list[Mesh]
    materials: list[Material]
    images: list[np.ndarray]                 # decoded, uint8 (H, W, 3) RGB, rows from the top

    def node(self, name: str) -> Node | None:
        return next((n for n in self.nodes if n.name == name), None)

    def nodes_named(self, prefix: str) -> list[Node]:
        return [n for n in self.nodes if n.name.startswith(prefix)]

    def mesh_of(self, node: Node) -> Mesh | None:
        return self.meshes[node.mesh] if node.mesh is not None else None


def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def _local_matrix(raw: dict) -> np.ndarray:
    if "matrix" in raw:
        return np.array(raw["matrix"], np.float64).reshape(4, 4).T          # glTF stores column-major
    m = np.eye(4)
    scale = raw.get("scale", [1.0, 1.0, 1.0])
    m[:3, :3] = _quat_matrix(*raw.get("rotation", [0.0, 0.0, 0.0, 1.0])) @ np.diag(scale)
    m[:3, 3] = raw.get("translation", [0.0, 0.0, 0.0])
    return m


def read(path: str | Path) -> Gltf:
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise GltfError("not a binary glTF (.glb) file")
    version, total = struct.unpack("<II", data[4:12])
    if version != 2:
        raise GltfError(f"glTF version {version} is not supported (need 2)")
    document, blob, pos = None, b"", 12
    while pos + 8 <= min(total, len(data)):
        length, kind = struct.unpack("<I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b"JSON":
            document = json.loads(chunk.decode("utf8"))
        elif kind == b"BIN\0":
            blob = chunk
        pos += 8 + length + (-length % 4)
    if document is None:
        raise GltfError("the file has no JSON chunk")

    views = document.get("bufferViews", [])
    accessors = document.get("accessors", [])

    def accessor(index: int) -> np.ndarray:
        acc = accessors[index]
        view = views[acc["bufferView"]]
        dtype = np.dtype(_COMPONENT[acc["componentType"]])
        width = _WIDTH[acc["type"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride", 0)
        count = acc["count"]
        if stride and stride != dtype.itemsize * width:                       # interleaved: gather row by row
            rows = np.frombuffer(blob, np.uint8, count=stride * count, offset=start).reshape(count, stride)
            out = np.ascontiguousarray(rows[:, :dtype.itemsize * width]).view(dtype).reshape(count, width)
        else:
            out = np.frombuffer(blob, dtype, count=count * width, offset=start).reshape(count, width)
        out = out.astype(np.float32) if dtype.kind == "f" else out
        if acc.get("normalized") and dtype.kind in "iu":
            out = out.astype(np.float32) / float(np.iinfo(dtype).max)
        return out[:, 0] if width == 1 else out

    meshes = []
    for raw in document.get("meshes", []):
        mesh = Mesh(raw.get("name", ""))
        for prim in raw.get("primitives", []):
            if prim.get("mode", 4) != 4:                                       # triangles only
                continue
            attrs = prim["attributes"]
            positions = accessor(attrs["POSITION"])
            if "indices" in prim:
                indices = accessor(prim["indices"]).astype(np.uint32)
            else:
                indices = np.arange(len(positions), dtype=np.uint32)
            mesh.primitives.append(Primitive(
                positions=positions,
                normals=accessor(attrs["NORMAL"]) if "NORMAL" in attrs else None,
                uv=accessor(attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else None,
                indices=indices.reshape(-1), material=prim.get("material")))
        meshes.append(mesh)

    import cv2

    images = []
    for raw in document.get("images", []):
        if "bufferView" in raw:
            view = views[raw["bufferView"]]
            encoded = np.frombuffer(blob, np.uint8, count=view["byteLength"], offset=view.get("byteOffset", 0))
        else:
            encoded = np.fromfile(Path(path).parent / raw["uri"], np.uint8)
        picture = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if picture is None:
            raise GltfError("an embedded image could not be decoded")
        images.append(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))

    textures = document.get("textures", [])
    materials = []
    for raw in document.get("materials", []):
        pbr = raw.get("pbrMetallicRoughness", {})
        tex = pbr.get("baseColorTexture")
        image = textures[tex["index"]].get("source") if tex is not None else None
        materials.append(Material(raw.get("name", ""), image, tuple(pbr.get("baseColorFactor", (1, 1, 1, 1)))))

    nodes: list[Node] = []
    raw_nodes = document.get("nodes", [])

    def walk(index: int, parent_matrix: np.ndarray, parent: int | None) -> None:
        raw = raw_nodes[index]
        world = parent_matrix @ _local_matrix(raw)
        mine = len(nodes)
        nodes.append(Node(raw.get("name", f"node{index}"), raw.get("mesh"), world, raw.get("extras") or {}, parent))
        for child in raw.get("children", []):
            walk(child, world, mine)

    scenes = document.get("scenes", [])
    roots = scenes[document.get("scene", 0)]["nodes"] if scenes else range(len(raw_nodes))
    for root in roots:
        walk(root, np.eye(4), None)
    return Gltf(nodes, meshes, materials, images)
