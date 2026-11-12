"""Export Depth Anything 3 Mono-Large (the app's optional High quality depth model) to ONNX.

Dev-time tooling, never imported by the app. Run it in a separate environment that has torch (the CPU build is
enough), the official package and a few of its light dependencies - not in the app's own venv:

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "numpy<2" einops omegaconf safetensors huggingface-hub opencv-python pillow addict onnx onnxruntime
    pip install --no-deps depth-anything-3
    python scripts/export_da3.py --out onnx_models

It writes ``Depth-Anything-3-Mono-Large.onnx`` (fp32, what DirectML / CPU run) and ``...fp16.onnx`` (what the NVIDIA
TensorRT-RTX add-on runs: 7.5 ms vs 26 ms for fp32 on an RTX 4080; DirectML is the other way round, 43 ms fp32 vs
250 ms fp16). Upload both to the ``models-v1`` release and put their SHA-256 in ``models.EXTRA``.

Licences: the DA3 code is Apache-2.0, and of the weights only DA3-Small, DA3-Base, DA3Mono-Large and DA3Metric-Large
are Apache-2.0 (Large, Giant and the Nested models are CC BY-NC 4.0: never ship those). Mono-Large is the one built
for single-picture depth; Small/Base are multi-view models and were worse than our V2 Small on films.

What the graph does: ImageNet-normalised RGB (1, 3, 280, 504) in - the same preprocessing as our V2 export - and
(1 / depth, sky) out. DA3 Mono predicts depth (exp of its head); the app maps it in ``depth._from_depth`` (log depth,
sky to the far end). 280 x 504 = 20 x 36 patches of 14 px: half the tokens of a 504 square, 2.6x faster, and films
are wide. Two things the stock code does not export: ``torch.cartesian_prod`` (RoPE grid; replaced by meshgrid
below) and the package's exporters/pose tools, which import open3d/moviepy/evo (stubbed: not needed for inference).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

REPO = "depth-anything/DA3MONO-LARGE"
NAME = "Depth-Anything-3-Mono-Large"
SIZE = (280, 504)


def _prepare_torch():
    import torch

    def cartesian_prod(a, b):
        ga, gb = torch.meshgrid(a, b, indexing="ij")
        return torch.stack((ga.reshape(-1), gb.reshape(-1)), dim=-1)

    torch.cartesian_prod = cartesian_prod
    for name, attrs in {"depth_anything_3.utils.export": ["export"],
                        "depth_anything_3.utils.pose_align": ["align_poses_umeyama"]}.items():
        stub = types.ModuleType(name)
        for attr in attrs:
            setattr(stub, attr, None)
        sys.modules[name] = stub
    return torch


def export(out_dir: Path) -> list[Path]:
    torch = _prepare_torch()
    from depth_anything_3.api import DepthAnything3

    net = DepthAnything3.from_pretrained(REPO).eval().model

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            h, w = x.shape[-2], x.shape[-1]
            feats, _ = self.net.backbone(x[:, None], cam_token=None, export_feat_layers=[])
            o = self.net.head(feats, h, w, patch_start_idx=0)
            inverse = 1.0 / o["depth"].reshape(1, h, w).clamp_min(1e-6)
            return inverse, o["sky"].reshape(1, h, w)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / f"{NAME}.onnx"
    x = torch.randn(1, 3, *SIZE)
    with torch.no_grad():
        torch.onnx.export(Wrap(net).eval(), (x,), str(fp32), input_names=["image"],
                          output_names=["inverse_depth", "sky"], opset_version=17, dynamo=False)

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16   # ORT's converter (onnxconverter-common breaks the Casts)

    fp16 = out_dir / f"{NAME}.fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))
    return [fp32, fp16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("onnx_models"))
    for path in export(parser.parse_args().out):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {path.stat().st_size} bytes, sha256 {digest}")


if __name__ == "__main__":
    main()
