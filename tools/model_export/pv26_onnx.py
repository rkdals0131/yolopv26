"""Export PV26 to ONNX with the manifest the C++ runtime (pv26_perception) reads.

The graph is the same fixed tensor view as the TorchScript export: network-pixel
xyxy/confidence/class detections and stride-4 road-marking logits. The manifest
carries everything the runtime must reproduce outside Python: letterbox, class
order, detection confidence, and the road-marking decode profile that travels
with the checkpoint (``run_metadata.train.roadmark_decode``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common.io import atomic_write_json
from common.schema import DEFAULT_IMAGE_HW, ROADMARK_CLASSES, ROADMARK_STRIDE, SIGNAL_CLASSES
from model.net.pv26 import PV26FocusedModel
from .common import ensure_writable_output
from .pv26_torchscript import FocusedExport

# decode_roadmark_points defaults; a checkpoint's profile overrides the keys it sets.
DEFAULT_ROADMARK_DECODE = {
    "threshold": [0.5, 0.5, 0.5],
    "min_points": 6,
    "localization": "grid",
    "orientation": "fixed",
    "max_gap": [3, 3, 3],
    "min_score": [0.0, 0.0, 0.0],
    "min_length_px": [0.0, 0.0, 0.0],
}


def _per_class(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value] * len(ROADMARK_CLASSES)


def roadmark_decode_profile(payload: dict) -> dict:
    configured = ((payload.get("run_metadata") or {}).get("train") or {}).get("roadmark_decode") or {}
    unknown = set(configured) - set(DEFAULT_ROADMARK_DECODE)
    if unknown:
        raise ValueError(f"unsupported roadmark_decode keys for the C++ runtime: {sorted(unknown)}")
    profile = {**DEFAULT_ROADMARK_DECODE, **configured}
    for key in ("threshold", "max_gap", "min_score", "min_length_px"):
        profile[key] = _per_class(profile[key])
    if profile["localization"] != "grid":
        raise ValueError("the C++ runtime implements grid localization only")
    return profile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_pv26_onnx(checkpoint: Path, *, output: Path, batch_size: int = 2,
                     opset: int = 17, overwrite: bool = False, verify: bool = True) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if checkpoint in (output, manifest_path):
        raise ValueError("export must not overwrite its source checkpoint")
    ensure_writable_output(output, overwrite=overwrite)
    ensure_writable_output(manifest_path, overwrite=overwrite)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PV26FocusedModel(**payload["model_config"])
    model.load_state_dict(payload["model"])
    wrapper = FocusedExport(model.eval()).eval()
    image_hw = tuple((payload.get("run_metadata") or {}).get("model", {}).get("image_hw", DEFAULT_IMAGE_HW))
    example = torch.rand(batch_size, 3, *image_hw, generator=torch.Generator().manual_seed(26))
    # no_grad, not inference_mode: the detect head caches anchors that the
    # exporter's traced forward must be able to reuse.
    with torch.no_grad():
        expected = [tensor.numpy() for tensor in wrapper(example)]
    torch.onnx.export(wrapper, (example,), str(output), opset_version=opset, dynamo=False,
                      input_names=["image"], output_names=["detections", "roadmark_logits"])
    manifest = {
        "format_version": 1,
        "model_type": "PV26FocusedModel",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "global_step": payload.get("global_step"),
        "onnx": output.name,
        "batch_size": batch_size,
        "input": {"name": "image", "layout": "NCHW", "dtype": "float32", "range": [0, 1],
                  "color": "rgb", "image_hw": list(image_hw),
                  "letterbox": {"pad_value": 114, "resize": "bilinear", "align": "center"}},
        "outputs": {
            "detections": {"shape": list(expected[0].shape),
                           "columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
                           "coordinates": "network_pixels"},
            "roadmark_logits": {"shape": list(expected[1].shape), "stride": ROADMARK_STRIDE},
        },
        "signal_classes": list(SIGNAL_CLASSES),
        "detection_confidence": 0.25,
        "roadmark_classes": list(ROADMARK_CLASSES),
        "roadmark_decode": roadmark_decode_profile(payload),
    }
    if verify:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"image": example.numpy()})
        manifest["verification"] = {
            "roadmark_logits_max_abs_error": float(np.abs(actual[1] - expected[1]).max()),
            "detections_max_abs_error": float(np.abs(actual[0] - expected[0]).max()),
        }
    atomic_write_json(manifest_path, manifest, ensure_ascii=False)
    return {"onnx": str(output), "manifest": str(manifest_path), **manifest.get("verification", {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_pv26_onnx(args.checkpoint, output=args.output, batch_size=args.batch_size,
                                      opset=args.opset, overwrite=args.overwrite,
                                      verify=not args.no_verify), indent=2))


if __name__ == "__main__":
    main()
