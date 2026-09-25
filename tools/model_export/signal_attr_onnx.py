"""Export SignalAttr to ONNX with the manifest the C++ runtime (pv26_perception) reads.

The batch axis is dynamic so the runtime can classify every signal crop of both
cameras in one call. The manifest records the crop, normalization and threshold
policy that ``SignalAttrRuntime.predict`` applies around the network.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.io import atomic_write_json
from model.signal_attr.classifier import BASE_COLORS, TL_BITS, load_signal_attr_classifier_checkpoint
from .common import ensure_writable_output
from .pv26_onnx import _sha256
from .signal_attr_torchscript import SignalAttrRuntimeExport


def export_signal_attr_onnx(checkpoint: Path, *, output: Path, max_batch: int = 16,
                            opset: int = 17, overwrite: bool = False, verify: bool = True) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if checkpoint in (output, manifest_path):
        raise ValueError("export must not overwrite its source checkpoint")
    ensure_writable_output(output, overwrite=overwrite)
    ensure_writable_output(manifest_path, overwrite=overwrite)
    loaded = load_signal_attr_classifier_checkpoint(checkpoint, device="cpu")
    wrapper = SignalAttrRuntimeExport(loaded["model"]).eval()
    size = int(loaded["model_config"].input_size)
    example = torch.randn(4, 3, size, size, generator=torch.Generator().manual_seed(26))
    with torch.inference_mode():
        expected = [tensor.numpy() for tensor in wrapper(example)]
    torch.onnx.export(wrapper, (example,), str(output), opset_version=opset, dynamo=False,
                      input_names=["crops"], output_names=["base_color_logits", "arrow_logit"],
                      dynamic_axes={"crops": {0: "batch"}, "base_color_logits": {0: "batch"},
                                    "arrow_logit": {0: "batch"}})
    payload = loaded["payload"]
    manifest = {
        "format_version": 1,
        "model_type": "SignalAttrCropClassifier",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx": output.name,
        "max_batch": max_batch,
        "input": {"name": "crops", "layout": "NCHW", "dtype": "float32", "color": "rgb",
                  "size": size, "normalization": loaded["crop_config"].get("normalization", "imagenet"),
                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "crop": loaded["crop_config"],
        "threshold_policy": asdict(loaded["threshold_policy"]),
        "state_semantics": payload.get("state_semantics", "legacy_arrow"),
        "all_off_is_valid": bool(payload.get("all_off_is_valid", False)),
        "base_colors": list(BASE_COLORS),
        "tl_bits": list(TL_BITS),
    }
    if verify:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"crops": example.numpy()})
        manifest["verification"] = {
            name: float(np.abs(a - e).max())
            for name, a, e in zip(("base_color_logits", "arrow_logit"), actual, expected)
        }
    atomic_write_json(manifest_path, manifest, ensure_ascii=False)
    return {"onnx": str(output), "manifest": str(manifest_path), **manifest.get("verification", {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_signal_attr_onnx(args.checkpoint, output=args.output, max_batch=args.max_batch,
                                             opset=args.opset, overwrite=args.overwrite,
                                             verify=not args.no_verify), indent=2))


if __name__ == "__main__":
    main()
