from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from common.schema import DEFAULT_IMAGE_HW, ROADMARK_CLASSES, SIGNAL_CLASSES
from common.io import atomic_write_json
from model.net.pv26 import PV26FocusedModel
from .common import artifact_paths_for_checkpoint, ensure_writable_output


class FocusedExport(nn.Module):
    def __init__(self, model: PV26FocusedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_export(image)


def export_focused_torchscript(checkpoint: Path, *, output: Path | None = None,
                              device: str = "cpu", batch_size: int = 2, overwrite: bool = False) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    default_output, _ = artifact_paths_for_checkpoint(checkpoint)
    output = output.expanduser().resolve() if output is not None else default_output
    if output == checkpoint:
        raise ValueError("export must not overwrite its source checkpoint")
    metadata_path = output.with_suffix(".meta.json")
    if metadata_path == checkpoint:
        raise ValueError("metadata must not overwrite its source checkpoint")
    ensure_writable_output(output, overwrite=overwrite)
    ensure_writable_output(metadata_path, overwrite=overwrite)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PV26FocusedModel(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    image_hw = tuple((payload.get("run_metadata") or {}).get("model", {}).get("image_hw", DEFAULT_IMAGE_HW))
    example = torch.zeros(batch_size, 3, *image_hw, device=device)
    wrapper = FocusedExport(model).eval()
    with torch.inference_mode():
        wrapper(example)
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    metadata = {"model_config": payload["model_config"], "image_hw": list(image_hw),
                "batch_size": batch_size, "input": {"layout": "NCHW", "dtype": "float32", "range": [0, 1]},
                "output_names": ["detections", "roadmark_logits"],
                "detection_columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
                "detection_coordinates": "network_pixels", "roadmark_stride": 4,
                "signal_classes": list(SIGNAL_CLASSES), "roadmark_classes": list(ROADMARK_CLASSES)}
    with tempfile.TemporaryDirectory(prefix=".focused_export_", dir=output.parent) as directory:
        staged = Path(directory) / "model.pt"
        traced.save(str(staged), _extra_files={"metadata.json": json.dumps(metadata, ensure_ascii=False)})
        os.replace(staged, output)
    atomic_write_json(metadata_path, metadata, ensure_ascii=False)
    return {"artifact_path": str(output), "metadata_path": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the focused PV26 detector and roadmark model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_focused_torchscript(args.checkpoint, output=args.output, device=args.device,
                                               batch_size=args.batch_size, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
