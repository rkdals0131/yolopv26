from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from model.signal_attr.classifier import (
    BASE_COLORS,
    TL_BITS,
    SignalAttrCropClassifier,
    load_signal_attr_classifier_checkpoint,
)

from .common import artifact_paths_for_checkpoint, ensure_writable_output


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "signal_attr" / "best_signal_attr.pt"


class SignalAttrRuntimeExport(nn.Module):
    """Tuple output adapter from the Plan B SignalAttr exporter."""

    def __init__(self, model: SignalAttrCropClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(crops)
        return outputs["base_color_logits"], outputs["arrow_logit"]


def export_signal_attr_torchscript(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    output_path: Path | None = None,
    meta_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    default_output, default_meta = artifact_paths_for_checkpoint(checkpoint_path)
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else default_output
    meta_path = Path(meta_path).expanduser().resolve() if meta_path is not None else default_meta
    if len({checkpoint_path, output_path, meta_path}) != 3:
        raise ValueError("checkpoint, TorchScript, and metadata paths must be different")
    ensure_writable_output(output_path, overwrite=overwrite)
    ensure_writable_output(meta_path, overwrite=overwrite)

    loaded = load_signal_attr_classifier_checkpoint(checkpoint_path, device="cpu")
    model = SignalAttrRuntimeExport(loaded["model"]).eval()
    scripted = torch.jit.script(model)
    config = loaded["model_config"]
    metadata = {
        "format_version": 1,
        "model_type": "SignalAttrCropClassifier",
        "source_checkpoint": os.path.relpath(checkpoint_path, meta_path.parent),
        "artifact_path": os.path.relpath(output_path, meta_path.parent),
        "model_config": asdict(config),
        "crop_config": loaded["crop_config"],
        "threshold_policy": asdict(loaded["threshold_policy"]),
        "state_semantics": loaded["payload"].get("state_semantics", "legacy_arrow"),
        "all_off_is_valid": loaded["payload"].get("all_off_is_valid"),
        "base_colors": list(BASE_COLORS),
        "tl_bits": list(TL_BITS),
        "input": {
            "shape": ["batch", 3, config.input_size, config.input_size],
            "dtype": "float32",
            "color_space": "rgb",
            "normalization": loaded["crop_config"].get("normalization", "imagenet"),
        },
        "output_names": ["base_color_logits", "arrow_logit"],
        "output_shapes": [["batch", len(BASE_COLORS)], ["batch"]],
    }
    with tempfile.TemporaryDirectory(prefix=".signal_attr_export_", dir=output_path.parent) as stage:
        staged_model = Path(stage) / "model.pt"
        scripted.save(str(staged_model), _extra_files={"metadata.json": json.dumps(metadata, ensure_ascii=False)})
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=meta_path.parent, delete=False) as handle:
            staged_meta = Path(handle.name)
            try:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            except BaseException:
                staged_meta.unlink(missing_ok=True)
                raise
        try:
            os.replace(staged_model, output_path)
            os.replace(staged_meta, meta_path)
        finally:
            staged_meta.unlink(missing_ok=True)
    return {"artifact_path": str(output_path), "meta_path": str(meta_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOPV26 SignalAttr to TorchScript.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = export_signal_attr_torchscript(
        args.checkpoint, output_path=args.output, meta_path=args.metadata, overwrite=args.overwrite
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
