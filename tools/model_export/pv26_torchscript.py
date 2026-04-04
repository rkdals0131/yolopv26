from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .common import artifact_paths_for_checkpoint, ensure_writable_output, resolve_torch_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OD_CLASSES = [
    "vehicle",
    "bike",
    "pedestrian",
    "traffic_cone",
    "obstacle",
    "traffic_light",
    "sign",
]
DEFAULT_TL_BITS = [
    "red",
    "yellow",
    "green",
    "arrow",
]
DEFAULT_LANE_CLASSES = [
    "white_lane",
    "yellow_lane",
    "blue_lane",
]
DEFAULT_LANE_TYPES = [
    "solid",
    "dotted",
]
YOLO26_VARIANT_BY_HEAD_CHANNELS = {
    (64, 128, 256): "n",
    (128, 256, 512): "s",
}


@dataclass(frozen=True)
class ExportVerification:
    name: str
    shape: list[int]
    max_abs_diff: float
    allclose: bool


class Pv26TorchscriptExportWrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        trunk_layers: list[torch.nn.Module],
        feature_source_indices: tuple[int, ...],
        heads: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.trunk_layers = torch.nn.ModuleList(trunk_layers)
        self.feature_source_indices = tuple(int(index) for index in feature_source_indices)
        self.max_feature_index = max(self.feature_source_indices)
        self.heads = heads

    def _forward_pyramid_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        current = image
        for layer_index, layer in enumerate(self.trunk_layers[: self.max_feature_index + 1]):
            from_index = getattr(layer, "f", -1)
            if from_index == -1:
                layer_input = current
            elif isinstance(from_index, int):
                layer_input = outputs[int(from_index)]
            else:
                layer_input = [current if int(index) == -1 else outputs[int(index)] for index in from_index]
            current = layer(layer_input)
            outputs.append(current)
        return [outputs[index] for index in self.feature_source_indices]

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = self.heads(self._forward_pyramid_features(image))
        return (
            predictions["det"],
            predictions["tl_attr"],
            predictions["lane"],
            predictions["stop_line"],
        )


def setup_pv26_imports(repo_root: Path) -> None:
    repo_root = repo_root.expanduser().resolve()
    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    site_packages = sorted((repo_root / ".venv" / "lib").glob("python*/site-packages"))
    for site_package in site_packages:
        site_path = str(site_package)
        if site_path not in sys.path:
            sys.path.insert(0, site_path)

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")


def import_pv26_symbols() -> tuple[Any, Any, Any]:
    errors: list[str] = []

    try:
        from model.net import PV26Heads
        from model.net import build_yolo26n_trunk
        from model.engine._loss_spec import build_loss_spec

        return PV26Heads, build_loss_spec, build_yolo26n_trunk
    except Exception as exc:
        errors.append(f"current layout import failed: {exc}")

    try:
        from model.heads import PV26Heads
        from model.loss.spec import build_loss_spec
        from model.trunk import build_yolo26n_trunk

        return PV26Heads, build_loss_spec, build_yolo26n_trunk
    except Exception as exc:
        errors.append(f"legacy layout import failed: {exc}")

    raise ImportError("unable to import PV26 modules after setup_pv26_imports: " + " | ".join(errors))


def resolve_default_checkpoint(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob("runs/pv26_*/*/checkpoints/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "unable to find a default checkpoint under runs/pv26_*/<run>/checkpoints/best.pt"
        )
    return candidates[0].resolve()


def infer_backbone_variant_from_head_channels(head_channels: tuple[int, int, int]) -> str | None:
    normalized = tuple(int(value) for value in head_channels)
    return YOLO26_VARIANT_BY_HEAD_CHANNELS.get(normalized)


def infer_backbone_variant_from_weights_path(weights_path: str | Path) -> str | None:
    filename = Path(weights_path).name.lower()
    for variant in ("n", "s"):
        if f"26{variant}" in filename:
            return variant
    return None


def resolve_trunk_weights(
    repo_root: Path,
    explicit: Path | None,
    *,
    checkpoint_variant: str | None = None,
) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
    else:
        variant = str(checkpoint_variant).strip().lower() if checkpoint_variant else "n"
        candidate = (repo_root / f"yolo26{variant}.pt").resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"missing trunk weights: {candidate}")
    resolved_variant = infer_backbone_variant_from_weights_path(candidate)
    if checkpoint_variant is not None and resolved_variant is not None and resolved_variant != checkpoint_variant:
        raise ValueError(
            "checkpoint expects a YOLO26 "
            f"{checkpoint_variant} backbone, but trunk weights resolve to {candidate.name}"
        )
    return candidate


def infer_head_channels(heads_state_dict: dict[str, Any]) -> tuple[int, int, int]:
    channels: list[int] = []
    for index in range(3):
        key = f"det_heads.{index}.block.0.weight"
        weight = heads_state_dict.get(key)
        if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 4:
            raise KeyError(f"checkpoint missing {key}")
        channels.append(int(weight.shape[1]))
    return tuple(channels)  # type: ignore[return-value]


def letterbox_example_image(image_bgr: np.ndarray, *, input_height: int, input_width: int) -> torch.Tensor:
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("invalid example image size")
    scale = min(float(input_width) / float(orig_w), float(input_height) / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (input_width - new_w) // 2
    pad_y = (input_height - new_h) // 2
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def build_example_input(
    *,
    input_height: int,
    input_width: int,
    device: torch.device,
    example_image: Path | None,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if example_image is not None:
        image_path = example_image.expanduser().resolve()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"failed to load example image: {image_path}")
        tensor = letterbox_example_image(
            image_bgr,
            input_height=input_height,
            input_width=input_width,
        )
        return tensor.to(device), {"kind": "image", "path": str(image_path)}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tensor = torch.rand((1, 3, input_height, input_width), generator=generator, dtype=torch.float32)
    return tensor.to(device), {"kind": "random", "seed": int(seed)}


def tensor_report(name: str, eager: torch.Tensor, scripted: torch.Tensor, *, atol: float, rtol: float) -> ExportVerification:
    return ExportVerification(
        name=name,
        shape=[int(v) for v in eager.shape],
        max_abs_diff=float((eager - scripted).abs().max().item()) if eager.numel() > 0 else 0.0,
        allclose=bool(torch.allclose(eager, scripted, atol=atol, rtol=rtol)),
    )


def export_metadata(
    *,
    checkpoint_path: Path,
    output_path: Path,
    trunk_weights: Path,
    input_height: int,
    input_width: int,
    det_shape: list[int],
    tl_attr_shape: list[int],
    lane_shape: list[int],
    stop_line_shape: list[int],
    od_classes: list[str],
    tl_bits: list[str],
    lane_classes: list[str],
    lane_types: list[str],
    det_feature_shapes: list[list[int]],
    det_feature_strides: list[int],
    example_info: dict[str, Any],
    verification: list[ExportVerification],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "artifact_type": "pv26_torchscript_raw_heads",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "artifact_path": str(output_path),
        "trunk_weights": str(trunk_weights),
        "input": {
            "layout": "NCHW",
            "dtype": "float32",
            "range": [0.0, 1.0],
            "color_order": "RGB",
            "height": int(input_height),
            "width": int(input_width),
            "preprocess": "letterbox_bgr_to_rgb_normalize_01",
        },
        "outputs": {
            "det": {
                "shape": ["batch"] + det_shape[1:],
                "dtype": "float32",
                "format": "ltrb_obj_cls_logits",
            },
            "tl_attr": {
                "shape": ["batch"] + tl_attr_shape[1:],
                "dtype": "float32",
                "format": "traffic_light_attribute_logits",
            },
            "lane": {
                "shape": ["batch"] + lane_shape[1:],
                "dtype": "float32",
                "format": "score_lane_class_lane_type_polyline_visibility",
            },
            "stop_line": {
                "shape": ["batch"] + stop_line_shape[1:],
                "dtype": "float32",
                "format": "score_polyline",
            },
        },
        "od_classes": od_classes,
        "tl_bits": tl_bits,
        "lane_classes": lane_classes,
        "lane_types": lane_types,
        "det_feature_shapes": det_feature_shapes,
        "det_feature_strides": det_feature_strides,
        "trace_example": example_info,
        "verification": [
            {
                "name": item.name,
                "shape": item.shape,
                "max_abs_diff": item.max_abs_diff,
                "allclose": item.allclose,
            }
            for item in verification
        ],
    }


def export_pv26_torchscript(
    *,
    checkpoint_path: Path,
    repo_root: Path | None = None,
    output_path: Path | None = None,
    meta_path: Path | None = None,
    trunk_weights: Path | None = None,
    device_name: str = "cpu",
    example_image: Path | None = None,
    seed: int = 1234,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    overwrite: bool = True,
) -> dict[str, Any]:
    repo_root = (Path(repo_root).expanduser().resolve() if repo_root is not None else REPO_ROOT)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"pv26 repo not found: {repo_root}")

    checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else resolve_default_checkpoint(repo_root)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    default_output_path, default_meta_path = artifact_paths_for_checkpoint(checkpoint_path)
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else default_output_path
    meta_path = Path(meta_path).expanduser().resolve() if meta_path is not None else default_meta_path
    example_image = Path(example_image).expanduser().resolve() if example_image is not None else None

    ensure_writable_output(output_path, overwrite=overwrite)
    ensure_writable_output(meta_path, overwrite=overwrite)
    setup_pv26_imports(repo_root)
    PV26Heads, build_loss_spec, build_yolo26n_trunk = import_pv26_symbols()

    device = resolve_torch_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "adapter_state_dict" not in checkpoint or "heads_state_dict" not in checkpoint:
        raise KeyError(
            f"checkpoint must contain adapter_state_dict and heads_state_dict: {checkpoint_path}"
        )

    spec = build_loss_spec()
    od_classes = list(spec["model_contract"]["od_classes"])
    tl_bits = list(spec["model_contract"]["tl_bits"])
    lane_classes = list(spec["model_contract"].get("lane_classes", DEFAULT_LANE_CLASSES))
    lane_types = list(spec["model_contract"].get("lane_types", DEFAULT_LANE_TYPES))
    head_channels = infer_head_channels(checkpoint["heads_state_dict"])
    checkpoint_variant = infer_backbone_variant_from_head_channels(head_channels)
    resolved_trunk_weights = resolve_trunk_weights(
        repo_root,
        trunk_weights,
        checkpoint_variant=checkpoint_variant,
    )

    adapter = build_yolo26n_trunk(str(resolved_trunk_weights))
    try:
        adapter.raw_model.load_state_dict(checkpoint["adapter_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "failed to load adapter_state_dict into the selected YOLO26 trunk. "
            f"checkpoint head channels={head_channels}, "
            f"inferred checkpoint variant={checkpoint_variant or 'unknown'}, "
            f"trunk weights={resolved_trunk_weights}"
        ) from exc
    adapter.raw_model = adapter.raw_model.to(device).eval()

    heads = PV26Heads(in_channels=head_channels).to(device).eval()
    heads.load_state_dict(checkpoint["heads_state_dict"])

    wrapper = Pv26TorchscriptExportWrapper(
        trunk_layers=list(adapter.raw_model.model),
        feature_source_indices=adapter.feature_source_indices,
        heads=heads,
    ).to(device).eval()

    example_input, example_info = build_example_input(
        input_height=608,
        input_width=800,
        device=device,
        example_image=example_image,
        seed=int(seed),
    )

    with torch.no_grad():
        feature_maps = wrapper._forward_pyramid_features(example_input)
        eager_out = wrapper(example_input)

    feature_shapes = [[int(feature.shape[-2]), int(feature.shape[-1])] for feature in feature_maps]
    if len(feature_shapes) != 3:
        raise RuntimeError(f"expected 3 pyramid feature maps, got {len(feature_shapes)}")
    feature_strides = [int(value) for value in getattr(heads, "feature_strides", (8, 16, 32))]
    if len(feature_strides) != len(feature_shapes):
        raise RuntimeError(
            "feature stride count does not match feature map count: "
            f"shapes={feature_shapes} strides={feature_strides}"
        )
    expected_det_rows = sum(int(height) * int(width) for height, width in feature_shapes)
    if expected_det_rows != int(eager_out[0].shape[1]):
        raise RuntimeError(
            "detector query count does not match feature metadata: "
            f"queries={int(eager_out[0].shape[1])} feature_shapes={feature_shapes}"
        )

    scripted = torch.jit.trace(wrapper, example_input, strict=False, check_trace=False)
    scripted = torch.jit.freeze(scripted.eval())

    with torch.no_grad():
        scripted_out = scripted(example_input)

    verification = [
        tensor_report("det", eager_out[0], scripted_out[0], atol=atol, rtol=rtol),
        tensor_report("tl_attr", eager_out[1], scripted_out[1], atol=atol, rtol=rtol),
        tensor_report("lane", eager_out[2], scripted_out[2], atol=atol, rtol=rtol),
        tensor_report("stop_line", eager_out[3], scripted_out[3], atol=atol, rtol=rtol),
    ]
    failed = [item for item in verification if not item.allclose]
    if failed:
        raise RuntimeError(
            "export verification failed: "
            + ", ".join(f"{item.name}(max_abs_diff={item.max_abs_diff:.6f})" for item in failed)
        )

    scripted.save(str(output_path))
    meta = export_metadata(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        trunk_weights=resolved_trunk_weights,
        input_height=int(example_input.shape[-2]),
        input_width=int(example_input.shape[-1]),
        det_shape=[int(v) for v in eager_out[0].shape],
        tl_attr_shape=[int(v) for v in eager_out[1].shape],
        lane_shape=[int(v) for v in eager_out[2].shape],
        stop_line_shape=[int(v) for v in eager_out[3].shape],
        od_classes=od_classes or list(DEFAULT_OD_CLASSES),
        tl_bits=tl_bits or list(DEFAULT_TL_BITS),
        lane_classes=lane_classes,
        lane_types=lane_types,
        det_feature_shapes=feature_shapes,
        det_feature_strides=feature_strides,
        example_info=example_info,
        verification=verification,
    )
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
        fp.write("\n")

    return {
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(output_path),
        "meta_path": str(meta_path),
        "trunk_weights": str(resolved_trunk_weights),
        "head_channels": list(head_channels),
        "checkpoint_variant": checkpoint_variant,
        "verification": [
            {
                "name": item.name,
                "shape": item.shape,
                "max_abs_diff": item.max_abs_diff,
                "allclose": item.allclose,
            }
            for item in verification
        ],
    }
