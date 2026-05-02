from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is an optional runtime dependency.
    Image = None
    ImageDraw = None
    ImageFont = None

from common.overlay import render_overlay
from common.pv26_schema import LANE_CLASSES, OD_CLASSES
from model.data import collate_pv26_samples
from .config import PhaseConfig, PreviewConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_points_to_list(item: dict[str, Any]) -> list[list[float]]:
    points = item.get("points_xy", [])
    if hasattr(points, "detach"):
        points = points.detach().cpu().tolist()
    return [[float(x), float(y)] for x, y in points]


def _gt_scene_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample["meta"]
    det_targets = sample.get("det_targets", {})
    lane_targets = sample.get("lane_targets", {})
    valid_mask = sample.get("valid_mask", {})

    scene: dict[str, Any] = {
        "source": {"image_path": str(meta["image_path"])},
        "detections": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "debug_rectangles": [],
    }

    boxes = det_targets.get("boxes_xyxy", [])
    classes = det_targets.get("classes", [])
    if hasattr(boxes, "detach"):
        boxes_list = boxes.detach().cpu().tolist()
    else:
        boxes_list = list(boxes)
    if hasattr(classes, "detach"):
        class_list = classes.detach().cpu().tolist()
    else:
        class_list = list(classes)
    for box, class_id in zip(boxes_list, class_list):
        class_name = OD_CLASSES[int(class_id)] if 0 <= int(class_id) < len(OD_CLASSES) else "unknown"
        item = {
            "bbox": [float(value) for value in box],
            "class_name": class_name,
        }
        if class_name == "traffic_light":
            scene["traffic_lights"].append(item)
        elif class_name == "sign":
            scene["traffic_signs"].append(item)
        else:
            scene["detections"].append(item)

    lane_valid = valid_mask.get("lane", [])
    if hasattr(lane_valid, "detach"):
        lane_valid = lane_valid.detach().cpu().tolist()
    stop_valid = valid_mask.get("stop_line", [])
    if hasattr(stop_valid, "detach"):
        stop_valid = stop_valid.detach().cpu().tolist()
    cross_valid = valid_mask.get("crosswalk", [])
    if hasattr(cross_valid, "detach"):
        cross_valid = cross_valid.detach().cpu().tolist()

    for index, lane in enumerate(lane_targets.get("lanes", [])):
        if index < len(lane_valid) and not bool(lane_valid[index]):
            continue
        raw_class_name = lane.get("class_name")
        if raw_class_name is None and lane.get("color") is not None:
            color_index = int(lane.get("color"))
            raw_class_name = LANE_CLASSES[color_index] if 0 <= color_index < len(LANE_CLASSES) else "lane"
        scene["lanes"].append(
            {
                "class_name": str(raw_class_name or "lane"),
                "points": _tensor_points_to_list(lane),
            }
        )
    for index, stop_line in enumerate(lane_targets.get("stop_lines", [])):
        if index < len(stop_valid) and not bool(stop_valid[index]):
            continue
        scene["stop_lines"].append({"points": _tensor_points_to_list(stop_line)})
    for index, crosswalk in enumerate(lane_targets.get("crosswalks", [])):
        if index < len(cross_valid) and not bool(cross_valid[index]):
            continue
        scene["crosswalks"].append({"points": _tensor_points_to_list(crosswalk)})
    return scene


def _prediction_to_overlay_scene(prediction: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample["meta"]
    scene: dict[str, Any] = {
        "source": {"image_path": str(meta["image_path"])},
        "detections": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "debug_rectangles": [],
    }
    for detection in prediction.get("detections", []):
        item = {
            "bbox": [float(value) for value in detection.get("box_xyxy", [])],
            "class_name": str(detection.get("class_name") or "unknown"),
        }
        if item["class_name"] == "traffic_light":
            scene["traffic_lights"].append(item)
        elif item["class_name"] == "sign":
            scene["traffic_signs"].append(item)
        else:
            scene["detections"].append(item)
    for lane in prediction.get("lanes", []):
        scene["lanes"].append(
            {
                "class_name": lane.get("class_name"),
                "points": [[float(x), float(y)] for x, y in lane.get("points_xy", [])],
            }
        )
    for stop_line in prediction.get("stop_lines", []):
        scene["stop_lines"].append({"points": [[float(x), float(y)] for x, y in stop_line.get("points_xy", [])]})
    for crosswalk in prediction.get("crosswalks", []):
        scene["crosswalks"].append({"points": [[float(x), float(y)] for x, y in crosswalk.get("points_xy", [])]})
    return scene


def _compose_pair(*, sample_id: str, gt_overlay_path: Path, pred_overlay_path: Path, output_path: Path) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for epoch comparison grids")
    gt = Image.open(gt_overlay_path).convert("RGB")
    pred = Image.open(pred_overlay_path).convert("RGB")
    width = max(gt.width, pred.width)
    height = max(gt.height, pred.height)
    header_height = 42
    canvas = Image.new("RGB", (width * 2, height + header_height), color=(12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas.paste(gt, (0, header_height))
    canvas.paste(pred, (width, header_height))
    draw.text((8, 6), sample_id, fill=(255, 255, 255), font=font)
    draw.text((8, 22), "ground_truth", fill=(255, 255, 255), font=font)
    draw.text((width + 8, 22), "prediction", fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _compose_grid(tile_paths: list[Path], *, columns: int, output_path: Path) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for epoch comparison grids")
    if not tile_paths:
        raise ValueError("cannot compose comparison grid with zero tiles")
    tiles = [Image.open(path).convert("RGB") for path in tile_paths]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = int(math.ceil(len(tiles) / max(columns, 1)))
    canvas = Image.new("RGB", (tile_width * columns, tile_height * rows), color=(8, 8, 8))
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        canvas.paste(tile, (col * tile_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_epoch_comparison_grid_callback(
    *,
    trainer: Any,
    phase: PhaseConfig,
    phase_dir: Path,
    preview_samples: list[dict[str, Any]],
    preview_config: PreviewConfig,
    log_fn: Callable[[str], None],
) -> Callable[[dict[str, Any]], None] | None:
    if not bool(preview_config.epoch_comparison_grid):
        return None
    if not preview_samples:
        log_fn("epoch comparison grid disabled: preview sample set is empty")
        return None
    every_n_epochs = max(1, int(preview_config.epoch_comparison_every_n_epochs))
    sample_count = max(1, int(preview_config.epoch_comparison_sample_count))
    columns = max(1, int(preview_config.epoch_comparison_columns))
    fixed_samples = list(preview_samples[:sample_count])
    output_root = phase_dir / "epoch_comparison_grids"
    _write_json(
        output_root / "manifest.json",
        {
            "enabled": True,
            "phase_name": phase.name,
            "phase_stage": phase.stage,
            "every_n_epochs": every_n_epochs,
            "sample_count": len(fixed_samples),
            "columns": columns,
            "split": preview_config.split,
            "sample_ids": [str(sample["meta"].get("sample_id")) for sample in fixed_samples],
        },
    )
    evaluator = trainer.build_evaluator()

    def _callback(epoch_summary: dict[str, Any]) -> None:
        epoch = int(epoch_summary.get("epoch", 0) or 0)
        if epoch <= 0 or epoch % every_n_epochs != 0:
            return
        try:
            epoch_dir = output_root / f"epoch_{epoch:03d}"
            tile_paths: list[Path] = []
            entries: list[dict[str, Any]] = []
            for index, sample in enumerate(fixed_samples, start=1):
                sample_meta = sample["meta"]
                sample_id = str(sample_meta["sample_id"])
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)[:120]
                prediction = evaluator.predict_batch(collate_pv26_samples([sample]))[0]
                sample_dir = epoch_dir / f"{index:02d}__{safe_id}"
                gt_overlay_path = sample_dir / "ground_truth.png"
                pred_overlay_path = sample_dir / "prediction.png"
                tile_path = sample_dir / "comparison.png"
                render_overlay(_gt_scene_from_sample(sample), gt_overlay_path)
                render_overlay(_prediction_to_overlay_scene(prediction, sample), pred_overlay_path)
                _compose_pair(
                    sample_id=sample_id,
                    gt_overlay_path=gt_overlay_path,
                    pred_overlay_path=pred_overlay_path,
                    output_path=tile_path,
                )
                tile_paths.append(tile_path)
                entries.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": str(sample_meta.get("dataset_key")),
                        "image_path": str(sample_meta.get("image_path")),
                        "comparison_path": str(tile_path),
                    }
                )
            grid_path = epoch_dir / "comparison_grid.png"
            _compose_grid(tile_paths, columns=columns, output_path=grid_path)
            _write_json(epoch_dir / "summary.json", {"epoch": epoch, "grid_path": str(grid_path), "samples": entries})
            epoch_summary.setdefault("artifacts", {})["epoch_comparison_grid"] = str(grid_path)
            log_fn(f"epoch comparison grid saved epoch={epoch} path={grid_path} samples={len(entries)}")
        except Exception as exc:  # pragma: no cover - visualization must not kill training.
            log_fn(f"epoch comparison grid failed epoch={epoch} error={exc}")

    return _callback


__all__ = ["build_epoch_comparison_grid_callback"]
