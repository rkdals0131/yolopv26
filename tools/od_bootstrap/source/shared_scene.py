from __future__ import annotations

from pathlib import Path
from typing import Any

from .raw_common import PairRecord
from .shared_raw import extract_filename, extract_image_size


DEFAULT_SCENE_VERSION = "pv26-scene-aihub-v1"


def bbox_to_yolo_line(class_id: int, bbox: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def sample_id(output_dataset_key: str, pair: PairRecord, *, safe_slug) -> str:
    return safe_slug(f"{output_dataset_key}_{pair.split}_{pair.relative_id}")


def build_base_scene(
    output_dataset_key: str,
    pair: PairRecord,
    raw: dict[str, Any],
    output_image_path: Path,
    *,
    scene_version: str = DEFAULT_SCENE_VERSION,
) -> tuple[int, int, dict[str, Any]]:
    assert pair.image_path is not None
    width, height = extract_image_size(raw, pair.image_path)
    scene = {
        "version": scene_version,
        "image": {
            "file_name": output_image_path.name,
            "width": width,
            "height": height,
            "original_file_name": extract_filename(raw, pair.image_file_name),
        },
        "source": {
            "dataset": output_dataset_key,
            "raw_id": pair.relative_id,
            "split": pair.split,
            "image_path": str(pair.image_path),
            "label_path": str(pair.label_path),
        },
    }
    return width, height, scene


__all__ = [
    "DEFAULT_SCENE_VERSION",
    "bbox_to_yolo_line",
    "build_base_scene",
    "sample_id",
]
