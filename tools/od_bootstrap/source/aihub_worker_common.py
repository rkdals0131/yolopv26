from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .raw_common import PairRecord, _extract_filename, _extract_image_size


SCENE_VERSION = "pv26-scene-aihub-v1"


@dataclass(frozen=True)
class StandardizeTask:
    dataset_kind: str
    output_dataset_key: str
    pair: PairRecord
    output_root: str


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _link_or_copy(source_path: Path, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return "existing"
    try:
        os.link(source_path, target_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"


def _bbox_to_yolo_line(class_id: int, bbox: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def _sample_id(output_dataset_key: str, pair: PairRecord, *, safe_slug) -> str:
    return safe_slug(f"{output_dataset_key}_{pair.split}_{pair.relative_id}")


def _base_scene(
    output_dataset_key: str,
    pair: PairRecord,
    raw: dict[str, Any],
    output_image_path: Path,
) -> tuple[int, int, dict[str, Any]]:
    assert pair.image_path is not None
    width, height = _extract_image_size(raw, pair.image_path)
    scene = {
        "version": SCENE_VERSION,
        "image": {
            "file_name": output_image_path.name,
            "width": width,
            "height": height,
            "original_file_name": _extract_filename(raw, pair.image_file_name),
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

