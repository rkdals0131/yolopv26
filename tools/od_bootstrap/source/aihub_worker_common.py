from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .raw_common import PairRecord
from .shared_io import link_or_copy, write_json, write_text
from .shared_scene import bbox_to_yolo_line, build_base_scene, sample_id
from .shared_summary import counter_to_dict


SCENE_VERSION = "pv26-scene-aihub-v1"


@dataclass(frozen=True)
class StandardizeTask:
    dataset_kind: str
    output_dataset_key: str
    pair: PairRecord
    output_root: str


def _write_text(path: Path, contents: str) -> None:
    write_text(path, contents)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return counter_to_dict(counter)


def _link_or_copy(source_path: Path, target_path: Path) -> str:
    return link_or_copy(source_path, target_path)


def _bbox_to_yolo_line(class_id: int, bbox: list[float], width: int, height: int) -> str:
    return bbox_to_yolo_line(class_id, bbox, width, height)


def _sample_id(output_dataset_key: str, pair: PairRecord, *, safe_slug) -> str:
    return sample_id(output_dataset_key, pair, safe_slug=safe_slug)


def _base_scene(
    output_dataset_key: str,
    pair: PairRecord,
    raw: dict[str, Any],
    output_image_path: Path,
) -> tuple[int, int, dict[str, Any]]:
    return build_base_scene(
        output_dataset_key,
        pair,
        raw,
        output_image_path,
        scene_version=SCENE_VERSION,
    )
