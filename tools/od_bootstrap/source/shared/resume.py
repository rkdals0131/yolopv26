from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from .summary import counter_to_dict


def count_held_annotation_reasons(
    held_annotations: Any,
    *,
    normalize_reason: Callable[[Any], str] | None = None,
) -> dict[str, int]:
    counter = Counter()
    if not isinstance(held_annotations, list):
        return {}
    for item in held_annotations:
        if not isinstance(item, dict):
            continue
        if normalize_reason is None:
            reason = str(item.get("reason") or "unknown").strip().lower()
        else:
            reason = normalize_reason(item.get("reason")) or "unknown"
        counter[reason] += 1
    return counter_to_dict(counter)


def load_existing_scene_output(
    *,
    output_root: Path,
    split: str,
    sample_id: str,
    image_suffix: str,
    load_json_fn: Callable[[Path], dict[str, Any]],
    scene_version: str | None = None,
    expected_dataset_key: str | None = None,
    expected_split: str | None = None,
    expected_image_file_name: str | None = None,
) -> dict[str, Any] | None:
    image_path = output_root / "images" / split / f"{sample_id}{image_suffix}"
    scene_path = output_root / "labels_scene" / split / f"{sample_id}.json"
    det_path = output_root / "labels_det" / split / f"{sample_id}.txt"
    if not image_path.is_file() or not scene_path.is_file():
        return None

    try:
        scene = load_json_fn(scene_path)
    except Exception:
        return None

    if scene_version is not None and str(scene.get("version") or "").strip() != scene_version:
        return None

    source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
    if expected_dataset_key is not None and str(source.get("dataset") or "").strip() != expected_dataset_key:
        return None
    if expected_split is not None and str(source.get("split") or "").strip() != expected_split:
        return None
    if expected_image_file_name is not None:
        image = scene.get("image") if isinstance(scene.get("image"), dict) else {}
        if str(image.get("file_name") or "").strip() != expected_image_file_name:
            return None

    return {
        "image_path": image_path,
        "scene_path": scene_path,
        "det_path": det_path,
        "scene": scene,
    }


def scene_detections_match_labels_det(
    detections: Any,
    det_path: Path,
    *,
    class_to_id: Mapping[str, int],
) -> bool:
    if not isinstance(detections, list):
        return False
    if not detections:
        return not det_path.is_file()
    if not det_path.is_file():
        return False

    try:
        det_rows = [line.split() for line in det_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return False
    if len(det_rows) != len(detections):
        return False

    for row_index, (detection, det_row) in enumerate(zip(detections, det_rows)):
        if not isinstance(detection, dict):
            return False
        try:
            detection_id = int(detection.get("id"))
        except (TypeError, ValueError):
            return False
        if detection_id != row_index:
            return False
        if len(det_row) != 5:
            return False
        class_name = str(detection.get("class_name") or "").strip()
        expected_class_id = class_to_id.get(class_name)
        if expected_class_id is None:
            return False
        try:
            det_class_id = int(det_row[0])
        except (TypeError, ValueError):
            return False
        if det_class_id != int(expected_class_id):
            return False
    return True


__all__ = [
    "count_held_annotation_reasons",
    "load_existing_scene_output",
    "scene_detections_match_labels_det",
]
