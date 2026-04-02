from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.pv26_schema import AIHUB_OBSTACLE_DATASET_KEY, OD_CLASS_TO_ID

from .aihub_worker_common import SCENE_VERSION, StandardizeTask
from .shared_io import link_or_copy, load_json, write_json, write_text
from .shared_raw import extract_annotations, normalize_text, safe_slug
from .shared_scene import bbox_to_yolo_line, build_base_scene, sample_id
from .shared_summary import counter_to_dict


OBSTACLE_CLASS_REMAP = {
    "traffic_cone": "traffic_cone",
    "animals(dolls)": "obstacle",
    "garbage_bag_&_sacks": "obstacle",
    "construction_signs_&_parking_prohibited_board": "obstacle",
    "box": "obstacle",
    "stones_on_road": "obstacle",
}
OBSTACLE_EXCLUDED_REASONS = {
    "person": "excluded_obstacle_person_policy",
    "manhole": "excluded_obstacle_manhole_policy",
    "pothole_on_road": "excluded_obstacle_pothole_policy",
    "filled_pothole": "excluded_obstacle_filled_pothole_policy",
}
DEBUG_VIS_REASON_COLORS = {
    "excluded": "#00e5ff",
    "unrecognized": "#ffe066",
    "invalid_bbox": "#ff5a5f",
}


def _obstacle_category_lookup(raw: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    categories = raw.get("categories")
    if not isinstance(categories, list):
        return lookup
    for item in categories:
        if not isinstance(item, dict):
            continue
        category_id = item.get("id")
        if category_id is None:
            continue
        lookup[int(category_id)] = normalize_text(item.get("name"))
    return lookup


def _obstacle_bbox(annotation: dict[str, Any], width: int, height: int) -> list[float] | None:
    box = annotation.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x_value = float(box[0])
    y_value = float(box[1])
    w_value = float(box[2])
    h_value = float(box[3])
    if w_value <= 0.0 or h_value <= 0.0:
        return None
    x1 = max(0.0, min(x_value, float(width)))
    y1 = max(0.0, min(y_value, float(height)))
    x2 = max(0.0, min(x_value + w_value, float(width)))
    y2 = max(0.0, min(y_value + h_value, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _debug_vis_color_for_reason(reason: str) -> str:
    if reason.startswith("excluded_obstacle_"):
        return DEBUG_VIS_REASON_COLORS["excluded"]
    if reason == "unrecognized_obstacle_annotation":
        return DEBUG_VIS_REASON_COLORS["unrecognized"]
    if reason.endswith("_invalid_bbox"):
        return DEBUG_VIS_REASON_COLORS["invalid_bbox"]
    return DEBUG_VIS_REASON_COLORS["excluded"]


def _obstacle_debug_rectangles(scene: dict[str, Any]) -> list[dict[str, Any]]:
    label_path_text = scene.get("source", {}).get("label_path")
    if not label_path_text:
        return []

    raw = load_json(Path(label_path_text))
    width = int(scene.get("image", {}).get("width") or 0)
    height = int(scene.get("image", {}).get("height") or 0)
    if width <= 0 or height <= 0:
        return []

    category_lookup = _obstacle_category_lookup(raw)
    rectangles: list[dict[str, Any]] = []
    for annotation in extract_annotations(raw):
        raw_category_id = annotation.get("category_id")
        raw_category = None
        if raw_category_id is not None:
            try:
                raw_category = category_lookup.get(int(raw_category_id))
            except (TypeError, ValueError):
                raw_category = None
        if raw_category is None:
            raw_category = normalize_text(annotation.get("class")) or "unknown"

        excluded_reason = OBSTACLE_EXCLUDED_REASONS.get(raw_category)
        if excluded_reason is not None:
            bbox = _obstacle_bbox(annotation, width, height)
            if bbox is None:
                continue
            rectangles.append(
                {
                    "bbox": bbox,
                    "color": _debug_vis_color_for_reason(excluded_reason),
                    "label": f"excluded:{raw_category}",
                }
            )
            continue

        mapped_class = OBSTACLE_CLASS_REMAP.get(raw_category)
        if mapped_class is None:
            bbox = _obstacle_bbox(annotation, width, height)
            if bbox is None:
                continue
            rectangles.append(
                {
                    "bbox": bbox,
                    "color": _debug_vis_color_for_reason("unrecognized_obstacle_annotation"),
                    "label": f"unrecognized:{raw_category}",
                }
            )

    return rectangles


def prepare_debug_scene_for_overlay(scene: dict[str, Any]) -> dict[str, Any]:
    if scene.get("source", {}).get("dataset") != AIHUB_OBSTACLE_DATASET_KEY:
        return scene

    debug_rectangles = _obstacle_debug_rectangles(scene)
    if not debug_rectangles:
        return scene

    overlay_scene = dict(scene)
    overlay_scene["debug_rectangles"] = debug_rectangles
    return overlay_scene


def obstacle_worker(task: StandardizeTask) -> dict[str, Any]:
    pair = task.pair
    assert pair.image_path is not None
    output_root = Path(task.output_root)
    raw = load_json(pair.label_path)
    scene_sample_id = sample_id(task.output_dataset_key, pair, safe_slug=safe_slug)
    image_output_path = output_root / "images" / pair.split / f"{scene_sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = link_or_copy(pair.image_path, image_output_path)
    width, height, scene = build_base_scene(
        task.output_dataset_key,
        pair,
        raw,
        image_output_path,
        scene_version=SCENE_VERSION,
    )

    category_lookup = _obstacle_category_lookup(raw)
    detections: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    det_lines: list[str] = []
    det_class_counts = Counter()
    held_reason_counts = Counter()

    for annotation in extract_annotations(raw):
        raw_category_id = annotation.get("category_id")
        raw_category = None
        if raw_category_id is not None:
            try:
                raw_category = category_lookup.get(int(raw_category_id))
            except (TypeError, ValueError):
                raw_category = None
        if raw_category is None:
            raw_category = normalize_text(annotation.get("class")) or "unknown"
        excluded_reason = OBSTACLE_EXCLUDED_REASONS.get(raw_category)
        if excluded_reason is not None:
            held_reason_counts[excluded_reason] += 1
            held_annotations.append({"raw_class": raw_category, "reason": excluded_reason})
            continue

        mapped_class = OBSTACLE_CLASS_REMAP.get(raw_category)
        if mapped_class is None:
            held_reason_counts["unrecognized_obstacle_annotation"] += 1
            held_annotations.append({"raw_class": raw_category, "reason": "unrecognized_obstacle_annotation"})
            continue

        bbox = _obstacle_bbox(annotation, width, height)
        if bbox is None:
            reason = f"{mapped_class}_invalid_bbox"
            held_reason_counts[reason] += 1
            held_annotations.append({"raw_class": raw_category, "reason": reason})
            continue

        detection_id = len(detections)
        detections.append(
            {
                "id": detection_id,
                "class_name": mapped_class,
                "bbox": bbox,
                "score": None,
                "meta": {"dataset_label": raw_category},
            }
        )
        det_class_counts[mapped_class] += 1
        det_lines.append(bbox_to_yolo_line(OD_CLASS_TO_ID[mapped_class], bbox, width, height))

    scene.update(
        {
            "tasks": {
                "has_det": int(bool(detections)),
                "has_lane": 0,
                "has_stop_line": 0,
                "has_crosswalk": 0,
                "has_tl_attr": 0,
            },
            "detections": detections,
            "traffic_lights": [],
            "traffic_signs": [],
            "auxiliary_annotations": [],
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
            "held_annotations": held_annotations,
            "notes": [
                "Obstacle source is standardized as a detector-only AIHUB source for PV26.",
                "Only traffic_cone and obstacle detector classes are retained from this source.",
            ],
        }
    )

    scene_path = output_root / "labels_scene" / pair.split / f"{scene_sample_id}.json"
    write_json(scene_path, scene)

    if det_lines:
        det_path = output_root / "labels_det" / pair.split / f"{scene_sample_id}.txt"
        write_text(det_path, "\n".join(det_lines) + "\n")

    return {
        "dataset_key": task.output_dataset_key,
        "split": pair.split,
        "sample_id": scene_sample_id,
        "scene_path": str(scene_path),
        "image_path": str(image_output_path),
        "image_materialization": image_materialization,
        "lane_count": 0,
        "stop_line_count": 0,
        "crosswalk_count": 0,
        "det_count": len(detections),
        "traffic_light_count": 0,
        "traffic_sign_count": 0,
        "tl_attr_valid_count": 0,
        "tl_attr_invalid_count": 0,
        "lane_class_counts": {},
        "lane_type_counts": {},
        "det_class_counts": counter_to_dict(det_class_counts),
        "tl_combo_counts": {},
        "tl_invalid_reason_counts": {},
        "held_reason_counts": counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


_prepare_debug_scene_for_overlay = prepare_debug_scene_for_overlay
_obstacle_worker = obstacle_worker
