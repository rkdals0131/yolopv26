from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.pv26_schema import AIHUB_OBSTACLE_DATASET_KEY, OD_CLASS_TO_ID, TL_BITS

from .aihub_worker_common import (
    StandardizeTask,
    _base_scene,
    _bbox_to_yolo_line,
    _counter_to_dict,
    _link_or_copy,
    _sample_id,
    _write_json,
    _write_text,
)
from .raw_common import (
    _extract_annotations,
    _extract_attribute_map,
    _extract_bbox,
    _extract_points,
    _extract_tl_state,
    _load_json,
    _normalize_text,
    _safe_slug,
)


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


def _lane_class_from_color(color: str) -> tuple[str | None, str | None]:
    if color == "white":
        return "white_lane", None
    if color == "yellow":
        return "yellow_lane", None
    if color == "blue":
        return "blue_lane", None
    return None, "lane_color_unmapped"


def _tl_bits_from_annotation(annotation: dict[str, Any]) -> tuple[dict[str, int], int, str]:
    bits = {bit: 0 for bit in TL_BITS}
    light_type = _normalize_text(annotation.get("type"))
    if light_type != "car":
        return bits, 0, "non_car_traffic_light"

    raw_attribute = annotation.get("attribute")
    candidate_items = raw_attribute if isinstance(raw_attribute, list) else [raw_attribute]
    attribute_map: dict[str, str] | None = None
    for item in candidate_items:
        if isinstance(item, dict):
            attribute_map = {str(key): str(value).strip().lower() for key, value in item.items()}
            break
    if attribute_map is None:
        return bits, 0, "missing_attribute_map"

    red_on = attribute_map.get("red") == "on"
    yellow_on = attribute_map.get("yellow") == "on"
    green_on = attribute_map.get("green") == "on"
    arrow_on = attribute_map.get("left_arrow") == "on" or attribute_map.get("others_arrow") == "on"
    x_light_on = attribute_map.get("x_light") == "on"

    bits["red"] = int(red_on)
    bits["yellow"] = int(yellow_on)
    bits["green"] = int(green_on)
    bits["arrow"] = int(arrow_on)

    base_on_count = sum(int(flag) for flag in (red_on, yellow_on, green_on))
    if x_light_on:
        return bits, 0, "x_light_active"
    if base_on_count > 1:
        return bits, 0, "multi_color_active"
    return bits, 1, "valid"


def _combo_name(bits: dict[str, int]) -> str:
    active = [key for key, value in bits.items() if value]
    return "+".join(active) if active else "off"


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
        lookup[int(category_id)] = _normalize_text(item.get("name"))
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

    raw = _load_json(Path(label_path_text))
    width = int(scene.get("image", {}).get("width") or 0)
    height = int(scene.get("image", {}).get("height") or 0)
    if width <= 0 or height <= 0:
        return []

    category_lookup = _obstacle_category_lookup(raw)
    rectangles: list[dict[str, Any]] = []
    for annotation in _extract_annotations(raw):
        raw_category_id = annotation.get("category_id")
        raw_category = None
        if raw_category_id is not None:
            try:
                raw_category = category_lookup.get(int(raw_category_id))
            except (TypeError, ValueError):
                raw_category = None
        if raw_category is None:
            raw_category = _normalize_text(annotation.get("class")) or "unknown"

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


def _prepare_debug_scene_for_overlay(scene: dict[str, Any]) -> dict[str, Any]:
    if scene.get("source", {}).get("dataset") != AIHUB_OBSTACLE_DATASET_KEY:
        return scene

    debug_rectangles = _obstacle_debug_rectangles(scene)
    if not debug_rectangles:
        return scene

    overlay_scene = dict(scene)
    overlay_scene["debug_rectangles"] = debug_rectangles
    return overlay_scene


def _lane_worker(task: StandardizeTask) -> dict[str, Any]:
    pair = task.pair
    assert pair.image_path is not None
    output_root = Path(task.output_root)
    raw = _load_json(pair.label_path)
    sample_id = _sample_id(task.output_dataset_key, pair, safe_slug=_safe_slug)
    image_output_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = _link_or_copy(pair.image_path, image_output_path)
    width, height, scene = _base_scene(task.output_dataset_key, pair, raw, image_output_path)
    del width, height

    lanes: list[dict[str, Any]] = []
    stop_lines: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    held_reason_counts = Counter()
    lane_class_counts = Counter()
    lane_type_counts = Counter()

    for annotation in _extract_annotations(raw):
        raw_class = _normalize_text(annotation.get("class"))
        if raw_class in {"traffic_lane", "lane", "lanes"}:
            points = _extract_points(annotation)
            attributes = _extract_attribute_map(annotation)
            lane_color = _normalize_text(attributes.get("lane_color"))
            lane_type = _normalize_text(attributes.get("lane_type"))
            class_name, error_reason = _lane_class_from_color(lane_color)
            if len(points) < 2:
                held_reason_counts["lane_requires_two_points"] += 1
                held_annotations.append({"raw_class": raw_class, "reason": "lane_requires_two_points"})
                continue
            if class_name is None:
                held_reason_counts[error_reason or "lane_color_unmapped"] += 1
                held_annotations.append(
                    {
                        "raw_class": raw_class,
                        "reason": error_reason or "lane_color_unmapped",
                        "raw_color": lane_color,
                    }
                )
                continue
            lanes.append(
                {
                    "id": len(lanes),
                    "class_name": class_name,
                    "source_style": lane_type or None,
                    "points": points,
                    "visibility": [1] * len(points),
                    "meta": {
                        "dataset_label": raw_class,
                        "raw_color": lane_color,
                        "raw_type": lane_type,
                    },
                }
            )
            lane_class_counts[class_name] += 1
            lane_type_counts[lane_type or "missing"] += 1
            continue

        if raw_class in {"stop_line", "stopline"}:
            points = _extract_points(annotation)
            if len(points) < 2:
                held_reason_counts["stop_line_requires_two_points"] += 1
                held_annotations.append({"raw_class": raw_class, "reason": "stop_line_requires_two_points"})
                continue
            stop_lines.append(
                {
                    "id": len(stop_lines),
                    "points": points,
                    "p1": points[0],
                    "p2": points[-1],
                    "meta": {"dataset_label": "stop_line"},
                }
            )
            continue

        if raw_class in {"crosswalk", "cross_walk"}:
            points = _extract_points(annotation)
            if len(points) < 3:
                held_reason_counts["crosswalk_requires_three_points"] += 1
                held_annotations.append({"raw_class": raw_class, "reason": "crosswalk_requires_three_points"})
                continue
            crosswalks.append(
                {
                    "id": len(crosswalks),
                    "class_name": "crosswalk",
                    "points": points,
                    "meta": {"dataset_label": "crosswalk"},
                }
            )
            continue

        held_reason_counts["unrecognized_lane_annotation"] += 1
        held_annotations.append({"raw_class": raw_class or "unclassified", "reason": "unrecognized_lane_annotation"})

    scene.update(
        {
            "tasks": {
                "has_det": 0,
                "has_lane": int(bool(lanes)),
                "has_stop_line": int(bool(stop_lines)),
                "has_crosswalk": int(bool(crosswalks)),
                "has_tl_attr": 0,
            },
            "detections": [],
            "traffic_lights": [],
            "traffic_signs": [],
            "auxiliary_annotations": [],
            "lanes": lanes,
            "stop_lines": stop_lines,
            "crosswalks": crosswalks,
            "held_annotations": held_annotations,
            "notes": [
                "Lane/stop/crosswalk outputs preserve AIHUB geometry and attributes for later target encoding.",
                "OD labels are intentionally omitted for lane-only source scenes.",
            ],
        }
    )

    scene_path = output_root / "labels_scene" / pair.split / f"{sample_id}.json"
    _write_json(scene_path, scene)

    return {
        "dataset_key": task.output_dataset_key,
        "split": pair.split,
        "sample_id": sample_id,
        "scene_path": str(scene_path),
        "image_path": str(image_output_path),
        "image_materialization": image_materialization,
        "lane_count": len(lanes),
        "stop_line_count": len(stop_lines),
        "crosswalk_count": len(crosswalks),
        "det_count": 0,
        "traffic_light_count": 0,
        "traffic_sign_count": 0,
        "tl_attr_valid_count": 0,
        "tl_attr_invalid_count": 0,
        "lane_class_counts": _counter_to_dict(lane_class_counts),
        "lane_type_counts": _counter_to_dict(lane_type_counts),
        "det_class_counts": {},
        "tl_combo_counts": {},
        "tl_invalid_reason_counts": {},
        "held_reason_counts": _counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


def _obstacle_worker(task: StandardizeTask) -> dict[str, Any]:
    pair = task.pair
    assert pair.image_path is not None
    output_root = Path(task.output_root)
    raw = _load_json(pair.label_path)
    sample_id = _sample_id(task.output_dataset_key, pair, safe_slug=_safe_slug)
    image_output_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = _link_or_copy(pair.image_path, image_output_path)
    width, height, scene = _base_scene(task.output_dataset_key, pair, raw, image_output_path)

    category_lookup = _obstacle_category_lookup(raw)
    detections: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    det_lines: list[str] = []
    det_class_counts = Counter()
    held_reason_counts = Counter()

    for annotation in _extract_annotations(raw):
        raw_category_id = annotation.get("category_id")
        raw_category = None
        if raw_category_id is not None:
            try:
                raw_category = category_lookup.get(int(raw_category_id))
            except (TypeError, ValueError):
                raw_category = None
        if raw_category is None:
            raw_category = _normalize_text(annotation.get("class")) or "unknown"
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
        det_lines.append(_bbox_to_yolo_line(OD_CLASS_TO_ID[mapped_class], bbox, width, height))

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

    scene_path = output_root / "labels_scene" / pair.split / f"{sample_id}.json"
    _write_json(scene_path, scene)

    if det_lines:
        det_path = output_root / "labels_det" / pair.split / f"{sample_id}.txt"
        _write_text(det_path, "\n".join(det_lines) + "\n")

    return {
        "dataset_key": task.output_dataset_key,
        "split": pair.split,
        "sample_id": sample_id,
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
        "det_class_counts": _counter_to_dict(det_class_counts),
        "tl_combo_counts": {},
        "tl_invalid_reason_counts": {},
        "held_reason_counts": _counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


def _traffic_worker(task: StandardizeTask) -> dict[str, Any]:
    pair = task.pair
    assert pair.image_path is not None
    output_root = Path(task.output_root)
    raw = _load_json(pair.label_path)
    sample_id = _sample_id(task.output_dataset_key, pair, safe_slug=_safe_slug)
    image_output_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = _link_or_copy(pair.image_path, image_output_path)
    width, height, scene = _base_scene(task.output_dataset_key, pair, raw, image_output_path)

    detections: list[dict[str, Any]] = []
    traffic_lights: list[dict[str, Any]] = []
    traffic_signs: list[dict[str, Any]] = []
    auxiliary_annotations: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    det_lines: list[str] = []
    det_class_counts = Counter()
    tl_combo_counts = Counter()
    tl_invalid_reason_counts = Counter()
    held_reason_counts = Counter()

    for annotation in _extract_annotations(raw):
        raw_class = _normalize_text(annotation.get("class"))
        if raw_class in {"traffic_light", "light"}:
            bbox = _extract_bbox(annotation, width, height)
            if bbox is None:
                held_reason_counts["traffic_light_invalid_bbox"] += 1
                held_annotations.append({"raw_class": raw_class, "reason": "traffic_light_invalid_bbox"})
                continue

            detection_id = len(detections)
            detections.append(
                {
                    "id": detection_id,
                    "class_name": "traffic_light",
                    "bbox": bbox,
                    "score": None,
                    "meta": {"dataset_label": "traffic_light"},
                }
            )
            det_class_counts["traffic_light"] += 1
            det_lines.append(_bbox_to_yolo_line(OD_CLASS_TO_ID["traffic_light"], bbox, width, height))

            bits, valid, status_reason = _tl_bits_from_annotation(annotation)
            traffic_lights.append(
                {
                    "id": len(traffic_lights),
                    "detection_id": detection_id,
                    "bbox": bbox,
                    "tl_bits": bits,
                    "tl_attr_valid": valid,
                    "collapse_reason": status_reason,
                    "state_hint": _extract_tl_state(annotation),
                    "light_count": annotation.get("light_count"),
                    "type": annotation.get("type"),
                    "direction": annotation.get("direction"),
                    "raw_attribute": annotation.get("attribute"),
                    "meta": {"dataset_label": "traffic_light"},
                }
            )
            if valid:
                tl_combo_counts[_combo_name(bits)] += 1
            else:
                tl_invalid_reason_counts[status_reason] += 1
            continue

        if raw_class in {"traffic_sign", "road_sign", "sign"}:
            bbox = _extract_bbox(annotation, width, height)
            if bbox is None:
                held_reason_counts["traffic_sign_invalid_bbox"] += 1
                held_annotations.append({"raw_class": raw_class, "reason": "traffic_sign_invalid_bbox"})
                continue

            detection_id = len(detections)
            detections.append(
                {
                    "id": detection_id,
                    "class_name": "sign",
                    "bbox": bbox,
                    "score": None,
                    "meta": {"dataset_label": "traffic_sign"},
                }
            )
            det_class_counts["sign"] += 1
            det_lines.append(_bbox_to_yolo_line(OD_CLASS_TO_ID["sign"], bbox, width, height))
            traffic_signs.append(
                {
                    "id": len(traffic_signs),
                    "detection_id": detection_id,
                    "bbox": bbox,
                    "shape": annotation.get("shape"),
                    "color": annotation.get("color"),
                    "kind": annotation.get("kind"),
                    "type": annotation.get("type"),
                    "text": annotation.get("text"),
                    "meta": {"dataset_label": "traffic_sign"},
                }
            )
            continue

        if raw_class in {"traffic_information", "traffic_info"}:
            auxiliary_annotations.append(
                {
                    "id": len(auxiliary_annotations),
                    "class_name": "traffic_information",
                    "bbox": _extract_bbox(annotation, width, height),
                    "type": annotation.get("type"),
                    "raw_annotation": annotation,
                }
            )
            continue

        held_reason_counts["unrecognized_traffic_annotation"] += 1
        held_annotations.append({"raw_class": raw_class or "unclassified", "reason": "unrecognized_traffic_annotation"})

    scene.update(
        {
            "tasks": {
                "has_det": int(bool(detections)),
                "has_lane": 0,
                "has_stop_line": 0,
                "has_crosswalk": 0,
                "has_tl_attr": int(any(item["tl_attr_valid"] for item in traffic_lights)),
            },
            "detections": detections,
            "traffic_lights": traffic_lights,
            "traffic_signs": traffic_signs,
            "auxiliary_annotations": auxiliary_annotations,
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
            "held_annotations": held_annotations,
            "notes": [
                "Traffic-light bbox stays generic for OD while TL attribute supervision is carried in tl_bits.",
                "AIHUB sign objects are normalized to detector class sign.",
            ],
        }
    )

    scene_path = output_root / "labels_scene" / pair.split / f"{sample_id}.json"
    _write_json(scene_path, scene)

    if det_lines:
        det_path = output_root / "labels_det" / pair.split / f"{sample_id}.txt"
        _write_text(det_path, "\n".join(det_lines) + "\n")

    return {
        "dataset_key": task.output_dataset_key,
        "split": pair.split,
        "sample_id": sample_id,
        "scene_path": str(scene_path),
        "image_path": str(image_output_path),
        "image_materialization": image_materialization,
        "lane_count": 0,
        "stop_line_count": 0,
        "crosswalk_count": 0,
        "det_count": len(detections),
        "traffic_light_count": len(traffic_lights),
        "traffic_sign_count": len(traffic_signs),
        "tl_attr_valid_count": sum(1 for item in traffic_lights if item["tl_attr_valid"]),
        "tl_attr_invalid_count": sum(1 for item in traffic_lights if not item["tl_attr_valid"]),
        "lane_class_counts": {},
        "lane_type_counts": {},
        "det_class_counts": _counter_to_dict(det_class_counts),
        "tl_combo_counts": _counter_to_dict(tl_combo_counts),
        "tl_invalid_reason_counts": _counter_to_dict(tl_invalid_reason_counts),
        "held_reason_counts": _counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


def _worker_entry(task: StandardizeTask) -> dict[str, Any]:
    if task.dataset_kind == "lane":
        return _lane_worker(task)
    if task.dataset_kind == "obstacle":
        return _obstacle_worker(task)
    if task.dataset_kind == "traffic":
        return _traffic_worker(task)
    raise ValueError(f"unsupported dataset kind: {task.dataset_kind}")


def _worker_chunk_entry(tasks: list[StandardizeTask]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        try:
            results.append({"summary": _worker_entry(task)})
        except Exception as exc:
            results.append(
                {
                    "failure": {
                        "dataset_kind": task.dataset_kind,
                        "dataset_key": task.output_dataset_key,
                        "split": task.pair.split,
                        "raw_id": task.pair.relative_id,
                        "image_path": str(task.pair.image_path),
                        "label_path": str(task.pair.label_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                }
            )
    return results


def _reason_counts_from_held_annotations(held_annotations: Any) -> dict[str, int]:
    counter = Counter()
    if not isinstance(held_annotations, list):
        return {}
    for item in held_annotations:
        if not isinstance(item, dict):
            continue
        reason = _normalize_text(item.get("reason")) or "unknown"
        counter[reason] += 1
    return _counter_to_dict(counter)


def _existing_output_summary(task: StandardizeTask) -> dict[str, Any] | None:
    sample_id = _sample_id(task.output_dataset_key, task.pair, safe_slug=_safe_slug)
    output_root = Path(task.output_root)
    image_output_path = output_root / "images" / task.pair.split / f"{sample_id}{task.pair.image_path.suffix.lower()}"
    scene_path = output_root / "labels_scene" / task.pair.split / f"{sample_id}.json"
    det_path = output_root / "labels_det" / task.pair.split / f"{sample_id}.txt"
    if not image_output_path.is_file() or not scene_path.is_file():
        return None

    try:
        scene = _load_json(scene_path)
    except Exception:
        return None

    detections = scene.get("detections") if isinstance(scene.get("detections"), list) else []
    lanes = scene.get("lanes") if isinstance(scene.get("lanes"), list) else []
    stop_lines = scene.get("stop_lines") if isinstance(scene.get("stop_lines"), list) else []
    crosswalks = scene.get("crosswalks") if isinstance(scene.get("crosswalks"), list) else []
    traffic_lights = scene.get("traffic_lights") if isinstance(scene.get("traffic_lights"), list) else []
    traffic_signs = scene.get("traffic_signs") if isinstance(scene.get("traffic_signs"), list) else []

    if detections and not det_path.is_file():
        return None

    det_class_counts = Counter()
    for item in detections:
        if isinstance(item, dict):
            det_class_counts[_normalize_text(item.get("class_name")) or "unknown"] += 1

    lane_class_counts = Counter()
    lane_type_counts = Counter()
    for item in lanes:
        if not isinstance(item, dict):
            continue
        lane_class_counts[_normalize_text(item.get("class_name")) or "unknown"] += 1
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        lane_type = _normalize_text(meta.get("raw_type") or item.get("source_style")) or "missing"
        lane_type_counts[lane_type] += 1

    tl_combo_counts = Counter()
    tl_invalid_reason_counts = Counter()
    tl_attr_valid_count = 0
    for item in traffic_lights:
        if not isinstance(item, dict):
            continue
        bits_raw = item.get("tl_bits") if isinstance(item.get("tl_bits"), dict) else {}
        bits = {bit: int(bits_raw.get(bit, 0)) for bit in TL_BITS}
        valid = int(item.get("tl_attr_valid", 0))
        reason = _normalize_text(item.get("collapse_reason")) or "unknown"
        if valid:
            tl_combo_counts[_combo_name(bits)] += 1
            tl_attr_valid_count += 1
        else:
            tl_invalid_reason_counts[reason] += 1

    return {
        "dataset_key": task.output_dataset_key,
        "split": task.pair.split,
        "sample_id": sample_id,
        "scene_path": str(scene_path),
        "image_path": str(image_output_path),
        "image_materialization": "resume_existing",
        "lane_count": len(lanes),
        "stop_line_count": len(stop_lines),
        "crosswalk_count": len(crosswalks),
        "det_count": len(detections),
        "traffic_light_count": len(traffic_lights),
        "traffic_sign_count": len(traffic_signs),
        "tl_attr_valid_count": tl_attr_valid_count,
        "tl_attr_invalid_count": len(traffic_lights) - tl_attr_valid_count,
        "lane_class_counts": _counter_to_dict(lane_class_counts),
        "lane_type_counts": _counter_to_dict(lane_type_counts),
        "det_class_counts": _counter_to_dict(det_class_counts),
        "tl_combo_counts": _counter_to_dict(tl_combo_counts),
        "tl_invalid_reason_counts": _counter_to_dict(tl_invalid_reason_counts),
        "held_reason_counts": _reason_counts_from_held_annotations(scene.get("held_annotations")),
        "resume_skipped": 1,
    }
