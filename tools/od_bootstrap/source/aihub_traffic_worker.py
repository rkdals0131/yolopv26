from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.pv26_schema import OD_CLASS_TO_ID, TL_BITS

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
    _extract_bbox,
    _extract_tl_state,
    _load_json,
    _normalize_text,
    _safe_slug,
)


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


def combo_name(bits: dict[str, int]) -> str:
    active = [key for key, value in bits.items() if value]
    return "+".join(active) if active else "off"


def traffic_worker(task: StandardizeTask) -> dict[str, Any]:
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
                tl_combo_counts[combo_name(bits)] += 1
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


_combo_name = combo_name
_traffic_worker = traffic_worker
