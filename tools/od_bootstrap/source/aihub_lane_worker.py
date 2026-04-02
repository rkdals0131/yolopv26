from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .aihub_worker_common import SCENE_VERSION, StandardizeTask
from .shared_io import link_or_copy, load_json, write_json
from .shared_raw import extract_annotations, extract_attribute_map, extract_points, normalize_text, safe_slug
from .shared_scene import build_base_scene, sample_id
from .shared_summary import counter_to_dict


def _lane_class_from_color(color: str) -> tuple[str | None, str | None]:
    if color == "white":
        return "white_lane", None
    if color == "yellow":
        return "yellow_lane", None
    if color == "blue":
        return "blue_lane", None
    return None, "lane_color_unmapped"


def lane_worker(task: StandardizeTask) -> dict[str, Any]:
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
    del width, height

    lanes: list[dict[str, Any]] = []
    stop_lines: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    held_reason_counts = Counter()
    lane_class_counts = Counter()
    lane_type_counts = Counter()

    for annotation in extract_annotations(raw):
        raw_class = normalize_text(annotation.get("class"))
        if raw_class in {"traffic_lane", "lane", "lanes"}:
            points = extract_points(annotation)
            attributes = extract_attribute_map(annotation)
            lane_color = normalize_text(attributes.get("lane_color"))
            lane_type = normalize_text(attributes.get("lane_type"))
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
            points = extract_points(annotation)
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
            points = extract_points(annotation)
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

    scene_path = output_root / "labels_scene" / pair.split / f"{scene_sample_id}.json"
    write_json(scene_path, scene)

    return {
        "dataset_key": task.output_dataset_key,
        "split": pair.split,
        "sample_id": scene_sample_id,
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
        "lane_class_counts": counter_to_dict(lane_class_counts),
        "lane_type_counts": counter_to_dict(lane_type_counts),
        "det_class_counts": {},
        "tl_combo_counts": {},
        "tl_invalid_reason_counts": {},
        "held_reason_counts": counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


_lane_worker = lane_worker
