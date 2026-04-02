from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.pv26_schema import TL_BITS

from .aihub_lane_worker import lane_worker
from .aihub_obstacle_worker import obstacle_worker, prepare_debug_scene_for_overlay
from .aihub_traffic_worker import combo_name, traffic_worker
from .aihub_worker_common import StandardizeTask
from .shared_io import load_json as _load_json
from .shared_raw import normalize_text as _normalize_text, safe_slug as _safe_slug
from .shared_scene import sample_id as _sample_id
from .shared_summary import counter_to_dict as _counter_to_dict

load_json = _load_json
normalize_text = _normalize_text
safe_slug = _safe_slug
sample_id = _sample_id
counter_to_dict = _counter_to_dict


def _worker_entry(task: StandardizeTask) -> dict[str, Any]:
    if task.dataset_kind == "lane":
        return lane_worker(task)
    if task.dataset_kind == "obstacle":
        return obstacle_worker(task)
    if task.dataset_kind == "traffic":
        return traffic_worker(task)
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
    return counter_to_dict(counter)


def _existing_output_summary(task: StandardizeTask) -> dict[str, Any] | None:
    sample_id_value = sample_id(task.output_dataset_key, task.pair, safe_slug=safe_slug)
    output_root = Path(task.output_root)
    image_output_path = output_root / "images" / task.pair.split / f"{sample_id_value}{task.pair.image_path.suffix.lower()}"
    scene_path = output_root / "labels_scene" / task.pair.split / f"{sample_id_value}.json"
    det_path = output_root / "labels_det" / task.pair.split / f"{sample_id_value}.txt"
    if not image_output_path.is_file() or not scene_path.is_file():
        return None

    try:
        scene = load_json(scene_path)
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
            det_class_counts[normalize_text(item.get("class_name")) or "unknown"] += 1

    lane_class_counts = Counter()
    lane_type_counts = Counter()
    for item in lanes:
        if not isinstance(item, dict):
            continue
        lane_class_counts[normalize_text(item.get("class_name")) or "unknown"] += 1
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        lane_type = normalize_text(meta.get("raw_type") or item.get("source_style")) or "missing"
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
        reason = normalize_text(item.get("collapse_reason")) or "unknown"
        if valid:
            tl_combo_counts[combo_name(bits)] += 1
            tl_attr_valid_count += 1
        else:
            tl_invalid_reason_counts[reason] += 1

    return {
        "dataset_key": task.output_dataset_key,
        "split": task.pair.split,
        "sample_id": sample_id_value,
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
        "lane_class_counts": counter_to_dict(lane_class_counts),
        "lane_type_counts": counter_to_dict(lane_type_counts),
        "det_class_counts": counter_to_dict(det_class_counts),
        "tl_combo_counts": counter_to_dict(tl_combo_counts),
        "tl_invalid_reason_counts": counter_to_dict(tl_invalid_reason_counts),
        "held_reason_counts": _reason_counts_from_held_annotations(scene.get("held_annotations")),
        "resume_skipped": 1,
    }


existing_output_summary = _existing_output_summary
worker_chunk_entry = _worker_chunk_entry
_prepare_debug_scene_for_overlay = prepare_debug_scene_for_overlay
