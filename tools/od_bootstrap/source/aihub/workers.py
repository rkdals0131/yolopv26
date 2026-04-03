from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.pv26_schema import TL_BITS

from .lane_worker import lane_worker
from .obstacle_worker import obstacle_worker, prepare_debug_scene_for_overlay
from .traffic_worker import combo_name, traffic_worker
from .worker_common import StandardizeTask
from ..shared.io import load_json as _load_json
from ..shared.raw import normalize_text as _normalize_text, safe_slug as _safe_slug
from ..shared.resume import (
    count_held_annotation_reasons as _count_held_annotation_reasons,
    load_existing_scene_output as _load_existing_scene_output,
)
from ..shared.scene import sample_id as _sample_id
from ..shared.summary import counter_to_dict as _counter_to_dict

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

def _existing_output_summary(task: StandardizeTask) -> dict[str, Any] | None:
    sample_id_value = sample_id(task.output_dataset_key, task.pair, safe_slug=safe_slug)
    output_root = Path(task.output_root)
    bundle = _load_existing_scene_output(
        output_root=output_root,
        split=task.pair.split,
        sample_id=sample_id_value,
        image_suffix=task.pair.image_path.suffix.lower(),
        load_json_fn=load_json,
    )
    if bundle is None:
        return None
    image_output_path = bundle["image_path"]
    scene_path = bundle["scene_path"]
    det_path = bundle["det_path"]
    scene = bundle["scene"]

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
        "held_reason_counts": _count_held_annotation_reasons(
            scene.get("held_annotations"),
            normalize_reason=normalize_text,
        ),
        "resume_skipped": 1,
    }


existing_output_summary = _existing_output_summary
worker_chunk_entry = _worker_chunk_entry
_prepare_debug_scene_for_overlay = prepare_debug_scene_for_overlay
