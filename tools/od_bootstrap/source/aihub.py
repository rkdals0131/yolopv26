from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TextIO

from common.pv26_schema import (
    AIHUB_LANE_DATASET_KEY,
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
    OD_CLASS_TO_ID,
    TL_BITS,
)
from .aihub_debug import (
    _generate_debug_vis as _generate_debug_vis_impl,
    _select_debug_vis_summaries as _select_debug_vis_summaries_impl,
)
from .aihub_reports import (
    aggregate_results as _aggregate_results,
    conversion_report_markdown as _conversion_report_markdown,
    det_class_map_yaml as _det_class_map_yaml,
    failure_manifest_markdown as _failure_manifest_markdown,
    qa_summary as _qa_summary,
    qa_summary_markdown as _qa_summary_markdown,
    scene_class_map_yaml as _scene_class_map_yaml,
)
from .aihub_source_meta import (
    build_source_inventory as _build_source_inventory,
    source_inventory_markdown as _source_inventory_markdown,
    source_root_for_dataset as _source_root_for_dataset,
    write_source_readmes as _write_source_readmes,
)
from .raw_common import (
    LANE_DATASET_KEY as DISCOVERY_LANE_KEY,
    OBSTACLE_DATASET_KEY as DISCOVERY_OBSTACLE_KEY,
    TRAFFIC_DATASET_KEY as DISCOVERY_TRAFFIC_KEY,
    PairRecord,
    _discover_pairs,
    _extract_annotations,
    _extract_attribute_map,
    _extract_bbox,
    _extract_image_size,
    _extract_points,
    _extract_tl_state,
    _extract_filename,
    _load_json,
    _env_path,
    _normalize_text,
    _now_iso,
    _repo_root,
    _safe_slug,
    _seg_dataset_root,
)

PIPELINE_VERSION = "pv26-aihub-standardize-v1"
SCENE_VERSION = "pv26-scene-aihub-v1"
DEFAULT_REPO_ROOT = _repo_root()
DEFAULT_SEG_DATASET_ROOT = _seg_dataset_root(DEFAULT_REPO_ROOT)
DEFAULT_AIHUB_ROOT = _env_path("PV26_AIHUB_ROOT", DEFAULT_SEG_DATASET_ROOT / "AIHUB")
DEFAULT_DOCS_ROOT = DEFAULT_AIHUB_ROOT / "docs"
DEFAULT_LANE_ROOT = DEFAULT_AIHUB_ROOT / "차선-횡단보도 인지 영상(수도권)"
DEFAULT_OBSTACLE_ROOT = DEFAULT_AIHUB_ROOT / "도로장애물·표면 인지 영상(수도권)"
DEFAULT_TRAFFIC_ROOT = DEFAULT_AIHUB_ROOT / "신호등-도로표지판 인지 영상(수도권)"
DEFAULT_OUTPUT_ROOT = _env_path("PV26_AIHUB_OUTPUT_ROOT", DEFAULT_AIHUB_ROOT.parent / "pv26_aihub_standardized")
CACHE_DIR_NAME = "_cache"
DEBUG_VIS_DIRNAME = "debug_vis"
DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26
OUTPUT_LANE_KEY = AIHUB_LANE_DATASET_KEY
OUTPUT_OBSTACLE_KEY = AIHUB_OBSTACLE_DATASET_KEY
OUTPUT_TRAFFIC_KEY = AIHUB_TRAFFIC_DATASET_KEY
DOCUMENTED_STATS = {
    OUTPUT_LANE_KEY: {
        "json_count_seoul": 1_147_048,
        "lane_objects_seoul": 6_115_856,
        "crosswalk_objects_seoul": 407_494,
        "stop_line_objects_seoul": 271_461,
        "white_lane_objects_seoul": 4_380_045,
        "yellow_lane_objects_seoul": 1_612_795,
        "blue_lane_objects_seoul": 123_016,
        "solid_lane_objects_seoul": 3_242_630,
        "dotted_lane_objects_seoul": 2_873_226,
        "doc_reference": "차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf",
    },
    OUTPUT_TRAFFIC_KEY: {
        "json_count_seoul": 1_106_612,
        "traffic_light_objects_seoul": 1_970_735,
        "traffic_sign_objects_seoul": 1_608_805,
        "traffic_light_red_seoul": 474_481,
        "traffic_light_yellow_seoul": 51_639,
        "traffic_light_green_seoul": 587_682,
        "traffic_light_left_arrow_seoul": 100_816,
        "traffic_sign_instruction_seoul": 598_752,
        "traffic_sign_caution_seoul": 138_707,
        "traffic_sign_restriction_seoul": 800_992,
        "doc_reference": "수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf",
    },
}
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
PARALLEL_SUBMIT_LOG_INTERVAL = 5_000
PARALLEL_WAIT_HEARTBEAT_SECONDS = 15.0
PARALLEL_MAX_TASKS_PER_CHUNK = 16
PARALLEL_INFLIGHT_CHUNKS_PER_WORKER = 2


@dataclass(frozen=True)
class StandardizeTask:
    dataset_kind: str
    output_dataset_key: str
    pair: PairRecord
    output_root: str


class LiveLogger:
    def __init__(self, stream: TextIO | None = None, throttle_seconds: float = 1.0) -> None:
        self.stream = stream or sys.stdout
        self.throttle_seconds = throttle_seconds
        self.stage_name = "idle"
        self.stage_started_at = time.monotonic()
        self.stage_total: int | None = None
        self.last_progress_at = 0.0

    def _emit(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.stream.write(f"[{timestamp}] {message}\n")
        self.stream.flush()

    def info(self, message: str) -> None:
        self._emit(message)

    def stage(self, name: str, why: str, total: int | None = None) -> None:
        self.stage_name = name
        self.stage_started_at = time.monotonic()
        self.stage_total = total
        self.last_progress_at = 0.0
        total_text = f", total={total}" if total is not None else ""
        self._emit(f"stage={name} 시작{total_text} | why={why}")

    def progress(self, completed: int, counters: dict[str, int], *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress_at < self.throttle_seconds:
            return
        elapsed = max(now - self.stage_started_at, 1e-6)
        rate = completed / elapsed
        eta_text = "eta=unknown"
        if self.stage_total is not None and completed > 0:
            remaining = max(self.stage_total - completed, 0)
            eta_seconds = remaining / rate if rate > 0 else 0.0
            eta_text = f"eta={eta_seconds:.1f}s"
        total_text = self.stage_total if self.stage_total is not None else "?"
        counters_text = " ".join(f"{key}={value}" for key, value in sorted(counters.items()))
        self._emit(
            f"stage={self.stage_name} progress={completed}/{total_text} rate={rate:.2f}/s {eta_text} {counters_text}".strip()
        )
        self.last_progress_at = now


def _default_workers() -> int:
    available = os.cpu_count() or 4
    return max(1, min(32, available - 1))


def _parallel_chunk_size(total_tasks: int, workers: int) -> int:
    if total_tasks <= 0:
        return 1
    target_inflight_tasks = max(1, workers * 32)
    return max(1, min(PARALLEL_MAX_TASKS_PER_CHUNK, (total_tasks + target_inflight_tasks - 1) // target_inflight_tasks))


def _iter_task_chunks(tasks: list[Any], chunk_size: int) -> Any:
    for start in range(0, len(tasks), chunk_size):
        yield tasks[start : start + chunk_size]


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _nested_counter_to_dict(data: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: _counter_to_dict(value) for key, value in sorted(data.items())}


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{seconds:.1f}s"


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


def _archive_paths(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".zip", ".tar"})


def _has_extracted_assets(dataset_root: Path) -> bool:
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"}:
            return True
    return False


def _extract_archives_if_needed(dataset_root: Path, cache_root: Path, logger: LiveLogger) -> Path:
    if _has_extracted_assets(dataset_root):
        logger.info(f"archive_extract skip | root={dataset_root} | reason=extracted assets already available")
        return dataset_root

    archives = _archive_paths(dataset_root)
    if not archives:
        raise FileNotFoundError(f"no extracted assets or archives found under {dataset_root}")

    extract_root = cache_root / _safe_slug(dataset_root.name)
    extract_root.mkdir(parents=True, exist_ok=True)
    logger.stage(
        f"extract:{dataset_root.name}",
        "원본 디렉터리에 추출본이 없어서 output cache에 archive를 풀어 작업 가능한 파일 트리를 만듭니다.",
        total=len(archives),
    )

    completed = 0
    for archive_path in archives:
        target_dir = extract_root / _safe_slug(str(archive_path.relative_to(dataset_root).with_suffix("")))
        done_marker = target_dir / ".done"
        if done_marker.is_file():
            completed += 1
            logger.progress(completed, {"archives": completed}, force=True)
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive_file:
                archive_file.extractall(target_dir)
        else:
            with tarfile.open(archive_path) as archive_file:
                archive_file.extractall(target_dir)
        done_marker.write_text("ok\n", encoding="utf-8")
        completed += 1
        logger.progress(completed, {"archives": completed}, force=True)

    return extract_root


def _sample_id(output_dataset_key: str, pair: PairRecord) -> str:
    return _safe_slug(f"{output_dataset_key}_{pair.split}_{pair.relative_id}")


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
    if scene.get("source", {}).get("dataset") != OUTPUT_OBSTACLE_KEY:
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
    sample_id = _sample_id(task.output_dataset_key, pair)
    image_output_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = _link_or_copy(pair.image_path, image_output_path)
    width, height, scene = _base_scene(task.output_dataset_key, pair, raw, image_output_path)

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
    sample_id = _sample_id(task.output_dataset_key, pair)
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
    sample_id = _sample_id(task.output_dataset_key, pair)
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
    sample_id = _sample_id(task.output_dataset_key, task.pair)
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


def _select_debug_vis_summaries(
    summaries: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    return _select_debug_vis_summaries_impl(
        summaries,
        count=count,
        seed=seed,
        obstacle_dataset_key=OUTPUT_OBSTACLE_KEY,
    )


def _generate_debug_vis(
    output_root: Path,
    summaries: list[dict[str, Any]],
    *,
    debug_vis_count: int,
    debug_vis_seed: int,
    logger: LiveLogger,
) -> dict[str, Path | None]:
    return _generate_debug_vis_impl(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        obstacle_dataset_key=OUTPUT_OBSTACLE_KEY,
        logger=logger,
        debug_vis_dirname=DEBUG_VIS_DIRNAME,
        now_iso_fn=_now_iso,
        write_json_fn=_write_json,
        load_json_fn=_load_json,
        prepare_scene_fn=_prepare_debug_scene_for_overlay,
    )


def run_standardization(
    *,
    lane_root: Path = DEFAULT_LANE_ROOT,
    obstacle_root: Path = DEFAULT_OBSTACLE_ROOT,
    traffic_root: Path = DEFAULT_TRAFFIC_ROOT,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int | None = None,
    max_samples_per_dataset: int | None = None,
    debug_vis_count: int = DEFAULT_DEBUG_VIS_COUNT,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    write_dataset_readmes: bool = True,
    force_reprocess: bool = False,
    fail_on_error: bool = False,
    log_stream: TextIO | None = None,
) -> dict[str, Path]:
    lane_root = lane_root.resolve()
    obstacle_root = obstacle_root.resolve()
    traffic_root = traffic_root.resolve()
    docs_root = docs_root.resolve()
    output_root = output_root.resolve()
    workers = workers or _default_workers()

    logger = LiveLogger(log_stream)
    logger.info(f"pv26_aihub_standardize version={PIPELINE_VERSION}")
    logger.info(f"lane_root={lane_root}")
    logger.info(f"obstacle_root={obstacle_root}")
    logger.info(f"traffic_root={traffic_root}")
    logger.info(f"output_root={output_root}")
    logger.info(
        f"workers={workers} max_samples_per_dataset={max_samples_per_dataset} debug_vis_count={debug_vis_count} "
        f"force_reprocess={force_reprocess} fail_on_error={fail_on_error}"
    )

    if not lane_root.is_dir():
        raise FileNotFoundError(f"lane root does not exist: {lane_root}")
    if not obstacle_root.is_dir():
        raise FileNotFoundError(f"obstacle root does not exist: {obstacle_root}")
    if not traffic_root.is_dir():
        raise FileNotFoundError(f"traffic root does not exist: {traffic_root}")
    if not docs_root.is_dir():
        raise FileNotFoundError(f"docs root does not exist: {docs_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)

    readme_paths = {"lane_readme": "", "traffic_readme": "", "obstacle_readme": ""}
    if write_dataset_readmes:
        readme_paths = _write_source_readmes(
            lane_root,
            traffic_root,
            obstacle_root,
            docs_root,
            logger,
            documented_stats=DOCUMENTED_STATS,
        )

    logger.stage(
        "source_inventory",
        "원본 데이터셋의 현재 추출 상태와 문서 참조 상태를 메타데이터로 남겨 나중에 재현성을 보장합니다.",
        total=1,
    )
    source_inventory = _build_source_inventory(
        lane_root,
        traffic_root,
        obstacle_root,
        docs_root,
        readme_paths,
        pipeline_version=PIPELINE_VERSION,
        documented_stats=DOCUMENTED_STATS,
    )
    logger.progress(1, {"docs": len(source_inventory["docs"]), "datasets": len(source_inventory["datasets"])}, force=True)

    working_lane_root = _extract_archives_if_needed(lane_root, cache_root, logger)
    working_obstacle_root = _extract_archives_if_needed(obstacle_root, cache_root, logger)
    working_traffic_root = _extract_archives_if_needed(traffic_root, cache_root, logger)

    logger.stage(
        "pair_discovery",
        "AIHUB 디렉터리 구조가 split/subfolder/archive 혼합이라 실제 image-label 짝을 먼저 확정해야 합니다.",
        total=3,
    )
    lane_discovery = _discover_pairs(DISCOVERY_LANE_KEY, working_lane_root)
    logger.progress(1, {"lane_pairs": len(lane_discovery.pairs)}, force=True)
    obstacle_discovery = _discover_pairs(DISCOVERY_OBSTACLE_KEY, working_obstacle_root)
    logger.progress(
        2,
        {"lane_pairs": len(lane_discovery.pairs), "obstacle_pairs": len(obstacle_discovery.pairs)},
        force=True,
    )
    traffic_discovery = _discover_pairs(DISCOVERY_TRAFFIC_KEY, working_traffic_root)
    logger.progress(
        3,
        {
            "lane_pairs": len(lane_discovery.pairs),
            "obstacle_pairs": len(obstacle_discovery.pairs),
            "traffic_pairs": len(traffic_discovery.pairs),
        },
        force=True,
    )

    lane_pairs = sorted(lane_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    obstacle_pairs = sorted(obstacle_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    traffic_pairs = sorted(traffic_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    if max_samples_per_dataset is not None:
        lane_pairs = lane_pairs[:max_samples_per_dataset]
        obstacle_pairs = obstacle_pairs[:max_samples_per_dataset]
        traffic_pairs = traffic_pairs[:max_samples_per_dataset]

    tasks = [
        StandardizeTask("lane", OUTPUT_LANE_KEY, pair, str(output_root))
        for pair in lane_pairs
    ] + [
        StandardizeTask("obstacle", OUTPUT_OBSTACLE_KEY, pair, str(output_root))
        for pair in obstacle_pairs
    ] + [
        StandardizeTask("traffic", OUTPUT_TRAFFIC_KEY, pair, str(output_root))
        for pair in traffic_pairs
    ]

    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pending_tasks: list[StandardizeTask] = []

    logger.stage(
        "resume_scan",
        "기존 표준화 결과가 있으면 재처리하지 않고 기존 산출물을 summary로 재사용해 장시간 작업을 이어갑니다.",
        total=len(tasks),
    )
    resume_progress = Counter()
    if tasks:
        for index, task in enumerate(tasks, start=1):
            existing = None if force_reprocess else _existing_output_summary(task)
            if existing is not None:
                summaries.append(existing)
                resume_progress["reused"] += 1
            else:
                pending_tasks.append(task)
                resume_progress["pending"] += 1
            logger.progress(index, dict(resume_progress))
        logger.progress(len(tasks), dict(resume_progress), force=True)
    else:
        logger.progress(0, {"pending": 0, "reused": 0}, force=True)

    logger.stage(
        "parallel_standardize",
        "JSON 파싱, class remap, scene/det 직렬화, 이미지 materialization을 프로세스 풀로 병렬 실행합니다.",
        total=len(pending_tasks),
    )
    completed = 0
    progress_counters = Counter()
    if pending_tasks:
        chunk_size = _parallel_chunk_size(len(pending_tasks), workers)
        max_inflight_chunks = max(1, workers * PARALLEL_INFLIGHT_CHUNKS_PER_WORKER)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            future_map: dict[Any, list[StandardizeTask]] = {}
            submitted = 0
            next_submit_log = PARALLEL_SUBMIT_LOG_INTERVAL
            chunk_iter = iter(_iter_task_chunks(pending_tasks, chunk_size))

            def submit_chunks() -> None:
                nonlocal submitted, next_submit_log
                while len(future_map) < max_inflight_chunks:
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        return
                    future_map[executor.submit(_worker_chunk_entry, chunk)] = chunk
                    submitted += len(chunk)
                    if submitted == len(chunk) or submitted == len(pending_tasks) or submitted >= next_submit_log:
                        logger.info(
                            f"stage=parallel_standardize submit_progress={submitted}/{len(pending_tasks)} "
                            f"workers={workers} chunk_size={chunk_size} inflight_chunks={len(future_map)}"
                        )
                        next_submit_log = ((submitted // PARALLEL_SUBMIT_LOG_INTERVAL) + 1) * PARALLEL_SUBMIT_LOG_INTERVAL

            submit_chunks()

            logger.info(
                f"stage=parallel_standardize waiting_for_results submitted={submitted}/{len(pending_tasks)} "
                f"completed={completed} chunk_size={chunk_size} inflight_chunks={len(future_map)} "
                f"heartbeat_interval_s={PARALLEL_WAIT_HEARTBEAT_SECONDS:.0f}"
            )

            while future_map:
                done, _ = wait(
                    set(future_map),
                    timeout=PARALLEL_WAIT_HEARTBEAT_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    logger.info(
                        f"stage=parallel_standardize heartbeat completed={completed}/{len(pending_tasks)} "
                        f"submitted={submitted}/{len(pending_tasks)} inflight_chunks={len(future_map)} workers={workers}"
                    )
                    continue

                for future in done:
                    chunk = future_map.pop(future)
                    try:
                        results = future.result()
                    except Exception as exc:
                        logger.info(
                            f"stage=parallel_standardize chunk_error size={len(chunk)} error={type(exc).__name__}: {exc}"
                        )
                        for task in chunk:
                            failures.append(
                                {
                                    "dataset_kind": task.dataset_kind,
                                    "dataset_key": task.output_dataset_key,
                                    "split": task.pair.split,
                                    "raw_id": task.pair.relative_id,
                                    "image_path": str(task.pair.image_path),
                                    "label_path": str(task.pair.label_path),
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                            )
                            completed += 1
                            progress_counters["failed"] += 1
                        logger.progress(completed, dict(progress_counters), force=True)
                        submit_chunks()
                        continue

                    for result in results:
                        failure = result.get("failure")
                        if failure is not None:
                            failures.append(failure)
                            logger.info(
                                f"stage=parallel_standardize error dataset={failure['dataset_kind']} "
                                f"sample={failure['raw_id']} error={failure['error_type']}: {failure['error_message']}"
                            )
                            completed += 1
                            progress_counters["failed"] += 1
                            logger.progress(completed, dict(progress_counters), force=True)
                            continue

                        summary = result["summary"]
                        summaries.append(summary)
                        completed += 1
                        progress_counters["samples"] = completed
                        progress_counters["detections"] += summary["det_count"]
                        progress_counters["lanes"] += summary["lane_count"]
                        progress_counters["stop_lines"] += summary["stop_line_count"]
                        progress_counters["crosswalks"] += summary["crosswalk_count"]
                        progress_counters["tl_valid"] += summary["tl_attr_valid_count"]
                        progress_counters["held"] += sum(summary["held_reason_counts"].values())
                        logger.progress(completed, dict(progress_counters))
                    submit_chunks()
        logger.progress(completed, dict(progress_counters), force=True)
    else:
        logger.progress(0, {"samples": 0, "failed": 0}, force=True)

    logger.stage(
        "report_write",
        "클래스 맵, 변환 리포트, source inventory를 남겨 다음 모델 설계와 데이터 검증의 입력으로 사용합니다.",
        total=8,
    )
    conversion_report = _aggregate_results(
        lane_root=lane_root,
        traffic_root=traffic_root,
        obstacle_root=obstacle_root,
        output_root=output_root,
        workers=workers,
        max_samples_per_dataset=max_samples_per_dataset,
        debug_vis_count=debug_vis_count,
        source_inventory=source_inventory,
        summaries=summaries,
        failures=failures,
        pipeline_version=PIPELINE_VERSION,
        scene_version=SCENE_VERSION,
        source_root_for_dataset=_source_root_for_dataset,
    )

    meta_root = output_root / "meta"
    conversion_json = meta_root / "conversion_report.json"
    conversion_md = meta_root / "conversion_report.md"
    inventory_json = meta_root / "source_inventory.json"
    inventory_md = meta_root / "source_inventory.md"
    det_map_yaml = meta_root / "class_map_det.yaml"
    scene_map_yaml = meta_root / "class_map_scene.yaml"
    failure_json = meta_root / "failure_manifest.json"
    failure_md = meta_root / "failure_manifest.md"

    _write_json(conversion_json, conversion_report)
    logger.progress(1, {"files_written": 1}, force=True)
    _write_text(conversion_md, _conversion_report_markdown(conversion_report))
    logger.progress(2, {"files_written": 2}, force=True)
    _write_json(inventory_json, source_inventory)
    logger.progress(3, {"files_written": 3}, force=True)
    _write_text(inventory_md, _source_inventory_markdown(source_inventory))
    logger.progress(4, {"files_written": 4}, force=True)
    _write_text(det_map_yaml, _det_class_map_yaml())
    logger.progress(5, {"files_written": 5}, force=True)
    _write_text(scene_map_yaml, _scene_class_map_yaml())
    logger.progress(6, {"files_written": 6}, force=True)
    failure_manifest = {
        "version": PIPELINE_VERSION,
        "generated_at": _now_iso(),
        "failure_count": len(failures),
        "items": failures,
    }
    _write_json(failure_json, failure_manifest)
    logger.progress(7, {"files_written": 7}, force=True)
    _write_text(failure_md, _failure_manifest_markdown(failure_manifest))
    logger.progress(8, {"files_written": 8}, force=True)

    debug_vis_outputs = _generate_debug_vis(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        logger=logger,
    )

    logger.stage(
        "qa_write",
        "resume/실패/debug-vis 선택 결과를 묶어 full-dataset 전처리 전 QA summary를 남깁니다.",
        total=2,
    )
    debug_vis_index = _load_json(debug_vis_outputs["debug_vis_index"])
    qa_json = meta_root / "qa_summary.json"
    qa_md = meta_root / "qa_summary.md"
    qa_summary = _qa_summary(conversion_report, debug_vis_index, failure_manifest)
    _write_json(qa_json, qa_summary)
    logger.progress(1, {"files_written": 1}, force=True)
    _write_text(qa_md, _qa_summary_markdown(qa_summary))
    logger.progress(2, {"files_written": 2}, force=True)

    logger.info("standardization complete")
    if failures and fail_on_error:
        raise RuntimeError(f"AIHUB standardization completed with failures: {len(failures)}")
    return {
        "output_root": output_root,
        "conversion_json": conversion_json,
        "conversion_md": conversion_md,
        "inventory_json": inventory_json,
        "inventory_md": inventory_md,
        "det_map_yaml": det_map_yaml,
        "scene_map_yaml": scene_map_yaml,
        "failure_json": failure_json,
        "failure_md": failure_md,
        "qa_json": qa_json,
        "qa_md": qa_md,
        "debug_vis_dir": debug_vis_outputs["debug_vis_dir"],
        "debug_vis_index": debug_vis_outputs["debug_vis_index"],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hardcoded AIHUB standardization pipeline.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional process pool size. Defaults to CPU count minus one.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional per-dataset sample limit for faster verification runs. Source inventory and README generation still scan the full source tree.",
    )
    parser.add_argument(
        "--debug-vis-count",
        type=int,
        default=DEFAULT_DEBUG_VIS_COUNT,
        help="Random QA overlay count written under meta/debug_vis after standardization. Set 0 to disable.",
    )
    parser.add_argument(
        "--skip-readmes",
        action="store_true",
        help="Skip dataset-local README generation under the AIHUB source roots.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore existing standardized outputs and rebuild every discovered sample from source.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero after writing failure manifests if any sample conversion fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    outputs = run_standardization(
        workers=args.workers,
        max_samples_per_dataset=args.max_samples_per_dataset,
        debug_vis_count=args.debug_vis_count,
        write_dataset_readmes=not args.skip_readmes,
        force_reprocess=args.force_reprocess,
        fail_on_error=args.fail_on_error,
    )
    print(f"output_root={outputs['output_root']}")
    print(f"conversion_json={outputs['conversion_json']}")
    print(f"inventory_json={outputs['inventory_json']}")
    print(f"failure_json={outputs['failure_json']}")
    print(f"qa_json={outputs['qa_json']}")
    print(f"debug_vis_dir={outputs['debug_vis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
