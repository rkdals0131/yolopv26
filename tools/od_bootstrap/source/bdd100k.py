from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TextIO

from .raw_common import (
    IMAGE_EXTENSIONS,
    PairRecord,
    _env_path,
    _now_iso,
    _probe_image_size,
    _repo_root,
    _safe_slug,
    _seg_dataset_root,
)
from .aihub import (
    PARALLEL_INFLIGHT_CHUNKS_PER_WORKER,
    PARALLEL_SUBMIT_LOG_INTERVAL,
    PARALLEL_WAIT_HEARTBEAT_SECONDS,
    LiveLogger,
    bbox_to_yolo_line as _bbox_to_yolo_line,
    counter_to_dict as _counter_to_dict,
    default_worker_count as _default_workers,
    det_class_map_yaml as _det_class_map_yaml,
    generate_debug_vis_outputs as _generate_debug_vis,
    iter_task_chunks as _iter_task_chunks,
    link_or_copy_file as _link_or_copy,
    parallel_chunk_size as _parallel_chunk_size,
    write_json_file as _write_json,
    write_text_file as _write_text,
)
from common.pv26_schema import BDD100K_DATASET_KEY, OD_CLASSES, OD_CLASS_TO_ID, TL_BITS

PIPELINE_VERSION = "pv26-bdd100k-standardize-v2"
SCENE_VERSION = "pv26-scene-bdd100k-v2"
DEFAULT_REPO_ROOT = _repo_root()
DEFAULT_SEG_DATASET_ROOT = _seg_dataset_root(DEFAULT_REPO_ROOT)
DEFAULT_BDD_ROOT = _env_path("PV26_BDD_ROOT", DEFAULT_SEG_DATASET_ROOT / "BDD100K")
DEFAULT_IMAGES_ROOT = DEFAULT_BDD_ROOT / "bdd100k_images_100k" / "100k"
DEFAULT_LABELS_ROOT = DEFAULT_BDD_ROOT / "bdd100k_labels" / "100k"
DEFAULT_OUTPUT_ROOT = _env_path("PV26_BDD_OUTPUT_ROOT", DEFAULT_BDD_ROOT.parent / "pv26_bdd100k_standardized")
DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26
OUTPUT_DATASET_KEY = BDD100K_DATASET_KEY
BDD_SPLITS = ("train", "val", "test")
HELD_ANNOTATION_LIMIT = 32
OFFICIAL_SPLIT_SIZES = {
    "train": 70_000,
    "val": 10_000,
    "test": 20_000,
}
BDD_CATEGORY_ALIASES = {
    "bike": "bicycle",
    "motor": "motorcycle",
    "person": "pedestrian",
    "other person": "pedestrian",
    "other vehicle": "car",
    "trailer": "truck",
    "caravan": "car",
    "van": "car",
}
BDD_TO_PV26_CLASS = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "bicycle": "bike",
    "motorcycle": "bike",
    "pedestrian": "pedestrian",
    "rider": "pedestrian",
}
BDD_EXCLUDED_REASON = {
    "traffic light": "excluded_bdd_traffic_light_policy",
    "traffic sign": "excluded_bdd_sign_policy",
}


@dataclass(frozen=True)
class BDDTask:
    pair: PairRecord
    output_root: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_bdd_category(value: Any) -> str:
    category = str(value or "").strip().lower()
    return BDD_CATEGORY_ALIASES.get(category, category)


def _sample_id(pair: PairRecord) -> str:
    return _safe_slug(f"{OUTPUT_DATASET_KEY}_{pair.split}_{pair.relative_id}")


def _count_files(root: Path, suffixes: set[str]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _tree_markdown(root: Path, *, max_depth: int = 3, max_lines: int = 96) -> str:
    lines = [f"{root.name}/"]

    def walk(current: Path, depth: int) -> None:
        if len(lines) >= max_lines or depth >= max_depth:
            return
        for child in sorted(current.iterdir(), key=lambda item: (item.is_file(), item.name)):
            if len(lines) >= max_lines:
                break
            indent = "  " * depth
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}- {child.name}{suffix}")
            if child.is_dir():
                walk(child, depth + 1)

    walk(root, 1)
    if len(lines) >= max_lines:
        lines.append("  - ...")
    return "\n".join(lines)


def _inventory_bdd_root(bdd_root: Path, images_root: Path, labels_root: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in BDD_SPLITS:
        image_dir = images_root / split
        label_dir = labels_root / split
        splits[split] = {
            "images_present": image_dir.is_dir(),
            "labels_present": label_dir.is_dir(),
            "images": _count_files(image_dir, IMAGE_EXTENSIONS) if image_dir.is_dir() else 0,
            "json_files": _count_files(label_dir, {".json"}) if label_dir.is_dir() else 0,
            "official_images": OFFICIAL_SPLIT_SIZES[split],
        }

    return {
        "root": str(bdd_root),
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "extra_assets": {
            "bdd100k_det_20_labels": (bdd_root / "bdd100k_det_20_labels").is_dir(),
            "bdd100k_drivable_maps": (bdd_root / "bdd100k_drivable_maps").is_dir(),
            "bdd100k_seg_maps": (bdd_root / "bdd100k_seg_maps").is_dir(),
            "bdd100k_gh_toolkit": (bdd_root / "bdd100k-gh").is_dir(),
        },
        "splits": splits,
    }


def _bdd_readme(bdd_root: Path, inventory: dict[str, Any]) -> str:
    lines = [
        "# BDD100K",
        "",
        "PV26에서 사용하는 BDD100K 원본 구조와 detection-only 표준화 관점을 정리한 원본용 README다.",
        "",
        "## PV26 사용 범위",
        "",
        "- 사용 목적: `7-class object detection` 중 non-signal class 보강",
        "- 사용 원천: `bdd100k_images_100k/100k/<split>` + `bdd100k_labels/100k/<split>/*.json`",
        "- 비사용 원천: drivable map, segmentation map, det_20 preview asset",
        "- BDD canonical output은 `vehicle / bike / pedestrian`만 detector supervision으로 남긴다.",
        "- `traffic light`, `traffic sign`는 AIHUB signal source가 담당하므로 BDD canonical output에서 제외한다.",
        "- TL 4-bit supervision은 BDD source에서 사용하지 않는다.",
        "",
        "## 공식 split 크기",
        "",
        f"- train: `{OFFICIAL_SPLIT_SIZES['train']:,}`",
        f"- val: `{OFFICIAL_SPLIT_SIZES['val']:,}`",
        f"- test: `{OFFICIAL_SPLIT_SIZES['test']:,}`",
        "",
        "## 현재 로컬 보유 상태",
        "",
        "| Split | Images Present | Labels Present | Local Images | Local JSON | Official Images |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split, item in inventory["splits"].items():
        lines.append(
            f"| {split} | {'yes' if item['images_present'] else 'no'} | {'yes' if item['labels_present'] else 'no'} "
            f"| {item['images']:,} | {item['json_files']:,} | {item['official_images']:,} |"
        )
    lines.extend(
        [
            "",
            "## 원본 어노테이션 스키마 요약",
            "",
            "- 파일 단위: 이미지 1장당 JSON 1개",
            "- top-level: `name`, `attributes`, `frames[]`",
            "- detection 객체: `frames[0].objects[]`",
            "- detection box: `box2d = {x1, y1, x2, y2}`",
            "- scene width/height와 YOLO normalization은 실제 image file size probe 기준으로 계산한다.",
            "- 문맥 메타: `attributes.weather`, `attributes.scene`, `attributes.timeofday`",
            "",
            "## PV26 클래스 collapse 규칙",
            "",
            "- `car/truck/bus/train/(other vehicle, van, caravan, trailer alias)` -> `vehicle`",
            "- `bike/motor/(bicycle, motorcycle alias)` -> `bike`",
            "- `person/rider/(other person alias)` -> `pedestrian`",
            "- `traffic light` -> excluded (`AIHUB-owned signal class`)",
            "- `traffic sign` -> excluded (`AIHUB-owned signal class`)",
            "- `lane/*`, `area/*` 등 non-box driving map 계열은 detector 표준화에서 제외",
            "",
            "## 로컬 디렉터리 구조",
            "",
            "```text",
            _tree_markdown(bdd_root),
            "```",
            "",
            "## 추가 자산 존재 여부",
            "",
        ]
    )
    for key, value in sorted(inventory["extra_assets"].items()):
        lines.append(f"- `{key}`: {'yes' if value else 'no'}")
    return "\n".join(lines) + "\n"


def _build_source_inventory(bdd_root: Path, images_root: Path, labels_root: Path, readme_path: str) -> dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "generated_at": _now_iso(),
        "dataset": {
            "dataset_key": OUTPUT_DATASET_KEY,
            "readme_path": readme_path,
            "local_inventory": _inventory_bdd_root(bdd_root, images_root, labels_root),
        },
    }


def _source_inventory_markdown(source_inventory: dict[str, Any]) -> str:
    dataset = source_inventory["dataset"]
    inventory = dataset["local_inventory"]
    lines = [
        "# PV26 BDD100K Source Inventory",
        "",
        f"- Generated: `{source_inventory['generated_at']}`",
        f"- Version: `{source_inventory['version']}`",
        f"- Dataset key: `{dataset['dataset_key']}`",
        f"- Root: `{inventory['root']}`",
        f"- README: `{dataset['readme_path']}`",
        "",
        "| Split | Images Present | Labels Present | Local Images | Local JSON | Official Images |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split, item in inventory["splits"].items():
        lines.append(
            f"| {split} | {'yes' if item['images_present'] else 'no'} | {'yes' if item['labels_present'] else 'no'} "
            f"| {item['images']:,} | {item['json_files']:,} | {item['official_images']:,} |"
        )
    lines.extend(
        [
            "",
            "## Extra Assets",
            "",
        ]
    )
    for key, value in sorted(inventory["extra_assets"].items()):
        lines.append(f"- `{key}`: {'yes' if value else 'no'}")
    return "\n".join(lines) + "\n"


def _discover_pairs(images_root: Path, labels_root: Path) -> dict[str, Any]:
    pairs: list[PairRecord] = []
    missing_images: list[dict[str, str]] = []
    missing_labels: list[dict[str, str]] = []
    per_split_counts: dict[str, dict[str, int]] = {}

    for split in BDD_SPLITS:
        image_dir = images_root / split
        label_dir = labels_root / split
        image_map = {
            path.stem: path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        } if image_dir.is_dir() else {}
        label_map = {
            path.stem: path
            for path in sorted(label_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".json"
        } if label_dir.is_dir() else {}

        matched = sorted(set(image_map) & set(label_map))
        for stem in matched:
            pairs.append(
                PairRecord(
                    dataset_key=OUTPUT_DATASET_KEY,
                    dataset_root=images_root.parent,
                    split=split,
                    image_path=image_map[stem],
                    label_path=label_map[stem],
                    image_file_name=image_map[stem].name,
                    relative_id=stem,
                )
            )
        for stem in sorted(set(label_map) - set(image_map)):
            missing_images.append(
                {
                    "split": split,
                    "image_file_name": f"{stem}.jpg",
                    "label_path": str(label_map[stem]),
                }
            )
        for stem in sorted(set(image_map) - set(label_map)):
            missing_labels.append(
                {
                    "split": split,
                    "image_path": str(image_map[stem]),
                    "image_file_name": image_map[stem].name,
                }
            )

        per_split_counts[split] = {
            "images": len(image_map),
            "json_files": len(label_map),
            "pairs": len(matched),
            "missing_images": len(set(label_map) - set(image_map)),
            "missing_labels": len(set(image_map) - set(label_map)),
        }

    return {
        "pairs": pairs,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "per_split_counts": per_split_counts,
    }


def _limit_pairs_per_split(pairs: list[PairRecord], max_samples_per_split: int | None) -> list[PairRecord]:
    if max_samples_per_split is None:
        return pairs
    grouped: dict[str, list[PairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.split].append(pair)
    limited: list[PairRecord] = []
    for split in BDD_SPLITS:
        limited.extend(sorted(grouped[split], key=lambda item: item.relative_id)[:max_samples_per_split])
    return limited


def _extract_bbox(box2d: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box2d, dict):
        return None
    try:
        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _maybe_append_held(
    held_annotations: list[dict[str, Any]],
    *,
    raw_category: str,
    reason: str,
    bbox_present: bool | None = None,
) -> None:
    if len(held_annotations) >= HELD_ANNOTATION_LIMIT:
        return
    payload: dict[str, Any] = {
        "raw_category": raw_category,
        "reason": reason,
    }
    if bbox_present is not None:
        payload["bbox_present"] = bbox_present
    held_annotations.append(payload)


def _worker_entry(task: BDDTask) -> dict[str, Any]:
    pair = task.pair
    assert pair.image_path is not None
    raw = _load_json(pair.label_path)
    output_root = Path(task.output_root)
    sample_id = _sample_id(pair)
    output_image_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    image_materialization = _link_or_copy(pair.image_path, output_image_path)

    width, height = _probe_image_size(pair.image_path)
    top_attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    frames = raw.get("frames") if isinstance(raw.get("frames"), list) else []
    frame = frames[0] if frames else {}
    objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []

    detections: list[dict[str, Any]] = []
    traffic_lights: list[dict[str, Any]] = []
    traffic_signs: list[dict[str, Any]] = []
    held_annotations: list[dict[str, Any]] = []
    det_lines: list[str] = []
    det_class_counts = Counter()
    raw_category_counts = Counter()
    tl_state_hint_counts = Counter()
    held_reason_counts = Counter()

    for obj in objects:
        raw_category = str(obj.get("category") or "").strip()
        canonical_category = _normalize_bdd_category(raw_category)
        raw_category_counts[canonical_category or "missing_category"] += 1
        excluded_reason = BDD_EXCLUDED_REASON.get(canonical_category)
        if excluded_reason is not None:
            held_reason_counts[excluded_reason] += 1
            _maybe_append_held(
                held_annotations,
                raw_category=canonical_category or "missing_category",
                reason=excluded_reason,
                bbox_present=obj.get("box2d") is not None,
            )
            continue
        mapped_class = BDD_TO_PV26_CLASS.get(canonical_category)
        box2d = obj.get("box2d")
        bbox = _extract_bbox(box2d, width, height) if box2d is not None else None

        if mapped_class is None:
            held_reason_counts["ignored_non_pv26_category"] += 1
            _maybe_append_held(
                held_annotations,
                raw_category=canonical_category or "missing_category",
                reason="ignored_non_pv26_category",
                bbox_present=box2d is not None,
            )
            continue

        if bbox is None:
            held_reason_counts["mapped_category_missing_box2d"] += 1
            _maybe_append_held(
                held_annotations,
                raw_category=canonical_category or "missing_category",
                reason="mapped_category_missing_box2d",
                bbox_present=box2d is not None,
            )
            continue

        detection_id = len(detections)
        detections.append(
            {
                "id": detection_id,
                "class_name": mapped_class,
                "bbox": bbox,
                "score": None,
                "meta": {
                    "dataset_label": canonical_category,
                    "bdd_id": obj.get("id"),
                },
            }
        )
        det_class_counts[mapped_class] += 1
        det_lines.append(_bbox_to_yolo_line(OD_CLASS_TO_ID[mapped_class], bbox, width, height))

    scene = {
        "version": SCENE_VERSION,
        "image": {
            "file_name": output_image_path.name,
            "width": width,
            "height": height,
            "original_file_name": pair.image_file_name,
        },
        "source": {
            "dataset": OUTPUT_DATASET_KEY,
            "raw_id": pair.relative_id,
            "split": pair.split,
            "image_path": str(pair.image_path),
            "label_path": str(pair.label_path),
            "raw_name": raw.get("name"),
        },
        "context": {
            "weather": top_attributes.get("weather"),
            "scene": top_attributes.get("scene"),
            "timeofday": top_attributes.get("timeofday"),
            "timestamp": frame.get("timestamp"),
        },
        "tasks": {
            "has_det": int(bool(detections)),
            "has_lane": 0,
            "has_stop_line": 0,
            "has_crosswalk": 0,
            "has_tl_attr": 0,
        },
        "detections": detections,
        "traffic_lights": traffic_lights,
        "traffic_signs": traffic_signs,
        "auxiliary_annotations": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "ignored_regions": [],
        "held_annotations": held_annotations,
        "notes": [
            "BDD100K is standardized as a detector-only source for PV26.",
            "BDD canonical output intentionally excludes traffic-light and sign supervision.",
            "Lane and area categories are intentionally excluded from the detector canonical set.",
        ],
    }

    scene_path = output_root / "labels_scene" / pair.split / f"{sample_id}.json"
    _write_json(scene_path, scene)
    if det_lines:
        det_path = output_root / "labels_det" / pair.split / f"{sample_id}.txt"
        _write_text(det_path, "\n".join(det_lines) + "\n")

    return {
        "dataset_key": OUTPUT_DATASET_KEY,
        "split": pair.split,
        "sample_id": sample_id,
        "scene_path": str(scene_path),
        "image_path": str(output_image_path),
        "image_materialization": image_materialization,
        "det_count": len(detections),
        "traffic_light_count": len(traffic_lights),
        "traffic_sign_count": len(traffic_signs),
        "det_class_counts": _counter_to_dict(det_class_counts),
        "raw_category_counts": _counter_to_dict(raw_category_counts),
        "tl_state_hint_counts": _counter_to_dict(tl_state_hint_counts),
        "held_reason_counts": _counter_to_dict(held_reason_counts),
        "resume_skipped": 0,
    }


def _worker_chunk_entry(tasks: list[BDDTask]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        try:
            results.append({"summary": _worker_entry(task)})
        except Exception as exc:
            results.append(
                {
                    "failure": {
                        "dataset_key": OUTPUT_DATASET_KEY,
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
        counter[str(item.get("reason") or "unknown").strip().lower()] += 1
    return _counter_to_dict(counter)


def _existing_output_summary(task: BDDTask) -> dict[str, Any] | None:
    pair = task.pair
    sample_id = _sample_id(pair)
    output_root = Path(task.output_root)
    image_path = output_root / "images" / pair.split / f"{sample_id}{pair.image_path.suffix.lower()}"
    scene_path = output_root / "labels_scene" / pair.split / f"{sample_id}.json"
    det_path = output_root / "labels_det" / pair.split / f"{sample_id}.txt"
    if not image_path.is_file() or not scene_path.is_file():
        return None

    try:
        scene = _load_json(scene_path)
    except Exception:
        return None
    if str(scene.get("version") or "").strip() != SCENE_VERSION:
        return None

    detections = scene.get("detections") if isinstance(scene.get("detections"), list) else []
    traffic_lights = scene.get("traffic_lights") if isinstance(scene.get("traffic_lights"), list) else []
    traffic_signs = scene.get("traffic_signs") if isinstance(scene.get("traffic_signs"), list) else []
    if traffic_lights or traffic_signs:
        return None
    if detections and not det_path.is_file():
        return None

    det_class_counts = Counter()
    raw_category_counts = Counter()
    tl_state_hint_counts = Counter()
    for item in detections:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or "unknown").strip().lower()
        if class_name in {"traffic_light", "sign"}:
            return None
        det_class_counts[class_name] += 1
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        raw_category_counts[str(meta.get("dataset_label") or "unknown").strip().lower()] += 1
    for item in traffic_lights:
        if not isinstance(item, dict):
            continue
        tl_state_hint_counts[str(item.get("state_hint") or "unknown").strip().lower()] += 1

    return {
        "dataset_key": OUTPUT_DATASET_KEY,
        "split": pair.split,
        "sample_id": sample_id,
        "scene_path": str(scene_path),
        "image_path": str(image_path),
        "image_materialization": "resume_existing",
        "det_count": len(detections),
        "traffic_light_count": len(traffic_lights),
        "traffic_sign_count": len(traffic_signs),
        "det_class_counts": _counter_to_dict(det_class_counts),
        "raw_category_counts": _counter_to_dict(raw_category_counts),
        "tl_state_hint_counts": _counter_to_dict(tl_state_hint_counts),
        "held_reason_counts": _reason_counts_from_held_annotations(scene.get("held_annotations")),
        "resume_skipped": 1,
    }


def _aggregate_results(
    *,
    bdd_root: Path,
    images_root: Path,
    labels_root: Path,
    output_root: Path,
    workers: int,
    max_samples_per_split: int | None,
    debug_vis_count: int,
    source_inventory: dict[str, Any],
    discovery: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    det_class_counts = Counter()
    raw_category_counts = Counter()
    tl_state_hint_counts = Counter()
    held_reason_counts = Counter()
    materialization_counts = Counter()
    total_detections = 0
    total_lights = 0
    total_signs = 0
    total_resume_skipped = 0
    empty_scene_count = 0

    for item in summaries:
        split = item["split"]
        split_counts[split]["samples"] += 1
        split_counts[split]["detections"] += item["det_count"]
        split_counts[split]["traffic_lights"] += item["traffic_light_count"]
        split_counts[split]["traffic_signs"] += item["traffic_sign_count"]
        total_detections += item["det_count"]
        total_lights += item["traffic_light_count"]
        total_signs += item["traffic_sign_count"]
        total_resume_skipped += int(item.get("resume_skipped", 0))
        materialization_counts[item["image_materialization"]] += 1
        det_class_counts.update(item["det_class_counts"])
        raw_category_counts.update(item["raw_category_counts"])
        tl_state_hint_counts.update(item["tl_state_hint_counts"])
        held_reason_counts.update(item["held_reason_counts"])
        if item["det_count"] == 0:
            empty_scene_count += 1

    return {
        "version": PIPELINE_VERSION,
        "scene_version": SCENE_VERSION,
        "generated_at": _now_iso(),
        "settings": {
            "workers": workers,
            "max_samples_per_split": max_samples_per_split,
            "debug_vis_count": debug_vis_count,
            "bdd_root": str(bdd_root),
            "images_root": str(images_root),
            "labels_root": str(labels_root),
            "output_root": str(output_root),
            "failure_count": len(failures),
        },
        "det_class_map": {str(index): class_name for index, class_name in enumerate(OD_CLASSES)},
        "tl_bits": TL_BITS,
        "dataset": {
            "dataset_key": OUTPUT_DATASET_KEY,
            "processed_samples": len(summaries),
            "fresh_processed_count": len(summaries) - total_resume_skipped,
            "resume_skipped_count": total_resume_skipped,
            "failure_count": len(failures),
            "empty_scene_count": empty_scene_count,
            "pair_discovery": discovery["per_split_counts"],
            "missing_images": discovery["missing_images"],
            "missing_labels": discovery["missing_labels"],
            "per_split_counts": {
                split: {key: value for key, value in sorted(counts.items())}
                for split, counts in sorted(split_counts.items())
            },
            "detection_count": total_detections,
            "traffic_light_count": total_lights,
            "traffic_sign_count": total_signs,
            "det_class_counts": _counter_to_dict(det_class_counts),
            "raw_category_counts": _counter_to_dict(raw_category_counts),
            "tl_state_hint_counts": _counter_to_dict(tl_state_hint_counts),
            "held_reason_counts": _counter_to_dict(held_reason_counts),
            "image_materialization": _counter_to_dict(materialization_counts),
        },
        "failures": failures,
        "source_inventory_snapshot": source_inventory,
    }


def _conversion_report_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# PV26 BDD100K Conversion Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Version: `{report['version']}`",
        f"- Output root: `{report['settings']['output_root']}`",
        f"- Workers: `{report['settings']['workers']}`",
        f"- Max samples per split: `{report['settings']['max_samples_per_split']}`",
        "",
        f"## {dataset['dataset_key']}",
        "",
        f"- Processed samples: `{dataset['processed_samples']}`",
        f"- Fresh processed samples: `{dataset['fresh_processed_count']}`",
        f"- Resume skipped samples: `{dataset['resume_skipped_count']}`",
        f"- Failure count: `{dataset['failure_count']}`",
        f"- Detection count: `{dataset['detection_count']}`",
        f"- Traffic light count: `{dataset['traffic_light_count']}`",
        f"- Traffic sign count: `{dataset['traffic_sign_count']}`",
        "",
        "| Split | Samples | Detections | Traffic lights | Traffic signs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, counts in dataset["per_split_counts"].items():
        lines.append(
            f"| {split} | {counts.get('samples', 0)} | {counts.get('detections', 0)} | "
            f"{counts.get('traffic_lights', 0)} | {counts.get('traffic_signs', 0)} |"
        )
    for title, payload in (
        ("Detection Class Counts", dataset["det_class_counts"]),
        ("Raw Category Counts", dataset["raw_category_counts"]),
        ("Traffic Light State Hints", dataset["tl_state_hint_counts"]),
        ("Held Reasons", dataset["held_reason_counts"]),
        ("Image Materialization", dataset["image_materialization"]),
    ):
        if not payload:
            continue
        lines.extend(["", f"### {title}", ""])
        for key, value in payload.items():
            lines.append(f"- `{key}`: {value}")
    if dataset["missing_images"]:
        lines.extend(["", "### Missing Images", ""])
        for item in dataset["missing_images"][:20]:
            lines.append(f"- split=`{item['split']}` label=`{item['label_path']}`")
    if dataset["missing_labels"]:
        lines.extend(["", "### Missing Labels", ""])
        for item in dataset["missing_labels"][:20]:
            lines.append(f"- split=`{item['split']}` image=`{item['image_path']}`")
    if report["failures"]:
        lines.extend(["", "### Failure Manifest", ""])
        for item in report["failures"][:32]:
            lines.append(
                f"- split=`{item['split']}` raw_id=`{item['raw_id']}` error=`{item['error_type']}`"
            )
    return "\n".join(lines) + "\n"


def _failure_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# PV26 BDD100K Failure Manifest",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Version: `{manifest['version']}`",
        f"- Failure count: `{manifest['failure_count']}`",
        "",
    ]
    for item in manifest["items"]:
        lines.extend(
            [
                f"## {item['split']} / {item['raw_id']}",
                "",
                f"- Error type: `{item['error_type']}`",
                f"- Error message: `{item['error_message']}`",
                f"- Image path: `{item['image_path']}`",
                f"- Label path: `{item['label_path']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _qa_summary(report: dict[str, Any], debug_vis_index: dict[str, Any], failure_manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = report["dataset"]
    return {
        "version": PIPELINE_VERSION,
        "generated_at": _now_iso(),
        "output_root": report["settings"]["output_root"],
        "debug_vis": {
            "selection_count": int(debug_vis_index.get("selection_count", 0)),
            "seed": int(debug_vis_index.get("seed", 0)),
        },
        "failure_count": int(failure_manifest["failure_count"]),
        "dataset": {
            "dataset_key": dataset["dataset_key"],
            "processed_samples": dataset["processed_samples"],
            "fresh_processed_count": dataset["fresh_processed_count"],
            "resume_skipped_count": dataset["resume_skipped_count"],
            "failure_count": dataset["failure_count"],
            "empty_scene_count": dataset["empty_scene_count"],
            "detection_count": dataset["detection_count"],
            "traffic_light_count": dataset["traffic_light_count"],
            "traffic_sign_count": dataset["traffic_sign_count"],
            "top_held_reasons": list(dataset["held_reason_counts"].items())[:5],
            "top_state_hints": list(dataset["tl_state_hint_counts"].items())[:5],
        },
    }


def _qa_summary_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# PV26 BDD100K QA Summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Debug vis selections: `{summary['debug_vis']['selection_count']}`",
        f"- Failure count: `{summary['failure_count']}`",
        "",
        f"## {dataset['dataset_key']}",
        "",
        f"- Processed samples: `{dataset['processed_samples']}`",
        f"- Fresh processed samples: `{dataset['fresh_processed_count']}`",
        f"- Resume skipped samples: `{dataset['resume_skipped_count']}`",
        f"- Failure count: `{dataset['failure_count']}`",
        f"- Empty scenes: `{dataset['empty_scene_count']}`",
        f"- Detection count: `{dataset['detection_count']}`",
        f"- Traffic lights: `{dataset['traffic_light_count']}`",
        f"- Traffic signs: `{dataset['traffic_sign_count']}`",
        "",
    ]
    if dataset["top_held_reasons"]:
        lines.append("### Top Held Reasons")
        lines.append("")
        for key, value in dataset["top_held_reasons"]:
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    if dataset["top_state_hints"]:
        lines.append("### Top Traffic Light State Hints")
        lines.append("")
        for key, value in dataset["top_state_hints"]:
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _scene_class_map_yaml() -> str:
    lines = [
        f"version: {SCENE_VERSION}",
        "supported_tasks:",
        "  has_det: true",
        "  has_tl_attr: false",
        "  has_lane: false",
        "  has_stop_line: false",
        "  has_crosswalk: false",
        "context_fields:",
        "  - weather",
        "  - scene",
        "  - timeofday",
        "  - timestamp",
    ]
    return "\n".join(lines) + "\n"


def _scan_existing_outputs(
    tasks: list[BDDTask],
    *,
    force_reprocess: bool,
    logger: LiveLogger,
) -> tuple[list[dict[str, Any]], list[BDDTask]]:
    summaries: list[dict[str, Any]] = []
    pending_tasks: list[BDDTask] = []
    logger.stage(
        "resume_scan",
        "기존 standardized outputs가 있으면 재처리하지 않고 summary만 재구성해 full run을 이어갑니다.",
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
    return summaries, pending_tasks


def _run_pending_standardization(
    pending_tasks: list[BDDTask],
    *,
    workers: int,
    logger: LiveLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logger.stage(
        "parallel_standardize",
        "BDD JSON 파싱, 7-class remap, YOLO txt 직렬화, image materialization을 프로세스 풀로 병렬 실행합니다.",
        total=len(pending_tasks),
    )
    progress = Counter()
    completed = 0
    if not pending_tasks:
        logger.progress(0, {"samples": 0}, force=True)
        return summaries, failures

    chunk_size = _parallel_chunk_size(len(pending_tasks), workers)
    max_inflight_chunks = max(1, workers * PARALLEL_INFLIGHT_CHUNKS_PER_WORKER)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        future_map: dict[Any, list[BDDTask]] = {}
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
                                "dataset_key": OUTPUT_DATASET_KEY,
                                "split": task.pair.split,
                                "raw_id": task.pair.relative_id,
                                "image_path": str(task.pair.image_path),
                                "label_path": str(task.pair.label_path),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        completed += 1
                        progress["failed"] += 1
                    logger.progress(completed, dict(progress), force=True)
                    submit_chunks()
                    continue

                for result in results:
                    failure = result.get("failure")
                    if failure is not None:
                        failures.append(failure)
                        logger.info(
                            f"stage=parallel_standardize error split={failure['split']} "
                            f"sample={failure['raw_id']} error={failure['error_type']}: {failure['error_message']}"
                        )
                        completed += 1
                        progress["failed"] += 1
                        logger.progress(completed, dict(progress), force=True)
                        continue

                    summary = result["summary"]
                    summaries.append(summary)
                    completed += 1
                    progress["samples"] = completed
                    progress["detections"] += summary["det_count"]
                    progress["traffic_lights"] += summary["traffic_light_count"]
                    progress["traffic_signs"] += summary["traffic_sign_count"]
                    progress["held"] += sum(summary["held_reason_counts"].values())
                    logger.progress(completed, dict(progress))
                submit_chunks()
    logger.progress(completed, dict(progress), force=True)
    return summaries, failures


def _write_standardization_outputs(
    *,
    bdd_root: Path,
    images_root: Path,
    labels_root: Path,
    output_root: Path,
    workers: int,
    max_samples_per_split: int | None,
    debug_vis_count: int,
    debug_vis_seed: int,
    source_inventory: dict[str, Any],
    discovery: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    logger: LiveLogger,
) -> dict[str, Path]:
    logger.stage(
        "report_write",
        "클래스 맵, 변환 리포트, source inventory를 기록해 loader가 동일 계약을 읽을 수 있게 만듭니다.",
        total=8,
    )
    report = _aggregate_results(
        bdd_root=bdd_root,
        images_root=images_root,
        labels_root=labels_root,
        output_root=output_root,
        workers=workers,
        max_samples_per_split=max_samples_per_split,
        debug_vis_count=debug_vis_count,
        source_inventory=source_inventory,
        discovery=discovery,
        summaries=summaries,
        failures=failures,
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

    _write_json(conversion_json, report)
    logger.progress(1, {"files_written": 1}, force=True)
    _write_text(conversion_md, _conversion_report_markdown(report))
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
        "resume/실패/debug-vis 결과를 묶어 BDD canonical QA summary를 남깁니다.",
        total=2,
    )
    debug_vis_index = _load_json(debug_vis_outputs["debug_vis_index"])
    qa_json = meta_root / "qa_summary.json"
    qa_md = meta_root / "qa_summary.md"
    qa_summary = _qa_summary(report, debug_vis_index, failure_manifest)
    _write_json(qa_json, qa_summary)
    logger.progress(1, {"files_written": 1}, force=True)
    _write_text(qa_md, _qa_summary_markdown(qa_summary))
    logger.progress(2, {"files_written": 2}, force=True)

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


def run_standardization(
    *,
    bdd_root: Path = DEFAULT_BDD_ROOT,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    labels_root: Path = DEFAULT_LABELS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int | None = None,
    max_samples_per_split: int | None = None,
    debug_vis_count: int = DEFAULT_DEBUG_VIS_COUNT,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    write_dataset_readme: bool = True,
    force_reprocess: bool = False,
    fail_on_error: bool = False,
    log_stream: TextIO | None = None,
) -> dict[str, Path]:
    bdd_root = bdd_root.resolve()
    images_root = images_root.resolve()
    labels_root = labels_root.resolve()
    output_root = output_root.resolve()
    workers = workers or _default_workers()

    logger = LiveLogger(log_stream)
    logger.info(f"pv26_bdd100k_standardize version={PIPELINE_VERSION}")
    logger.info(f"bdd_root={bdd_root}")
    logger.info(f"images_root={images_root}")
    logger.info(f"labels_root={labels_root}")
    logger.info(f"output_root={output_root}")
    logger.info(
        f"workers={workers} max_samples_per_split={max_samples_per_split} debug_vis_count={debug_vis_count} "
        f"force_reprocess={force_reprocess} fail_on_error={fail_on_error}"
    )

    if not bdd_root.is_dir():
        raise FileNotFoundError(f"bdd root does not exist: {bdd_root}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"images root does not exist: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels root does not exist: {labels_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    inventory = _inventory_bdd_root(bdd_root, images_root, labels_root)
    readme_path = ""
    if write_dataset_readme:
        logger.stage(
            "source_readme",
            "원본 BDD100K 구조와 PV26 collapse 규칙을 source-local README로 고정합니다.",
            total=1,
        )
        readme = bdd_root / "README.md"
        _write_text(readme, _bdd_readme(bdd_root, inventory))
        readme_path = str(readme)
        logger.progress(1, {"written": 1}, force=True)

    logger.stage(
        "source_inventory",
        "원본 split별 image/json 상태와 부가 자산 존재 여부를 메타데이터로 기록합니다.",
        total=1,
    )
    source_inventory = _build_source_inventory(bdd_root, images_root, labels_root, readme_path)
    logger.progress(1, {"datasets": 1}, force=True)

    logger.stage(
        "pair_discovery",
        "BDD100K image/json이 split별 flat 구조라 stem 기준으로 짝을 맞추고 누락을 확인합니다.",
        total=len(BDD_SPLITS),
    )
    discovery = _discover_pairs(images_root, labels_root)
    for index, split in enumerate(BDD_SPLITS, start=1):
        counts = discovery["per_split_counts"][split]
        logger.progress(
            index,
            {
                "pairs": counts["pairs"],
                "missing_images": counts["missing_images"],
                "missing_labels": counts["missing_labels"],
            },
            force=True,
        )

    pairs = _limit_pairs_per_split(
        sorted(discovery["pairs"], key=lambda item: (item.split, item.relative_id)),
        max_samples_per_split=max_samples_per_split,
    )
    tasks = [BDDTask(pair=pair, output_root=str(output_root)) for pair in pairs]
    summaries, pending_tasks = _scan_existing_outputs(
        tasks,
        force_reprocess=force_reprocess,
        logger=logger,
    )
    fresh_summaries, failures = _run_pending_standardization(
        pending_tasks,
        workers=workers,
        logger=logger,
    )
    summaries.extend(fresh_summaries)
    outputs = _write_standardization_outputs(
        bdd_root=bdd_root,
        images_root=images_root,
        labels_root=labels_root,
        output_root=output_root,
        workers=workers,
        max_samples_per_split=max_samples_per_split,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        source_inventory=source_inventory,
        discovery=discovery,
        summaries=summaries,
        failures=failures,
        logger=logger,
    )

    logger.info("standardization complete")
    if failures and fail_on_error:
        raise RuntimeError(f"BDD100K standardization completed with failures: {len(failures)}")
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hardcoded BDD100K standardization pipeline.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional process pool size. Defaults to CPU count minus one.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional per-split sample limit applied independently to train/val/test for faster verification runs.",
    )
    parser.add_argument(
        "--debug-vis-count",
        type=int,
        default=DEFAULT_DEBUG_VIS_COUNT,
        help="Random QA overlay count written under meta/debug_vis after standardization. Set 0 to disable.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip dataset-local README generation under the BDD100K source root.",
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
        max_samples_per_split=args.max_samples_per_split,
        debug_vis_count=args.debug_vis_count,
        write_dataset_readme=not args.skip_readme,
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
