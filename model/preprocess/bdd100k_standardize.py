from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .aihub_common import IMAGE_EXTENSIONS, PairRecord, _now_iso, _safe_slug
from .aihub_standardize import (
    LiveLogger,
    TL_BITS,
    _bbox_to_yolo_line,
    _counter_to_dict,
    _default_workers,
    _det_class_map_yaml,
    _generate_debug_vis,
    _link_or_copy,
    _write_json,
    _write_text,
)

PIPELINE_VERSION = "pv26-bdd100k-standardize-v1"
SCENE_VERSION = "pv26-scene-bdd100k-v1"
DEFAULT_REPO_ROOT = Path("/home/user1/ROS2_Workspace/ros2_ws/src/yolopv26")
DEFAULT_BDD_ROOT = DEFAULT_REPO_ROOT / "seg_dataset" / "BDD100K"
DEFAULT_IMAGES_ROOT = DEFAULT_BDD_ROOT / "bdd100k_images_100k" / "100k"
DEFAULT_LABELS_ROOT = DEFAULT_BDD_ROOT / "bdd100k_labels" / "100k"
DEFAULT_OUTPUT_ROOT = DEFAULT_BDD_ROOT.parent / "pv26_bdd100k_standardized"
DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26
OUTPUT_DATASET_KEY = "bdd100k_det_100k"
BDD_SPLITS = ("train", "val", "test")
BDD_IMAGE_WIDTH = 1280
BDD_IMAGE_HEIGHT = 720
HELD_ANNOTATION_LIMIT = 32
OFFICIAL_SPLIT_SIZES = {
    "train": 70_000,
    "val": 10_000,
    "test": 20_000,
}
OD_CLASSES = [
    "vehicle",
    "bike",
    "pedestrian",
    "traffic_cone",
    "obstacle",
    "traffic_light",
    "sign",
]
OD_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(OD_CLASSES)}
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
    "traffic light": "traffic_light",
    "traffic sign": "sign",
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
        "- 사용 목적: `7-class object detection` 보강",
        "- 사용 원천: `bdd100k_images_100k/100k/<split>` + `bdd100k_labels/100k/<split>/*.json`",
        "- 비사용 원천: drivable map, segmentation map, det_20 preview asset",
        "- TL 4-bit supervision은 BDD source에서 사용하지 않는다. BDD traffic light는 generic bbox만 detector supervision에 사용한다.",
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
            "- 문맥 메타: `attributes.weather`, `attributes.scene`, `attributes.timeofday`",
            "",
            "## PV26 클래스 collapse 규칙",
            "",
            "- `car/truck/bus/train/(other vehicle, van, caravan, trailer alias)` -> `vehicle`",
            "- `bike/motor/(bicycle, motorcycle alias)` -> `bike`",
            "- `person/rider/(other person alias)` -> `pedestrian`",
            "- `traffic light` -> `traffic_light`",
            "- `traffic sign` -> `sign`",
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

    width = BDD_IMAGE_WIDTH
    height = BDD_IMAGE_HEIGHT
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

        if mapped_class == "traffic_light":
            state_hint = str(obj.get("attributes", {}).get("trafficLightColor") or "unknown").strip().lower()
            tl_state_hint_counts[state_hint or "unknown"] += 1
            traffic_lights.append(
                {
                    "id": len(traffic_lights),
                    "detection_id": detection_id,
                    "bbox": bbox,
                    "tl_bits": {bit: 0 for bit in TL_BITS},
                    "tl_attr_valid": 0,
                    "collapse_reason": "bdd_det_only_source",
                    "state_hint": state_hint or "unknown",
                    "raw_attributes": obj.get("attributes"),
                    "meta": {
                        "dataset_label": canonical_category,
                    },
                }
            )
        elif mapped_class == "sign":
            traffic_signs.append(
                {
                    "id": len(traffic_signs),
                    "detection_id": detection_id,
                    "bbox": bbox,
                    "raw_attributes": obj.get("attributes"),
                    "meta": {
                        "dataset_label": canonical_category,
                    },
                }
            )

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
            "Traffic-light raw color hints are preserved as state_hint but do not enable TL attribute supervision.",
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

    for item in summaries:
        split = item["split"]
        split_counts[split]["samples"] += 1
        split_counts[split]["detections"] += item["det_count"]
        split_counts[split]["traffic_lights"] += item["traffic_light_count"]
        split_counts[split]["traffic_signs"] += item["traffic_sign_count"]
        total_detections += item["det_count"]
        total_lights += item["traffic_light_count"]
        total_signs += item["traffic_sign_count"]
        materialization_counts[item["image_materialization"]] += 1
        det_class_counts.update(item["det_class_counts"])
        raw_category_counts.update(item["raw_category_counts"])
        tl_state_hint_counts.update(item["tl_state_hint_counts"])
        held_reason_counts.update(item["held_reason_counts"])

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
        },
        "det_class_map": {str(index): class_name for index, class_name in enumerate(OD_CLASSES)},
        "tl_bits": TL_BITS,
        "dataset": {
            "dataset_key": OUTPUT_DATASET_KEY,
            "processed_samples": len(summaries),
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
        f"workers={workers} max_samples_per_split={max_samples_per_split} debug_vis_count={debug_vis_count}"
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

    logger.stage(
        "parallel_standardize",
        "BDD JSON 파싱, 7-class remap, YOLO txt 직렬화, image materialization을 프로세스 풀로 병렬 실행합니다.",
        total=len(tasks),
    )
    summaries: list[dict[str, Any]] = []
    progress = Counter()
    completed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_worker_entry, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    logger.info(
                        f"stage=parallel_standardize error split={task.pair.split} sample={task.pair.relative_id} error={exc}"
                    )
                    raise
                summaries.append(summary)
                completed += 1
                progress["samples"] = completed
                progress["detections"] += summary["det_count"]
                progress["traffic_lights"] += summary["traffic_light_count"]
                progress["traffic_signs"] += summary["traffic_sign_count"]
                progress["held"] += sum(summary["held_reason_counts"].values())
                logger.progress(completed, dict(progress))
        logger.progress(completed, dict(progress), force=True)
    else:
        logger.progress(0, {"samples": 0}, force=True)

    logger.stage(
        "report_write",
        "클래스 맵, 변환 리포트, source inventory를 기록해 loader가 동일 계약을 읽을 수 있게 만듭니다.",
        total=6,
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
    )
    meta_root = output_root / "meta"
    conversion_json = meta_root / "conversion_report.json"
    conversion_md = meta_root / "conversion_report.md"
    inventory_json = meta_root / "source_inventory.json"
    inventory_md = meta_root / "source_inventory.md"
    det_map_yaml = meta_root / "class_map_det.yaml"
    scene_map_yaml = meta_root / "class_map_scene.yaml"

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

    debug_vis_outputs = _generate_debug_vis(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        logger=logger,
    )

    logger.info("standardization complete")
    return {
        "output_root": output_root,
        "conversion_json": conversion_json,
        "conversion_md": conversion_md,
        "inventory_json": inventory_json,
        "inventory_md": inventory_md,
        "det_map_yaml": det_map_yaml,
        "scene_map_yaml": scene_map_yaml,
        "debug_vis_dir": debug_vis_outputs["debug_vis_dir"],
        "debug_vis_index": debug_vis_outputs["debug_vis_index"],
    }


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
        help="Optional smoke-run limit applied independently to train/val/test.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    outputs = run_standardization(
        workers=args.workers,
        max_samples_per_split=args.max_samples_per_split,
        debug_vis_count=args.debug_vis_count,
        write_dataset_readme=not args.skip_readme,
    )
    print(f"output_root={outputs['output_root']}")
    print(f"conversion_json={outputs['conversion_json']}")
    print(f"inventory_json={outputs['inventory_json']}")
    print(f"debug_vis_dir={outputs['debug_vis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
