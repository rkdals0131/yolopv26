from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TypedDict

from common.io import now_iso, write_json, write_text
from common.pv26_schema import LANE_CLASSES, LANE_TYPES, OD_CLASSES, TL_BITS

FINAL_DATASET_STATS_NAME = "final_dataset_stats.json"
FINAL_DATASET_STATS_MARKDOWN_NAME = "final_dataset_stats.md"

TL_COMBO_OFF = "off"
FINAL_DATASET_FOCUS_NAMES = (*OD_CLASSES, "lane", "stop_line", "crosswalk", "tl_attr")


class FinalDatasetSceneRow(TypedDict):
    final_sample_id: str
    split: str
    source_dataset_key: str
    source_kind: str
    scene_path: str
    image_path: str
    det_count: int
    det_class_counts: dict[str, int]
    traffic_light_count: int
    tl_attr_valid_count: int
    lane_count: int
    lane_class_counts: dict[str, int]
    lane_type_counts: dict[str, int]
    stop_line_count: int
    crosswalk_count: int


class FinalDatasetAuditSummary(TypedDict):
    manifest_found: bool
    manifest_sample_count: int
    scanned_scene_count: int
    manifest_scene_path_valid_count: int
    manifest_scene_path_invalid_count: int
    manifest_image_path_valid_count: int
    manifest_image_path_invalid_count: int
    manifest_det_path_present_count: int
    manifest_det_path_valid_count: int
    manifest_det_path_invalid_count: int
    manifest_source_dataset_counts: dict[str, int]
    scanned_source_dataset_counts: dict[str, int]
    rebuild_needed: bool


class FinalDatasetStats(TypedDict):
    version: str
    generated_at: str
    dataset_root: str
    scene_root: str
    manifest_path: str | None
    stats_path: str
    stats_markdown_path: str
    sample_count: int
    dataset_counts: dict[str, int]
    split_counts: dict[str, int]
    dataset_split_counts: dict[str, int]
    source_kind_counts: dict[str, int]
    presence_counts: dict[str, int]
    detector: dict[str, Any]
    traffic_light_attr: dict[str, Any]
    lane: dict[str, Any]
    stop_line: dict[str, int]
    crosswalk: dict[str, int]
    audit: FinalDatasetAuditSummary
    warnings: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return payload


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _combo_name(bits: Iterable[Any]) -> str:
    active = [bit_name for bit_name, value in zip(TL_BITS, bits) if bool(value)]
    return "+".join(active) if active else TL_COMBO_OFF


def _det_box_bucket(bbox: list[float]) -> str:
    if len(bbox) != 4:
        return "unknown"
    x1, y1, x2, y2 = [float(value) for value in bbox]
    min_side = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
    if min_side < 16.0:
        return "tiny"
    if min_side < 32.0:
        return "small"
    return "medium_plus"


def _scene_rows(dataset_root: Path) -> list[FinalDatasetSceneRow]:
    rows: list[FinalDatasetSceneRow] = []
    scene_root = dataset_root / "labels_scene"
    if not scene_root.is_dir():
        return rows
    for scene_path in sorted(scene_root.rglob("*.json"), key=lambda item: (item.parent.name, item.stem)):
        scene = _load_json(scene_path)
        source = scene.get("source", {}) if isinstance(scene.get("source"), dict) else {}
        image = scene.get("image", {}) if isinstance(scene.get("image"), dict) else {}
        split = str(source.get("split") or scene_path.parent.name).strip() or scene_path.parent.name
        dataset_key = str(source.get("dataset") or "unknown_dataset").strip() or "unknown_dataset"
        source_kind = str(source.get("source_kind") or "").strip()
        if not source_kind:
            if dataset_key == "aihub_lane_seoul":
                source_kind = "lane"
            elif dataset_key.startswith("pv26_exhaustive_"):
                source_kind = "exhaustive_od"
            else:
                source_kind = "unknown"
        image_name = str(image.get("file_name") or "").strip()
        image_path = (dataset_root / "images" / split / image_name).resolve()

        det_class_counts = Counter(
            str(item.get("class_name") or "unknown")
            for item in scene.get("detections", [])
            if isinstance(item, dict)
        )
        lane_class_counts = Counter(
            str(item.get("class_name") or "unknown")
            for item in scene.get("lanes", [])
            if isinstance(item, dict)
        )
        lane_type_counts = Counter(
            str(item.get("source_style") or item.get("meta", {}).get("raw_type") or "missing")
            for item in scene.get("lanes", [])
            if isinstance(item, dict)
        )
        traffic_lights = [
            item for item in scene.get("traffic_lights", [])
            if isinstance(item, dict)
        ]

        rows.append(
            {
                "final_sample_id": str(source.get("final_sample_id") or scene_path.stem),
                "split": split,
                "source_dataset_key": dataset_key,
                "source_kind": source_kind,
                "scene_path": str(scene_path.resolve()),
                "image_path": str(image_path),
                "det_count": int(sum(det_class_counts.values())),
                "det_class_counts": _counter_dict(det_class_counts),
                "traffic_light_count": len(traffic_lights),
                "tl_attr_valid_count": int(sum(1 for item in traffic_lights if bool(item.get("tl_attr_valid")))),
                "lane_count": int(sum(lane_class_counts.values())),
                "lane_class_counts": _counter_dict(lane_class_counts),
                "lane_type_counts": _counter_dict(lane_type_counts),
                "stop_line_count": int(len(scene.get("stop_lines", []) or [])),
                "crosswalk_count": int(len(scene.get("crosswalks", []) or [])),
            }
        )
    return rows


def _load_manifest_rows(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "meta" / "final_dataset_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = _load_json(manifest_path)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        return []
    return [dict(item) for item in samples if isinstance(item, dict)]


def _build_audit(
    *,
    dataset_root: Path,
    scene_rows: list[FinalDatasetSceneRow],
    manifest_rows: list[dict[str, Any]],
) -> FinalDatasetAuditSummary:
    manifest_source_counts = Counter(
        str(row.get("source_dataset_key") or "unknown_dataset")
        for row in manifest_rows
    )
    scanned_source_counts = Counter(
        str(row["source_dataset_key"])
        for row in scene_rows
    )
    scene_valid = 0
    image_valid = 0
    det_present = 0
    det_valid = 0
    for row in manifest_rows:
        scene_path = Path(str(row.get("scene_path") or ""))
        image_path = Path(str(row.get("image_path") or ""))
        det_path_value = row.get("det_path")
        if scene_path.is_file():
            scene_valid += 1
        if image_path.is_file():
            image_valid += 1
        if det_path_value not in {None, ""}:
            det_present += 1
            if Path(str(det_path_value)).is_file():
                det_valid += 1
    manifest_found = bool(manifest_rows)
    scene_invalid = max(0, len(manifest_rows) - scene_valid)
    image_invalid = max(0, len(manifest_rows) - image_valid)
    det_invalid = max(0, det_present - det_valid)
    rebuild_needed = bool(scene_invalid or image_invalid or det_invalid or len(manifest_rows) != len(scene_rows))
    return {
        "manifest_found": manifest_found,
        "manifest_sample_count": int(len(manifest_rows)),
        "scanned_scene_count": int(len(scene_rows)),
        "manifest_scene_path_valid_count": int(scene_valid),
        "manifest_scene_path_invalid_count": int(scene_invalid),
        "manifest_image_path_valid_count": int(image_valid),
        "manifest_image_path_invalid_count": int(image_invalid),
        "manifest_det_path_present_count": int(det_present),
        "manifest_det_path_valid_count": int(det_valid),
        "manifest_det_path_invalid_count": int(det_invalid),
        "manifest_source_dataset_counts": _counter_dict(manifest_source_counts),
        "scanned_source_dataset_counts": _counter_dict(scanned_source_counts),
        "rebuild_needed": rebuild_needed,
    }


def _build_detector_stats(
    dataset_root: Path,
    scene_rows: list[FinalDatasetSceneRow],
) -> dict[str, Any]:
    class_image_counts = {class_name: Counter() for class_name in OD_CLASSES}
    class_instance_counts = {class_name: Counter() for class_name in OD_CLASSES}
    bucket_counts = Counter()
    bucket_image_sets: dict[str, set[str]] = defaultdict(set)
    scene_root = dataset_root / "labels_scene"

    for row in scene_rows:
        split = str(row["split"])
        sample_id = str(row["final_sample_id"])
        det_class_counts = row["det_class_counts"]
        for class_name in OD_CLASSES:
            instance_count = int(det_class_counts.get(class_name, 0))
            if instance_count <= 0:
                continue
            class_instance_counts[class_name][split] += instance_count
            class_instance_counts[class_name]["total"] += instance_count
            class_image_counts[class_name][split] += 1
            class_image_counts[class_name]["total"] += 1

        scene_path = scene_root / split / f"{sample_id}.json"
        if not scene_path.is_file():
            continue
        scene = _load_json(scene_path)
        for det_item in scene.get("detections", []):
            if not isinstance(det_item, dict):
                continue
            if str(det_item.get("class_name") or "") != "traffic_light":
                continue
            bbox = det_item.get("bbox")
            bucket = _det_box_bucket(list(bbox) if isinstance(bbox, list) else [])
            bucket_counts[bucket] += 1
            bucket_image_sets[bucket].add(sample_id)

    return {
        "classes": {
            class_name: {
                "image_count": int(class_image_counts[class_name]["total"]),
                "instance_count": int(class_instance_counts[class_name]["total"]),
                "split_image_counts": _counter_dict(class_image_counts[class_name]),
                "split_instance_counts": _counter_dict(class_instance_counts[class_name]),
            }
            for class_name in OD_CLASSES
        },
        "traffic_light_bbox_buckets": {
            bucket_name: {
                "image_count": int(len(bucket_image_sets[bucket_name])),
                "instance_count": int(bucket_counts[bucket_name]),
            }
            for bucket_name in ("tiny", "small", "medium_plus", "unknown")
            if bucket_counts[bucket_name] > 0 or bucket_image_sets[bucket_name]
        },
    }


def _build_tl_attr_stats(dataset_root: Path, scene_rows: list[FinalDatasetSceneRow]) -> dict[str, Any]:
    combo_counts = Counter()
    invalid_reason_counts = Counter()
    bit_positive_counts = Counter()
    split_valid_counts = Counter()
    split_image_counts = Counter()
    scene_root = dataset_root / "labels_scene"
    valid_image_ids: set[str] = set()

    for row in scene_rows:
        split = str(row["split"])
        sample_id = str(row["final_sample_id"])
        scene_path = scene_root / split / f"{sample_id}.json"
        if not scene_path.is_file():
            continue
        scene = _load_json(scene_path)
        sample_has_valid = False
        for item in scene.get("traffic_lights", []):
            if not isinstance(item, dict):
                continue
            valid = bool(item.get("tl_attr_valid"))
            if valid:
                bits = list(item.get("tl_bits") or [0, 0, 0, 0])
                combo_counts[_combo_name(bits)] += 1
                for bit_name, bit_value in zip(TL_BITS, bits):
                    if bool(bit_value):
                        bit_positive_counts[bit_name] += 1
                split_valid_counts[split] += 1
                sample_has_valid = True
            else:
                invalid_reason_counts[str(item.get("collapse_reason") or "unknown")] += 1
        if sample_has_valid:
            split_image_counts[split] += 1
            valid_image_ids.add(sample_id)

    return {
        "valid_count": int(sum(combo_counts.values())),
        "invalid_count": int(sum(invalid_reason_counts.values())),
        "valid_image_count": int(len(valid_image_ids)),
        "split_valid_counts": _counter_dict(split_valid_counts),
        "split_valid_image_counts": _counter_dict(split_image_counts),
        "bit_positive_counts": _counter_dict(bit_positive_counts),
        "combo_counts": _counter_dict(combo_counts),
        "invalid_reason_counts": _counter_dict(invalid_reason_counts),
    }


def _build_lane_stats(scene_rows: list[FinalDatasetSceneRow]) -> dict[str, Any]:
    class_image_counts = {class_name: Counter() for class_name in LANE_CLASSES}
    class_instance_counts = {class_name: Counter() for class_name in LANE_CLASSES}
    type_image_counts = {lane_type: Counter() for lane_type in (*LANE_TYPES, "missing")}
    type_instance_counts = {lane_type: Counter() for lane_type in (*LANE_TYPES, "missing")}

    for row in scene_rows:
        split = str(row["split"])
        for class_name in LANE_CLASSES:
            instance_count = int(row["lane_class_counts"].get(class_name, 0))
            if instance_count <= 0:
                continue
            class_instance_counts[class_name]["total"] += instance_count
            class_instance_counts[class_name][split] += instance_count
            class_image_counts[class_name]["total"] += 1
            class_image_counts[class_name][split] += 1
        for lane_type in (*LANE_TYPES, "missing"):
            instance_count = int(row["lane_type_counts"].get(lane_type, 0))
            if instance_count <= 0:
                continue
            type_instance_counts[lane_type]["total"] += instance_count
            type_instance_counts[lane_type][split] += instance_count
            type_image_counts[lane_type]["total"] += 1
            type_image_counts[lane_type][split] += 1

    return {
        "classes": {
            class_name: {
                "image_count": int(class_image_counts[class_name]["total"]),
                "instance_count": int(class_instance_counts[class_name]["total"]),
                "split_image_counts": _counter_dict(class_image_counts[class_name]),
                "split_instance_counts": _counter_dict(class_instance_counts[class_name]),
            }
            for class_name in LANE_CLASSES
        },
        "types": {
            lane_type: {
                "image_count": int(type_image_counts[lane_type]["total"]),
                "instance_count": int(type_instance_counts[lane_type]["total"]),
                "split_image_counts": _counter_dict(type_image_counts[lane_type]),
                "split_instance_counts": _counter_dict(type_instance_counts[lane_type]),
            }
            for lane_type in (*LANE_TYPES, "missing")
        },
    }


def _presence_counts(scene_rows: list[FinalDatasetSceneRow]) -> dict[str, int]:
    output = Counter()
    for row in scene_rows:
        if int(row["det_count"]) > 0:
            output["det"] += 1
        if int(row["traffic_light_count"]) > 0:
            output["traffic_light"] += 1
        if int(row["tl_attr_valid_count"]) > 0:
            output["tl_attr_valid"] += 1
        if int(row["lane_count"]) > 0:
            output["lane"] += 1
        if int(row["stop_line_count"]) > 0:
            output["stop_line"] += 1
        if int(row["crosswalk_count"]) > 0:
            output["crosswalk"] += 1
    return _counter_dict(output)


def _warnings(*, stats: FinalDatasetStats) -> list[str]:
    warnings: list[str] = []
    detector_total = sum(
        int(payload["instance_count"])
        for payload in stats["detector"]["classes"].values()
    )
    if detector_total <= 0:
        warnings.append("detector_supervision_missing")
    if stats["dataset_counts"] == {"aihub_lane_seoul": stats["sample_count"]}:
        warnings.append("lane_only_final_dataset")
    audit = stats["audit"]
    if int(audit["manifest_scene_path_invalid_count"]) > 0 or int(audit["manifest_image_path_invalid_count"]) > 0:
        warnings.append("manifest_paths_stale")
    if bool(audit["rebuild_needed"]):
        warnings.append("rebuild_recommended")
    tl_val_count = int(stats["detector"]["classes"]["traffic_light"]["split_image_counts"].get("val", 0))
    if tl_val_count <= 0:
        warnings.append("traffic_light_val_missing")
    if int(audit["manifest_det_path_present_count"]) <= 0:
        warnings.append("manifest_det_paths_missing")
    return warnings


def _stats_markdown(stats: FinalDatasetStats) -> str:
    detector_classes = stats["detector"]["classes"]
    traffic_light = detector_classes["traffic_light"]
    lane_class_summary = ", ".join(
        f"{name}:{stats['lane']['classes'][name]['instance_count']}"
        for name in LANE_CLASSES
    )
    lane_type_summary = ", ".join(
        f"{name}:{stats['lane']['types'][name]['instance_count']}"
        for name in (*LANE_TYPES, "missing")
    )
    audit = stats["audit"]
    lines = [
        "# Final Dataset Stats",
        "",
        f"- Generated: `{stats['generated_at']}`",
        f"- Dataset root: `{stats['dataset_root']}`",
        f"- Samples: `{stats['sample_count']}`",
        f"- Dataset counts: `{stats['dataset_counts']}`",
        f"- Split counts: `{stats['split_counts']}`",
        f"- Source kinds: `{stats['source_kind_counts']}`",
        f"- Presence counts: `{stats['presence_counts']}`",
        f"- Warnings: `{stats['warnings']}`",
        "",
        "## Detector",
        "",
    ]
    for class_name in OD_CLASSES:
        class_stats = detector_classes[class_name]
        lines.append(
            f"- `{class_name}`: images={class_stats['image_count']}, instances={class_stats['instance_count']}, "
            f"splits={class_stats['split_image_counts']}"
        )
    lines.extend(
        [
            "",
            "## Traffic Light Attr",
            "",
            f"- valid={stats['traffic_light_attr']['valid_count']}, invalid={stats['traffic_light_attr']['invalid_count']}, "
            f"valid_images={stats['traffic_light_attr']['valid_image_count']}",
            f"- combo_counts={stats['traffic_light_attr']['combo_counts']}",
            f"- bit_positive_counts={stats['traffic_light_attr']['bit_positive_counts']}",
            "",
            "## Lane Family",
            "",
            f"- lane classes={{{lane_class_summary}}}",
            f"- lane types={{{lane_type_summary}}}",
            f"- stop_line={stats['stop_line']}",
            f"- crosswalk={stats['crosswalk']}",
            "",
            "## Audit",
            "",
            f"- manifest_found={audit['manifest_found']}",
            f"- manifest_scene_valid={audit['manifest_scene_path_valid_count']}/{audit['manifest_sample_count']}",
            f"- manifest_image_valid={audit['manifest_image_path_valid_count']}/{audit['manifest_sample_count']}",
            f"- manifest_det_valid={audit['manifest_det_path_valid_count']}/{audit['manifest_det_path_present_count']}",
            f"- rebuild_needed={audit['rebuild_needed']}",
            "",
            "## TL Focus",
            "",
            f"- traffic_light images={traffic_light['image_count']}, val_images={traffic_light['split_image_counts'].get('val', 0)}",
            f"- bbox buckets={stats['detector']['traffic_light_bbox_buckets']}",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_final_dataset(
    *,
    dataset_root: Path,
    write_artifacts: bool = True,
) -> FinalDatasetStats:
    resolved_root = dataset_root.resolve()
    meta_root = resolved_root / "meta"
    scene_rows = _scene_rows(resolved_root)
    dataset_counts = Counter(str(row["source_dataset_key"]) for row in scene_rows)
    split_counts = Counter(str(row["split"]) for row in scene_rows)
    dataset_split_counts = Counter(
        f"{row['source_dataset_key']}::{row['split']}"
        for row in scene_rows
    )
    source_kind_counts = Counter(str(row["source_kind"]) for row in scene_rows)
    manifest_rows = _load_manifest_rows(resolved_root)
    audit = _build_audit(
        dataset_root=resolved_root,
        scene_rows=scene_rows,
        manifest_rows=manifest_rows,
    )
    stats_path = meta_root / FINAL_DATASET_STATS_NAME
    stats_markdown_path = meta_root / FINAL_DATASET_STATS_MARKDOWN_NAME
    stats: FinalDatasetStats = {
        "version": "pv26-final-dataset-stats-v1",
        "generated_at": now_iso(),
        "dataset_root": str(resolved_root),
        "scene_root": str((resolved_root / "labels_scene").resolve()),
        "manifest_path": str((resolved_root / "meta" / "final_dataset_manifest.json").resolve())
        if (resolved_root / "meta" / "final_dataset_manifest.json").is_file()
        else None,
        "stats_path": str(stats_path),
        "stats_markdown_path": str(stats_markdown_path),
        "sample_count": int(len(scene_rows)),
        "dataset_counts": _counter_dict(dataset_counts),
        "split_counts": _counter_dict(split_counts),
        "dataset_split_counts": _counter_dict(dataset_split_counts),
        "source_kind_counts": _counter_dict(source_kind_counts),
        "presence_counts": _presence_counts(scene_rows),
        "detector": _build_detector_stats(resolved_root, scene_rows),
        "traffic_light_attr": _build_tl_attr_stats(resolved_root, scene_rows),
        "lane": _build_lane_stats(scene_rows),
        "stop_line": {
            "image_count": int(sum(1 for row in scene_rows if int(row["stop_line_count"]) > 0)),
            "instance_count": int(sum(int(row["stop_line_count"]) for row in scene_rows)),
        },
        "crosswalk": {
            "image_count": int(sum(1 for row in scene_rows if int(row["crosswalk_count"]) > 0)),
            "instance_count": int(sum(int(row["crosswalk_count"]) for row in scene_rows)),
        },
        "audit": audit,
        "warnings": [],
    }
    stats["warnings"] = _warnings(stats=stats)
    if write_artifacts:
        write_json(stats_path, stats)
        write_text(stats_markdown_path, _stats_markdown(stats))
    return stats


def load_final_dataset_stats(dataset_root: Path) -> FinalDatasetStats | None:
    stats_path = dataset_root.resolve() / "meta" / FINAL_DATASET_STATS_NAME
    if not stats_path.is_file():
        return None
    payload = _load_json(stats_path)
    return payload if isinstance(payload, dict) else None


def _seeded_row_key(row: FinalDatasetSceneRow, *, focus: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{focus}:{row['source_dataset_key']}:{row['split']}:{row['final_sample_id']}".encode("utf-8")
    ).hexdigest()
    return digest, str(row["final_sample_id"])


def _focus_matches(row: FinalDatasetSceneRow, *, focus: str) -> bool:
    if focus in OD_CLASSES:
        return int(row["det_class_counts"].get(focus, 0)) > 0
    if focus == "lane":
        return int(row["lane_count"]) > 0
    if focus == "stop_line":
        return int(row["stop_line_count"]) > 0
    if focus == "crosswalk":
        return int(row["crosswalk_count"]) > 0
    if focus == "tl_attr":
        return int(row["tl_attr_valid_count"]) > 0
    raise ValueError(f"unsupported final dataset focus: {focus}")


def select_final_dataset_focus_rows(
    *,
    dataset_root: Path,
    focus: str,
    split: str,
    count: int,
    seed: int | None = None,
) -> list[FinalDatasetSceneRow]:
    normalized_focus = str(focus).strip()
    if normalized_focus not in FINAL_DATASET_FOCUS_NAMES:
        raise ValueError(
            f"unsupported final dataset focus: {focus}; expected one of {FINAL_DATASET_FOCUS_NAMES}"
        )
    normalized_split = str(split).strip()
    if count <= 0:
        raise ValueError("count must be > 0")
    grouped: dict[str, list[FinalDatasetSceneRow]] = defaultdict(list)
    for row in _scene_rows(dataset_root.resolve()):
        if str(row["split"]) != normalized_split:
            continue
        if not _focus_matches(row, focus=normalized_focus):
            continue
        grouped[str(row["source_dataset_key"])].append(row)
    if not grouped:
        raise RuntimeError(f"no final dataset samples found for focus={normalized_focus} split={normalized_split}")
    if seed is not None:
        for dataset_key in grouped:
            grouped[dataset_key] = sorted(
                grouped[dataset_key],
                key=lambda item: _seeded_row_key(item, focus=normalized_focus, seed=int(seed)),
            )
    else:
        for dataset_key in grouped:
            grouped[dataset_key] = sorted(grouped[dataset_key], key=lambda item: str(item["final_sample_id"]))
    selected: list[FinalDatasetSceneRow] = []
    while len(selected) < count:
        made_progress = False
        for dataset_key in sorted(grouped):
            rows = grouped[dataset_key]
            if not rows:
                continue
            selected.append(rows.pop(0))
            made_progress = True
            if len(selected) >= count:
                break
        if not made_progress:
            break
    if not selected:
        raise RuntimeError(f"no selectable final dataset samples for focus={normalized_focus} split={normalized_split}")
    return selected


__all__ = [
    "FINAL_DATASET_FOCUS_NAMES",
    "FINAL_DATASET_STATS_MARKDOWN_NAME",
    "FINAL_DATASET_STATS_NAME",
    "FinalDatasetAuditSummary",
    "FinalDatasetSceneRow",
    "FinalDatasetStats",
    "analyze_final_dataset",
    "load_final_dataset_stats",
    "select_final_dataset_focus_rows",
]
