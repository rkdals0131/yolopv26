from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypedDict

from common.io import write_json
from common.overlay import render_overlay


DEFAULT_REVIEW_QUOTAS: dict[str, int] = {
    "pv26_exhaustive_bdd100k_det_100k": 75,
    "pv26_exhaustive_aihub_traffic_seoul": 75,
    "pv26_exhaustive_aihub_obstacle_seoul": 75,
    "aihub_lane_seoul": 75,
}
REVIEW_BUNDLE_INDEX_NAME = "index.json"


class ReviewSampleRow(TypedDict):
    final_sample_id: str
    source_dataset_key: str
    split: str
    scene_path: str
    image_path: str


class ReviewBundleEntry(TypedDict):
    dataset_key: str
    split: str
    final_sample_id: str
    scene_path: str
    image_path: str
    overlay_path: str


class ReviewBundleSummary(TypedDict):
    summary_path: str
    index_path: str
    version: str
    manifest_path: str
    output_root: str
    split: str
    quotas: dict[str, int]
    seed: int | None
    image_count: int
    entries: list[ReviewBundleEntry]


def _coerce_bbox(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [
            float(value.get("x1", 0.0)),
            float(value.get("y1", 0.0)),
            float(value.get("x2", 0.0)),
            float(value.get("y2", 0.0)),
        ]
    if isinstance(value, list):
        return [float(item) for item in value[:4]]
    return []


def _coerce_points(value: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for item in value or []:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                points.append([float(item["x"]), float(item["y"])])
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append([float(item[0]), float(item[1])])
    return points


def _box_key(bbox: list[float]) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    return tuple(round(float(value), 3) for value in bbox)


def canonical_scene_to_overlay_scene(scene: dict[str, Any], *, image_path: Path) -> dict[str, Any]:
    overlay_scene: dict[str, Any] = {
        "image": {"source_path": str(Path(image_path).resolve())},
        "detections": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "ignored_regions": [],
        "debug_rectangles": list(scene.get("debug_rectangles") or []),
    }
    traffic_light_keys: set[tuple[float, float, float, float]] = set()
    traffic_sign_keys: set[tuple[float, float, float, float]] = set()

    for light in scene.get("traffic_lights", []):
        bbox = _coerce_bbox(light.get("bbox"))
        key = _box_key(bbox)
        if key is None or key in traffic_light_keys:
            continue
        overlay_scene["traffic_lights"].append({"bbox": bbox})
        traffic_light_keys.add(key)

    for sign in scene.get("traffic_signs", []):
        bbox = _coerce_bbox(sign.get("bbox"))
        key = _box_key(bbox)
        if key is None or key in traffic_sign_keys:
            continue
        overlay_scene["traffic_signs"].append({"bbox": bbox})
        traffic_sign_keys.add(key)

    for detection in scene.get("detections", []):
        class_name = str(detection.get("class_name") or "").strip()
        bbox = _coerce_bbox(detection.get("bbox"))
        key = _box_key(bbox)
        if class_name == "traffic_light":
            if key is not None and key not in traffic_light_keys:
                overlay_scene["traffic_lights"].append({"bbox": bbox})
                traffic_light_keys.add(key)
            continue
        if class_name == "sign":
            if key is not None and key not in traffic_sign_keys:
                overlay_scene["traffic_signs"].append({"bbox": bbox})
                traffic_sign_keys.add(key)
            continue
        overlay_scene["detections"].append({"class_name": class_name, "bbox": bbox})

    for lane in scene.get("lanes", []):
        overlay_scene["lanes"].append(
            {
                "class_name": lane.get("class_name"),
                "points": _coerce_points(lane.get("points")),
            }
        )
    for stop_line in scene.get("stop_lines", []):
        overlay_scene["stop_lines"].append({"points": _coerce_points(stop_line.get("points"))})
    for crosswalk in scene.get("crosswalks", []):
        overlay_scene["crosswalks"].append({"points": _coerce_points(crosswalk.get("points"))})
    for ignored_region in scene.get("ignored_regions", []):
        overlay_scene["ignored_regions"].append({"points": _coerce_points(ignored_region.get("points"))})
    return overlay_scene


def _normalize_review_quotas(quotas: Mapping[str, int] | None) -> dict[str, int]:
    payload = quotas or DEFAULT_REVIEW_QUOTAS
    normalized = {}
    for dataset_key, count in payload.items():
        dataset_name = str(dataset_key).strip()
        quota = int(count)
        if not dataset_name:
            raise ValueError("dataset_key must not be empty")
        if quota <= 0:
            raise ValueError(f"quota must be > 0 for dataset {dataset_name}")
        normalized[dataset_name] = quota
    return normalized


def _seeded_review_order_key(row: Mapping[str, Any], *, dataset_key: str, seed: int) -> tuple[str, str, str]:
    sample_id = str(row.get("final_sample_id") or "")
    scene_path = str(row.get("scene_path") or "")
    digest = hashlib.sha256(f"{seed}:{dataset_key}:{sample_id}:{scene_path}".encode("utf-8")).hexdigest()
    return digest, sample_id, scene_path


def select_review_rows(
    manifest: dict[str, Any],
    *,
    split: str = "val",
    quotas: Mapping[str, int] | None = None,
    seed: int | None = None,
) -> list[ReviewSampleRow]:
    normalized_quotas = _normalize_review_quotas(quotas)
    rows = sorted(
        manifest.get("samples") or [],
        key=lambda item: (
            str(item.get("source_dataset_key") or ""),
            str(item.get("split") or ""),
            str(item.get("final_sample_id") or ""),
            str(item.get("scene_path") or ""),
        ),
    )
    rows_by_dataset: dict[str, list[ReviewSampleRow]] = {dataset_key: [] for dataset_key in normalized_quotas}
    for row in rows:
        dataset_key = str(row.get("source_dataset_key") or "").strip()
        if dataset_key not in rows_by_dataset:
            continue
        if str(row.get("split") or "").strip() != split:
            continue
        rows_by_dataset[dataset_key].append(dict(row))

    selected: list[ReviewSampleRow] = []
    counts = Counter()
    for dataset_key, target in normalized_quotas.items():
        dataset_rows = rows_by_dataset.get(dataset_key, [])
        if seed is not None:
            dataset_rows = sorted(
                dataset_rows,
                key=lambda item: _seeded_review_order_key(item, dataset_key=dataset_key, seed=int(seed)),
            )
        picked = dataset_rows[:target]
        selected.extend(picked)
        counts[dataset_key] = len(picked)
    missing = [
        f"{dataset_key}={counts[dataset_key]}/{target}"
        for dataset_key, target in normalized_quotas.items()
        if counts[dataset_key] < target
    ]
    if missing:
        raise RuntimeError(f"review bundle is missing required samples: {', '.join(missing)}")
    return selected


def render_review_bundle(
    *,
    manifest_path: Path,
    output_root: Path,
    split: str = "val",
    quotas: Mapping[str, int] | None = None,
    seed: int | None = None,
) -> ReviewBundleSummary:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    selected_rows = select_review_rows(manifest, split=split, quotas=quotas, seed=seed)
    resolved_output_root = Path(output_root).resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    entries: list[ReviewBundleEntry] = []
    for row in selected_rows:
        scene_path = Path(str(row["scene_path"])).resolve()
        image_path = Path(str(row["image_path"])).resolve()
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        overlay_scene = canonical_scene_to_overlay_scene(scene, image_path=image_path)
        dataset_key = str(row["source_dataset_key"])
        sample_id = str(row["final_sample_id"])
        output_path = resolved_output_root / dataset_key / f"{sample_id}.png"
        render_overlay(overlay_scene, output_path)
        entries.append(
            {
                "dataset_key": dataset_key,
                "split": str(row["split"]),
                "final_sample_id": sample_id,
                "scene_path": str(scene_path),
                "image_path": str(image_path),
                "overlay_path": str(output_path),
            }
        )

    index_path = resolved_output_root / REVIEW_BUNDLE_INDEX_NAME
    summary: ReviewBundleSummary = {
        "summary_path": str(index_path),
        "index_path": str(index_path),
        "version": "od-bootstrap-review-v1",
        "manifest_path": str(resolved_manifest_path),
        "output_root": str(resolved_output_root),
        "split": split,
        "quotas": _normalize_review_quotas(quotas),
        "seed": int(seed) if seed is not None else None,
        "image_count": len(entries),
        "entries": entries,
    }
    write_json(index_path, summary)
    return summary


__all__ = [
    "DEFAULT_REVIEW_QUOTAS",
    "REVIEW_BUNDLE_INDEX_NAME",
    "canonical_scene_to_overlay_scene",
    "render_overlay",
    "render_review_bundle",
    "select_review_rows",
]
