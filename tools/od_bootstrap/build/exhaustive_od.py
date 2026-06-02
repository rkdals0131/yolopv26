from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, TypedDict

import yaml

from common.io import read_json, write_json, write_text
from common.pv26_schema import EXHAUSTIVE_DATASET_KEY_BY_SOURCE, OD_CLASS_TO_ID, OD_CLASSES

from .artifacts import BoxProvenance
from .image_list import ImageListEntry
from ..teacher.policy import apply_policy_to_predictions

EXHAUSTIVE_MATERIALIZATION_MANIFEST_NAME = "materialization_manifest.json"
EXHAUSTIVE_MATERIALIZATION_SUMMARY_NAME = "materialization_summary.json"


@dataclass(frozen=True)
class MaterializedSample:
    sample_id: str
    dataset_key: str
    scene_path: Path
    det_path: Path
    image_path: Path
    raw_detection_count: int
    bootstrap_detection_count: int


class TeacherPredictionRow(TypedDict):
    sample_id: str
    sample_uid: str
    image_path: str
    scene_path: str
    dataset_key: str
    split: str
    teacher_name: str
    model_version: str
    class_name: str
    confidence: float
    xyxy: list[float]
    box_index: int
    image_width: int
    image_height: int


class ExhaustiveSampleRow(TypedDict):
    sample_id: str
    sample_uid: str
    source_dataset_key: str
    exhaustive_dataset_key: str
    split: str
    source_scene_path: str
    source_image_path: str
    source_det_path: str | None
    scene_path: str
    det_path: str
    image_path: str
    raw_detection_count: int
    bootstrap_detection_count: int


class ExhaustiveMaterializationManifest(TypedDict):
    version: str
    run_id: str
    generated_at: str
    dataset_root: str
    sample_count: int
    class_counts: dict[str, int]
    samples: list[ExhaustiveSampleRow]


class ExhaustiveMaterializationSummary(TypedDict):
    dataset_root: str
    manifest_path: str
    summary_path: str
    run_id: str
    generated_at: str
    sample_count: int
    class_counts: dict[str, int]


def _build_materialization_summary(
    *,
    dataset_root: Path,
    run_id: str,
    created_at: str,
    sample_count: int,
    class_counts: dict[str, int],
) -> ExhaustiveMaterializationSummary:
    manifest_path = dataset_root / "meta" / EXHAUSTIVE_MATERIALIZATION_MANIFEST_NAME
    summary_path = dataset_root / "meta" / EXHAUSTIVE_MATERIALIZATION_SUMMARY_NAME
    return {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "run_id": run_id,
        "generated_at": created_at,
        "sample_count": sample_count,
        "class_counts": dict(sorted(class_counts.items())),
    }


def _bbox_from_scene_detection(detection: dict[str, Any]) -> list[float]:
    bbox = detection.get("bbox") or {}
    if isinstance(bbox, dict):
        return [
            float(bbox.get("x1", 0.0)),
            float(bbox.get("y1", 0.0)),
            float(bbox.get("x2", 0.0)),
            float(bbox.get("y2", 0.0)),
        ]
    if isinstance(bbox, (list, tuple)):
        values = [float(item) for item in bbox[:4]]
        if len(values) == 4:
            return values
    return [0.0, 0.0, 0.0, 0.0]


def _bbox_to_mapping(box: list[float]) -> dict[str, float]:
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
    }


def _bbox_to_yolo_line(class_name: str, box: list[float], width: int, height: int) -> str:
    class_id = OD_CLASS_TO_ID[class_name]
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) * 0.5) / float(width)
    center_y = ((y1 + y2) * 0.5) / float(height)
    box_w = max(0.0, x2 - x1) / float(width)
    box_h = max(0.0, y2 - y1) / float(height)
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"

def _materialize_image(source_path: Path, target_path: Path, *, copy_images: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if copy_images:
        shutil.copy2(source_path, target_path)
        return
    try:
        target_path.hardlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _default_io_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _validate_materialization_entries(entries: tuple[ImageListEntry, ...]) -> None:
    seen_sample_uids: set[str] = set()
    for entry in entries:
        sample_uid = str(entry.sample_uid).strip()
        if not sample_uid:
            raise ValueError("materialization entry sample_uid must not be empty")
        if sample_uid in seen_sample_uids:
            raise ValueError(f"duplicate sample_uid in materialization entries: {sample_uid}")
        seen_sample_uids.add(sample_uid)


def _raw_provenance(*, run_id: str, created_at: str) -> BoxProvenance:
    return BoxProvenance(
        label_origin="raw_source",
        teacher_name="raw_source",
        confidence=1.0,
        model_version="source_label",
        run_id=run_id,
        created_at=created_at,
    )


def _load_scene(scene_path: Path) -> dict[str, Any]:
    payload = read_json(scene_path)
    if not isinstance(payload, dict):
        raise TypeError(f"scene root must be an object: {scene_path}")
    return payload


def _resolve_entry_source_metadata(scene: dict[str, Any], entry: ImageListEntry) -> tuple[str, str]:
    source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
    source_dataset_key = str(source.get("dataset") or "").strip()
    if not source_dataset_key:
        raise ValueError(f"scene source.dataset missing: {entry.scene_path}")
    if source_dataset_key != entry.dataset_key:
        raise ValueError(
            f"scene source.dataset must match image list dataset_key: {source_dataset_key} != {entry.dataset_key} ({entry.scene_path})"
        )
    if source_dataset_key not in EXHAUSTIVE_DATASET_KEY_BY_SOURCE:
        raise ValueError(f"unsupported exhaustive OD source dataset: {source_dataset_key} ({entry.scene_path})")

    split = str(source.get("split") or "").strip()
    if not split:
        raise ValueError(f"scene source.split missing: {entry.scene_path}")
    if split != entry.split:
        raise ValueError(f"scene source.split must match image list split: {split} != {entry.split} ({entry.scene_path})")
    return source_dataset_key, split


def _assert_raw_detection_row_ids(raw_detections: list[Any], *, scene_path: Path) -> None:
    for row_index, detection in enumerate(raw_detections):
        if not isinstance(detection, dict):
            raise TypeError(f"scene detections[{row_index}] must be an object: {scene_path}")
        detection_id = detection.get("id")
        if isinstance(detection_id, bool) or not isinstance(detection_id, int):
            raise ValueError(f"scene detections[{row_index}].id must match detection row order: {scene_path}")
        if detection_id != row_index:
            raise ValueError(
                f"scene detections[{row_index}].id must match detection row order: "
                f"id={detection_id} expected={row_index} ({scene_path})"
            )


def _materialize_sample(
    *,
    entry: ImageListEntry,
    predictions_by_sample_uid: dict[str, list[TeacherPredictionRow]],
    class_policy: dict[str, Any],
    dataset_root: Path,
    run_id: str,
    created_at: str,
    copy_images: bool,
) -> tuple[MaterializedSample, dict[str, int], ExhaustiveSampleRow]:
    class_counts = Counter()
    scene = _load_scene(entry.scene_path)
    source_dataset_key, split = _resolve_entry_source_metadata(scene, entry)
    exhaustive_dataset_key = EXHAUSTIVE_DATASET_KEY_BY_SOURCE[source_dataset_key]
    original_image_name = str(scene.get("image", {}).get("file_name") or entry.image_path.name)
    materialized_image_name = f"{entry.sample_uid}{entry.image_path.suffix.lower()}"
    raw_detections = list(scene.get("detections") or [])
    _assert_raw_detection_row_ids(raw_detections, scene_path=entry.scene_path)

    final_scene = deepcopy(scene)
    final_scene["source"]["dataset"] = exhaustive_dataset_key
    final_scene["source"]["bootstrap_run_id"] = run_id
    final_scene["source"]["bootstrap_created_at"] = created_at
    final_scene["source"]["bootstrap_original_dataset"] = source_dataset_key
    final_scene["source"]["bootstrap_original_sample_id"] = entry.sample_id
    final_scene["source"]["bootstrap_sample_uid"] = entry.sample_uid
    final_detections: list[dict[str, Any]] = []
    raw_boxes_by_class: dict[str, list[list[float]]] = defaultdict(list)

    raw_provenance = _raw_provenance(run_id=run_id, created_at=created_at).to_dict()
    for detection in raw_detections:
        detection_copy = deepcopy(detection)
        box = _bbox_from_scene_detection(detection_copy)
        class_name = str(detection_copy.get("class_name") or "")
        raw_boxes_by_class[class_name].append(box)
        detection_copy["provenance"] = raw_provenance
        detection_copy.setdefault("meta", {})
        detection_copy["meta"]["bootstrap_label_origin"] = "raw_source"
        final_detections.append(detection_copy)
        class_counts[class_name] += 1

    kept_predictions = apply_policy_to_predictions(
        rows=predictions_by_sample_uid.get(entry.sample_uid, []),
        class_policy=class_policy,
        dataset_key=source_dataset_key,
        image_width=int(final_scene["image"]["width"]),
        image_height=int(final_scene["image"]["height"]),
        raw_boxes_by_class=raw_boxes_by_class,
    )

    next_detection_id = len(final_detections)
    for row in sorted(kept_predictions, key=lambda item: (str(item["class_name"]), -float(item["confidence"]))):
        box = [float(value) for value in row["xyxy"]]
        provenance = BoxProvenance(
            label_origin="bootstrap",
            teacher_name=str(row["teacher_name"]),
            confidence=float(row["confidence"]),
            model_version=str(row["model_version"]),
            run_id=run_id,
            created_at=created_at,
        ).to_dict()
        final_detections.append(
            {
                "id": next_detection_id,
                "class_name": str(row["class_name"]),
                "bbox": _bbox_to_mapping(box),
                "score": float(row["confidence"]),
                "meta": {
                    "bootstrap_label_origin": "bootstrap",
                    "teacher_name": str(row["teacher_name"]),
                    "model_version": str(row["model_version"]),
                },
                "provenance": provenance,
            }
        )
        class_counts[str(row["class_name"])] += 1
        next_detection_id += 1

    final_scene["detections"] = final_detections
    final_scene.setdefault("notes", [])
    final_scene["notes"].append("OD bootstrap exhaustive detector labels were materialized from raw source labels plus teacher predictions.")
    final_scene.setdefault("tasks", {})
    final_scene["tasks"]["has_det"] = int(bool(final_detections))

    scene_output_path = dataset_root / "labels_scene" / split / f"{entry.sample_uid}.json"
    det_output_path = dataset_root / "labels_det" / split / f"{entry.sample_uid}.txt"
    image_output_path = dataset_root / "images" / split / materialized_image_name
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    det_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_scene["image"]["original_file_name"] = original_image_name
    final_scene["image"]["file_name"] = image_output_path.name
    write_json(scene_output_path, final_scene)
    det_lines = [
        _bbox_to_yolo_line(
            str(detection["class_name"]),
            _bbox_from_scene_detection(detection),
            int(final_scene["image"]["width"]),
            int(final_scene["image"]["height"]),
        )
        for detection in final_detections
    ]
    write_text(det_output_path, ("\n".join(det_lines) + "\n") if det_lines else "")
    _materialize_image(entry.image_path, image_output_path, copy_images=copy_images)

    materialized = MaterializedSample(
        sample_id=entry.sample_uid,
        dataset_key=exhaustive_dataset_key,
        scene_path=scene_output_path,
        det_path=det_output_path,
        image_path=image_output_path,
        raw_detection_count=len(raw_detections),
        bootstrap_detection_count=len(kept_predictions),
    )
    sample_row: ExhaustiveSampleRow = {
        "sample_id": entry.sample_id,
        "sample_uid": entry.sample_uid,
        "source_dataset_key": source_dataset_key,
        "exhaustive_dataset_key": exhaustive_dataset_key,
        "split": split,
        "source_scene_path": str(entry.scene_path),
        "source_image_path": str(entry.image_path),
        "source_det_path": str(entry.det_path) if entry.det_path is not None else None,
        "scene_path": str(scene_output_path),
        "det_path": str(det_output_path),
        "image_path": str(image_output_path),
        "raw_detection_count": len(raw_detections),
        "bootstrap_detection_count": len(kept_predictions),
    }
    return materialized, dict(class_counts), sample_row


def materialize_exhaustive_od_dataset(
    *,
    image_entries: Iterable[ImageListEntry],
    predictions_by_sample_uid: dict[str, list[TeacherPredictionRow]],
    class_policy: dict[str, Any],
    output_root: Path,
    run_id: str,
    created_at: str,
    copy_images: bool,
    log_fn: Any | None = None,
) -> ExhaustiveMaterializationSummary:
    dataset_root = output_root.resolve() / run_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    sample_rows: list[ExhaustiveSampleRow] = []
    materialized: list[MaterializedSample] = []
    entries = tuple(image_entries)
    _validate_materialization_entries(entries)
    total_entries = len(entries)
    workers = _default_io_workers()
    completed = 0
    bootstrap_boxes = 0
    start_time = time.monotonic()
    log_every = max(250, workers * 50)
    if log_fn is not None:
        log_fn(f"materialize start samples={total_entries} workers={workers} copy_images={copy_images}")
    if total_entries:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="od_materialize") as executor:
            futures = [
                executor.submit(
                    _materialize_sample,
                    entry=entry,
                    predictions_by_sample_uid=predictions_by_sample_uid,
                    class_policy=class_policy,
                    dataset_root=dataset_root,
                    run_id=run_id,
                    created_at=created_at,
                    copy_images=copy_images,
                )
                for entry in entries
            ]
            for future in as_completed(futures):
                materialized_sample, sample_class_counts, sample_row = future.result()
                materialized.append(materialized_sample)
                sample_rows.append(sample_row)
                class_counts.update(sample_class_counts)
                bootstrap_boxes += int(sample_row["bootstrap_detection_count"])
                completed += 1
                if (
                    log_fn is not None
                    and (completed == total_entries or completed == 1 or completed % log_every == 0)
                ):
                    elapsed = max(time.monotonic() - start_time, 1e-6)
                    rate = completed / elapsed
                    log_fn(
                        f"materialize progress {completed}/{total_entries} "
                        f"samples ({rate:.1f} samples/s, bootstrap_boxes={bootstrap_boxes})"
                    )

    materialized.sort(key=lambda item: item.sample_id)
    sample_rows.sort(key=lambda item: str(item["sample_uid"]))

    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    write_text(
        meta_root / "class_map_det.yaml",
        yaml.safe_dump({str(index): class_name for index, class_name in enumerate(OD_CLASSES)}, sort_keys=False),
    )
    manifest_payload: ExhaustiveMaterializationManifest = {
        "version": "od-bootstrap-exhaustive-od-v1",
        "run_id": run_id,
        "generated_at": created_at,
        "dataset_root": str(dataset_root),
        "sample_count": len(materialized),
        "class_counts": dict(sorted(class_counts.items())),
        "samples": sample_rows,
    }
    manifest_path = meta_root / EXHAUSTIVE_MATERIALIZATION_MANIFEST_NAME
    write_json(manifest_path, manifest_payload)
    summary_payload = _build_materialization_summary(
        dataset_root=dataset_root,
        run_id=run_id,
        created_at=created_at,
        sample_count=len(materialized),
        class_counts=dict(class_counts),
    )
    summary_path = meta_root / EXHAUSTIVE_MATERIALIZATION_SUMMARY_NAME
    write_json(summary_path, summary_payload)
    return summary_payload
