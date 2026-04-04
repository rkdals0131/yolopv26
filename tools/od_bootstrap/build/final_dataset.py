from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Literal, TypedDict

import yaml

from common.io import write_json as _write_common_json
from common.paths import resolve_latest_root
from common.pv26_schema import OD_CLASSES
from .final_dataset_stats import (
    FINAL_DATASET_STATS_MARKDOWN_NAME,
    FINAL_DATASET_STATS_NAME,
    analyze_final_dataset,
)

FINAL_DATASET_MANIFEST_NAME = "final_dataset_manifest.json"
FINAL_DATASET_SUMMARY_NAME = "final_dataset_summary.json"
FINAL_DATASET_PUBLISH_MARKER = "final_dataset_publish_state.json"
FINAL_DATASET_RERUN_MODE = "atomic_overwrite"
FinalDatasetPublishStatus = Literal["building", "ready", "completed"]
FinalDatasetSourceKind = Literal["exhaustive_od", "lane"]


class FinalDatasetPublishMarker(TypedDict, total=False):
    artifact: str
    status: FinalDatasetPublishStatus
    rerun_mode: str
    final_output_root: str
    staging_root: str
    sample_count: int
    dataset_counts: dict[str, int]


class FinalSampleOwner(TypedDict):
    scene_path: str
    source_kind: FinalDatasetSourceKind


class FinalizeSampleTask(TypedDict):
    scene_output_path: Path
    final_scene: dict[str, Any]
    det_source_path: Path | None
    det_output_path: Path
    source_image_path: Path
    image_output_path: Path
    copy_images: bool
    final_sample_id: str
    source_kind: FinalDatasetSourceKind
    dataset_key: str
    split: str


class FinalDatasetSampleRow(TypedDict):
    final_sample_id: str
    source_kind: FinalDatasetSourceKind
    source_dataset_key: str
    split: str
    scene_path: str
    det_path: str | None
    image_path: str


class FinalDatasetManifest(TypedDict):
    version: str
    exhaustive_od_root: str
    aihub_canonical_root: str
    output_root: str
    sample_count: int
    dataset_counts: dict[str, int]
    rerun_mode: FinalDatasetRerunMode
    samples: list[FinalDatasetSampleRow]


class FinalDatasetBuildSummary(TypedDict):
    output_root: str
    exhaustive_od_root: str
    aihub_canonical_root: str
    manifest_path: str
    summary_path: str
    publish_marker_path: str
    stats_path: str
    stats_markdown_path: str
    rerun_mode: FinalDatasetRerunMode
    sample_count: int
    dataset_counts: dict[str, int]
    warnings: list[str]


def _default_io_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> None:
    """Keep final-dataset image publication local: hardlink/copy only and never overwrite."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"target path already exists: {target_path}")
    if copy_images:
        shutil.copy2(source_path, target_path)
        return
    try:
        target_path.hardlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _copy_json(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"target path already exists: {target_path}")
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"target path already exists: {path}")
    _write_common_json(path, payload)


def _write_json_replace(path: Path, payload: Any) -> None:
    _write_common_json(path, payload)


def _copy_optional(source_path: Path, target_path: Path) -> bool:
    if source_path.is_file():
        _copy_json(source_path, target_path)
        return True
    if target_path.exists():
        raise FileExistsError(f"target path already exists without source file: {target_path}")
    return False


def _staging_root_for(output_root: Path) -> Path:
    parent = output_root.parent
    stamp = int(time.time() * 1_000_000)
    return parent / f".{output_root.name}.staging.{os.getpid()}.{stamp}"


def _resolve_exhaustive_dataset_root(exhaustive_od_root: Path) -> Path:
    resolved_root = resolve_latest_root(exhaustive_od_root)
    if (resolved_root / "labels_scene").is_dir():
        return resolved_root
    candidates = sorted(
        (
            child for child in resolved_root.iterdir()
            if child.is_dir() and (child / "labels_scene").is_dir()
        ),
        key=lambda item: item.name,
    )
    if candidates:
        return candidates[-1].resolve()
    return resolved_root


def _write_publish_marker(
    root: Path,
    *,
    status: FinalDatasetPublishStatus,
    final_output_root: Path,
    sample_count: int | None = None,
    dataset_counts: dict[str, int] | None = None,
) -> Path:
    marker_path = root / "meta" / FINAL_DATASET_PUBLISH_MARKER
    payload: FinalDatasetPublishMarker = {
        "artifact": "pv26_final_dataset",
        "status": status,
        "rerun_mode": FINAL_DATASET_RERUN_MODE,
        "final_output_root": str(final_output_root),
        "staging_root": str(root),
    }
    if sample_count is not None:
        payload["sample_count"] = int(sample_count)
    if dataset_counts is not None:
        payload["dataset_counts"] = dict(sorted(dataset_counts.items()))
    _write_json_replace(marker_path, payload)
    return marker_path


def _build_final_dataset_summary(
    *,
    output_root: Path,
    exhaustive_od_root: Path,
    aihub_canonical_root: Path,
    publish_marker_path: Path,
    stats_path: Path,
    stats_markdown_path: Path,
    sample_count: int,
    dataset_counts: dict[str, int],
    warnings: list[str],
) -> FinalDatasetBuildSummary:
    manifest_path = output_root / "meta" / FINAL_DATASET_MANIFEST_NAME
    summary_path = output_root / "meta" / FINAL_DATASET_SUMMARY_NAME
    return {
        "output_root": str(output_root),
        "exhaustive_od_root": str(exhaustive_od_root),
        "aihub_canonical_root": str(aihub_canonical_root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "publish_marker_path": str(publish_marker_path),
        "stats_path": str(stats_path),
        "stats_markdown_path": str(stats_markdown_path),
        "rerun_mode": FINAL_DATASET_RERUN_MODE,
        "sample_count": sample_count,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "warnings": list(warnings),
    }


def _publish_staging_root(*, staging_root: Path, output_root: Path) -> None:
    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    backup_root: Path | None = None
    if output_root.exists():
        backup_root = output_parent / f".{output_root.name}.backup.{os.getpid()}.{int(time.time() * 1_000_000)}"
        output_root.rename(backup_root)
    try:
        staging_root.rename(output_root)
    except Exception:
        if backup_root is not None and backup_root.exists() and not output_root.exists():
            backup_root.rename(output_root)
        raise
    if backup_root is not None and backup_root.exists():
        shutil.rmtree(backup_root)


def _reserve_final_sample_id(
    final_sample_id: str,
    *,
    seen_ids: dict[str, FinalSampleOwner],
    scene_path: Path,
    source_kind: FinalDatasetSourceKind,
) -> None:
    existing = seen_ids.get(final_sample_id)
    if existing is not None:
        raise ValueError(
            "duplicate final_sample_id detected: "
            f"{final_sample_id} ({existing['source_kind']}:{existing['scene_path']} vs {source_kind}:{scene_path})"
        )
    seen_ids[final_sample_id] = {
        "scene_path": str(scene_path),
        "source_kind": source_kind,
    }


def _load_scene(scene_path: Path) -> dict[str, Any]:
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"scene root must be an object: {scene_path}")
    return payload


def _coerce_split(scene: dict[str, Any], *, scene_path: Path) -> str:
    split = str(scene.get("source", {}).get("split") or scene_path.parent.name).strip()
    if not split:
        raise ValueError(f"scene split must not be empty: {scene_path}")
    return split


def _coerce_dataset_key(scene: dict[str, Any], *, scene_path: Path) -> str:
    dataset_key = str(scene.get("source", {}).get("dataset") or "").strip()
    if not dataset_key:
        raise ValueError(f"scene source.dataset must not be empty: {scene_path}")
    return dataset_key


def _coerce_image_name(scene: dict[str, Any], *, scene_path: Path) -> str:
    image_name = str(scene.get("image", {}).get("file_name") or "").strip()
    if not image_name:
        raise ValueError(f"scene image.file_name must not be empty: {scene_path}")
    return image_name


def _build_final_scene(
    *,
    scene: dict[str, Any],
    final_sample_id: str,
    source_kind: FinalDatasetSourceKind,
    original_image_name: str,
    final_image_name: str,
) -> dict[str, Any]:
    final_scene = deepcopy(scene)
    final_scene.setdefault("source", {})
    final_scene.setdefault("image", {})
    original_file_name = str(final_scene["image"].get("original_file_name") or original_image_name)
    final_scene["source"]["final_sample_id"] = final_sample_id
    final_scene["source"]["source_kind"] = source_kind
    final_scene["image"]["original_file_name"] = original_file_name
    final_scene["image"]["file_name"] = final_image_name
    return final_scene


def _finalize_sample_task(
    *,
    scene_output_path: Path,
    final_scene: dict[str, Any],
    det_source_path: Path | None,
    det_output_path: Path,
    source_image_path: Path,
    image_output_path: Path,
    copy_images: bool,
    final_sample_id: str,
    source_kind: FinalDatasetSourceKind,
    dataset_key: str,
    split: str,
) -> FinalDatasetSampleRow:
    _write_json(scene_output_path, final_scene)
    has_det = False
    if det_source_path is not None:
        has_det = _copy_optional(det_source_path, det_output_path)
    _link_or_copy(source_image_path, image_output_path, copy_images=copy_images)
    return {
        "final_sample_id": final_sample_id,
        "source_kind": source_kind,
        "source_dataset_key": dataset_key,
        "split": split,
        "scene_path": str(scene_output_path),
        "det_path": str(det_output_path) if has_det else None,
        "image_path": str(image_output_path),
    }


def _manifest_row_for_output_root(*, row: FinalDatasetSampleRow, output_root: Path) -> FinalDatasetSampleRow:
    split = str(row["split"])
    final_sample_id = str(row["final_sample_id"])
    image_suffix = Path(str(row["image_path"])).suffix.lower()
    manifest_row: FinalDatasetSampleRow = {
        "final_sample_id": final_sample_id,
        "source_kind": row["source_kind"],
        "source_dataset_key": str(row["source_dataset_key"]),
        "split": split,
        "scene_path": str((output_root / "labels_scene" / split / f"{final_sample_id}.json").resolve()),
        "det_path": None,
        "image_path": str((output_root / "images" / split / f"{final_sample_id}{image_suffix}").resolve()),
    }
    if row["det_path"] is not None:
        manifest_row["det_path"] = str((output_root / "labels_det" / split / f"{final_sample_id}.txt").resolve())
    return manifest_row


def build_pv26_exhaustive_od_lane_dataset(
    *,
    exhaustive_od_root: Path,
    aihub_canonical_root: Path,
    output_root: Path,
    copy_images: bool = False,
    log_fn: Any | None = None,
) -> FinalDatasetBuildSummary:
    resolved_exhaustive_root = _resolve_exhaustive_dataset_root(exhaustive_od_root)
    resolved_aihub_root = aihub_canonical_root.resolve()
    resolved_output_root = output_root.resolve()
    staging_root = _staging_root_for(resolved_output_root)
    _write_publish_marker(
        staging_root,
        status="building",
        final_output_root=resolved_output_root,
    )

    dataset_counts = Counter()
    sample_rows: list[FinalDatasetSampleRow] = []
    seen_final_sample_ids: dict[str, FinalSampleOwner] = {}
    exhaustive_tasks: list[FinalizeSampleTask] = []
    lane_tasks: list[FinalizeSampleTask] = []

    if log_fn is not None:
        log_fn(f"finalize output_root={resolved_output_root}")
        log_fn(f"finalize staging_root={staging_root}")
        log_fn(f"finalize exhaustive_od_root={resolved_exhaustive_root}")
        log_fn(f"finalize aihub_canonical_root={resolved_aihub_root}")

    for scene_path in sorted((resolved_exhaustive_root / "labels_scene").rglob("*.json")):
        scene = _load_scene(scene_path)
        split = _coerce_split(scene, scene_path=scene_path)
        dataset_key = _coerce_dataset_key(scene, scene_path=scene_path)
        image_name = _coerce_image_name(scene, scene_path=scene_path)
        final_sample_id = scene_path.stem
        bootstrap_sample_uid = str(scene.get("source", {}).get("bootstrap_sample_uid") or "").strip()
        if bootstrap_sample_uid and bootstrap_sample_uid != final_sample_id:
            raise ValueError(
                f"bootstrap_sample_uid mismatch for {scene_path}: "
                f"{bootstrap_sample_uid} != {final_sample_id}"
            )
        _reserve_final_sample_id(
            final_sample_id,
            seen_ids=seen_final_sample_ids,
            scene_path=scene_path,
            source_kind="exhaustive_od",
        )
        source_image_path = resolved_exhaustive_root / "images" / split / image_name
        final_image_name = f"{final_sample_id}{source_image_path.suffix.lower()}"
        scene_output_path = staging_root / "labels_scene" / split / f"{final_sample_id}.json"
        det_output_path = staging_root / "labels_det" / split / f"{final_sample_id}.txt"
        image_output_path = staging_root / "images" / split / final_image_name
        final_scene = _build_final_scene(
            scene=scene,
            final_sample_id=final_sample_id,
            source_kind="exhaustive_od",
            original_image_name=image_name,
            final_image_name=final_image_name,
        )
        exhaustive_tasks.append(
            {
                "scene_output_path": scene_output_path,
                "final_scene": final_scene,
                "det_source_path": resolved_exhaustive_root / "labels_det" / split / f"{final_sample_id}.txt",
                "det_output_path": det_output_path,
                "source_image_path": source_image_path,
                "image_output_path": image_output_path,
                "copy_images": copy_images,
                "final_sample_id": final_sample_id,
                "source_kind": "exhaustive_od",
                "dataset_key": dataset_key,
                "split": split,
            }
        )

    for scene_path in sorted((resolved_aihub_root / "labels_scene").rglob("*.json")):
        scene = _load_scene(scene_path)
        split = _coerce_split(scene, scene_path=scene_path)
        final_sample_id = scene_path.stem
        dataset_key = _coerce_dataset_key(scene, scene_path=scene_path)
        if dataset_key != "aihub_lane_seoul":
            continue
        image_name = _coerce_image_name(scene, scene_path=scene_path)
        _reserve_final_sample_id(
            final_sample_id,
            seen_ids=seen_final_sample_ids,
            scene_path=scene_path,
            source_kind="lane",
        )
        source_image_path = resolved_aihub_root / "images" / split / image_name
        final_image_name = f"{final_sample_id}{source_image_path.suffix.lower()}"
        scene_output_path = staging_root / "labels_scene" / split / f"{final_sample_id}.json"
        det_output_path = staging_root / "labels_det" / split / f"{final_sample_id}.txt"
        image_output_path = staging_root / "images" / split / final_image_name
        final_scene = _build_final_scene(
            scene=scene,
            final_sample_id=final_sample_id,
            source_kind="lane",
            original_image_name=image_name,
            final_image_name=final_image_name,
        )
        lane_tasks.append(
            {
                "scene_output_path": scene_output_path,
                "final_scene": final_scene,
                "det_source_path": resolved_aihub_root / "labels_det" / split / f"{final_sample_id}.txt",
                "det_output_path": det_output_path,
                "source_image_path": source_image_path,
                "image_output_path": image_output_path,
                "copy_images": copy_images,
                "final_sample_id": final_sample_id,
                "source_kind": "lane",
                "dataset_key": dataset_key,
                "split": split,
            }
        )

    workers = _default_io_workers()

    def _run_tasks(tasks: list[FinalizeSampleTask], *, stage_name: str) -> None:
        if not tasks:
            return
        start_time = time.monotonic()
        completed = 0
        log_every = max(250, workers * 50)
        stage_rows: list[FinalDatasetSampleRow] = []
        if log_fn is not None:
            log_fn(f"finalize {stage_name} start samples={len(tasks)} workers={workers}")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"finalize_{stage_name}") as executor:
            futures = [executor.submit(_finalize_sample_task, **task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                stage_rows.append(row)
                dataset_counts[str(row["source_dataset_key"])] += 1
                completed += 1
                if (
                    log_fn is not None
                    and (completed == len(tasks) or completed == 1 or completed % log_every == 0)
                ):
                    elapsed = max(time.monotonic() - start_time, 1e-6)
                    rate = completed / elapsed
                    log_fn(
                        f"finalize {stage_name} progress {completed}/{len(tasks)} "
                        f"samples ({rate:.1f} samples/s)"
                    )
        stage_rows.sort(key=lambda row: str(row["final_sample_id"]))
        sample_rows.extend(stage_rows)

    _run_tasks(exhaustive_tasks, stage_name="exhaustive_od")
    _run_tasks(lane_tasks, stage_name="lane")
    copied_samples = len(sample_rows)
    manifest_rows = [_manifest_row_for_output_root(row=row, output_root=resolved_output_root) for row in sample_rows]

    meta_root = staging_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "class_map_det.yaml").write_text(
        yaml.safe_dump({str(index): class_name for index, class_name in enumerate(OD_CLASSES)}, sort_keys=False),
        encoding="utf-8",
    )
    manifest_payload: FinalDatasetManifest = {
        "version": "pv26-exhaustive-od-lane-v2",
        "exhaustive_od_root": str(resolved_exhaustive_root),
        "aihub_canonical_root": str(resolved_aihub_root),
        "output_root": str(resolved_output_root),
        "sample_count": copied_samples,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "rerun_mode": FINAL_DATASET_RERUN_MODE,
        "samples": manifest_rows,
    }
    manifest_path = meta_root / FINAL_DATASET_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_publish_marker(
        staging_root,
        status="ready",
        final_output_root=resolved_output_root,
        sample_count=copied_samples,
        dataset_counts=dict(dataset_counts),
    )
    _publish_staging_root(staging_root=staging_root, output_root=resolved_output_root)
    publish_marker_path = _write_publish_marker(
        resolved_output_root,
        status="completed",
        final_output_root=resolved_output_root,
        sample_count=copied_samples,
        dataset_counts=dict(dataset_counts),
    )
    stats_payload = analyze_final_dataset(dataset_root=resolved_output_root, write_artifacts=True)
    build_summary = _build_final_dataset_summary(
        output_root=resolved_output_root,
        exhaustive_od_root=resolved_exhaustive_root,
        aihub_canonical_root=resolved_aihub_root,
        publish_marker_path=publish_marker_path,
        stats_path=resolved_output_root / "meta" / FINAL_DATASET_STATS_NAME,
        stats_markdown_path=resolved_output_root / "meta" / FINAL_DATASET_STATS_MARKDOWN_NAME,
        sample_count=copied_samples,
        dataset_counts=dict(dataset_counts),
        warnings=list(stats_payload.get("warnings", [])),
    )
    _write_json_replace(resolved_output_root / "meta" / FINAL_DATASET_SUMMARY_NAME, build_summary)
    return build_summary
