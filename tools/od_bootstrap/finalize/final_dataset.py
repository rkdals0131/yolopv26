from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from model.preprocess.aihub_standardize import OD_CLASSES


def _default_io_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _resolve_latest_root(root: Path) -> Path:
    if root.name != "latest":
        return root.resolve()
    parent = root.parent.resolve()
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no exhaustive OD runs found under {parent}")
    return sorted(candidates, key=lambda item: item.name)[-1]


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"target path already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _copy_optional(source_path: Path, target_path: Path) -> bool:
    if source_path.is_file():
        _copy_json(source_path, target_path)
        return True
    if target_path.exists():
        raise FileExistsError(f"target path already exists without source file: {target_path}")
    return False


def _reserve_final_sample_id(
    final_sample_id: str,
    *,
    seen_ids: dict[str, dict[str, str]],
    scene_path: Path,
    source_kind: str,
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
    source_kind: str,
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
    source_kind: str,
    dataset_key: str,
    split: str,
) -> dict[str, Any]:
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


def build_pv26_exhaustive_od_lane_dataset(
    *,
    exhaustive_od_root: Path,
    aihub_canonical_root: Path,
    output_root: Path,
    copy_images: bool = False,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    resolved_exhaustive_root = _resolve_latest_root(exhaustive_od_root)
    resolved_aihub_root = aihub_canonical_root.resolve()
    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    dataset_counts = Counter()
    sample_rows: list[dict[str, Any]] = []
    seen_final_sample_ids: dict[str, dict[str, str]] = {}
    exhaustive_tasks: list[dict[str, Any]] = []
    lane_tasks: list[dict[str, Any]] = []

    if log_fn is not None:
        log_fn(f"finalize output_root={resolved_output_root}")
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
        scene_output_path = resolved_output_root / "labels_scene" / split / f"{final_sample_id}.json"
        det_output_path = resolved_output_root / "labels_det" / split / f"{final_sample_id}.txt"
        image_output_path = resolved_output_root / "images" / split / final_image_name
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
        scene_output_path = resolved_output_root / "labels_scene" / split / f"{final_sample_id}.json"
        det_output_path = resolved_output_root / "labels_det" / split / f"{final_sample_id}.txt"
        image_output_path = resolved_output_root / "images" / split / final_image_name
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

    def _run_tasks(tasks: list[dict[str, Any]], *, stage_name: str) -> None:
        if not tasks:
            return
        start_time = time.monotonic()
        completed = 0
        log_every = max(250, workers * 50)
        stage_rows: list[dict[str, Any]] = []
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

    meta_root = resolved_output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "class_map_det.yaml").write_text(
        yaml.safe_dump({str(index): class_name for index, class_name in enumerate(OD_CLASSES)}, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "version": "pv26-exhaustive-od-lane-v2",
        "exhaustive_od_root": str(resolved_exhaustive_root),
        "aihub_canonical_root": str(resolved_aihub_root),
        "output_root": str(resolved_output_root),
        "sample_count": copied_samples,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "samples": sample_rows,
    }
    summary_path = meta_root / "final_dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "output_root": str(resolved_output_root),
        "manifest_path": str(summary_path),
        "sample_count": copied_samples,
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }
