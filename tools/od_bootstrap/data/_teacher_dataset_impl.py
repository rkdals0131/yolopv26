from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Callable

from common.pv26_schema import OD_CLASSES
from .image_list import build_sample_uid

from .debug_vis import DEFAULT_DEBUG_VIS_SEED, generate_teacher_dataset_debug_vis
from .source_prep import BOOTSTRAP_SOURCE_KEYS, CanonicalSourceBundle


TEACHER_DATASET_SPECS = {
    "mobility": {
        "source_dataset_keys": ("bdd100k_det_100k",),
        "class_names": ("vehicle", "bike", "pedestrian"),
    },
    "signal": {
        "source_dataset_keys": ("aihub_traffic_seoul",),
        "class_names": ("traffic_light", "sign"),
    },
    "obstacle": {
        "source_dataset_keys": ("aihub_obstacle_seoul",),
        "class_names": ("traffic_cone", "obstacle"),
    },
}


@dataclass(frozen=True)
class TeacherDatasetSpec:
    name: str
    source_dataset_keys: tuple[str, ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class TeacherDatasetBuildConfig:
    output_root: Path
    copy_images: bool = False
    workers: int = 1
    log_every: int = 250
    debug_vis_count: int = 0
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED


@dataclass(frozen=True)
class TeacherDatasetBuildResult:
    teacher_name: str
    output_root: Path
    dataset_root: Path
    manifest_path: Path
    debug_vis_manifest_path: Path
    sample_count: int
    detection_count: int
    class_counts: dict[str, int]
    source_dataset_keys: tuple[str, ...]


@dataclass(frozen=True)
class TeacherDatasetTask:
    source_dataset_key: str
    split: str
    sample_id: str
    scene_path: Path
    image_src: Path
    image_dst: Path
    label_src: Path
    label_dst: Path


@dataclass(frozen=True)
class TeacherDatasetTaskResult:
    manifest_row: dict[str, Any]
    detection_count: int
    class_counts: dict[str, int]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return "existing"
    if copy_images:
        import shutil

        shutil.copy2(source_path, target_path)
        return "copy"
    try:
        target_path.hardlink_to(source_path)
        return "hardlink"
    except Exception:
        import shutil

        shutil.copy2(source_path, target_path)
        return "copy"


def _parse_det_row(line: str) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    return class_id, [float(value) for value in parts[1:]]


def _global_class_name(class_id: int) -> str:
    try:
        return OD_CLASSES[class_id]
    except IndexError as exc:
        raise ValueError(f"unsupported OD class id: {class_id}") from exc


def _resolve_spec(spec: TeacherDatasetSpec | str) -> TeacherDatasetSpec:
    if isinstance(spec, TeacherDatasetSpec):
        return spec
    payload = TEACHER_DATASET_SPECS[spec]
    return TeacherDatasetSpec(
        name=spec,
        source_dataset_keys=tuple(payload["source_dataset_keys"]),
        class_names=tuple(payload["class_names"]),
    )


def _iter_scene_paths(source_root: Path) -> list[Path]:
    labels_scene_root = source_root / "labels_scene"
    if not labels_scene_root.is_dir():
        return []
    return sorted(labels_scene_root.rglob("*.json"), key=lambda item: (item.parent.name, item.stem))


def _validate_positive_int(value: int, *, field_name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return resolved


def _discover_teacher_tasks(
    source_bundle: CanonicalSourceBundle,
    resolved_spec: TeacherDatasetSpec,
    dataset_root: Path,
    *,
    log_fn: Callable[[str], None] | None,
) -> list[TeacherDatasetTask]:
    seen_samples: set[tuple[str, str]] = set()
    seen_image_targets: dict[Path, tuple[str, str]] = {}
    seen_label_targets: dict[Path, tuple[str, str]] = {}
    source_roots = (source_bundle.bdd_root, source_bundle.aihub_root)
    tasks: list[TeacherDatasetTask] = []

    for source_root in source_roots:
        scene_paths = _iter_scene_paths(source_root)
        if log_fn is not None:
            log_fn(
                f"[teacher:{resolved_spec.name}] discovered {len(scene_paths)} scene labels under {source_root}"
            )
        matched_count = 0
        for scene_path in scene_paths:
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            source_dataset_key = str(scene.get("source", {}).get("dataset") or "")
            if source_dataset_key not in resolved_spec.source_dataset_keys:
                continue
            split = str(scene.get("source", {}).get("split") or scene_path.parent.name)
            sample_id = scene_path.stem
            sample_key = (source_dataset_key, sample_id)
            if sample_key in seen_samples:
                continue
            seen_samples.add(sample_key)

            image_file_name = str(scene.get("image", {}).get("file_name") or "")
            if not image_file_name:
                raise ValueError(f"teacher dataset image.file_name missing: {scene_path}")
            image_src = source_root / "images" / split / image_file_name
            label_src = source_root / "labels_det" / split / f"{sample_id}.txt"
            image_dst = dataset_root / "images" / split / image_file_name
            label_dst = dataset_root / "labels" / split / f"{sample_id}.txt"

            prior_image_owner = seen_image_targets.get(image_dst)
            if prior_image_owner is not None and prior_image_owner != sample_key:
                raise ValueError(
                    f"teacher dataset image target collision: {image_dst} owned by {prior_image_owner} and {sample_key}"
                )
            seen_image_targets[image_dst] = sample_key

            prior_label_owner = seen_label_targets.get(label_dst)
            if prior_label_owner is not None and prior_label_owner != sample_key:
                raise ValueError(
                    f"teacher dataset label target collision: {label_dst} owned by {prior_label_owner} and {sample_key}"
                )
            seen_label_targets[label_dst] = sample_key

            tasks.append(
                TeacherDatasetTask(
                    source_dataset_key=source_dataset_key,
                    split=split,
                    sample_id=sample_id,
                    scene_path=scene_path,
                    image_src=image_src,
                    image_dst=image_dst,
                    label_src=label_src,
                    label_dst=label_dst,
                )
            )
            matched_count += 1
        if log_fn is not None and matched_count:
            log_fn(f"[teacher:{resolved_spec.name}] queued {matched_count} samples from {source_root.name}")
    return sorted(tasks, key=lambda item: (item.split, item.sample_id, item.source_dataset_key))


def _process_teacher_task(
    task: TeacherDatasetTask,
    *,
    class_to_local_id: dict[str, int],
    copy_images: bool,
) -> TeacherDatasetTaskResult:
    if not task.image_src.is_file():
        raise FileNotFoundError(f"teacher dataset image missing: {task.image_src}")

    image_action = _link_or_copy(task.image_src, task.image_dst, copy_images=copy_images)

    filtered_rows: list[str] = []
    class_counts: dict[str, int] = {}
    if task.label_src.is_file():
        for line in task.label_src.read_text(encoding="utf-8").splitlines():
            parsed = _parse_det_row(line)
            if parsed is None:
                continue
            class_id, values = parsed
            class_name = _global_class_name(class_id)
            if class_name not in class_to_local_id:
                continue
            filtered_rows.append(f"{class_to_local_id[class_name]} " + " ".join(f"{value:.6f}" for value in values))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    task.label_dst.parent.mkdir(parents=True, exist_ok=True)
    task.label_dst.write_text(("\n".join(filtered_rows) + "\n") if filtered_rows else "", encoding="utf-8")
    return TeacherDatasetTaskResult(
        manifest_row={
            "source_dataset_key": task.source_dataset_key,
            "split": task.split,
            "sample_id": task.sample_id,
            "sample_uid": build_sample_uid(
                dataset_key=task.source_dataset_key,
                split=task.split,
                sample_id=task.sample_id,
            ),
            "source_scene_path": str(task.scene_path),
            "source_image_path": str(task.image_src),
            "output_image_path": str(task.image_dst),
            "output_label_path": str(task.label_dst),
            "detection_count": len(filtered_rows),
            "image_action": image_action,
        },
        detection_count=len(filtered_rows),
        class_counts=class_counts,
    )


def build_teacher_dataset(
    source_bundle: CanonicalSourceBundle,
    spec: TeacherDatasetSpec | str,
    config: TeacherDatasetBuildConfig,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> TeacherDatasetBuildResult:
    resolved_spec = _resolve_spec(spec)
    output_root = config.output_root.resolve()
    dataset_root = output_root / resolved_spec.name
    dataset_root.mkdir(parents=True, exist_ok=True)
    workers = _validate_positive_int(config.workers, field_name="build.workers")
    log_every = _validate_positive_int(config.log_every, field_name="build.log_every")

    class_to_local_id = {class_name: index for index, class_name in enumerate(resolved_spec.class_names)}
    class_counts: dict[str, int] = {class_name: 0 for class_name in resolved_spec.class_names}
    tasks = _discover_teacher_tasks(source_bundle, resolved_spec, dataset_root, log_fn=log_fn)
    total_tasks = len(tasks)
    if log_fn is not None:
        log_fn(
            f"[teacher:{resolved_spec.name}] building {total_tasks} samples with workers={workers} copy_images={config.copy_images}"
        )

    detection_count = 0
    manifest_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    completed = 0

    if total_tasks:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{resolved_spec.name}_dataset") as executor:
            futures = [
                executor.submit(
                    _process_teacher_task,
                    task,
                    class_to_local_id=class_to_local_id,
                    copy_images=config.copy_images,
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                task_result = future.result()
                completed += 1
                detection_count += task_result.detection_count
                for class_name, count in task_result.class_counts.items():
                    class_counts[class_name] += count
                manifest_row = {"teacher_name": resolved_spec.name, **task_result.manifest_row}
                manifest_rows.append(manifest_row)

                if log_fn is not None and (completed == total_tasks or completed == 1 or completed % log_every == 0):
                    elapsed = max(time.monotonic() - start_time, 1e-6)
                    rate = completed / elapsed
                    log_fn(
                        f"[teacher:{resolved_spec.name}] progress {completed}/{total_tasks} samples "
                        f"({rate:.1f} samples/s, detections={detection_count})"
                    )

    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"]), str(row["source_dataset_key"])))
    sample_count = len(manifest_rows)
    elapsed = max(time.monotonic() - start_time, 1e-6)
    debug_vis_outputs = generate_teacher_dataset_debug_vis(
        dataset_root=dataset_root,
        teacher_name=resolved_spec.name,
        class_names=resolved_spec.class_names,
        manifest_rows=manifest_rows,
        debug_vis_count=int(config.debug_vis_count),
        debug_vis_seed=int(config.debug_vis_seed),
        log_fn=log_fn,
    )

    manifest_payload = {
        "version": "od-bootstrap-teacher-dataset-v1",
        "generated_at": _now_iso(),
        "teacher_name": resolved_spec.name,
        "source_dataset_keys": list(resolved_spec.source_dataset_keys),
        "class_names": list(resolved_spec.class_names),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "copy_images": bool(config.copy_images),
        "workers": workers,
        "log_every": log_every,
        "debug_vis_count": int(config.debug_vis_count),
        "debug_vis_seed": int(config.debug_vis_seed),
        "debug_vis_manifest_path": str(debug_vis_outputs["debug_vis_manifest"]),
        "sample_count": sample_count,
        "detection_count": detection_count,
        "elapsed_seconds": round(elapsed, 3),
        "class_counts": class_counts,
        "bootstrap_source_keys": list(BOOTSTRAP_SOURCE_KEYS),
        "samples": manifest_rows,
    }
    manifest_path = _write_json(dataset_root / "meta" / "teacher_dataset_manifest.json", manifest_payload)
    _write_json(
        dataset_root / "meta" / "teacher_dataset_summary.json",
        {key: value for key, value in manifest_payload.items() if key != "samples"},
    )

    if log_fn is not None:
        class_summary = ", ".join(f"{name}={class_counts[name]}" for name in resolved_spec.class_names)
        log_fn(
            f"[teacher:{resolved_spec.name}] done samples={sample_count} detections={detection_count} "
            f"elapsed={elapsed:.1f}s classes[{class_summary}]"
        )

    return TeacherDatasetBuildResult(
        teacher_name=resolved_spec.name,
        output_root=output_root,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        debug_vis_manifest_path=Path(str(debug_vis_outputs["debug_vis_manifest"])),
        sample_count=sample_count,
        detection_count=detection_count,
        class_counts=class_counts,
        source_dataset_keys=resolved_spec.source_dataset_keys,
    )


def build_teacher_datasets(
    source_bundle: CanonicalSourceBundle,
    output_root: Path,
    *,
    copy_images: bool = False,
    workers: int = 1,
    log_every: int = 250,
    debug_vis_count: int = 0,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
    teacher_specs: dict[str, TeacherDatasetSpec] | None = None,
) -> dict[str, TeacherDatasetBuildResult]:
    spec_map = teacher_specs or {
        name: TeacherDatasetSpec(
            name=name,
            source_dataset_keys=tuple(payload["source_dataset_keys"]),
            class_names=tuple(payload["class_names"]),
        )
        for name, payload in TEACHER_DATASET_SPECS.items()
    }
    config = TeacherDatasetBuildConfig(
        output_root=output_root,
        copy_images=copy_images,
        workers=workers,
        log_every=log_every,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
    )
    results: dict[str, TeacherDatasetBuildResult] = {}
    for name, spec in spec_map.items():
        results[name] = build_teacher_dataset(source_bundle, spec, config, log_fn=log_fn)
    return results
