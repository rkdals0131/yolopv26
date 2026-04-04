from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
import time
from typing import Any, Iterable, TypedDict, cast

import torch

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

from common.boxes import nms_rows
from common.io import now_iso as _now_iso
from common.io import timestamp_token as _timestamp_token
from common.train_runtime import format_duration, join_status_segments, timing_profile
from .artifacts import (
    RunManifest,
    TeacherJobManifest,
    TeacherJobManifestPayload,
    teacher_output_dir,
    write_image_list_snapshot,
    write_run_manifest,
    write_teacher_job_manifest,
    write_teacher_predictions,
)
from .exhaustive_od import (
    ExhaustiveMaterializationSummary,
    materialize_exhaustive_od_dataset,
)
from .image_list import ImageListEntry, load_image_list
from ..teacher.policy import row_passes_policy
from .sweep_types import BootstrapSweepScenario, ClassPolicy, TeacherConfig, TeacherPredictionRow


class ModelCentricSweepSummary(TypedDict):
    run_id: str
    run_dir: str
    image_count: int
    teacher_names: list[str]
    class_policy_path: str
    teacher_jobs: list[TeacherJobManifestPayload]
    materialization: ExhaustiveMaterializationSummary


def _log_bootstrap(message: str) -> None:
    print(f"[od_bootstrap.sweep] {message}", flush=True)


def _safe_name(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value.strip())
    return normalized.strip("_") or "od_bootstrap"


def _resolve_run_id(scenario_path: Path) -> str:
    return f"{_timestamp_token()}_{_safe_name(scenario_path.stem)}"


def _batched(items: Iterable[ImageListEntry], batch_size: int) -> Iterable[list[ImageListEntry]]:
    batch: list[ImageListEntry] = []
    for item in items:
        batch.append(item)
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _decode_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), copy=True)


def _submit_decode_batch(
    executor: ThreadPoolExecutor,
    batch_entries: list[ImageListEntry],
) -> list[Any]:
    return [executor.submit(_decode_image_array, entry.image_path) for entry in batch_entries]


def _build_teacher_job_manifest(
    *,
    teacher: TeacherConfig,
    run_id: str,
    created_at: str,
    image_count: int,
    predictions_path: Path,
) -> TeacherJobManifest:
    return TeacherJobManifest(
        run_id=run_id,
        created_at=created_at,
        teacher_name=teacher.name,
        base_model=teacher.base_model,
        model_version=teacher.model_version,
        checkpoint_path=str(teacher.checkpoint_path),
        classes=teacher.classes,
        image_count=image_count,
        predictions_path=str(predictions_path),
    )


def _extract_teacher_rows(
    *,
    teacher: TeacherConfig,
    batch_entries: list[ImageListEntry],
    results: list[Any],
    class_policy: dict[str, ClassPolicy],
) -> list[TeacherPredictionRow]:
    rows: list[TeacherPredictionRow] = []
    entry_by_path = {str(entry.image_path.resolve()): entry for entry in batch_entries}
    for result_index, result in enumerate(results):
        result_path = str(Path(str(getattr(result, "path", ""))).resolve())
        entry = entry_by_path.get(result_path)
        if entry is None and result_index < len(batch_entries):
            # Ultralytics may rewrite per-item list inputs to synthetic names such as image0.jpg.
            entry = batch_entries[result_index]
        if entry is None:
            continue
        orig_shape = getattr(result, "orig_shape", None)
        if not isinstance(orig_shape, (list, tuple)) or len(orig_shape) < 2:
            raise ValueError("teacher prediction result must expose orig_shape=(height, width)")
        image_height = int(orig_shape[0])
        image_width = int(orig_shape[1])
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_rows = torch.as_tensor(getattr(boxes, "xyxy", None)).tolist()
        cls_rows = torch.as_tensor(getattr(boxes, "cls", None)).tolist()
        conf_rows = torch.as_tensor(getattr(boxes, "conf", None)).tolist()
        filtered: list[TeacherPredictionRow] = []
        for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
            class_name = str(names.get(int(cls_index), str(int(cls_index))))
            if class_name not in teacher.classes:
                continue
            policy = class_policy[class_name]
            row: TeacherPredictionRow = {
                "sample_id": entry.sample_id,
                "sample_uid": entry.sample_uid,
                "image_path": str(entry.image_path),
                "scene_path": str(entry.scene_path),
                "dataset_key": entry.dataset_key,
                "split": entry.split,
                "teacher_name": teacher.name,
                "model_version": teacher.model_version,
                "class_name": class_name,
                "confidence": float(confidence),
                "xyxy": [float(value) for value in box],
                "box_index": box_index,
                "image_width": image_width,
                "image_height": image_height,
            }
            if not row_passes_policy(
                row=row,
                policy=policy,
                dataset_key=entry.dataset_key,
                image_width=image_width,
                image_height=image_height,
            ):
                continue
            filtered.append(row)

        by_class: dict[str, list[TeacherPredictionRow]] = {}
        for row in filtered:
            by_class.setdefault(str(row["class_name"]), []).append(row)
        for class_name, class_rows in by_class.items():
            rows.extend(
                cast(
                    list[TeacherPredictionRow],
                    nms_rows(class_rows, iou_threshold=float(class_policy[class_name].nms_iou_threshold)),
                )
            )
    return rows


def _run_teacher_inference(
    *,
    teacher: TeacherConfig,
    entries: tuple[ImageListEntry, ...],
    scenario: BootstrapSweepScenario,
) -> list[TeacherPredictionRow]:
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed")
    if not teacher.checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher.checkpoint_path}")
    model = YOLO(str(teacher.checkpoint_path))
    rows: list[TeacherPredictionRow] = []
    total_images = len(entries)
    processed_images = 0
    last_logged = 0
    start_time = time.perf_counter()
    batch_size = max(1, int(scenario.run.batch_size))
    decode_workers = max(1, min(int(scenario.run.decode_workers), batch_size))
    profile_window = deque(maxlen=max(1, int(scenario.run.profile_window)))
    log_every_images = max(batch_size * 10, 320)
    batches = tuple(_batched(entries, batch_size))
    _log_bootstrap(
        f"teacher={teacher.name} inference start images={total_images} batch={batch_size} "
        f"decode_workers={decode_workers} profile_window={profile_window.maxlen} checkpoint={teacher.checkpoint_path}"
    )
    if not batches:
        return rows
    with ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="od_decode") as decode_executor:
        pending_futures = _submit_decode_batch(decode_executor, batches[0])
        pending_entries = batches[0]
        for batch_index, _ in enumerate(batches):
            batch_started_at = time.perf_counter()
            decode_started_at = batch_started_at
            batch_images = [future.result() for future in pending_futures]
            decode_finished_at = time.perf_counter()

            next_batch_index = batch_index + 1
            if next_batch_index < len(batches):
                pending_entries = batches[next_batch_index]
                pending_futures = _submit_decode_batch(decode_executor, pending_entries)
            else:
                pending_entries = []
                pending_futures = []

            infer_started_at = time.perf_counter()
            results = model.predict(
                source=batch_images,
                imgsz=scenario.run.imgsz,
                device=scenario.run.device,
                conf=scenario.run.predict_conf,
                iou=scenario.run.predict_iou,
                verbose=False,
                save=False,
                stream=False,
            )
            infer_finished_at = time.perf_counter()
            postprocess_started_at = infer_finished_at
            current_batch_entries = batches[batch_index]
            rows.extend(
                _extract_teacher_rows(
                    teacher=teacher,
                    batch_entries=current_batch_entries,
                    results=list(results),
                    class_policy=scenario.class_policy,
                )
            )
            batch_finished_at = time.perf_counter()
            profile_window.append(
                {
                    "iteration_sec": max(0.0, batch_finished_at - batch_started_at),
                    "decode_sec": max(0.0, decode_finished_at - decode_started_at),
                    "infer_sec": max(0.0, infer_finished_at - infer_started_at),
                    "postprocess_sec": max(0.0, batch_finished_at - postprocess_started_at),
                }
            )
            processed_images += len(current_batch_entries)
            if (
                processed_images == total_images
                or processed_images == len(current_batch_entries)
                or processed_images - last_logged >= log_every_images
            ):
                last_logged = processed_images
                profile_summary = timing_profile(
                    list(profile_window),
                    keys=("iteration_sec", "decode_sec", "infer_sec", "postprocess_sec"),
                )
                iteration_profile = profile_summary["iteration_sec"]
                remaining_images = max(0, total_images - processed_images)
                eta_sec = float(iteration_profile["mean"]) * (remaining_images / float(batch_size)) if processed_images else None
                elapsed_sec = max(0.0, time.perf_counter() - start_time)
                profile_postfix = join_status_segments(
                    f"elapsed={format_duration(elapsed_sec)}",
                    f"eta={format_duration(eta_sec)}",
                    (
                        "iter_ms="
                        f"{iteration_profile['mean'] * 1000.0:.1f}/"
                        f"{iteration_profile['p50'] * 1000.0:.1f}/"
                        f"{iteration_profile['p99'] * 1000.0:.1f}"
                    ),
                    (
                        "timing_ms="
                        f"decode:{profile_summary['decode_sec']['mean'] * 1000.0:.1f},"
                        f"infer:{profile_summary['infer_sec']['mean'] * 1000.0:.1f},"
                        f"post:{profile_summary['postprocess_sec']['mean'] * 1000.0:.1f},"
                        f"gap:{max(0.0, (iteration_profile['mean'] - profile_summary['decode_sec']['mean'] - profile_summary['infer_sec']['mean'] - profile_summary['postprocess_sec']['mean']) * 1000.0):.1f}"
                    ),
                )
                _log_bootstrap(
                    f"teacher={teacher.name} inference progress {processed_images}/{total_images} images "
                    f"predictions={len(rows)} "
                    f"{profile_postfix}"
                )
    return rows


def run_model_centric_sweep_scenario(
    scenario: BootstrapSweepScenario,
    *,
    scenario_path: Path,
) -> ModelCentricSweepSummary:
    entries = load_image_list(scenario.image_list.manifest_path)
    created_at = _now_iso()
    run_id = _resolve_run_id(scenario_path)
    run_dir = scenario.run.output_root / run_id
    teacher_names = tuple(teacher.name for teacher in scenario.teachers)

    _log_bootstrap(f"loaded {len(entries)} images from {scenario.image_list.manifest_path}")
    _log_bootstrap(f"creating run directory: {run_dir}")

    run_manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        scenario_path=str(scenario_path),
        execution_mode=scenario.run.execution_mode,
        run_dir=str(run_dir),
        image_pool_manifest=str(scenario.image_list.manifest_path),
        image_count=len(entries),
        teacher_names=teacher_names,
    )
    write_run_manifest(run_dir, run_manifest)
    write_image_list_snapshot(run_dir, entries)

    predictions_by_sample_uid: dict[str, list[TeacherPredictionRow]] = {}
    teacher_jobs: list[TeacherJobManifestPayload] = []
    for teacher in scenario.teachers:
        predictions_path = teacher_output_dir(run_dir, teacher.name) / "predictions.jsonl"
        job_manifest = _build_teacher_job_manifest(
            teacher=teacher,
            run_id=run_id,
            created_at=created_at,
            image_count=len(entries),
            predictions_path=predictions_path,
        )
        write_teacher_job_manifest(run_dir, job_manifest)
        teacher_jobs.append(job_manifest.to_dict())
        teacher_rows = _run_teacher_inference(teacher=teacher, entries=entries, scenario=scenario)
        write_teacher_predictions(run_dir, teacher.name, teacher_rows)
        for row in teacher_rows:
            predictions_by_sample_uid.setdefault(str(row["sample_uid"]), []).append(row)
        _log_bootstrap(f"teacher={teacher.name} predictions={len(teacher_rows)}")

    materialization_summary = materialize_exhaustive_od_dataset(
        image_entries=entries,
        predictions_by_sample_uid=predictions_by_sample_uid,
        class_policy=scenario.class_policy,
        output_root=scenario.materialization.output_root,
        run_id=run_id,
        created_at=created_at,
        copy_images=scenario.materialization.copy_images,
        log_fn=_log_bootstrap,
    )

    summary: ModelCentricSweepSummary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "image_count": len(entries),
        "teacher_names": list(teacher_names),
        "class_policy_path": str(scenario.class_policy_path),
        "teacher_jobs": teacher_jobs,
        "materialization": materialization_summary,
    }
    _log_bootstrap(f"completed model-centric sweep for {run_id}")
    return summary


__all__ = [
    "BootstrapSweepScenario",
    "ClassPolicy",
    "TeacherConfig",
    "YOLO",
    "_extract_teacher_rows",
    "run_model_centric_sweep_scenario",
]
