from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.common import box_size, nms_rows
from tools.od_bootstrap.sweep.image_list import ImageListEntry, load_image_list
from tools.od_bootstrap.sweep.materialize import materialize_exhaustive_od_dataset
from tools.od_bootstrap.sweep.scenario import BootstrapSweepScenario, ClassPolicy, TeacherConfig, load_sweep_scenario
from tools.od_bootstrap.sweep.schema import RunManifest, TeacherJobManifest
from tools.od_bootstrap.sweep.writer import (
    teacher_output_dir,
    write_image_list_snapshot,
    write_run_manifest,
    write_teacher_job_manifest,
    write_teacher_predictions,
)


DEFAULT_SCENARIO_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "sweep" / "model_centric.default.yaml"


@dataclass(frozen=True)
class EntryConfig:
    scenario_path: Path = DEFAULT_SCENARIO_PATH


ENTRY_CONFIG = EntryConfig()


def _log_bootstrap(message: str) -> None:
    print(f"[od_bootstrap.sweep] {message}", flush=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _build_teacher_job_manifest(
    *,
    teacher: TeacherConfig,
    run_id: str,
    created_at: str,
    image_count: int,
    predictions_path: Path,
    dry_run: bool,
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
        dry_run=dry_run,
    )


def _extract_teacher_rows(
    *,
    teacher: TeacherConfig,
    batch_entries: list[ImageListEntry],
    results: list[Any],
    class_policy: dict[str, ClassPolicy],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entry_by_path = {str(entry.image_path.resolve()): entry for entry in batch_entries}
    for result in results:
        result_path = str(Path(str(getattr(result, "path", ""))).resolve())
        entry = entry_by_path.get(result_path)
        if entry is None:
            continue
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_rows = torch.as_tensor(getattr(boxes, "xyxy", None)).tolist()
        cls_rows = torch.as_tensor(getattr(boxes, "cls", None)).tolist()
        conf_rows = torch.as_tensor(getattr(boxes, "conf", None)).tolist()
        filtered: list[dict[str, Any]] = []
        for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
            class_name = str(names.get(int(cls_index), str(int(cls_index))))
            if class_name not in teacher.classes:
                continue
            policy = class_policy[class_name]
            width_px, height_px = box_size([float(value) for value in box])
            if float(confidence) < float(policy.score_threshold):
                continue
            if min(width_px, height_px) < int(policy.min_box_size):
                continue
            filtered.append(
                {
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
                }
            )

        by_class: dict[str, list[dict[str, Any]]] = {}
        for row in filtered:
            by_class.setdefault(str(row["class_name"]), []).append(row)
        for class_name, class_rows in by_class.items():
            rows.extend(nms_rows(class_rows, iou_threshold=float(class_policy[class_name].nms_iou_threshold)))
    return rows


def _run_teacher_inference(
    *,
    teacher: TeacherConfig,
    entries: tuple[ImageListEntry, ...],
    scenario: BootstrapSweepScenario,
) -> list[dict[str, Any]]:
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed")
    if not teacher.checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher.checkpoint_path}")
    model = YOLO(str(teacher.checkpoint_path))
    rows: list[dict[str, Any]] = []
    for batch_entries in _batched(entries, scenario.run.batch_size):
        results = model.predict(
            source=[str(entry.image_path) for entry in batch_entries],
            imgsz=scenario.run.imgsz,
            device=scenario.run.device,
            conf=scenario.run.predict_conf,
            iou=scenario.run.predict_iou,
            verbose=False,
            save=False,
            stream=False,
        )
        rows.extend(
            _extract_teacher_rows(
                teacher=teacher,
                batch_entries=batch_entries,
                results=list(results),
                class_policy=scenario.class_policy,
            )
        )
    return rows


def run_model_centric_sweep_scenario(scenario: BootstrapSweepScenario, *, scenario_path: Path) -> dict[str, Any]:
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
        dry_run=scenario.run.dry_run,
        run_dir=str(run_dir),
        image_pool_manifest=str(scenario.image_list.manifest_path),
        image_count=len(entries),
        teacher_names=teacher_names,
    )
    write_run_manifest(run_dir, run_manifest)
    write_image_list_snapshot(run_dir, entries)

    predictions_by_sample_uid: dict[str, list[dict[str, Any]]] = {}
    teacher_jobs: list[dict[str, Any]] = []
    for teacher in scenario.teachers:
        predictions_path = teacher_output_dir(run_dir, teacher.name) / "predictions.jsonl"
        job_manifest = _build_teacher_job_manifest(
            teacher=teacher,
            run_id=run_id,
            created_at=created_at,
            image_count=len(entries),
            predictions_path=predictions_path,
            dry_run=scenario.run.dry_run,
        )
        write_teacher_job_manifest(run_dir, job_manifest)
        teacher_jobs.append(job_manifest.to_dict())
        if scenario.run.dry_run:
            write_teacher_predictions(run_dir, teacher.name, [])
            _log_bootstrap(f"dry-run: skip teacher inference for {teacher.name}")
            continue

        teacher_rows = _run_teacher_inference(teacher=teacher, entries=entries, scenario=scenario)
        write_teacher_predictions(run_dir, teacher.name, teacher_rows)
        for row in teacher_rows:
            predictions_by_sample_uid.setdefault(str(row["sample_uid"]), []).append(row)
        _log_bootstrap(f"teacher={teacher.name} predictions={len(teacher_rows)}")

    materialization_summary = None
    if not scenario.run.dry_run:
        materialization_summary = materialize_exhaustive_od_dataset(
            image_entries=entries,
            predictions_by_sample_uid=predictions_by_sample_uid,
            class_policy=scenario.class_policy,
            output_root=scenario.materialization.output_root,
            run_id=run_id,
            created_at=created_at,
            copy_images=scenario.materialization.copy_images,
        )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "image_count": len(entries),
        "teacher_names": list(teacher_names),
        "dry_run": scenario.run.dry_run,
        "class_policy_path": str(scenario.class_policy_path),
        "teacher_jobs": teacher_jobs,
        "materialization": materialization_summary,
    }
    _log_bootstrap(f"completed model-centric sweep for {run_id}")
    return summary


def load_and_run_default_scenario() -> dict[str, Any]:
    scenario_path = Path(ENTRY_CONFIG.scenario_path).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"bootstrap sweep scenario not found: {scenario_path}")
    scenario = load_sweep_scenario(scenario_path)
    return run_model_centric_sweep_scenario(scenario, scenario_path=scenario_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the OD bootstrap model-centric teacher sweep.")
    parser.add_argument("--config", default=str(ENTRY_CONFIG.scenario_path), help="Path to a sweep scenario YAML.")
    args = parser.parse_args(argv)
    scenario_path = Path(args.config).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"bootstrap sweep scenario not found: {scenario_path}")
    scenario = load_sweep_scenario(scenario_path)
    run_model_centric_sweep_scenario(scenario, scenario_path=scenario_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
