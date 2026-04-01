from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .image_list import ImageListEntry


RUN_MANIFEST_VERSION = "od-bootstrap-run-v2"
JOB_MANIFEST_VERSION = "od-bootstrap-job-v2"
LABEL_ORIGINS = {"raw_source", "bootstrap"}


@dataclass(frozen=True)
class BoxProvenance:
    label_origin: str
    teacher_name: str
    confidence: float
    model_version: str
    run_id: str
    created_at: str

    def __post_init__(self) -> None:
        if self.label_origin not in LABEL_ORIGINS:
            raise ValueError(f"unsupported label_origin: {self.label_origin}")
        if not self.teacher_name:
            raise ValueError("teacher_name must not be empty")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.created_at:
            raise ValueError("created_at must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    scenario_path: str
    execution_mode: str
    run_dir: str
    image_pool_manifest: str
    image_count: int
    teacher_names: tuple[str, ...]
    manifest_version: str = RUN_MANIFEST_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherJobManifest:
    run_id: str
    created_at: str
    teacher_name: str
    base_model: str
    model_version: str
    checkpoint_path: str
    classes: tuple[str, ...]
    image_count: int
    predictions_path: str
    manifest_version: str = JOB_MANIFEST_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text((serialized + "\n") if serialized else "", encoding="utf-8")
    return path


def teacher_output_dir(run_dir: Path, teacher_name: str) -> Path:
    return run_dir / "teachers" / teacher_name


def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    return _write_json(run_dir / "manifest.json", manifest.to_dict())


def write_image_list_snapshot(run_dir: Path, entries: Iterable[ImageListEntry]) -> Path:
    return _write_jsonl(run_dir / "image_list.jsonl", (entry.to_dict() for entry in entries))


def write_teacher_job_manifest(run_dir: Path, manifest: TeacherJobManifest) -> Path:
    return _write_json(teacher_output_dir(run_dir, manifest.teacher_name) / "job_manifest.json", manifest.to_dict())


def write_teacher_predictions(run_dir: Path, teacher_name: str, rows: Iterable[dict[str, object]]) -> Path:
    return _write_jsonl(teacher_output_dir(run_dir, teacher_name) / "predictions.jsonl", rows)
