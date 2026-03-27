from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .image_list import ImageListEntry
from .schema import RunManifest, TeacherJobManifest


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
