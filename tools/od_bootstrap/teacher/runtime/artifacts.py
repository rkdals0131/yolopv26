from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from common.io import now_iso, remove_path, write_json
from .tensorboard import resolve_tensorboard_status, tensorboard_event_files

__all__ = [
    "build_teacher_runtime_artifact_paths",
    "build_teacher_train_summary",
    "finalize_teacher_train_artifacts",
    "publish_teacher_train_summary",
    "refresh_latest_teacher_artifacts",
]

def _link_or_copy_file(source: Path, destination: Path) -> str:
    """Refresh alias weights locally: hardlink first, then symlink, then copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        remove_path(destination)
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        try:
            os.symlink(source, destination)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def build_teacher_runtime_artifact_paths(run_dir: Path) -> dict[str, Path]:
    tensorboard_dir = run_dir / "tensorboard"
    return {
        "best_checkpoint": run_dir / "weights" / "best.pt",
        "last_checkpoint": run_dir / "weights" / "last.pt",
        "profile_log": run_dir / "profile_log.jsonl",
        "profile_summary": run_dir / "profile_summary.json",
        "tensorboard_dir": tensorboard_dir,
        "train_summary": run_dir / "train_summary.json",
    }


def build_teacher_train_summary(
    *,
    teacher_name: str,
    resolved_weights: str,
    dataset_yaml: Path,
    teacher_root: Path,
    run_dir: Path,
    runtime_params: dict[str, Any],
    train_kwargs: dict[str, Any],
    train_result: Any,
    trainer: Any,
) -> dict[str, Any]:
    artifact_paths = build_teacher_runtime_artifact_paths(run_dir)
    tensorboard_dir = artifact_paths["tensorboard_dir"]
    return {
        "teacher_name": teacher_name,
        "weights": resolved_weights,
        "dataset_yaml": str(dataset_yaml),
        "teacher_root": str(teacher_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(artifact_paths["best_checkpoint"]),
        "last_checkpoint": str(artifact_paths["last_checkpoint"]),
        "profile_log_path": str(artifact_paths["profile_log"]),
        "profile_summary_path": str(artifact_paths["profile_summary"]),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_status": resolve_tensorboard_status(trainer, tensorboard_dir),
        "tensorboard_event_files": tensorboard_event_files(tensorboard_dir),
        "runtime": dict(runtime_params),
        "train_kwargs": train_kwargs,
        "train_result_type": type(train_result).__name__,
    }


def publish_teacher_train_summary(*, summary_path: Path, summary: dict[str, Any]) -> Path:
    write_json(summary_path, summary)
    return summary_path


def finalize_teacher_train_artifacts(
    *,
    teacher_root: Path,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    latest_artifacts = refresh_latest_teacher_artifacts(
        teacher_root=teacher_root,
        run_dir=run_dir,
        summary=summary,
    )
    summary["latest_artifacts"] = latest_artifacts
    artifact_paths = build_teacher_runtime_artifact_paths(run_dir)
    publish_teacher_train_summary(summary_path=artifact_paths["train_summary"], summary=summary)
    publish_teacher_train_summary(
        summary_path=Path(latest_artifacts["train_summary_path"]),
        summary=summary,
    )
    return latest_artifacts


def refresh_latest_teacher_artifacts(
    *,
    teacher_root: Path,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    latest_summary_path = teacher_root / "train_summary.json"
    write_json(latest_summary_path, summary)

    alias_weights_root = teacher_root / "weights"
    alias_actions: dict[str, str] = {}
    for checkpoint_name in ("best.pt", "last.pt", "last_resume.pt"):
        source_path = run_dir / "weights" / checkpoint_name
        if not source_path.is_file():
            continue
        alias_actions[checkpoint_name] = _link_or_copy_file(source_path, alias_weights_root / checkpoint_name)

    latest_run_payload = {
        "teacher_root": str(teacher_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "updated_at": now_iso(),
        "alias_actions": alias_actions,
    }
    latest_run_path = teacher_root / "latest_run.json"
    write_json(latest_run_path, latest_run_payload)
    return {
        "train_summary_path": str(latest_summary_path),
        "latest_run_path": str(latest_run_path),
        "weights_root": str(alias_weights_root),
        "alias_actions": alias_actions,
    }


_refresh_latest_teacher_artifacts = refresh_latest_teacher_artifacts
_build_teacher_runtime_artifact_paths = build_teacher_runtime_artifact_paths
_build_teacher_train_summary = build_teacher_train_summary
_finalize_teacher_train_artifacts = finalize_teacher_train_artifacts
_publish_teacher_train_summary = publish_teacher_train_summary
