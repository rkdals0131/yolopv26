from __future__ import annotations

from pathlib import Path
# Keep a module-level time import for existing patch targets in tests.
import time
from typing import Any

from . import runtime_progress
from .runtime_artifacts import (
    build_teacher_train_summary,
    finalize_teacher_train_artifacts,
)
from .runtime_resume import coerce_weights_name, extract_run_dir, resolve_resume_argument
from .runtime_trainer import make_teacher_trainer as _make_teacher_trainer

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    YOLO = None


_build_teacher_train_summary = build_teacher_train_summary
_emit_log = runtime_progress.emit_log
_finalize_teacher_train_artifacts = finalize_teacher_train_artifacts
_timestamp_token = runtime_progress.timestamp_token


def _build_teacher_train_kwargs(
    *,
    dataset_yaml: Path,
    teacher_root: Path,
    run_name: str,
    train_params: dict[str, Any],
    exist_ok: bool,
) -> dict[str, Any]:
    return {
        "data": str(dataset_yaml),
        "project": str(teacher_root),
        "name": run_name,
        "exist_ok": bool(exist_ok),
        "pretrained": True,
        **train_params,
    }


def train_teacher_with_ultralytics(
    *,
    teacher_name: str,
    dataset_yaml: Path,
    output_root: Path,
    model_size: str,
    weights: str | None,
    train_params: dict[str, Any],
    runtime_params: dict[str, Any],
    exist_ok: bool,
) -> dict[str, Any]:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    resolved_weights = coerce_weights_name(model_size, weights)
    log_fn = _emit_log
    trainer_cls, callbacks = _make_teacher_trainer(runtime_params=runtime_params, log_fn=log_fn)
    teacher_root = output_root / teacher_name
    train_params = dict(train_params)
    train_params["resume"] = resolve_resume_argument(
        train_params.get("resume", False),
        teacher_name=teacher_name,
        teacher_root=teacher_root,
    )
    model_source = train_params["resume"] if train_params["resume"] else resolved_weights
    model = YOLO(model_source)
    if hasattr(model, "add_callback"):
        for event_name, callback in callbacks.items():
            model.add_callback(event_name, callback)
    run_name = _timestamp_token()
    train_kwargs = _build_teacher_train_kwargs(
        dataset_yaml=dataset_yaml,
        teacher_root=teacher_root,
        run_name=run_name,
        train_params=train_params,
        exist_ok=exist_ok,
    )
    train_result = model.train(trainer=trainer_cls, **train_kwargs)
    run_dir = extract_run_dir(train_result, teacher_root / run_name)
    trainer = getattr(model, "trainer", None)
    summary = _build_teacher_train_summary(
        teacher_name=teacher_name,
        resolved_weights=resolved_weights,
        dataset_yaml=dataset_yaml,
        teacher_root=teacher_root,
        run_dir=run_dir,
        runtime_params=runtime_params,
        train_kwargs=train_kwargs,
        train_result=train_result,
        trainer=trainer,
    )
    _finalize_teacher_train_artifacts(teacher_root=teacher_root, run_dir=run_dir, summary=summary)
    return summary
