from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    YOLO = None


def _coerce_weights_name(model_size: str, weights: str | None) -> str:
    if weights:
        return weights
    size = model_size.strip().lower() or "n"
    return f"yolo26{size}.pt"


def _extract_run_dir(train_result: Any, fallback_dir: Path) -> Path:
    for candidate in (
        getattr(train_result, "save_dir", None),
        getattr(getattr(train_result, "trainer", None), "save_dir", None),
    ):
        if candidate:
            return Path(candidate)
    return fallback_dir


def train_teacher_with_ultralytics(
    *,
    teacher_name: str,
    dataset_yaml: Path,
    output_root: Path,
    model_size: str,
    weights: str | None,
    train_params: dict[str, Any],
    exist_ok: bool,
) -> dict[str, Any]:
    if YOLO is None:  # pragma: no cover - exercised only when dependency is missing.
        raise RuntimeError("ultralytics is not installed")

    resolved_weights = _coerce_weights_name(model_size, weights)
    model = YOLO(resolved_weights)

    train_kwargs = {
        "data": str(dataset_yaml),
        "project": str(output_root),
        "name": teacher_name,
        "exist_ok": bool(exist_ok),
        "pretrained": True,
        **train_params,
    }
    train_result = model.train(**train_kwargs)
    run_dir = _extract_run_dir(train_result, output_root / teacher_name)
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"

    summary = {
        "teacher_name": teacher_name,
        "weights": resolved_weights,
        "dataset_yaml": str(dataset_yaml),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "train_kwargs": train_kwargs,
        "train_result_type": type(train_result).__name__,
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary
