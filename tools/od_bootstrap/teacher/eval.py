from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    YOLO = None

from .data_yaml import build_teacher_data_yaml, resolve_teacher_dataset_root
from .eval_types import CheckpointEvalScenario


def _log_eval(message: str) -> None:
    print(f"[od_bootstrap.eval] {message}", flush=True)


@dataclass(frozen=True)
class PredictionSummary:
    class_counts: dict[str, int]
    confidence: dict[str, float]
    prediction_count: int
    sample_count: int


def _build_resolved_runtime_summary(scenario: CheckpointEvalScenario) -> dict[str, Any]:
    params = scenario.eval
    return {
        "imgsz": params.imgsz,
        "batch": params.batch,
        "device": params.device,
        "conf": params.conf,
        "iou": params.iou,
    }


def _collect_sample_images(root: Path, *, split: str, image_dir: str, limit: int) -> list[Path]:
    image_root = root / image_dir / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"evaluation image root does not exist: {image_root}")
    images = [path for path in sorted(image_root.rglob("*")) if path.is_file()]
    return images[: max(0, int(limit))]


def _extract_prediction_rows(results: list[Any]) -> tuple[list[dict[str, Any]], PredictionSummary]:
    class_counts: dict[str, int] = {}
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []

    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        path = str(getattr(result, "path", ""))
        if boxes is None:
            continue

        xyxy = getattr(boxes, "xyxy", None)
        cls_tensor = getattr(boxes, "cls", None)
        conf_tensor = getattr(boxes, "conf", None)
        if xyxy is None or cls_tensor is None or conf_tensor is None:
            continue

        xyxy_rows = torch.as_tensor(xyxy).tolist()
        cls_rows = torch.as_tensor(cls_tensor).tolist()
        conf_rows = torch.as_tensor(conf_tensor).tolist()

        for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
            class_name = names.get(int(cls_index), str(int(cls_index)))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(float(confidence))
            rows.append(
                {
                    "image_path": path,
                    "box_index": box_index,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in box],
                }
            )

    confidence_summary = {
        "min": min(confidences) if confidences else 0.0,
        "max": max(confidences) if confidences else 0.0,
        "mean": mean(confidences) if confidences else 0.0,
    }
    summary = PredictionSummary(
        class_counts=dict(sorted(class_counts.items())),
        confidence=confidence_summary,
        prediction_count=len(rows),
        sample_count=len(results),
    )
    return rows, summary


def eval_teacher_checkpoint(
    *,
    scenario: CheckpointEvalScenario,
    scenario_path: Path,
) -> dict[str, Any]:
    if YOLO is None:  # pragma: no cover - exercised only when dependency is missing.
        raise RuntimeError("ultralytics is not installed")

    checkpoint_path = scenario.model.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    _log_eval(f"scenario={scenario_path}")
    _log_eval(f"teacher={scenario.teacher_name} checkpoint={checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    dataset_root = resolve_teacher_dataset_root(
        source_root=scenario.dataset.root,
        image_dir=scenario.dataset.image_dir,
        label_dir=scenario.dataset.label_dir,
        train_split="train",
        val_split=scenario.dataset.split,
    )

    data_yaml_path = build_teacher_data_yaml(
        dataset_root=dataset_root,
        class_names=scenario.model.class_names,
        output_path=scenario.run.output_root / scenario.teacher_name / "data.yaml",
        train_split="train",
        val_split=scenario.dataset.split,
    )
    _log_eval(f"teacher={scenario.teacher_name} dataset_root={dataset_root}")

    predict_rows: list[dict[str, Any]] = []
    prediction_summary = PredictionSummary({}, {"min": 0.0, "max": 0.0, "mean": 0.0}, 0, 0)
    if scenario.eval.predict:
        sample_images = _collect_sample_images(
            dataset_root,
            split=scenario.dataset.split,
            image_dir=scenario.dataset.image_dir,
            limit=scenario.dataset.sample_limit,
        )
        _log_eval(
            f"teacher={scenario.teacher_name} predict start images={len(sample_images)} "
            f"imgsz={scenario.eval.imgsz} batch={scenario.eval.batch}"
        )
        predict_results = model.predict(
            source=[str(path) for path in sample_images],
            imgsz=scenario.eval.imgsz,
            conf=scenario.eval.conf,
            iou=scenario.eval.iou,
            save=False,
            verbose=scenario.eval.verbose,
            device=scenario.eval.device,
            stream=False,
        )
        predict_rows, prediction_summary = _extract_prediction_rows(list(predict_results))
        _log_eval(
            f"teacher={scenario.teacher_name} predict done samples={prediction_summary.sample_count} "
            f"predictions={prediction_summary.prediction_count}"
        )

    val_summary: dict[str, Any] = {}
    if scenario.eval.val:
        _log_eval(
            f"teacher={scenario.teacher_name} val start split={scenario.dataset.split} "
            f"imgsz={scenario.eval.imgsz} batch={scenario.eval.batch}"
        )
        val_result = model.val(
            data=str(data_yaml_path),
            split=scenario.dataset.split,
            imgsz=scenario.eval.imgsz,
            batch=scenario.eval.batch,
            device=scenario.eval.device,
            verbose=scenario.eval.verbose,
        )
        val_summary = _normalize_val_result(val_result)
        _log_eval(f"teacher={scenario.teacher_name} val done")

    output_dir = scenario.run.output_root / scenario.teacher_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in predict_rows) + ("\n" if predict_rows else ""),
        encoding="utf-8",
    )

    summary = {
        "scenario_path": str(scenario_path),
        "teacher_name": scenario.teacher_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "data_yaml_path": str(data_yaml_path),
        "prediction_summary": asdict(prediction_summary),
        "val_summary": val_summary,
        "predictions_path": str(predictions_path),
        "eval": asdict(scenario.eval),
        "model": asdict(scenario.model),
        "run": asdict(scenario.run),
        "resolved_runtime": _build_resolved_runtime_summary(scenario),
    }
    summary_path = output_dir / "checkpoint_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    _log_eval(f"teacher={scenario.teacher_name} summary={summary_path}")
    return summary


def _normalize_val_result(val_result: Any) -> dict[str, Any]:
    if isinstance(val_result, dict):
        return val_result
    results_dict = getattr(val_result, "results_dict", None)
    if isinstance(results_dict, dict):
        return results_dict
    payload: dict[str, Any] = {}
    for attr in ("box", "box_map", "box_p", "box_r", "maps"):
        value = getattr(val_result, attr, None)
        if value is not None:
            payload[attr] = value
    payload["type"] = type(val_result).__name__
    return payload


__all__ = ["CheckpointEvalScenario", "PredictionSummary", "eval_teacher_checkpoint"]
