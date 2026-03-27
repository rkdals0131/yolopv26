from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

from tools.od_bootstrap.common import greedy_match_boxes
from tools.od_bootstrap.sweep.policy import apply_policy_to_predictions, class_policy_to_dict
from tools.od_bootstrap.sweep.scenario import ClassPolicy

from .scenario import CalibrationScenario, CalibrationTeacherConfig


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text((serialized + "\n") if serialized else "", encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _collect_sample_images(teacher: CalibrationTeacherConfig) -> list[Path]:
    image_root = teacher.dataset.root / teacher.dataset.image_dir / teacher.dataset.split
    if not image_root.is_dir():
        raise FileNotFoundError(f"teacher calibration image root does not exist: {image_root}")
    return [path for path in sorted(image_root.rglob("*")) if path.is_file()]


def _coerce_shape(result: Any) -> tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
        return int(orig_shape[1]), int(orig_shape[0])
    raise ValueError("prediction result must expose orig_shape=(height, width) for calibration")


def _yolo_to_xyxy(values: list[float], *, width: int, height: int) -> list[float]:
    center_x = float(values[0]) * float(width)
    center_y = float(values[1]) * float(height)
    box_w = float(values[2]) * float(width)
    box_h = float(values[3]) * float(height)
    half_w = box_w * 0.5
    half_h = box_h * 0.5
    return [
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    ]


def _load_ground_truth_by_class(
    teacher: CalibrationTeacherConfig,
    *,
    sample_id: str,
    width: int,
    height: int,
) -> dict[str, list[list[float]]]:
    label_path = teacher.dataset.root / teacher.dataset.label_dir / teacher.dataset.split / f"{sample_id}.txt"
    ground_truth: dict[str, list[list[float]]] = {class_name: [] for class_name in teacher.classes}
    if not label_path.is_file():
        return ground_truth
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_index = int(parts[0])
        if not 0 <= class_index < len(teacher.classes):
            continue
        class_name = teacher.classes[class_index]
        ground_truth[class_name].append(
            _yolo_to_xyxy([float(value) for value in parts[1:]], width=width, height=height)
        )
    return ground_truth


def _extract_teacher_rows(
    *,
    teacher: CalibrationTeacherConfig,
    results: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    samples: dict[str, dict[str, Any]] = {}
    for result in results:
        image_path = Path(str(getattr(result, "path", ""))).resolve()
        sample_id = image_path.stem
        width, height = _coerce_shape(result)
        sample_record = samples.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "ground_truth": _load_ground_truth_by_class(teacher, sample_id=sample_id, width=width, height=height),
            },
        )
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_rows = torch.as_tensor(getattr(boxes, "xyxy", None)).tolist()
        cls_rows = torch.as_tensor(getattr(boxes, "cls", None)).tolist()
        conf_rows = torch.as_tensor(getattr(boxes, "conf", None)).tolist()
        for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
            class_name = str(names.get(int(cls_index), teacher.classes[int(cls_index)] if int(cls_index) < len(teacher.classes) else int(cls_index)))
            if class_name not in teacher.classes:
                continue
            row = {
                "teacher_name": teacher.name,
                "model_version": teacher.model_version,
                "sample_id": sample_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "class_name": class_name,
                "confidence": float(confidence),
                "xyxy": [float(value) for value in box],
                "box_index": box_index,
            }
            rows.append(row)
        sample_record["raw_prediction_count"] = sum(1 for row in rows if row["sample_id"] == sample_id)
    return rows, samples


def _run_teacher_predictions(
    *,
    teacher: CalibrationTeacherConfig,
    scenario: CalibrationScenario,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed")
    if not teacher.checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher.checkpoint_path}")
    sample_images = _collect_sample_images(teacher)
    model = YOLO(str(teacher.checkpoint_path))
    results: list[Any] = []
    for batch_start in range(0, len(sample_images), max(1, int(scenario.run.batch_size))):
        batch = sample_images[batch_start : batch_start + max(1, int(scenario.run.batch_size))]
        results.extend(
            list(
                model.predict(
                    source=[str(path) for path in batch],
                    imgsz=scenario.run.imgsz,
                    device=scenario.run.device,
                    conf=scenario.run.predict_conf,
                    iou=scenario.run.predict_iou,
                    verbose=False,
                    save=False,
                    stream=False,
                )
            )
        )
    return _extract_teacher_rows(teacher=teacher, results=results)


def _f_beta(*, precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def _evaluate_candidate(
    *,
    samples: dict[str, dict[str, Any]],
    class_name: str,
    match_iou: float,
    class_policy: dict[str, ClassPolicy],
    target_policy: ClassPolicy,
    dataset_key: str,
) -> dict[str, Any]:
    tp = 0
    fp = 0
    fn = 0
    raw_prediction_count = 0
    gt_count = 0
    for sample in samples.values():
        sample_policy = dict(class_policy)
        sample_policy[class_name] = target_policy
        gt_boxes = list(sample["ground_truth"].get(class_name, []))
        gt_count += len(gt_boxes)
        all_predictions = list(sample.get("predictions", []))
        raw_prediction_count += sum(1 for row in all_predictions if row["class_name"] == class_name)
        kept_rows = apply_policy_to_predictions(
            rows=all_predictions,
            class_policy=sample_policy,
            dataset_key=dataset_key,
            image_width=int(sample["width"]),
            image_height=int(sample["height"]),
            raw_boxes_by_class={},
        )
        class_rows = [row for row in kept_rows if row["class_name"] == class_name]
        sample_tp, sample_fp, sample_fn = greedy_match_boxes(class_rows, gt_boxes, match_iou=match_iou)
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f0_5 = _f_beta(precision=precision, recall=recall, beta=0.5)
    return {
        "score_threshold": float(target_policy.score_threshold),
        "nms_iou_threshold": float(target_policy.nms_iou_threshold),
        "min_box_size": int(target_policy.min_box_size),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "ground_truth_count": int(gt_count),
        "raw_prediction_count": int(raw_prediction_count),
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
    }


def _select_best_candidate(
    *,
    candidates: list[dict[str, Any]],
    min_precision: float,
) -> dict[str, Any]:
    meets_floor = [item for item in candidates if float(item["precision"]) >= float(min_precision)]
    if meets_floor:
        best = max(
            meets_floor,
            key=lambda item: (
                float(item["recall"]),
                float(item["f0_5"]),
                float(item["precision"]),
                -float(item["score_threshold"]),
                -float(item["nms_iou_threshold"]),
                -int(item["min_box_size"]),
            ),
        )
        best["meets_precision_floor"] = True
        return best
    best = max(
        candidates,
        key=lambda item: (
            float(item["precision"]),
            float(item["recall"]),
            float(item["f0_5"]),
            -float(item["score_threshold"]),
            -float(item["nms_iou_threshold"]),
            -int(item["min_box_size"]),
        ),
    )
    best["meets_precision_floor"] = False
    return best


def _build_default_policy_template(scenario: CalibrationScenario) -> dict[str, ClassPolicy]:
    default_policy = ClassPolicy(
        score_threshold=float(scenario.search.score_thresholds[0]),
        nms_iou_threshold=float(scenario.search.nms_iou_thresholds[0]),
        min_box_size=int(scenario.search.min_box_sizes[0]),
    )
    return {
        class_name: default_policy
        for teacher in scenario.teachers
        for class_name in teacher.classes
    }


def calibrate_class_policy_scenario(
    scenario: CalibrationScenario,
    *,
    scenario_path: Path,
) -> dict[str, Any]:
    created_at = _now_iso()
    output_root = scenario.run.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    teacher_summaries: list[dict[str, Any]] = []
    samples_by_teacher: dict[str, dict[str, dict[str, Any]]] = {}
    teacher_by_class: dict[str, CalibrationTeacherConfig] = {}
    policy_template = dict(scenario.policy_template or _build_default_policy_template(scenario))

    for teacher in scenario.teachers:
        rows, samples = _run_teacher_predictions(teacher=teacher, scenario=scenario)
        for sample in samples.values():
            sample["predictions"] = [row for row in rows if row["sample_id"] == sample["sample_id"]]
        teacher_dir = output_root / "teachers" / teacher.name
        predictions_path = _write_jsonl(teacher_dir / "predictions.jsonl", rows)
        teacher_summary = {
            "teacher_name": teacher.name,
            "model_version": teacher.model_version,
            "checkpoint_path": str(teacher.checkpoint_path),
            "dataset_root": str(teacher.dataset.root),
            "split": teacher.dataset.split,
            "sample_count": len(samples),
            "prediction_count": len(rows),
            "predictions_path": str(predictions_path),
        }
        _write_json(teacher_dir / "summary.json", teacher_summary)
        teacher_summaries.append(teacher_summary)
        samples_by_teacher[teacher.name] = samples
        for class_name in teacher.classes:
            teacher_by_class[class_name] = teacher

    report_classes: dict[str, Any] = {}
    class_policy_payload: dict[str, Any] = {}
    for class_name in sorted(teacher_by_class):
        teacher = teacher_by_class[class_name]
        teacher_samples = samples_by_teacher[teacher.name]
        base_policy = policy_template[class_name]
        candidates: list[dict[str, Any]] = []
        for score_threshold in scenario.search.score_thresholds:
            for nms_iou_threshold in scenario.search.nms_iou_thresholds:
                for min_box_size in scenario.search.min_box_sizes:
                    candidate_policy = replace(
                        base_policy,
                        score_threshold=float(score_threshold),
                        nms_iou_threshold=float(nms_iou_threshold),
                        min_box_size=int(min_box_size),
                    )
                    candidates.append(
                        _evaluate_candidate(
                            samples=teacher_samples,
                            class_name=class_name,
                            match_iou=float(scenario.search.match_iou),
                            class_policy=policy_template,
                            target_policy=candidate_policy,
                            dataset_key=teacher.dataset.source_dataset_key,
                        )
                    )
        best_candidate = _select_best_candidate(candidates=candidates, min_precision=float(scenario.search.min_precision))
        selected_policy = replace(
            base_policy,
            score_threshold=float(best_candidate["score_threshold"]),
            nms_iou_threshold=float(best_candidate["nms_iou_threshold"]),
            min_box_size=int(best_candidate["min_box_size"]),
        )
        policy_template[class_name] = selected_policy
        class_policy_payload[class_name] = class_policy_to_dict(selected_policy)
        report_classes[class_name] = {
            "selected_policy": class_policy_payload[class_name],
            "metrics": {
                "precision": best_candidate["precision"],
                "recall": best_candidate["recall"],
                "f0_5": best_candidate["f0_5"],
                "tp": best_candidate["tp"],
                "fp": best_candidate["fp"],
                "fn": best_candidate["fn"],
                "ground_truth_count": best_candidate["ground_truth_count"],
                "raw_prediction_count": best_candidate["raw_prediction_count"],
            },
            "meets_precision_floor": bool(best_candidate["meets_precision_floor"]),
            "candidate_count": len(candidates),
        }

    class_policy_path = _write_yaml(output_root / "class_policy.yaml", class_policy_payload)
    report = {
        "scenario_path": str(scenario_path),
        "generated_at": created_at,
        "output_root": str(output_root),
        "class_policy_path": str(class_policy_path),
        "run": asdict(scenario.run),
        "search": asdict(scenario.search),
        "teachers": teacher_summaries,
        "classes": report_classes,
    }
    report_path = _write_json(output_root / "calibration_report.json", report)
    return {
        "output_root": str(output_root),
        "class_policy_path": str(class_policy_path),
        "report_path": str(report_path),
        "classes": report_classes,
    }
