from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

from tools.od_bootstrap.common import iou
from tools.od_bootstrap.sweep.policy import apply_policy_to_predictions, class_policy_to_dict
from tools.od_bootstrap.sweep.scenario import ClassPolicy

from .scenario import CalibrationScenario, CalibrationTeacherConfig, HardNegativeConfig


def _log_calibration(message: str) -> None:
    print(f"[od_bootstrap.calibration] {message}", flush=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _prediction_sort_key(row: dict[str, Any]) -> float:
    return float(row["confidence"])


def _split_predictions_against_gt(
    *,
    predictions: list[dict[str, Any]],
    ground_truth_boxes: list[list[float]],
    match_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    gt_rows = [[float(value) for value in box] for box in ground_truth_boxes]
    matched_gt = [False] * len(gt_rows)
    matched_predictions: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for row in sorted(predictions, key=_prediction_sort_key, reverse=True):
        candidate_box = [float(value) for value in row["xyxy"]]
        best_index = -1
        best_iou = 0.0
        for index, gt_box in enumerate(gt_rows):
            if matched_gt[index]:
                continue
            overlap = iou(candidate_box, gt_box)
            if overlap >= match_iou and overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index >= 0:
            matched_gt[best_index] = True
            matched_predictions.append(row)
            continue
        false_positives.append(row)
    return matched_predictions, false_positives, len(gt_rows) - len(matched_predictions)


def _resolve_hard_negative_config(scenario: CalibrationScenario) -> HardNegativeConfig:
    return scenario.hard_negative or HardNegativeConfig()


def _resolve_manifest_image_path(*, manifest_path: Path, raw_path: str) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path.resolve()
    return (manifest_path.parent / image_path).resolve()


def _load_hard_negative_manifest(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes") or {}
    if not isinstance(classes, dict):
        raise TypeError("hard negative manifest classes must be a mapping")
    resolved: dict[str, list[dict[str, Any]]] = {}
    for class_name, raw_samples in classes.items():
        if not isinstance(raw_samples, list):
            raise TypeError(f"hard negative manifest class '{class_name}' must contain a list")
        entries: list[dict[str, Any]] = []
        for index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, dict):
                raise TypeError(f"hard negative manifest class '{class_name}' sample[{index}] must be a mapping")
            raw_image_path = str(raw_sample.get("image_path") or "").strip()
            if not raw_image_path:
                raise ValueError(f"hard negative manifest class '{class_name}' sample[{index}] missing image_path")
            image_path = _resolve_manifest_image_path(manifest_path=path, raw_path=raw_image_path)
            entries.append(
                {
                    "sample_id": str(raw_sample.get("sample_id") or image_path.stem),
                    "image_path": str(image_path),
                    "dataset_key": str(raw_sample.get("dataset_key") or "").strip(),
                    "teacher_name": str(raw_sample.get("teacher_name") or "").strip(),
                }
            )
        resolved[str(class_name)] = entries
    return resolved


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


def _append_prediction_rows(
    *,
    teacher: CalibrationTeacherConfig,
    result: Any,
    samples: dict[str, dict[str, Any]],
    prediction_stream: Any | None = None,
) -> int:
    image_path = Path(str(getattr(result, "path", ""))).resolve()
    sample_id = image_path.stem
    sample_key = str(image_path)
    width, height = _coerce_shape(result)
    sample_record = samples.setdefault(
        sample_key,
        {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "predictions": [],
            "raw_prediction_count": 0,
        },
    )
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    xyxy_rows = torch.as_tensor(getattr(boxes, "xyxy", None)).tolist()
    cls_rows = torch.as_tensor(getattr(boxes, "cls", None)).tolist()
    conf_rows = torch.as_tensor(getattr(boxes, "conf", None)).tolist()
    appended = 0
    for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
        class_name = str(
            names.get(
                int(cls_index),
                teacher.classes[int(cls_index)] if int(cls_index) < len(teacher.classes) else int(cls_index),
            )
        )
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
        sample_record["predictions"].append(row)
        appended += 1
        if prediction_stream is not None:
            prediction_stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    sample_record["raw_prediction_count"] = len(sample_record["predictions"])
    return appended


def _run_teacher_predictions_for_images(
    *,
    teacher: CalibrationTeacherConfig,
    scenario: CalibrationScenario,
    image_paths: list[Path],
    predictions_path: Path | None = None,
    log_fn: Any | None = None,
) -> tuple[int, dict[str, dict[str, Any]]]:
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed")
    if not teacher.checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher.checkpoint_path}")
    model = YOLO(str(teacher.checkpoint_path))
    ordered_images = sorted({path.resolve() for path in image_paths})
    batch_size = max(1, int(scenario.run.batch_size))
    total_images = len(ordered_images)
    samples: dict[str, dict[str, Any]] = {}
    prediction_count = 0
    processed_images = 0
    last_logged = 0
    log_every_images = max(batch_size * 50, 500)
    if log_fn is not None:
        log_fn(f"teacher={teacher.name} predict start images={total_images} batch={batch_size}")
    stream_context = (
        predictions_path.open("w", encoding="utf-8")
        if predictions_path is not None
        else nullcontext(None)
    )
    with stream_context as prediction_stream:
        for batch_start in range(0, total_images, batch_size):
            batch = ordered_images[batch_start : batch_start + batch_size]
            batch_prediction_count = 0
            for result in model.predict(
                source=[str(path) for path in batch],
                imgsz=scenario.run.imgsz,
                device=scenario.run.device,
                conf=scenario.run.predict_conf,
                iou=scenario.run.predict_iou,
                verbose=False,
                save=False,
                stream=True,
            ):
                batch_prediction_count += _append_prediction_rows(
                    teacher=teacher,
                    result=result,
                    samples=samples,
                    prediction_stream=prediction_stream,
                )
            processed_images += len(batch)
            prediction_count += batch_prediction_count
            if (
                log_fn is not None
                and (
                    processed_images == total_images
                    or processed_images == len(batch)
                    or processed_images - last_logged >= log_every_images
                )
            ):
                last_logged = processed_images
                log_fn(
                    f"teacher={teacher.name} predict progress {processed_images}/{total_images} "
                    f"images predictions={prediction_count}"
                )
    return prediction_count, samples


def _collect_hard_negative_samples_by_class(
    *,
    scenario: CalibrationScenario,
    teacher_by_class: dict[str, CalibrationTeacherConfig],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    config = _resolve_hard_negative_config(scenario)
    manifest = _load_hard_negative_manifest(config.manifest_path)
    if not manifest:
        return {}, {}

    manifest_by_class_and_image: dict[str, dict[str, dict[str, Any]]] = {}
    for class_name, samples in manifest.items():
        if class_name not in teacher_by_class:
            continue
        manifest_by_class_and_image[class_name] = {
            str(Path(str(item["image_path"])).resolve()): item for item in samples
        }

    samples_by_class: dict[str, dict[str, dict[str, Any]]] = {}
    for teacher in {teacher_by_class[class_name] for class_name in manifest_by_class_and_image}:
        teacher_classes = [class_name for class_name, owner in teacher_by_class.items() if owner == teacher and class_name in manifest_by_class_and_image]
        teacher_image_paths = sorted(
            {
                Path(image_path)
                for class_name in teacher_classes
                for image_path in manifest_by_class_and_image[class_name]
            }
        )
        if teacher_image_paths:
            _log_calibration(
                f"teacher={teacher.name} hard-negative replay images={len(teacher_image_paths)} classes={teacher_classes}"
            )
        _, predicted_samples = _run_teacher_predictions_for_images(
            teacher=teacher,
            scenario=scenario,
            image_paths=teacher_image_paths,
            log_fn=_log_calibration,
        )
        for class_name in teacher_classes:
            class_samples: dict[str, dict[str, Any]] = {}
            for image_path, manifest_entry in manifest_by_class_and_image[class_name].items():
                sample = predicted_samples.get(image_path)
                if sample is None:
                    continue
                class_samples[image_path] = {
                    **sample,
                    "dataset_key": str(manifest_entry.get("dataset_key") or teacher.dataset.source_dataset_key),
                    "predictions": list(sample.get("predictions", [])),
                }
            samples_by_class[class_name] = class_samples
    return manifest, samples_by_class


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
    hard_negative_samples: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tp = 0
    fp = 0
    fn = 0
    raw_prediction_count = 0
    gt_count = 0
    sample_policy = dict(class_policy)
    sample_policy[class_name] = target_policy
    for sample in samples.values():
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
        matched_rows, false_positives, sample_fn = _split_predictions_against_gt(
            predictions=class_rows,
            ground_truth_boxes=gt_boxes,
            match_iou=match_iou,
        )
        tp += len(matched_rows)
        fp += len(false_positives)
        fn += sample_fn

    hard_negative_sample_count = 0
    hard_negative_failures = 0
    hard_negative_prediction_count = 0
    for sample in (hard_negative_samples or {}).values():
        hard_negative_sample_count += 1
        kept_rows = apply_policy_to_predictions(
            rows=list(sample.get("predictions", [])),
            class_policy=sample_policy,
            dataset_key=str(sample.get("dataset_key") or dataset_key),
            image_width=int(sample["width"]),
            image_height=int(sample["height"]),
            raw_boxes_by_class={},
        )
        class_rows = [row for row in kept_rows if row["class_name"] == class_name]
        if not class_rows:
            continue
        hard_negative_failures += 1
        hard_negative_prediction_count += len(class_rows)
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
        "hard_negative_sample_count": int(hard_negative_sample_count),
        "hard_negative_failures": int(hard_negative_failures),
        "hard_negative_prediction_count": int(hard_negative_prediction_count),
    }


def _select_best_candidate(
    *,
    candidates: list[dict[str, Any]],
    min_precision: float,
) -> dict[str, Any]:
    meets_floor = [item for item in candidates if float(item["precision"]) >= float(min_precision)]
    if meets_floor:
        best = min(
            meets_floor,
            key=lambda item: (
                int(item["hard_negative_failures"]),
                int(item["hard_negative_prediction_count"]),
                -float(item["recall"]),
                -float(item["f0_5"]),
                -float(item["precision"]),
                float(item["score_threshold"]),
                float(item["nms_iou_threshold"]),
                int(item["min_box_size"]),
            ),
        )
        best["meets_precision_floor"] = True
        return best
    best = min(
        candidates,
        key=lambda item: (
            -float(item["precision"]),
            int(item["hard_negative_failures"]),
            int(item["hard_negative_prediction_count"]),
            -float(item["recall"]),
            -float(item["f0_5"]),
            float(item["score_threshold"]),
            float(item["nms_iou_threshold"]),
            int(item["min_box_size"]),
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


def _build_hard_negative_manifest_payload(
    *,
    scenario: CalibrationScenario,
    class_policy: dict[str, ClassPolicy],
    samples_by_teacher: dict[str, dict[str, dict[str, Any]]],
    teacher_by_class: dict[str, CalibrationTeacherConfig],
    source_manifest_path: Path | None,
    created_at: str,
    scenario_path: Path,
) -> dict[str, Any]:
    config = _resolve_hard_negative_config(scenario)
    focus_classes = list(config.focus_classes or sorted(teacher_by_class))
    classes_payload: dict[str, list[dict[str, Any]]] = {}
    for class_name in focus_classes:
        teacher = teacher_by_class[class_name]
        sample_rows: list[dict[str, Any]] = []
        for sample in samples_by_teacher[teacher.name].values():
            kept_rows = apply_policy_to_predictions(
                rows=list(sample.get("predictions", [])),
                class_policy=class_policy,
                dataset_key=teacher.dataset.source_dataset_key,
                image_width=int(sample["width"]),
                image_height=int(sample["height"]),
                raw_boxes_by_class={},
            )
            class_rows = [row for row in kept_rows if row["class_name"] == class_name]
            if not class_rows:
                continue
            _, false_positives, _ = _split_predictions_against_gt(
                predictions=class_rows,
                ground_truth_boxes=list(sample["ground_truth"].get(class_name, [])),
                match_iou=float(scenario.search.match_iou),
            )
            if not false_positives:
                continue
            sample_rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "image_path": str(sample["image_path"]),
                    "dataset_key": teacher.dataset.source_dataset_key,
                    "teacher_name": teacher.name,
                    "false_positive_count": len(false_positives),
                    "max_confidence": max(float(row["confidence"]) for row in false_positives),
                    "predictions": [
                        {
                            "confidence": float(row["confidence"]),
                            "xyxy": [float(value) for value in row["xyxy"]],
                        }
                        for row in sorted(false_positives, key=_prediction_sort_key, reverse=True)
                    ],
                }
            )
        classes_payload[class_name] = sorted(
            sample_rows,
            key=lambda item: (
                -int(item["false_positive_count"]),
                -float(item["max_confidence"]),
                str(item["sample_id"]),
            ),
        )[: max(1, int(config.top_k_per_class))]

    return {
        "version": "od-bootstrap-hard-negative-v1",
        "generated_at": created_at,
        "scenario_path": str(scenario_path),
        "source_manifest_path": str(source_manifest_path) if source_manifest_path is not None else None,
        "top_k_per_class": int(config.top_k_per_class),
        "focus_classes": focus_classes,
        "classes": classes_payload,
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
    _log_calibration(f"scenario={scenario_path}")
    _log_calibration(f"output_root={output_root}")

    for teacher in scenario.teachers:
        teacher_dir = output_root / "teachers" / teacher.name
        teacher_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = teacher_dir / "predictions.jsonl"
        _log_calibration(f"teacher={teacher.name} calibration start checkpoint={teacher.checkpoint_path}")
        prediction_count, samples = _run_teacher_predictions_for_images(
            teacher=teacher,
            scenario=scenario,
            image_paths=_collect_sample_images(teacher),
            predictions_path=predictions_path,
            log_fn=_log_calibration,
        )
        _log_calibration(f"teacher={teacher.name} loading ground truth for {len(samples)} samples")
        for sample in samples.values():
            sample["ground_truth"] = _load_ground_truth_by_class(
                teacher,
                sample_id=str(sample["sample_id"]),
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
        teacher_summary = {
            "teacher_name": teacher.name,
            "model_version": teacher.model_version,
            "checkpoint_path": str(teacher.checkpoint_path),
            "dataset_root": str(teacher.dataset.root),
            "split": teacher.dataset.split,
            "sample_count": len(samples),
            "prediction_count": int(prediction_count),
            "predictions_path": str(predictions_path),
        }
        _write_json(teacher_dir / "summary.json", teacher_summary)
        teacher_summaries.append(teacher_summary)
        samples_by_teacher[teacher.name] = samples
        _log_calibration(
            f"teacher={teacher.name} calibration ready samples={len(samples)} predictions={prediction_count}"
        )
        for class_name in teacher.classes:
            teacher_by_class[class_name] = teacher

    hard_negative_config = _resolve_hard_negative_config(scenario)
    if hard_negative_config.manifest_path is not None:
        _log_calibration(f"hard-negative input manifest={hard_negative_config.manifest_path}")
    input_hard_negative_manifest, hard_negative_samples_by_class = _collect_hard_negative_samples_by_class(
        scenario=scenario,
        teacher_by_class=teacher_by_class,
    )

    report_classes: dict[str, Any] = {}
    class_policy_payload: dict[str, Any] = {}
    for class_name in sorted(teacher_by_class):
        teacher = teacher_by_class[class_name]
        teacher_samples = samples_by_teacher[teacher.name]
        base_policy = policy_template[class_name]
        class_min_precision = float(scenario.search.min_precision_by_class.get(class_name, scenario.search.min_precision))
        candidates: list[dict[str, Any]] = []
        candidate_count = (
            len(scenario.search.score_thresholds)
            * len(scenario.search.nms_iou_thresholds)
            * len(scenario.search.min_box_sizes)
        )
        _log_calibration(
            f"class={class_name} search start teacher={teacher.name} samples={len(teacher_samples)} "
            f"candidates={candidate_count} min_precision={class_min_precision:.2f}"
        )
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
                            hard_negative_samples=hard_negative_samples_by_class.get(class_name, {}),
                        )
                    )
        best_candidate = _select_best_candidate(candidates=candidates, min_precision=class_min_precision)
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
                "hard_negative_sample_count": best_candidate["hard_negative_sample_count"],
                "hard_negative_failures": best_candidate["hard_negative_failures"],
                "hard_negative_prediction_count": best_candidate["hard_negative_prediction_count"],
            },
            "meets_precision_floor": bool(best_candidate["meets_precision_floor"]),
            "min_precision_target": class_min_precision,
            "candidate_count": len(candidates),
        }
        _log_calibration(
            f"class={class_name} selected score={best_candidate['score_threshold']:.2f} "
            f"nms={best_candidate['nms_iou_threshold']:.2f} min_box={best_candidate['min_box_size']} "
            f"precision={best_candidate['precision']:.4f} recall={best_candidate['recall']:.4f}"
        )

    class_policy_path = _write_yaml(output_root / "class_policy.yaml", class_policy_payload)
    hard_negative_manifest_path = _write_json(
        output_root / "hard_negative_manifest.json",
        _build_hard_negative_manifest_payload(
            scenario=scenario,
            class_policy=policy_template,
            samples_by_teacher=samples_by_teacher,
            teacher_by_class=teacher_by_class,
            source_manifest_path=hard_negative_config.manifest_path,
            created_at=created_at,
            scenario_path=scenario_path,
        ),
    )
    report = {
        "scenario_path": str(scenario_path),
        "generated_at": created_at,
        "output_root": str(output_root),
        "class_policy_path": str(class_policy_path),
        "hard_negative": {
            "input_manifest_path": str(hard_negative_config.manifest_path) if hard_negative_config.manifest_path is not None else None,
            "input_sample_count_by_class": {
                class_name: len(samples)
                for class_name, samples in sorted(input_hard_negative_manifest.items())
            },
            "output_manifest_path": str(hard_negative_manifest_path),
            "top_k_per_class": int(hard_negative_config.top_k_per_class),
            "focus_classes": list(hard_negative_config.focus_classes or sorted(teacher_by_class)),
        },
        "run": asdict(scenario.run),
        "search": asdict(scenario.search),
        "teachers": teacher_summaries,
        "classes": report_classes,
    }
    report_path = _write_json(output_root / "calibration_report.json", report)
    _log_calibration(f"class policy written to {class_policy_path}")
    _log_calibration(f"hard-negative manifest written to {hard_negative_manifest_path}")
    _log_calibration(f"report written to {report_path}")
    return {
        "output_root": str(output_root),
        "class_policy_path": str(class_policy_path),
        "hard_negative_manifest_path": str(hard_negative_manifest_path),
        "report_path": str(report_path),
        "classes": report_classes,
    }
