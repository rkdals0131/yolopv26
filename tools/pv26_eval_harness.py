from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import site
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

from common.pv26_schema import TL_BITS
from model.data.dataset import PV26CanonicalDataset, collate_pv26_samples
from model.engine.metrics import _extract_gt_samples
from model.engine.metrics import summarize_pv26_metrics
from model.engine.metrics import summarize_pv26_tensorboard_histograms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FILENAMES = (
    "yolopv26.torchscript.pt",
    "yolopv26best.torchscript.pt",
    "best.torchscript.pt",
)
REQUIRED_OUTPUT_NAMES = ("det", "tl_attr", "lane", "stop_line")
TASK_SCORE_PATHS = {
    "detector": ("detector", "map50_95"),
    "traffic_light": ("traffic_light", "mean_f1"),
    "lane": ("lane", "f1"),
    "stop_line": ("stop_line", "f1"),
    "crosswalk": ("crosswalk", "f1"),
}


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    weights: Path
    model_meta: Path


@dataclass(frozen=True)
class HarnessOptions:
    dataset_root: Path
    output_root: Path
    candidates: tuple[ModelCandidate, ...]
    spade_root: Path
    pv26_repo_root: Path
    device_name: str = "auto"
    split: str | None = None
    limit: int | None = None
    overlay_count: int = 10
    write_plots: bool = True
    write_overlays: bool = True
    det_conf_thres: float = 0.30
    det_iou_thres: float = 0.70
    lane_obj_thres: float = 0.50
    lane_visibility_thres: float = 0.50
    stop_line_obj_thres: float = 0.50


def _safe_name(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._") or "model"


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw_name in names:
        base = _safe_name(raw_name)
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique.append(base if count == 1 else f"{base}_{count}")
    return tuple(unique)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_floats(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is not None:
            result.append(number)
    return result


def _summary_stats(values: Any) -> dict[str, Any]:
    numbers = _finite_floats(values)
    if not numbers:
        return {"count": 0}
    array = np.asarray(numbers, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _summarize_histograms(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _summarize_histograms(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return _summary_stats(payload)
    return payload


def _normalized_timing(timing: Any) -> dict[str, float]:
    if not isinstance(timing, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in timing.items():
        number = _finite_float(value)
        if number is not None:
            normalized[str(key)] = number
    if "total_ms" not in normalized:
        parts = [
            normalized.get("preprocess_ms"),
            normalized.get("forward_ms"),
            normalized.get("postprocess_ms"),
        ]
        if any(value is not None for value in parts):
            normalized["total_ms"] = float(sum(value for value in parts if value is not None))
    return normalized


def summarize_timing_records(timing_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({str(key) for record in timing_records for key in record})
    return {key: _summary_stats([record.get(key) for record in timing_records]) for key in keys}


def default_candidates(spade_root: Path) -> tuple[ModelCandidate, ...]:
    weights_root = Path(spade_root).resolve() / "data" / "weights"
    candidates: list[ModelCandidate] = []
    for filename in DEFAULT_MODEL_FILENAMES:
        weights = weights_root / filename
        meta = _default_meta_for_weights(weights)
        candidates.append(ModelCandidate(name=weights.stem, weights=weights, model_meta=meta))
    return tuple(candidates)


def _default_meta_for_weights(weights: Path) -> Path:
    name = weights.name
    if name.endswith(".torchscript.pt"):
        return weights.with_name(name[: -len(".torchscript.pt")] + ".torchscript.meta.json")
    return weights.with_suffix(weights.suffix + ".meta.json")


def build_candidates(
    weights: Sequence[str],
    metas: Sequence[str],
    *,
    spade_root: Path,
    names: Sequence[str] | None = None,
) -> tuple[ModelCandidate, ...]:
    if not weights:
        if metas or names:
            raise ValueError("--model-name/--model-meta require paired --weights")
        return default_candidates(spade_root)
    if metas and len(metas) != len(weights):
        raise ValueError("--model-meta count must match --weights count")
    if names and len(names) != len(weights):
        raise ValueError("--model-name count must match --weights count")
    candidate_names = _unique_names(
        list(names) if names else [Path(raw_weights).expanduser().resolve().stem for raw_weights in weights]
    )
    candidates: list[ModelCandidate] = []
    for index, raw_weights in enumerate(weights):
        weights_path = Path(raw_weights).expanduser().resolve()
        meta_path = Path(metas[index]).expanduser().resolve() if metas else _default_meta_for_weights(weights_path)
        candidates.append(
            ModelCandidate(
                name=candidate_names[index],
                weights=weights_path,
                model_meta=meta_path,
            )
        )
    return tuple(candidates)


def model_contract_status(meta: dict[str, Any]) -> tuple[bool, str | None]:
    output_names = meta.get("output_names")
    outputs = meta.get("outputs")
    if isinstance(output_names, list):
        present = {str(item) for item in output_names}
    elif isinstance(outputs, dict):
        present = {str(item) for item in outputs}
    else:
        return False, "metadata has neither output_names nor outputs"

    missing = [name for name in REQUIRED_OUTPUT_NAMES if name not in present]
    if missing:
        return False, "missing required outputs: " + ", ".join(missing)
    if not isinstance(outputs, dict):
        return True, None

    shapes: dict[str, list[Any]] = {}
    for name in REQUIRED_OUTPUT_NAMES:
        payload = outputs.get(name)
        shape = payload.get("shape") if isinstance(payload, dict) else None
        if not isinstance(shape, list) or len(shape) < 3:
            return False, f"output {name} must provide rank-3+ shape metadata"
        shapes[name] = shape
    if shapes["det"][1] != shapes["tl_attr"][1]:
        return False, "det and tl_attr query counts differ"
    tl_bits = meta.get("tl_bits", list(TL_BITS))
    if int(shapes["tl_attr"][-1]) != len(tl_bits):
        return False, "tl_attr bit dimension differs from tl_bits"
    return True, None


def _load_spade_runtime(spade_root: Path) -> Any:
    scripts_dir = Path(spade_root).resolve() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("pv26_inference_runtime")


def _selected_indices(dataset: PV26CanonicalDataset, *, split: str | None, limit: int | None) -> list[int]:
    indices: list[int] = []
    for index, record in enumerate(dataset.records):
        if split is not None and str(record.split) != str(split):
            continue
        indices.append(index)
        if limit is not None and len(indices) >= int(limit):
            break
    return indices


def _load_samples(dataset: PV26CanonicalDataset, indices: Sequence[int]) -> list[dict[str, Any]]:
    samples = [dataset[index] for index in indices]
    if not samples:
        raise ValueError("no dataset samples selected for evaluation")
    return samples


def _read_raw_image(meta: dict[str, Any]) -> np.ndarray:
    image_path = Path(str(meta["image_path"]))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read sample image: {image_path}")
    return image


def _runtime_outputs_to_prediction(meta: dict[str, Any], outputs: Any) -> dict[str, Any]:
    detections = [dict(item) for item in getattr(outputs, "detections")]
    for det_index, detection in enumerate(detections):
        if "tl_attr_scores" not in detection:
            raise ValueError(
                "runtime detection missing tl_attr_scores: "
                f"sample={meta.get('sample_id')} index={det_index}"
            )
    return {
        "meta": dict(meta),
        "detections": detections,
        "lanes": [dict(item) for item in getattr(outputs, "lanes")],
        "stop_lines": [dict(item) for item in getattr(outputs, "stop_lines")],
        "crosswalks": [dict(item) for item in getattr(outputs, "crosswalks", [])],
    }


def run_runtime_predictions(
    runtime: Any,
    samples: Sequence[dict[str, Any]],
    *,
    timings_out: list[dict[str, float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        meta = dict(sample["meta"])
        image_bgr = _read_raw_image(meta)
        outputs, timing = runtime.infer(image_bgr)
        predictions.append(_runtime_outputs_to_prediction(meta, outputs))
        if timings_out is not None:
            timings_out.append(_normalized_timing(timing))
        if progress is not None:
            progress(f"inferred {sample_index}/{len(samples)}: {meta.get('sample_id')}")
    return predictions


def supervised_task_names(raw_batch: dict[str, Any]) -> list[str]:
    source_masks = list(raw_batch.get("source_mask") or [])
    valid_masks = list(raw_batch.get("valid_mask") or [])
    tasks: list[str] = []
    if any(bool(mask.get("det")) for mask in source_masks):
        tasks.append("detector")
    if any(bool(mask.get("tl_attr")) for mask in source_masks) and any(
        bool(mask.get("tl_attr", torch.zeros(0, dtype=torch.bool)).any().item())
        for mask in valid_masks
        if isinstance(mask.get("tl_attr"), torch.Tensor)
    ):
        tasks.append("traffic_light")
    for source_name, task_name in (("lane", "lane"), ("stop_line", "stop_line"), ("crosswalk", "crosswalk")):
        if any(bool(mask.get(source_name)) for mask in source_masks):
            tasks.append(task_name)
    return tasks


def representative_scores(metrics: dict[str, Any], *, tasks: Sequence[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in tasks:
        path = TASK_SCORE_PATHS[task]
        cursor: Any = metrics
        for key in path:
            cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
        scores[task] = float(cursor) if isinstance(cursor, (int, float)) and math.isfinite(float(cursor)) else 0.0
    return scores


def composite_score(task_scores: dict[str, float]) -> float:
    if not task_scores:
        return 0.0
    return float(sum(float(value) for value in task_scores.values()) / len(task_scores))


def _counts_for_gt(gt_sample: dict[str, Any]) -> dict[str, int]:
    return {
        "detections": len(gt_sample["detections"]),
        "tl_attr": sum(1 for item in gt_sample["detections"] if bool(item.get("tl_valid"))),
        "lanes": len(gt_sample["lanes"]),
        "stop_lines": len(gt_sample["stop_lines"]),
        "crosswalks": len(gt_sample["crosswalks"]),
    }


def _counts_for_prediction(prediction: dict[str, Any]) -> dict[str, int]:
    return {
        "detections": len(prediction["detections"]),
        "lanes": len(prediction["lanes"]),
        "stop_lines": len(prediction["stop_lines"]),
        "crosswalks": len(prediction["crosswalks"]),
    }


def _sample_score_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = _finite_floats([record.get("composite_score") for record in records])
    summary = _summary_stats(scores)
    ranked = sorted(
        records,
        key=lambda item: float(item.get("composite_score", 0.0))
        if isinstance(item.get("composite_score"), (int, float))
        else 0.0,
    )
    summary["worst_samples"] = [
        {
            "sample_id": item.get("sample_id"),
            "dataset_key": item.get("dataset_key"),
            "composite_score": item.get("composite_score"),
            "representative_scores": item.get("representative_scores", {}),
        }
        for item in ranked[:10]
    ]
    summary["best_samples"] = [
        {
            "sample_id": item.get("sample_id"),
            "dataset_key": item.get("dataset_key"),
            "composite_score": item.get("composite_score"),
            "representative_scores": item.get("representative_scores", {}),
        }
        for item in reversed(ranked[-10:])
    ]
    return summary


def write_predictions_jsonl(
    path: Path,
    *,
    samples: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    timings: Sequence[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_records: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as fp:
        for sample_index, (sample, prediction) in enumerate(zip(samples, predictions)):
            raw_batch = collate_pv26_samples([sample])
            gt_sample = _extract_gt_samples(raw_batch)[0]
            sample_metrics = summarize_pv26_metrics([prediction], raw_batch)
            tasks = supervised_task_names(raw_batch)
            scores = representative_scores(sample_metrics, tasks=tasks)
            timing = dict(timings[sample_index]) if timings is not None and sample_index < len(timings) else {}
            record = {
                "sample_id": sample["meta"]["sample_id"],
                "dataset_key": sample["meta"]["dataset_key"],
                "split": sample["meta"]["split"],
                "image_path": sample["meta"]["image_path"],
                "gt_counts": _counts_for_gt(gt_sample),
                "prediction_counts": _counts_for_prediction(prediction),
                "representative_scores": scores,
                "composite_score": composite_score(scores),
                "timing_ms": timing,
                "predictions": {
                    "detections": prediction["detections"],
                    "lanes": prediction["lanes"],
                    "stop_lines": prediction["stop_lines"],
                    "crosswalks": prediction["crosswalks"],
                },
            }
            fp.write(json.dumps(_jsonable(record), ensure_ascii=True, sort_keys=True) + "\n")
            compact_records.append({key: value for key, value in record.items() if key != "predictions"})
    return compact_records


def _sample_scores(
    samples: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> list[tuple[float, int, dict[str, Any]]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for sample_index, (sample, prediction) in enumerate(zip(samples, predictions)):
        raw_batch = collate_pv26_samples([sample])
        metrics = summarize_pv26_metrics([prediction], raw_batch)
        scores = representative_scores(metrics, tasks=supervised_task_names(raw_batch))
        scored.append((composite_score(scores), sample_index, metrics))
    return sorted(scored, key=lambda item: item[0])


def _draw_box(
    image: np.ndarray,
    box: Sequence[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_polyline(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    color: tuple[int, int, int],
    closed: bool = False,
) -> None:
    if len(points) < 2:
        return
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
    cv2.polylines(image, [array], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def write_failure_overlays(
    output_dir: Path,
    *,
    samples: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    count: int,
) -> list[str]:
    if count <= 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rank, (score, sample_index, _metrics) in enumerate(_sample_scores(samples, predictions)[:count], start=1):
        sample = samples[sample_index]
        prediction = predictions[sample_index]
        raw_batch = collate_pv26_samples([sample])
        gt_sample = _extract_gt_samples(raw_batch)[0]
        image = _read_raw_image(sample["meta"])
        for det in gt_sample["detections"]:
            _draw_box(image, det["box_xyxy"], (0, 0, 255), f"GT {det['class_name']}")
        for det in prediction["detections"]:
            _draw_box(
                image,
                det["box_xyxy"],
                (0, 255, 0),
                f"P {det['class_name']} {float(det.get('score', 0.0)):.2f}",
            )
        for lane in gt_sample["lanes"]:
            _draw_polyline(image, lane["points_xy"], (0, 0, 255))
        for lane in prediction["lanes"]:
            _draw_polyline(image, lane["points_xy"], (0, 255, 0))
        for stop_line in gt_sample["stop_lines"]:
            _draw_polyline(image, stop_line["points_xy"], (0, 0, 255))
        for stop_line in prediction["stop_lines"]:
            _draw_polyline(image, stop_line["points_xy"], (0, 255, 0))
        for crosswalk in gt_sample["crosswalks"]:
            _draw_polyline(image, crosswalk["points_xy"], (0, 0, 255), closed=True)
        for crosswalk in prediction["crosswalks"]:
            _draw_polyline(image, crosswalk["points_xy"], (0, 255, 0), closed=True)
        cv2.putText(
            image,
            f"rank={rank} composite={score:.3f} red=GT green=pred",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = output_dir / f"{rank:03d}_{_safe_name(str(sample['meta']['sample_id']))}.jpg"
        cv2.imwrite(str(path), image)
        written.append(str(path))
    return written


def _metric_value(payload: dict[str, Any], path: Sequence[str], default: float = 0.0) -> float:
    cursor: Any = payload
    for key in path:
        cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
    number = _finite_float(cursor)
    return float(default) if number is None else number


def _plot_grouped_bars(
    ax: Any,
    *,
    names: Sequence[str],
    groups: Sequence[str],
    values_by_name: dict[str, Sequence[float]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(groups), dtype=np.float32)
    width = min(0.8 / max(len(names), 1), 0.25)
    offsets = (np.arange(len(names), dtype=np.float32) - (len(names) - 1) / 2.0) * width
    for name, offset in zip(names, offsets):
        ax.bar(x + offset, list(values_by_name.get(name, [])), width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(names) > 1:
        ax.legend(fontsize=8)


def _write_summary_metrics_plot(output_dir: Path, evaluated: dict[str, dict[str, Any]]) -> str | None:
    names = list(evaluated)
    task_names = sorted({task for summary in evaluated.values() for task in summary.get("representative_scores", {})})
    if not names or not task_names:
        return None
    import matplotlib.pyplot as plt

    values_by_name = {
        name: [float(evaluated[name].get("representative_scores", {}).get(task, 0.0)) for task in task_names]
        for name in names
    }
    fig, ax = plt.subplots(figsize=(max(6.5, len(task_names) * max(len(names), 1) * 0.7), 4.2))
    _plot_grouped_bars(
        ax,
        names=names,
        groups=task_names,
        values_by_name=values_by_name,
        ylabel="score",
        ylim=(0.0, 1.0),
    )
    ax.set_title("Task representative scores")
    fig.tight_layout()
    path = output_dir / "summary_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _write_detector_per_class_plot(output_dir: Path, evaluated: dict[str, dict[str, Any]]) -> str | None:
    names = list(evaluated)
    classes = sorted(
        {
            class_name
            for summary in evaluated.values()
            for class_name in (
                summary.get("metrics", {})
                .get("detector", {})
                .get("per_class", {})
                .keys()
            )
        }
    )
    if not names or not classes:
        return None
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, len(classes) * max(len(names), 1) * 0.45), 7.0), sharex=True)
    for ax, metric_name in zip(axes, ("ap50", "f1")):
        values_by_name = {
            name: [
                _metric_value(
                    evaluated[name],
                    ("metrics", "detector", "per_class", class_name, metric_name),
                )
                for class_name in classes
            ]
            for name in names
        }
        _plot_grouped_bars(
            ax,
            names=names,
            groups=classes,
            values_by_name=values_by_name,
            ylabel=metric_name,
            ylim=(0.0, 1.0),
        )
        ax.set_title(f"Detector per-class {metric_name}")
    fig.tight_layout()
    path = output_dir / "detector_per_class.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _hist_values(summary: dict[str, Any], path: Sequence[str]) -> list[float]:
    cursor: Any = summary.get("_plot_histograms", summary.get("histograms", {}))
    for key in path:
        cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
    return _finite_floats(cursor)


def _write_task_error_distribution_plot(output_dir: Path, evaluated: dict[str, dict[str, Any]]) -> str | None:
    import matplotlib.pyplot as plt

    specs = (
        ("detector IoU", ("detector", "matched_positive_iou"), (0.0, 1.0)),
        ("detector conf", ("detector", "prediction_confidence"), (0.0, 1.0)),
        ("TL attr conf", ("traffic_light", "attr_confidence"), (0.0, 1.0)),
        ("lane dist px", ("lane", "mean_point_distance"), None),
        ("stop angle deg", ("stop_line", "mean_angle_error"), None),
        ("crosswalk IoU", ("crosswalk", "mean_polygon_iou"), (0.0, 1.0)),
    )
    if not any(_hist_values(summary, path) for summary in evaluated.values() for _, path, _ in specs):
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5))
    for ax, (title, path, xlim) in zip(axes.flat, specs):
        plotted = False
        for name, summary in evaluated.items():
            values = _hist_values(summary, path)
            if not values:
                continue
            ax.hist(values, bins=min(30, max(5, int(math.sqrt(len(values))) + 1)), alpha=0.45, label=name)
            plotted = True
        ax.set_title(title)
        ax.set_ylabel("count")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if plotted and len(evaluated) > 1:
            ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "task_error_distributions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _write_sample_score_distribution_plot(output_dir: Path, evaluated: dict[str, dict[str, Any]]) -> str | None:
    import matplotlib.pyplot as plt

    score_by_name = {
        name: _finite_floats([record.get("composite_score") for record in summary.get("_plot_sample_records", [])])
        for name, summary in evaluated.items()
    }
    score_by_name = {name: values for name, values in score_by_name.items() if values}
    if not score_by_name:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for name, values in score_by_name.items():
        axes[0].hist(values, bins=min(30, max(5, int(math.sqrt(len(values))) + 1)), alpha=0.45, label=name)
    axes[0].set_title("Sample composite distribution")
    axes[0].set_xlabel("composite score")
    axes[0].set_ylabel("count")
    axes[0].set_xlim(0.0, 1.0)
    if len(score_by_name) > 1:
        axes[0].legend(fontsize=8)
    axes[1].boxplot(
        list(score_by_name.values()),
        tick_labels=list(score_by_name),
        orientation="vertical",
    )
    axes[1].set_title("Sample composite spread")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output_dir / "sample_score_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _write_latency_plot(output_dir: Path, evaluated: dict[str, dict[str, Any]]) -> str | None:
    import matplotlib.pyplot as plt

    names = [
        name
        for name, summary in evaluated.items()
        if any((stats.get("count") or 0) > 0 for stats in summary.get("timing_summary", {}).values())
    ]
    if not names:
        return None
    components = ("preprocess_ms", "forward_ms", "postprocess_ms")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    bottom = np.zeros(len(names), dtype=np.float64)
    x = np.arange(len(names), dtype=np.float32)
    for component in components:
        values = [
            _metric_value(evaluated[name], ("timing_summary", component, "mean"))
            for name in names
        ]
        axes[0].bar(x, values, bottom=bottom, label=component.replace("_ms", ""))
        bottom += np.asarray(values, dtype=np.float64)
    axes[0].set_title("Mean latency components")
    axes[0].set_ylabel("ms")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].legend(fontsize=8)

    values_by_name = {
        name: [
            _metric_value(evaluated[name], ("timing_summary", "total_ms", percentile))
            for percentile in ("p50", "p95", "p99")
        ]
        for name in names
    }
    _plot_grouped_bars(
        axes[1],
        names=names,
        groups=("p50", "p95", "p99"),
        values_by_name=values_by_name,
        ylabel="total latency ms",
    )
    axes[1].set_title("Total latency percentiles")
    fig.tight_layout()
    path = output_dir / "latency.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def write_metric_plots(output_dir: Path, model_summaries: dict[str, dict[str, Any]]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluated = {
        name: summary for name, summary in model_summaries.items() if summary.get("status") == "evaluated"
    }
    if not evaluated:
        return []

    written: list[str] = []
    names = list(evaluated)
    composites = [float(evaluated[name]["composite_score"]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 1.6), 4.0))
    ax.bar(names, composites, color="#3874b8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("composite")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output_dir / "composite_scores.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    task_names = sorted({task for summary in evaluated.values() for task in summary.get("representative_scores", {})})
    for task in task_names:
        values = [float(evaluated[name].get("representative_scores", {}).get(task, 0.0)) for name in names]
        fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 1.6), 4.0))
        ax.bar(names, values, color="#4b9f68")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(task)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = output_dir / f"{_safe_name(task)}_scores.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    for plot_path in (
        _write_summary_metrics_plot(output_dir, evaluated),
        _write_detector_per_class_plot(output_dir, evaluated),
        _write_task_error_distribution_plot(output_dir, evaluated),
        _write_sample_score_distribution_plot(output_dir, evaluated),
        _write_latency_plot(output_dir, evaluated),
    ):
        if plot_path is not None:
            written.append(plot_path)
    return written


def _instantiate_runtime(runtime_module: Any, candidate: ModelCandidate, options: HarnessOptions) -> Any:
    return runtime_module.Pv26InferenceRuntime(
        weights=str(candidate.weights),
        model_meta=str(candidate.model_meta),
        pv26_repo_root=str(options.pv26_repo_root),
        device_name=str(options.device_name),
        det_conf_thres=float(options.det_conf_thres),
        det_iou_thres=float(options.det_iou_thres),
        lane_obj_thres=float(options.lane_obj_thres),
        lane_visibility_thres=float(options.lane_visibility_thres),
        stop_line_obj_thres=float(options.stop_line_obj_thres),
    )


def evaluate_models(
    options: HarnessOptions,
    *,
    runtime_module: Any | None = None,
    runtime_factory: Callable[[ModelCandidate, HarnessOptions], Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dataset = PV26CanonicalDataset([options.dataset_root])
    indices = _selected_indices(dataset, split=options.split, limit=options.limit)
    samples = _load_samples(dataset, indices)
    raw_batch = collate_pv26_samples(samples)
    sample_count = len(samples)
    supervised_tasks = supervised_task_names(raw_batch)
    options.output_root.mkdir(parents=True, exist_ok=True)

    if runtime_module is None and runtime_factory is None:
        runtime_module = _load_spade_runtime(options.spade_root)

    model_summaries: dict[str, dict[str, Any]] = {}
    model_names = _unique_names([candidate.name for candidate in options.candidates])
    for candidate, model_name in zip(options.candidates, model_names):
        if not candidate.weights.is_file():
            model_summaries[model_name] = {
                "status": "skipped",
                "reason": f"weights not found: {candidate.weights}",
                "weights": str(candidate.weights),
                "model_meta": str(candidate.model_meta),
            }
            continue
        if not candidate.model_meta.is_file():
            model_summaries[model_name] = {
                "status": "skipped",
                "reason": f"model metadata not found: {candidate.model_meta}",
                "weights": str(candidate.weights),
                "model_meta": str(candidate.model_meta),
            }
            continue
        meta = _read_json(candidate.model_meta)
        supported, reason = model_contract_status(meta)
        if not supported:
            model_summaries[model_name] = {
                "status": "skipped",
                "reason": reason,
                "weights": str(candidate.weights),
                "model_meta": str(candidate.model_meta),
            }
            continue

        if progress is not None:
            progress(f"evaluating {model_name} on {sample_count} samples")
        runtime = (
            runtime_factory(candidate, options)
            if runtime_factory is not None
            else _instantiate_runtime(runtime_module, candidate, options)
        )
        timing_records: list[dict[str, float]] = []
        predictions = run_runtime_predictions(runtime, samples, timings_out=timing_records, progress=progress)
        metrics = summarize_pv26_metrics(predictions, raw_batch)
        histograms = summarize_pv26_tensorboard_histograms(predictions, raw_batch)
        scores = representative_scores(metrics, tasks=supervised_tasks)
        pred_path = options.output_root / "predictions" / f"{model_name}.jsonl"
        sample_records = write_predictions_jsonl(
            pred_path,
            samples=samples,
            predictions=predictions,
            timings=timing_records,
        )
        overlay_paths: list[str] = []
        if options.write_overlays:
            overlay_paths = write_failure_overlays(
                options.output_root / "overlays" / model_name,
                samples=samples,
                predictions=predictions,
                count=int(options.overlay_count),
            )
        model_summaries[model_name] = {
            "status": "evaluated",
            "weights": str(candidate.weights),
            "model_meta": str(candidate.model_meta),
            "sample_count": int(sample_count),
            "supervised_tasks": list(supervised_tasks),
            "metrics": metrics,
            "histogram_summary": _summarize_histograms(histograms),
            "representative_scores": scores,
            "composite_score": composite_score(scores),
            "sample_score_summary": _sample_score_summary(sample_records),
            "timing_summary": summarize_timing_records(timing_records),
            "predictions_jsonl": str(pred_path),
            "overlays": overlay_paths,
            "_plot_histograms": histograms,
            "_plot_sample_records": sample_records,
        }

    plot_paths: list[str] = []
    if options.write_plots:
        plot_paths = write_metric_plots(options.output_root / "plots", model_summaries)

    summary_path = options.output_root / "summary.json"
    public_model_summaries = {
        name: {
            key: value
            for key, value in model_summary.items()
            if not str(key).startswith("_plot_")
        }
        for name, model_summary in model_summaries.items()
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(options.dataset_root),
        "output_root": str(options.output_root),
        "summary_json": str(summary_path),
        "sample_count": int(sample_count),
        "split": options.split,
        "limit": options.limit,
        "models": public_model_summaries,
        "plots": plot_paths,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(_jsonable(summary), fp, indent=2, ensure_ascii=True, sort_keys=True)
        fp.write("\n")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOPV26 TorchScript models on a canonical PV26 dataset root."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="PV26 canonical/eval dataset root.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory for summary, predictions, plots, and overlays.",
    )
    parser.add_argument(
        "--spade-root",
        type=Path,
        default=REPO_ROOT.parent / "spade",
        help="spade package root containing scripts/ and data/weights/.",
    )
    parser.add_argument(
        "--pv26-repo-root",
        type=Path,
        default=REPO_ROOT,
        help="yolopv26 repository root for named-output postprocess imports.",
    )
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        help="TorchScript weights path. Repeat for multiple models.",
    )
    parser.add_argument(
        "--model-meta",
        action="append",
        default=[],
        help="Metadata JSON path paired with --weights. Repeat in the same order.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Summary/output name paired with --weights. Repeat in the same order.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to Pv26InferenceRuntime, e.g. auto, cpu, cuda:0.",
    )
    parser.add_argument("--split", default=None, help="Optional dataset split filter such as val.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of selected samples.")
    parser.add_argument("--overlay-count", type=int, default=10, help="Number of worst samples to visualize per model.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--no-overlays", action="store_true", help="Skip overlay image generation.")
    parser.add_argument("--det-conf-thres", type=float, default=0.30)
    parser.add_argument("--det-iou-thres", type=float, default=0.70)
    parser.add_argument("--lane-obj-thres", type=float, default=0.50)
    parser.add_argument("--lane-visibility-thres", type=float, default=0.50)
    parser.add_argument("--stop-line-obj-thres", type=float, default=0.50)
    return parser


def options_from_args(args: argparse.Namespace) -> HarnessOptions:
    spade_root = Path(args.spade_root).expanduser().resolve()
    return HarnessOptions(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        candidates=build_candidates(
            args.weights,
            args.model_meta,
            spade_root=spade_root,
            names=args.model_name,
        ),
        spade_root=spade_root,
        pv26_repo_root=Path(args.pv26_repo_root).expanduser().resolve(),
        device_name=str(args.device),
        split=args.split,
        limit=args.limit,
        overlay_count=int(args.overlay_count),
        write_plots=not bool(args.no_plots),
        write_overlays=not bool(args.no_overlays),
        det_conf_thres=float(args.det_conf_thres),
        det_iou_thres=float(args.det_iou_thres),
        lane_obj_thres=float(args.lane_obj_thres),
        lane_visibility_thres=float(args.lane_visibility_thres),
        stop_line_obj_thres=float(args.stop_line_obj_thres),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = evaluate_models(options_from_args(args), progress=lambda message: print(message, flush=True))
    evaluated = [name for name, item in summary["models"].items() if item.get("status") == "evaluated"]
    if not evaluated:
        print(f"no compatible models were evaluated; summary={summary['summary_json']}", file=sys.stderr)
        return 2
    print(f"wrote summary: {summary['summary_json']}")
    return 0


if str(REPO_ROOT) not in sys.path:
    site.addsitedir(str(REPO_ROOT))


__all__ = [
    "HarnessOptions",
    "ModelCandidate",
    "build_candidates",
    "composite_score",
    "default_candidates",
    "evaluate_models",
    "main",
    "model_contract_status",
    "representative_scores",
    "run_runtime_predictions",
    "supervised_task_names",
]
