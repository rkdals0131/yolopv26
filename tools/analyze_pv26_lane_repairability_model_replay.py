from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FEATURES = (
    "pred_point_count",
    "pred_polyline_length",
    "pred_bbox_width",
    "pred_bbox_height",
    "pred_bbox_aspect",
    "pred_track_pixels",
    "pred_center_mask_mean",
    "pred_center_mask_q10",
    "pred_center_mask_active05",
    "pred_center_point_mean",
    "pred_center_point_q10",
    "pred_center_point_active05",
    "pred_center_point_low_run05",
    "pred_support_point_mean",
    "pred_support_point_q10",
    "pred_support_mask_mean",
    "nearest_other_pred_distance",
    "sample_pred_lane_count",
)

DEFAULT_LABELS = (
    "repairable_le80_center050",
    "repairable_le120_any_center",
)

DEFAULT_BASELINE_TP = 4518
DEFAULT_BASELINE_FP = 2206
DEFAULT_BASELINE_FN = 4959


@dataclass(frozen=True)
class BaselineCounts:
    tp: int = DEFAULT_BASELINE_TP
    fp: int = DEFAULT_BASELINE_FP
    fn: int = DEFAULT_BASELINE_FN


@dataclass(frozen=True)
class LogisticModel:
    features: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only learned replay for lane unmatched-prediction repairability. "
            "It trains a small no-GT feature ranker out-of-fold, then replays "
            "how many existing false-positive lanes would become true positives "
            "under GT-derived repair labels."
        )
    )
    parser.add_argument("--unmatched-rows", required=True, help="lane_unmatched_prediction_repair_rows.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-tp", type=int, default=DEFAULT_BASELINE_TP)
    parser.add_argument("--baseline-fp", type=int, default=DEFAULT_BASELINE_FP)
    parser.add_argument("--baseline-fn", type=int, default=DEFAULT_BASELINE_FN)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser.parse_args()


def _float_value(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sample_key(row: dict[str, str]) -> int:
    batch = int(_float_value(row.get("batch_index"), 0.0))
    sample = int(_float_value(row.get("sample_index"), _float_value(row.get("sample_batch_index"), 0.0)))
    return batch * 10_000 + sample


def _split_id(row: dict[str, str]) -> int:
    return _sample_key(row) % 2


def _safe_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom <= 0 else float((2 * tp) / denom)


def _roc_auc(labels: list[bool], scores: list[float]) -> float:
    pairs = [(float(score), bool(label)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives <= 0 or negatives <= 0:
        return math.nan
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        index = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(labels: list[bool], scores: list[float]) -> float:
    pairs = [(float(score), bool(label)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = sum(1 for _, label in pairs if label)
    if positives <= 0:
        return math.nan
    pairs.sort(key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            precision_sum += float(tp / rank)
    return float(precision_sum / positives)


def _feature_matrix(
    rows: list[dict[str, str]], features: tuple[str, ...], means: tuple[float, ...] | None = None, scales: tuple[float, ...] | None = None
) -> tuple[list[list[float]], tuple[float, ...], tuple[float, ...]]:
    raw_columns: list[list[float]] = []
    for feature in features:
        values = [_float_value(row.get(feature)) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        fill_value = 0.0 if not finite_values else sum(finite_values) / len(finite_values)
        raw_columns.append([value if math.isfinite(value) else fill_value for value in values])

    if means is None:
        means = tuple(sum(column) / len(column) if column else 0.0 for column in raw_columns)
    if scales is None:
        computed_scales: list[float] = []
        for column, mean in zip(raw_columns, means):
            variance = sum((value - mean) ** 2 for value in column) / max(1, len(column))
            scale = math.sqrt(variance)
            computed_scales.append(scale if scale > 1e-6 else 1.0)
        scales = tuple(computed_scales)

    matrix: list[list[float]] = []
    for row_index in range(len(rows)):
        matrix.append(
            [
                (raw_columns[col_index][row_index] - means[col_index]) / scales[col_index]
                for col_index in range(len(features))
            ]
        )
    return matrix, means, scales


def _train_logistic(
    rows: list[dict[str, str]],
    *,
    label_name: str,
    features: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    l2: float,
) -> LogisticModel:
    labels = [1.0 if _bool_value(row.get(label_name)) else 0.0 for row in rows]
    matrix, means, scales = _feature_matrix(rows, features)
    if not matrix:
        return LogisticModel(features=features, means=means, scales=scales, weights=tuple(0.0 for _ in features), bias=0.0)

    positive_rate = min(0.99, max(0.01, sum(labels) / len(labels)))
    bias = math.log(positive_rate / (1.0 - positive_rate))
    weights = [0.0 for _ in features]
    n_rows = float(len(matrix))

    for _ in range(max(1, epochs)):
        grad_weights = [0.0 for _ in weights]
        grad_bias = 0.0
        for values, label in zip(matrix, labels):
            score = bias + sum(weight * value for weight, value in zip(weights, values))
            error = _safe_sigmoid(score) - label
            grad_bias += error
            for index, value in enumerate(values):
                grad_weights[index] += error * value
        bias -= learning_rate * grad_bias / n_rows
        for index, weight in enumerate(weights):
            grad = grad_weights[index] / n_rows + l2 * weight
            weights[index] -= learning_rate * grad

    return LogisticModel(features=features, means=means, scales=scales, weights=tuple(weights), bias=bias)


def _score_rows(model: LogisticModel, rows: list[dict[str, str]]) -> list[float]:
    matrix, _, _ = _feature_matrix(rows, model.features, model.means, model.scales)
    return [
        _safe_sigmoid(model.bias + sum(weight * value for weight, value in zip(model.weights, values)))
        for values in matrix
    ]


def _oof_scores(
    rows: list[dict[str, str]],
    *,
    label_name: str,
    features: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores = [math.nan for _ in rows]
    model_rows: list[dict[str, Any]] = []
    for holdout_split in (0, 1):
        train_indexed = [(index, row) for index, row in enumerate(rows) if _split_id(row) != holdout_split]
        holdout_indexed = [(index, row) for index, row in enumerate(rows) if _split_id(row) == holdout_split]
        train_rows = [row for _, row in train_indexed]
        holdout_rows = [row for _, row in holdout_indexed]
        model = _train_logistic(
            train_rows,
            label_name=label_name,
            features=features,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        holdout_scores = _score_rows(model, holdout_rows)
        for (index, _), score in zip(holdout_indexed, holdout_scores):
            scores[index] = score
        model_rows.extend(_model_weight_rows(model, label_name=label_name, holdout_split=holdout_split))
    return scores, model_rows


def _model_weight_rows(model: LogisticModel, *, label_name: str, holdout_split: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature, weight in sorted(zip(model.features, model.weights), key=lambda item: abs(item[1]), reverse=True):
        rows.append(
            {
                "label": label_name,
                "holdout_split": holdout_split,
                "feature": feature,
                "weight": weight,
            }
        )
    return rows


def _model_parameters(model: LogisticModel) -> dict[str, Any]:
    return {
        "features": list(model.features),
        "means": list(model.means),
        "scales": list(model.scales),
        "weights": list(model.weights),
        "bias": float(model.bias),
    }


def _topk_values(row_count: int, positive_count: int) -> list[int]:
    candidates = {
        max(1, int(round(positive_count * ratio)))
        for ratio in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
        if positive_count > 0
    }
    candidates.update({50, 100, 250, 500, 750, 1000, positive_count, row_count})
    return sorted(k for k in candidates if 0 < k <= row_count)


def _replay_topk(
    *,
    labels: list[bool],
    scores: list[float],
    baseline: BaselineCounts,
    topk_values: list[int],
) -> list[dict[str, Any]]:
    pairs = [(score, label) for score, label in zip(scores, labels) if math.isfinite(score)]
    pairs.sort(key=lambda item: item[0], reverse=True)
    total_positive = sum(1 for _, label in pairs if label)
    rows: list[dict[str, Any]] = []
    for topk in topk_values:
        selected = pairs[:topk]
        repaired = sum(1 for _, label in selected if label)
        tp = baseline.tp + repaired
        fp = max(0, baseline.fp - repaired)
        fn = max(0, baseline.fn - repaired)
        rows.append(
            {
                "topk": int(len(selected)),
                "selected_repairable": int(repaired),
                "selected_not_repairable": int(len(selected) - repaired),
                "selection_precision": 0.0 if not selected else float(repaired / len(selected)),
                "repair_recall": 0.0 if total_positive <= 0 else float(repaired / total_positive),
                "lane_tp": int(tp),
                "lane_fp": int(fp),
                "lane_fn": int(fn),
                "lane_f1": _f1(tp, fp, fn),
            }
        )
    return rows


def build_model_replay(
    rows: list[dict[str, str]],
    *,
    baseline: BaselineCounts = BaselineCounts(),
    features: tuple[str, ...] = DEFAULT_FEATURES,
    label_names: tuple[str, ...] = DEFAULT_LABELS,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict[str, Any]:
    label_summaries: dict[str, Any] = {}
    replay_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for label_name in label_names:
        labels = [_bool_value(row.get(label_name)) for row in rows]
        positive_count = sum(1 for label in labels if label)
        scores, model_rows = _oof_scores(
            rows,
            label_name=label_name,
            features=features,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        label_replay_rows = _replay_topk(
            labels=labels,
            scores=scores,
            baseline=baseline,
            topk_values=_topk_values(len(rows), positive_count),
        )
        full_model = _train_logistic(
            rows,
            label_name=label_name,
            features=features,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        for row in label_replay_rows:
            row["label"] = label_name
        best_replay = max(label_replay_rows, key=lambda row: float(row["lane_f1"])) if label_replay_rows else None
        positive_budget = next((row for row in label_replay_rows if row["topk"] == positive_count), None)
        label_summaries[label_name] = {
            "positive_count": int(positive_count),
            "row_count": int(len(rows)),
            "oof_auc": _roc_auc(labels, scores),
            "oof_average_precision": _average_precision(labels, scores),
            "positive_count_budget_replay": positive_budget,
            "best_replay_by_f1": best_replay,
            "full_model": _model_parameters(full_model),
        }
        replay_rows.extend(label_replay_rows)
        weight_rows.extend(model_rows)

    return {
        "row_count": int(len(rows)),
        "features": list(features),
        "baseline": {
            "lane_tp": int(baseline.tp),
            "lane_fp": int(baseline.fp),
            "lane_fn": int(baseline.fn),
            "lane_f1": _f1(baseline.tp, baseline.fp, baseline.fn),
        },
        "labels": label_summaries,
        "replay_rows": replay_rows,
        "model_weight_rows": weight_rows,
        "interpretation": (
            "Read-only GT-labeled premise audit. Scores are out-of-fold no-GT feature ranks. "
            "Replay assumes a selected repairable unmatched prediction can replace one existing FP "
            "with one TP and remove one FN. This is not production lane success."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    *,
    unmatched_rows: Path,
    output_dir: Path,
    baseline: BaselineCounts = BaselineCounts(),
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict[str, Any]:
    with unmatched_rows.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    summary = build_model_replay(
        rows,
        baseline=baseline,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    summary["unmatched_rows"] = str(unmatched_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "repairability_model_replay.csv", list(summary["replay_rows"]))
    _write_csv(output_dir / "repairability_model_weights.csv", list(summary["model_weight_rows"]))
    parameters = {
        label_name: payload["full_model"]
        for label_name, payload in dict(summary["labels"]).items()
        if isinstance(payload, dict) and isinstance(payload.get("full_model"), dict)
    }
    (output_dir / "repairability_model_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_audit(
        unmatched_rows=Path(args.unmatched_rows).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        baseline=BaselineCounts(tp=args.baseline_tp, fp=args.baseline_fp, fn=args.baseline_fn),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
