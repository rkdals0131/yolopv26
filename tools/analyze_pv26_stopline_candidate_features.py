from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_INPUT = SOURCE_RUN / "analysis_exports" / "stopline_candidate_pool_val512_epoch2" / "candidate_features.csv"
BASE_FEATURE_NAMES = (
    "score",
    "length",
    "proposal_rank",
    "inverse_rank",
    "score_x_length",
    "log1p_length",
)
RAW_BASE_FEATURE_NAMES = {"score", "length", "proposal_rank"}
EXCLUDED_FEATURE_COLUMNS = {
    "batch_index",
    "sample_index",
    "proposal_source",
    "nearest_gt_distance",
    "nearest_gt_angle_error",
    "nearest_gt_index",
    "is_oracle_positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether existing non-GT stop-line candidate features "
            "can separate oracle-positive from false-positive candidates."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_label(row: dict[str, Any]) -> int:
    value = str(row.get("is_oracle_positive", "")).strip().lower()
    return int(value in {"1", "true", "yes", "y"})


def _can_parse_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _extra_numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for key in rows[0]:
        if key in EXCLUDED_FEATURE_COLUMNS or key in RAW_BASE_FEATURE_NAMES or key in BASE_FEATURE_NAMES:
            continue
        values = [row.get(key, "") for row in rows]
        non_empty = [value for value in values if str(value).strip()]
        if non_empty and all(_can_parse_float(value) for value in non_empty):
            names.append(key)
    return names


def _feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    extra_names = _extra_numeric_feature_names(rows)
    names = list(BASE_FEATURE_NAMES) + extra_names
    matrix: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        score = _float(row, "score")
        length = _float(row, "length")
        rank = max(1.0, _float(row, "proposal_rank", 1.0))
        values = [
            score,
            length,
            rank,
            1.0 / rank,
            score * length,
            float(np.log1p(max(0.0, length))),
        ]
        values.extend(_float(row, name) for name in extra_names)
        matrix.append(values)
        labels.append(_bool_label(row))
    return np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=np.float64), names


def _split_by_batch(rows: list[dict[str, Any]], train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    batch_indices = np.asarray([int(_float(row, "batch_index", 0.0)) for row in rows], dtype=np.int64)
    if batch_indices.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    min_batch = int(batch_indices.min())
    max_batch = int(batch_indices.max())
    cutoff = min_batch + int(round((max_batch - min_batch + 1) * float(train_fraction))) - 1
    train_mask = batch_indices <= cutoff
    test_mask = ~train_mask
    return train_mask, test_mask


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return (x - mean) / std, mean, std


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, *, steps: int, lr: float) -> tuple[np.ndarray, float]:
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    pos = max(float(train_y.sum()), 1.0)
    neg = max(float(train_y.shape[0] - train_y.sum()), 1.0)
    sample_weights = np.where(train_y > 0.5, 0.5 / pos, 0.5 / neg)
    for _ in range(max(1, int(steps))):
        probs = _sigmoid(x @ weights)
        grad = x.T @ ((probs - train_y) * sample_weights)
        weights -= float(lr) * grad
    return weights[1:], float(weights[0])


def _predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return _sigmoid(x @ weights + float(bias))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0.5
    n_pos = int(positives.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.shape[0] + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[positives].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0.5
    n_pos = int(positives.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = positives[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
    return float((precision * sorted_labels).sum() / float(n_pos))


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= float(threshold)
    actual = labels > 0.5
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1.0e-12, precision + recall))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_count": int(predicted.sum()),
    }


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    if scores.size == 0:
        return _threshold_metrics(scores, labels, 1.0)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best = _threshold_metrics(scores, labels, float(candidates[0]))
    for threshold in candidates:
        metrics = _threshold_metrics(scores, labels, float(threshold))
        if (float(metrics["f1"]), float(metrics["precision"])) > (float(best["f1"]), float(best["precision"])):
            best = metrics
    return best


def _score_report(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        selected = _best_threshold(scores, labels)
    else:
        selected = _threshold_metrics(scores, labels, threshold)
    return {
        "auc": float(_auc(scores, labels)),
        "average_precision": float(_average_precision(scores, labels)),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "count": int(labels.shape[0]),
        "positive_count": int(labels.sum()),
        "threshold": selected,
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else input_path.parent / "candidate_feature_validator"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_path)
    features, labels, feature_names = _feature_matrix(rows)
    train_mask, test_mask = _split_by_batch(rows, args.train_fraction)
    train_x = features[train_mask]
    train_y = labels[train_mask]
    test_x = features[test_mask]
    test_y = labels[test_mask]
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError("candidate feature audit requires non-empty train and test splits")
    train_x_std, mean, std = _standardize(train_x, train_x)
    test_x_std = (test_x - mean) / std
    weights, bias = _fit_logistic(train_x_std, train_y, steps=args.steps, lr=args.lr)
    train_scores = _predict(train_x_std, weights, bias)
    test_scores = _predict(test_x_std, weights, bias)
    train_threshold = float(_best_threshold(train_scores, train_y)["threshold"])

    feature_reports: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        raw_train = train_x[:, index]
        raw_test = test_x[:, index]
        if name == "proposal_rank":
            raw_train = -raw_train
            raw_test = -raw_test
        best_train = _best_threshold(raw_train, train_y)
        feature_reports.append(
            {
                "feature": name,
                "train_auc": _auc(raw_train, train_y),
                "train_ap": _average_precision(raw_train, train_y),
                "train_best_f1": best_train["f1"],
                "train_threshold": best_train["threshold"],
                "test_auc": _auc(raw_test, test_y),
                "test_ap": _average_precision(raw_test, test_y),
                "test_f1_at_train_threshold": _threshold_metrics(raw_test, test_y, float(best_train["threshold"]))["f1"],
            }
        )

    summary = {
        "input": str(input_path),
        "row_count": int(len(rows)),
        "feature_names": feature_names,
        "train": _score_report(train_scores, train_y),
        "test": _score_report(test_scores, test_y, threshold=train_threshold),
        "test_oracle_best_threshold": _best_threshold(test_scores, test_y),
        "weights": {name: float(weight) for name, weight in zip(feature_names, weights)},
        "bias": float(bias),
        "split": {
            "train_fraction": float(args.train_fraction),
            "train_rows": int(train_y.shape[0]),
            "test_rows": int(test_y.shape[0]),
            "train_positive_count": int(train_y.sum()),
            "test_positive_count": int(test_y.sum()),
        },
        "feature_reports": feature_reports,
        "interpretation": (
            "This is a candidate-level feature audit, not a production stop-line decoder. "
            "Good candidate-level AUC/AP would only justify a learned instance validator; it does not prove task F1."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "feature_reports.csv", feature_reports)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
