from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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

from model.engine.metrics import STOP_LINE_POINT_COUNT, _hungarian_from_cost, _mean_point_distance, _segment_angle_error
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key
from tools.probe_pv26_stopline_angle_mask_extent import _write_csv


DEFAULT_SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_lane_head_transplant_original_stop_pca_20260512"
    / "analysis_exports"
    / "stopline_current_candidate_pool_manifest_val512_epoch2"
)


@dataclass(frozen=True)
class FragmentUnionVariant:
    name: str
    min_gap: float
    top_k: int
    min_score: float
    angle_threshold_deg: float
    offset_threshold_px: float
    min_cluster_count: int
    extension_px: float = 0.0
    fallback_top: bool = False


DEFAULT_VARIANTS = (
    FragmentUnionVariant(
        "union_a8_o24_s080_c2",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=8.0,
        offset_threshold_px=24.0,
        min_cluster_count=2,
    ),
    FragmentUnionVariant(
        "union_a12_o36_s080_c2",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=12.0,
        offset_threshold_px=36.0,
        min_cluster_count=2,
    ),
    FragmentUnionVariant(
        "union_a16_o48_s080_c2",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
    ),
    FragmentUnionVariant(
        "union_a12_o36_s050_c2",
        min_gap=4.0,
        top_k=50,
        min_score=0.50,
        angle_threshold_deg=12.0,
        offset_threshold_px=36.0,
        min_cluster_count=2,
    ),
    FragmentUnionVariant(
        "union_a12_o36_s080_c1",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=12.0,
        offset_threshold_px=36.0,
        min_cluster_count=1,
    ),
    FragmentUnionVariant(
        "union_a12_o36_s080_c2_fallback_top",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=12.0,
        offset_threshold_px=36.0,
        min_cluster_count=2,
        fallback_top=True,
    ),
)


@dataclass(frozen=True)
class SegmentCandidate:
    sample_id: str
    proposal_rank: int
    score: float
    length: float
    points_xy: list[list[float]]
    center_xy: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    offset: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a stop-line fragment-union readout from exported candidate_features.csv. "
            "This is read-only: it uses GT only for evaluation, never for prediction."
        )
    )
    parser.add_argument("--candidate-features", default=str(DEFAULT_SOURCE_RUN / "candidate_features.csv"))
    parser.add_argument("--summary", default=str(DEFAULT_SOURCE_RUN / "summary.json"))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SOURCE_RUN.parent / "stopline_fragment_union_readout_val512_epoch2"),
    )
    parser.add_argument("--reference-variant", default="baseline")
    return parser.parse_args()


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _parse_points_json(value: str) -> list[list[float]]:
    if not value:
        return []
    parsed = json.loads(value)
    points = np.asarray(parsed, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points.tolist()]


def _parse_gt_points_json(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    parsed = json.loads(value)
    rows: list[dict[str, Any]] = []
    for item in parsed:
        points = np.asarray(item, dtype=np.float32).reshape(-1, 2)
        rows.append({"points_xy": [[float(x), float(y)] for x, y in points.tolist()]})
    return rows


def _interpolated_stop_line_points(start_xy: np.ndarray, end_xy: np.ndarray) -> list[list[float]]:
    count = max(2, int(STOP_LINE_POINT_COUNT))
    points = [
        start_xy + (end_xy - start_xy) * float(index) / float(count - 1)
        for index in range(count)
    ]
    return [[float(point[0]), float(point[1])] for point in points]


def _normalize_axis(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6 or not np.isfinite(norm):
        return None
    axis = vector.astype(np.float32) / norm
    if axis[0] < 0.0 or (abs(float(axis[0])) <= 1.0e-6 and axis[1] < 0.0):
        axis = -axis
    return axis


def _segment_candidate(row: dict[str, str]) -> SegmentCandidate | None:
    points_xy = _parse_points_json(row.get("candidate_points_json", ""))
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2 or not bool(np.isfinite(points).all()):
        return None
    axis = _normalize_axis(points[-1] - points[0])
    if axis is None:
        return None
    normal = np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)
    center_xy = points.mean(axis=0).astype(np.float32)
    length = float(np.linalg.norm(points[-1] - points[0]))
    return SegmentCandidate(
        sample_id=str(row.get("sample_id", "")),
        proposal_rank=_int_value(row.get("proposal_rank"), default=10**6),
        score=_float_value(row.get("score")),
        length=length,
        points_xy=points_xy,
        center_xy=center_xy,
        axis=axis,
        normal=normal,
        offset=float(np.dot(center_xy, normal)),
    )


def _angle_error_deg(left: SegmentCandidate, right: SegmentCandidate) -> float:
    cosine = float(np.clip(abs(float(np.dot(left.axis, right.axis))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _offset_error_px(left: SegmentCandidate, right: SegmentCandidate) -> float:
    # Use both normals because tiny axis differences can move the absolute line offset.
    left_error = abs(float(np.dot(right.center_xy, left.normal) - left.offset))
    right_error = abs(float(np.dot(left.center_xy, right.normal) - right.offset))
    return float(max(left_error, right_error))


def _union_find_groups(candidates: list[SegmentCandidate], variant: FragmentUnionVariant) -> list[list[SegmentCandidate]]:
    if not candidates:
        return []
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if _angle_error_deg(left, right) > float(variant.angle_threshold_deg):
                continue
            if _offset_error_px(left, right) > float(variant.offset_threshold_px):
                continue
            union(left_index, right_index)

    groups_by_root: dict[int, list[SegmentCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups_by_root.setdefault(find(index), []).append(candidate)
    return list(groups_by_root.values())


def _merge_group(group: list[SegmentCandidate], variant: FragmentUnionVariant) -> dict[str, Any] | None:
    if len(group) < max(1, int(variant.min_cluster_count)):
        return None
    reference_axis = group[0].axis
    weighted_axis = np.zeros(2, dtype=np.float32)
    weighted_offset = 0.0
    weight_sum = 0.0
    for candidate in group:
        axis = candidate.axis
        if float(np.dot(axis, reference_axis)) < 0.0:
            axis = -axis
        weight = max(1.0e-3, float(candidate.score))
        weighted_axis += axis * weight
        weight_sum += weight
    axis = _normalize_axis(weighted_axis)
    if axis is None:
        return None
    normal = np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)
    for candidate in group:
        weighted_offset += float(np.dot(candidate.center_xy, normal)) * max(1.0e-3, float(candidate.score))
    offset = weighted_offset / max(1.0e-6, weight_sum)

    projections: list[float] = []
    for candidate in group:
        points = np.asarray(candidate.points_xy, dtype=np.float32).reshape(-1, 2)
        projections.extend([float(np.dot(point, axis)) for point in points])
    if not projections:
        return None
    start_t = min(projections) - float(variant.extension_px)
    end_t = max(projections) + float(variant.extension_px)
    if end_t <= start_t:
        return None
    start_xy = axis * start_t + normal * offset
    end_xy = axis * end_t + normal * offset
    points_xy = _interpolated_stop_line_points(start_xy.astype(np.float32), end_xy.astype(np.float32))
    length = float(np.linalg.norm(end_xy - start_xy))
    max_score = max(float(candidate.score) for candidate in group)
    mean_score = float(sum(float(candidate.score) for candidate in group) / max(1, len(group)))
    score = max_score + 0.02 * float(max(0, len(group) - 1))
    return {
        "score": float(score),
        "center_score": float(mean_score),
        "length": float(length),
        "points_xy": points_xy,
        "fragment_count": int(len(group)),
        "source_ranks": ";".join(str(candidate.proposal_rank) for candidate in sorted(group, key=lambda item: item.proposal_rank)),
    }


def _fallback_top_candidate(candidates: list[SegmentCandidate]) -> dict[str, Any] | None:
    if not candidates:
        return None
    candidate = max(candidates, key=lambda item: (float(item.score), float(item.length)))
    return {
        "score": float(candidate.score),
        "center_score": float(candidate.score),
        "length": float(candidate.length),
        "points_xy": candidate.points_xy,
        "fragment_count": 1,
        "source_ranks": str(candidate.proposal_rank),
    }


def _union_predictions(rows: list[dict[str, str]], variant: FragmentUnionVariant) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[SegmentCandidate] = []
    for row in rows:
        if str(row.get("proposal_source", "")) != "max":
            continue
        if abs(_float_value(row.get("proposal_min_gap")) - float(variant.min_gap)) > 1.0e-6:
            continue
        if _int_value(row.get("proposal_rank"), default=10**6) > int(variant.top_k):
            continue
        if _float_value(row.get("score")) < float(variant.min_score):
            continue
        candidate = _segment_candidate(row)
        if candidate is not None:
            candidates.append(candidate)
    groups = _union_find_groups(candidates, variant)
    predictions = [merged for group in groups if (merged := _merge_group(group, variant)) is not None]
    if not predictions and bool(variant.fallback_top):
        fallback = _fallback_top_candidate(candidates)
        if fallback is not None:
            predictions.append(fallback)
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[:1], {
        "candidate_count": int(len(candidates)),
        "cluster_count": int(len(groups)),
        "merged_prediction_count": int(len(predictions[:1])),
    }


def _evaluate_predictions(
    sample_records: dict[str, dict[str, Any]],
    predictions_by_sample: dict[str, list[dict[str, Any]]],
    *,
    missing_gt_count: int,
) -> dict[str, float | int]:
    total_tp = 0
    total_fp = 0
    total_fn = int(missing_gt_count)
    prediction_count = 0
    matched_distances: list[float] = []
    matched_angle_errors: list[float] = []
    for sample_id, record in sample_records.items():
        predictions = list(predictions_by_sample.get(sample_id, []))
        gt_stop_lines = list(record.get("gt_stop_lines", []))
        prediction_count += len(predictions)
        if predictions and gt_stop_lines:
            cost_matrix = np.zeros((len(predictions), len(gt_stop_lines)), dtype=np.float32)
            for pred_index, prediction in enumerate(predictions):
                for gt_index, gt in enumerate(gt_stop_lines):
                    cost_matrix[pred_index, gt_index] = _mean_point_distance(
                        prediction.get("points_xy", []),
                        gt.get("points_xy", []),
                        STOP_LINE_POINT_COUNT,
                    )
            matches = _hungarian_from_cost(cost_matrix, max_cost=40.0)
            matched_distances.extend(float(cost_matrix[pred_index, gt_index]) for pred_index, gt_index in matches)
            matched_angle_errors.extend(
                float(
                    _segment_angle_error(
                        predictions[pred_index].get("points_xy", []),
                        gt_stop_lines[gt_index].get("points_xy", []),
                        STOP_LINE_POINT_COUNT,
                    )
                )
                for pred_index, gt_index in matches
            )
        else:
            matches = []
        total_tp += len(matches)
        total_fp += len(predictions) - len(matches)
        total_fn += len(gt_stop_lines) - len(matches)
    precision = float(total_tp / max(1, total_tp + total_fp))
    recall = float(total_tp / max(1, total_tp + total_fn))
    f1 = float(2.0 * precision * recall / max(1.0e-12, precision + recall))
    return {
        "pred_stop_line_count": int(prediction_count),
        "stop_line_precision": precision,
        "stop_line_recall": recall,
        "stop_line_f1": f1,
        "stop_line_tp": int(total_tp),
        "stop_line_fp": int(total_fp),
        "stop_line_fn": int(total_fn),
        "stop_line_support": int(total_tp + total_fn),
        "stop_line_mean_point_distance": float(np.mean(matched_distances)) if matched_distances else 0.0,
        "stop_line_mean_angle_error": float(np.mean(matched_angle_errors)) if matched_angle_errors else 0.0,
    }


def _reference_row(summary: dict[str, Any], reference_variant: str) -> dict[str, Any]:
    rows = summary.get("variants", [])
    if not isinstance(rows, list):
        raise ValueError("summary variants must be a list")
    for row in rows:
        if isinstance(row, dict) and str(row.get("variant")) == str(reference_variant):
            return dict(row)
    raise ValueError(f"reference variant not found: {reference_variant}")


def _stop_line_support_from_summary(summary: dict[str, Any]) -> int:
    rows = summary.get("variants", [])
    if not isinstance(rows, list):
        return 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        tp = _int_value(row.get("stop_line_tp"))
        fn = _int_value(row.get("stop_line_fn"))
        if tp + fn > 0:
            return int(tp + fn)
    return 0


def _load_candidate_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, str]]]]:
    sample_records: dict[str, dict[str, Any]] = {}
    rows_by_sample: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                continue
            rows_by_sample.setdefault(sample_id, []).append(row)
            if sample_id not in sample_records:
                sample_records[sample_id] = {
                    "sample_id": sample_id,
                    "gt_stop_lines": _parse_gt_points_json(row.get("gt_stop_line_points_json", "")),
                    "gt_stop_line_count": _int_value(row.get("gt_stop_line_count")),
                }
    return sample_records, rows_by_sample


def _row_from_summary(reference_row: dict[str, Any], variant: FragmentUnionVariant, stop_line: dict[str, float | int]) -> dict[str, Any]:
    row = dict(reference_row)
    row["variant"] = variant.name
    row["fragment_union_min_gap"] = float(variant.min_gap)
    row["fragment_union_top_k"] = int(variant.top_k)
    row["fragment_union_min_score"] = float(variant.min_score)
    row["fragment_union_angle_threshold_deg"] = float(variant.angle_threshold_deg)
    row["fragment_union_offset_threshold_px"] = float(variant.offset_threshold_px)
    row["fragment_union_min_cluster_count"] = int(variant.min_cluster_count)
    row["fragment_union_extension_px"] = float(variant.extension_px)
    row["fragment_union_fallback_top"] = bool(variant.fallback_top)
    row.update(stop_line)
    lane_f1 = _float_value(row.get("lane_f1"))
    stop_f1 = _float_value(row.get("stop_line_f1"))
    cross_f1 = _float_value(row.get("crosswalk_f1"))
    row["lane_family_mean_f1"] = float((lane_f1 + stop_f1 + cross_f1) / 3.0)
    row["lane_family_min_f1"] = float(min(lane_f1, stop_f1, cross_f1))
    row["phase4_objective_proxy"] = float(0.50 * lane_f1 + 0.30 * stop_f1 + 0.20 * cross_f1)
    row.pop("phase_objective", None)
    return row


def run_replay(
    *,
    candidate_features_path: Path,
    summary_path: Path,
    output_dir: Path,
    reference_variant: str,
    variants: tuple[FragmentUnionVariant, ...] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    reference = _reference_row(summary, reference_variant)
    total_stop_line_support = _stop_line_support_from_summary(summary)
    sample_records, rows_by_sample = _load_candidate_records(candidate_features_path)
    candidate_bearing_gt_count = sum(int(record.get("gt_stop_line_count", 0)) for record in sample_records.values())
    missing_gt_count = max(0, int(total_stop_line_support) - int(candidate_bearing_gt_count))

    rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for variant in variants:
        predictions_by_sample: dict[str, list[dict[str, Any]]] = {}
        for sample_id, candidate_rows in rows_by_sample.items():
            predictions, stats = _union_predictions(candidate_rows, variant)
            predictions_by_sample[sample_id] = predictions
            cluster_rows.append(
                {
                    "variant": variant.name,
                    "sample_id": sample_id,
                    **stats,
                    "gt_stop_line_count": int(sample_records[sample_id].get("gt_stop_line_count", 0)),
                    "pred_stop_line_count": int(len(predictions)),
                }
            )
        stop_line = _evaluate_predictions(sample_records, predictions_by_sample, missing_gt_count=missing_gt_count)
        rows.append(_row_from_summary(reference, variant, stop_line))

    rows.sort(
        key=lambda row: (
            _float_value(row.get("stop_line_f1")),
            _float_value(row.get("stop_line_precision")),
            -_float_value(row.get("stop_line_fp")),
        ),
        reverse=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fragment_union_variants.csv", rows)
    _write_csv(output_dir / "fragment_union_clusters.csv", cluster_rows)
    payload = {
        "candidate_features": str(candidate_features_path),
        "source_summary": str(summary_path),
        "reference_variant": str(reference_variant),
        "total_stop_line_support": int(total_stop_line_support),
        "candidate_bearing_sample_count": int(len(sample_records)),
        "candidate_bearing_gt_count": int(candidate_bearing_gt_count),
        "missing_gt_count": int(missing_gt_count),
        "variants": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    result = run_replay(
        candidate_features_path=Path(args.candidate_features).expanduser().resolve(),
        summary_path=Path(args.summary).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        reference_variant=str(args.reference_variant),
    )
    best = result["variants"][0] if result["variants"] else {}
    print(
        "[stopline_fragment_union] "
        f"best={best.get('variant')} "
        f"stop_f1={_float_value(best.get('stop_line_f1')):.4f} "
        f"tp/fp/fn={best.get('stop_line_tp')}/{best.get('stop_line_fp')}/{best.get('stop_line_fn')} "
        f"missing_gt={result.get('missing_gt_count')}"
    )
    print(f"[stopline_fragment_union] wrote {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
