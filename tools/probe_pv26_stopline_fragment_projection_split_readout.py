from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key
from tools.probe_pv26_stopline_angle_mask_extent import _write_csv
from tools.probe_pv26_stopline_fragment_union_readout import (
    DEFAULT_SOURCE_RUN,
    FragmentUnionVariant,
    SegmentCandidate,
    _evaluate_predictions,
    _fallback_top_candidate,
    _float_value,
    _int_value,
    _load_candidate_records,
    _merge_group,
    _normalize_axis,
    _reference_row,
    _segment_candidate,
    _stop_line_support_from_summary,
    _union_find_groups,
)


@dataclass(frozen=True)
class ProjectionSplitVariant:
    name: str
    min_gap: float
    top_k: int
    min_score: float
    angle_threshold_deg: float
    offset_threshold_px: float
    min_cluster_count: int
    projection_gap_px: float
    max_predictions: int
    extension_px: float = 0.0
    fallback_top: bool = True
    second_min_score: float = 0.0
    second_min_fragment_count: int = 0
    second_min_length_ratio: float = 0.0


DEFAULT_VARIANTS = (
    ProjectionSplitVariant(
        "proj_split_a16_o48_gap320_c2_top2_second_lenratio070",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        max_predictions=2,
        second_min_length_ratio=0.70,
    ),
    ProjectionSplitVariant(
        "proj_split_a16_o48_gap240_c2_top2_second_lenratio070",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=240.0,
        max_predictions=2,
        second_min_length_ratio=0.70,
    ),
    ProjectionSplitVariant(
        "proj_split_a16_o48_gap320_c2_top2_second_score108",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        max_predictions=2,
        second_min_score=1.08,
    ),
    ProjectionSplitVariant(
        "proj_split_a16_o48_gap320_c2_top2_raw",
        min_gap=4.0,
        top_k=50,
        min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        max_predictions=2,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a stop-line fragment-union readout that splits over-merged same-line "
            "fragment groups by large projection gaps before merging predictions."
        )
    )
    parser.add_argument("--candidate-features", default=str(DEFAULT_SOURCE_RUN / "candidate_features.csv"))
    parser.add_argument("--summary", default=str(DEFAULT_SOURCE_RUN / "summary.json"))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SOURCE_RUN.parent / "stopline_fragment_projection_split_readout_val512_epoch2"),
    )
    parser.add_argument("--reference-variant", default="baseline")
    return parser.parse_args()


def _as_union_variant(variant: ProjectionSplitVariant) -> FragmentUnionVariant:
    return FragmentUnionVariant(
        variant.name,
        min_gap=variant.min_gap,
        top_k=variant.top_k,
        min_score=variant.min_score,
        angle_threshold_deg=variant.angle_threshold_deg,
        offset_threshold_px=variant.offset_threshold_px,
        min_cluster_count=variant.min_cluster_count,
        extension_px=variant.extension_px,
        fallback_top=variant.fallback_top,
    )


def _group_axis(group: list[SegmentCandidate]) -> np.ndarray | None:
    if not group:
        return None
    reference_axis = group[0].axis
    weighted_axis = np.zeros(2, dtype=np.float32)
    for candidate in group:
        axis = candidate.axis
        if float(np.dot(axis, reference_axis)) < 0.0:
            axis = -axis
        weighted_axis += axis * max(1.0e-3, float(candidate.score))
    return _normalize_axis(weighted_axis)


def _projection_interval(candidate: SegmentCandidate, axis: np.ndarray) -> tuple[float, float]:
    points = np.asarray(candidate.points_xy, dtype=np.float32).reshape(-1, 2)
    projections = [float(np.dot(point, axis)) for point in points]
    return min(projections), max(projections)


def _split_group_by_projection_gap(group: list[SegmentCandidate], projection_gap_px: float) -> list[list[SegmentCandidate]]:
    if len(group) <= 1:
        return [group]
    axis = _group_axis(group)
    if axis is None:
        return [group]
    intervals = [
        (*_projection_interval(candidate, axis), candidate)
        for candidate in group
    ]
    intervals.sort(key=lambda item: item[0])

    parts: list[list[SegmentCandidate]] = []
    current: list[SegmentCandidate] = []
    current_hi: float | None = None
    for lo, hi, candidate in intervals:
        if current and current_hi is not None and float(lo) - float(current_hi) > float(projection_gap_px):
            parts.append(current)
            current = []
        current.append(candidate)
        current_hi = max(float(current_hi) if current_hi is not None else float(hi), float(hi))
    if current:
        parts.append(current)
    return parts


def _keep_additional_prediction(
    prediction: dict[str, Any],
    primary_prediction: dict[str, Any],
    variant: ProjectionSplitVariant,
) -> bool:
    if _float_value(prediction.get("score")) < float(variant.second_min_score):
        return False
    if _int_value(prediction.get("fragment_count")) < int(variant.second_min_fragment_count):
        return False
    primary_length = max(1.0e-6, _float_value(primary_prediction.get("length")))
    length_ratio = _float_value(prediction.get("length")) / primary_length
    if length_ratio < float(variant.second_min_length_ratio):
        return False
    return True


def _apply_prediction_cap(predictions: list[dict[str, Any]], variant: ProjectionSplitVariant) -> list[dict[str, Any]]:
    max_predictions = max(1, int(variant.max_predictions))
    if len(predictions) <= 1 or max_predictions <= 1:
        return predictions[:1]
    capped = [predictions[0]]
    for prediction in predictions[1:]:
        if len(capped) >= max_predictions:
            break
        if _keep_additional_prediction(prediction, predictions[0], variant):
            capped.append(prediction)
    return capped


def _projection_split_predictions(
    rows: list[dict[str, str]],
    variant: ProjectionSplitVariant,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    union_variant = _as_union_variant(variant)
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

    groups = _union_find_groups(candidates, union_variant)
    split_groups: list[list[SegmentCandidate]] = []
    for group in groups:
        split_groups.extend(_split_group_by_projection_gap(group, variant.projection_gap_px))
    predictions = [
        merged
        for group in split_groups
        if (merged := _merge_group(group, union_variant)) is not None
    ]
    if not predictions and bool(variant.fallback_top):
        fallback = _fallback_top_candidate(candidates)
        if fallback is not None:
            predictions.append(fallback)

    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    capped = _apply_prediction_cap(predictions, variant)
    return capped, {
        "candidate_count": int(len(candidates)),
        "cluster_count": int(len(groups)),
        "split_group_count": int(len(split_groups)),
        "raw_prediction_count": int(len(predictions)),
        "merged_prediction_count": int(len(capped)),
    }


def _row_from_summary(
    reference_row: dict[str, Any],
    variant: ProjectionSplitVariant,
    stop_line: dict[str, float | int],
) -> dict[str, Any]:
    row = dict(reference_row)
    row["variant"] = variant.name
    row["fragment_projection_split_min_gap"] = float(variant.min_gap)
    row["fragment_projection_split_top_k"] = int(variant.top_k)
    row["fragment_projection_split_min_score"] = float(variant.min_score)
    row["fragment_projection_split_angle_threshold_deg"] = float(variant.angle_threshold_deg)
    row["fragment_projection_split_offset_threshold_px"] = float(variant.offset_threshold_px)
    row["fragment_projection_split_min_cluster_count"] = int(variant.min_cluster_count)
    row["fragment_projection_split_projection_gap_px"] = float(variant.projection_gap_px)
    row["fragment_projection_split_max_predictions"] = int(variant.max_predictions)
    row["fragment_projection_split_second_min_score"] = float(variant.second_min_score)
    row["fragment_projection_split_second_min_fragment_count"] = int(variant.second_min_fragment_count)
    row["fragment_projection_split_second_min_length_ratio"] = float(variant.second_min_length_ratio)
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
    variants: tuple[ProjectionSplitVariant, ...] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    reference = _reference_row(summary, reference_variant)
    total_stop_line_support = _stop_line_support_from_summary(summary)
    sample_records, rows_by_sample = _load_candidate_records(candidate_features_path)
    candidate_bearing_gt_count = sum(int(record.get("gt_stop_line_count", 0)) for record in sample_records.values())
    missing_gt_count = max(0, int(total_stop_line_support) - int(candidate_bearing_gt_count))

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for variant in variants:
        predictions_by_sample: dict[str, list[dict[str, Any]]] = {}
        for sample_id, candidate_rows in rows_by_sample.items():
            predictions, stats = _projection_split_predictions(candidate_rows, variant)
            predictions_by_sample[sample_id] = predictions
            sample_rows.append(
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
    _write_csv(output_dir / "fragment_projection_split_variants.csv", rows)
    _write_csv(output_dir / "fragment_projection_split_samples.csv", sample_rows)
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
        "[stopline_fragment_projection_split] "
        f"best={best.get('variant')} "
        f"stop_f1={_float_value(best.get('stop_line_f1')):.4f} "
        f"tp/fp/fn={best.get('stop_line_tp')}/{best.get('stop_line_fp')}/{best.get('stop_line_fn')} "
        f"missing_gt={result.get('missing_gt_count')}"
    )
    print(f"[stopline_fragment_projection_split] wrote {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
