from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.engine.postprocess import _dedupe_stop_line_predictions
from tools.probe_pv26_stopline_angle_mask_extent import _write_csv
from tools.probe_pv26_stopline_fragment_projection_split_readout import _split_group_by_projection_gap
from tools.probe_pv26_stopline_fragment_union_readout import (
    DEFAULT_SOURCE_RUN,
    FragmentUnionVariant,
    SegmentCandidate,
    _evaluate_predictions,
    _float_value,
    _int_value,
    _load_candidate_records,
    _merge_group,
    _reference_row,
    _segment_candidate,
    _stop_line_support_from_summary,
    _union_find_groups,
)


@dataclass(frozen=True)
class ProjectionCompetitionVariant:
    name: str
    min_gap: float
    top_k: int
    union_min_score: float
    single_min_score: float
    angle_threshold_deg: float
    offset_threshold_px: float
    min_cluster_count: int
    projection_gap_px: float
    rank_feature: str
    max_predictions: int
    extension_px: float = 0.0
    second_min_score: float = 0.0
    second_min_fragment_count: int = 0
    second_min_length_ratio: float = 0.0


DEFAULT_VARIANTS = (
    ProjectionCompetitionVariant(
        "proj_comp_length_s090_top2_second_frag5",
        min_gap=4.0,
        top_k=50,
        union_min_score=0.80,
        single_min_score=0.90,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        rank_feature="length",
        max_predictions=2,
        second_min_fragment_count=5,
    ),
    ProjectionCompetitionVariant(
        "proj_comp_length_s080_top2_second_frag5",
        min_gap=4.0,
        top_k=50,
        union_min_score=0.80,
        single_min_score=0.80,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        rank_feature="length",
        max_predictions=2,
        second_min_fragment_count=5,
    ),
    ProjectionCompetitionVariant(
        "proj_comp_component_length_s090_top2_second_frag5",
        min_gap=4.0,
        top_k=50,
        union_min_score=0.80,
        single_min_score=0.90,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        rank_feature="component_svd_length",
        max_predictions=2,
        second_min_fragment_count=5,
    ),
    ProjectionCompetitionVariant(
        "proj_comp_length_s090_top1",
        min_gap=4.0,
        top_k=50,
        union_min_score=0.80,
        single_min_score=0.90,
        angle_threshold_deg=16.0,
        offset_threshold_px=48.0,
        min_cluster_count=2,
        projection_gap_px=320.0,
        rank_feature="length",
        max_predictions=1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a projection-split stop-line readout where split union groups and "
            "high-confidence single candidates compete by a no-GT rank feature."
        )
    )
    parser.add_argument("--candidate-features", default=str(DEFAULT_SOURCE_RUN / "candidate_features.csv"))
    parser.add_argument("--summary", default=str(DEFAULT_SOURCE_RUN / "summary.json"))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SOURCE_RUN.parent / "stopline_fragment_projection_competition_readout_val512_epoch2"),
    )
    parser.add_argument("--reference-variant", default="baseline")
    return parser.parse_args()


def _as_union_variant(variant: ProjectionCompetitionVariant) -> FragmentUnionVariant:
    return FragmentUnionVariant(
        variant.name,
        min_gap=variant.min_gap,
        top_k=variant.top_k,
        min_score=variant.union_min_score,
        angle_threshold_deg=variant.angle_threshold_deg,
        offset_threshold_px=variant.offset_threshold_px,
        min_cluster_count=variant.min_cluster_count,
        extension_px=variant.extension_px,
        fallback_top=False,
    )


def _candidate_rows(
    rows: list[dict[str, str]],
    *,
    min_gap: float,
    top_k: int,
    min_score: float,
) -> list[tuple[SegmentCandidate, dict[str, str]]]:
    candidates: list[tuple[SegmentCandidate, dict[str, str]]] = []
    for row in rows:
        if str(row.get("proposal_source", "")) != "max":
            continue
        if abs(_float_value(row.get("proposal_min_gap")) - float(min_gap)) > 1.0e-6:
            continue
        if _int_value(row.get("proposal_rank"), default=10**6) > int(top_k):
            continue
        if _float_value(row.get("score")) < float(min_score):
            continue
        candidate = _segment_candidate(row)
        if candidate is not None:
            candidates.append((candidate, row))
    return candidates


def _single_prediction(
    candidate: SegmentCandidate,
    row: dict[str, str],
    variant: ProjectionCompetitionVariant,
) -> dict[str, Any]:
    return {
        "score": float(candidate.score),
        "center_score": float(candidate.score),
        "length": float(candidate.length),
        "points_xy": candidate.points_xy,
        "fragment_count": 1,
        "source_ranks": str(candidate.proposal_rank),
        "_proposal_kind": "single",
        "_rank_feature": _float_value(row.get(variant.rank_feature)),
    }


def _keep_additional_prediction(
    prediction: dict[str, Any],
    primary_prediction: dict[str, Any],
    variant: ProjectionCompetitionVariant,
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


def _apply_prediction_cap(predictions: list[dict[str, Any]], variant: ProjectionCompetitionVariant) -> list[dict[str, Any]]:
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


def _clean_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(prediction)
    cleaned.pop("_proposal_kind", None)
    cleaned.pop("_rank_feature", None)
    return cleaned


def _projection_competition_predictions(
    rows: list[dict[str, str]],
    variant: ProjectionCompetitionVariant,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    union_variant = _as_union_variant(variant)
    union_items = _candidate_rows(
        rows,
        min_gap=variant.min_gap,
        top_k=variant.top_k,
        min_score=variant.union_min_score,
    )
    row_by_candidate_id = {id(candidate): row for candidate, row in union_items}
    union_groups = _union_find_groups([candidate for candidate, _row in union_items], union_variant)

    proposals: list[dict[str, Any]] = []
    split_group_count = 0
    for group in union_groups:
        split_groups = _split_group_by_projection_gap(group, variant.projection_gap_px)
        split_group_count += len(split_groups)
        for split_group in split_groups:
            if len(split_group) < max(1, int(variant.min_cluster_count)):
                continue
            merged = _merge_group(split_group, union_variant)
            if merged is None:
                continue
            group_rows = [row_by_candidate_id[id(candidate)] for candidate in split_group]
            feature_values = [_float_value(row.get(variant.rank_feature)) for row in group_rows]
            merged["_proposal_kind"] = "union"
            merged["_rank_feature"] = max(feature_values) if feature_values else 0.0
            proposals.append(merged)

    single_items = _candidate_rows(
        rows,
        min_gap=variant.min_gap,
        top_k=variant.top_k,
        min_score=variant.single_min_score,
    )
    for candidate, row in single_items:
        proposals.append(_single_prediction(candidate, row, variant))

    if not proposals:
        return [], {
            "union_candidate_count": int(len(union_items)),
            "single_candidate_count": int(len(single_items)),
            "union_group_count": int(len(union_groups)),
            "split_group_count": int(split_group_count),
            "proposal_count": 0,
            "selected_kind": "",
        }

    proposals.sort(
        key=lambda proposal: (
            _float_value(proposal.get("_rank_feature")),
            1 if str(proposal.get("_proposal_kind")) == "union" else 0,
            _float_value(proposal.get("score")),
            _float_value(proposal.get("length")),
        ),
        reverse=True,
    )
    proposals = _dedupe_stop_line_predictions(proposals)
    capped = _apply_prediction_cap(proposals, variant)
    selected_kind = str(capped[0].get("_proposal_kind", "")) if capped else ""
    return [_clean_prediction(prediction) for prediction in capped], {
        "union_candidate_count": int(len(union_items)),
        "single_candidate_count": int(len(single_items)),
        "union_group_count": int(len(union_groups)),
        "split_group_count": int(split_group_count),
        "proposal_count": int(len(proposals)),
        "selected_kind": selected_kind,
    }


def _row_from_summary(
    reference_row: dict[str, Any],
    variant: ProjectionCompetitionVariant,
    stop_line: dict[str, float | int],
) -> dict[str, Any]:
    row = dict(reference_row)
    row["variant"] = variant.name
    row["fragment_projection_comp_min_gap"] = float(variant.min_gap)
    row["fragment_projection_comp_top_k"] = int(variant.top_k)
    row["fragment_projection_comp_union_min_score"] = float(variant.union_min_score)
    row["fragment_projection_comp_single_min_score"] = float(variant.single_min_score)
    row["fragment_projection_comp_angle_threshold_deg"] = float(variant.angle_threshold_deg)
    row["fragment_projection_comp_offset_threshold_px"] = float(variant.offset_threshold_px)
    row["fragment_projection_comp_min_cluster_count"] = int(variant.min_cluster_count)
    row["fragment_projection_comp_projection_gap_px"] = float(variant.projection_gap_px)
    row["fragment_projection_comp_rank_feature"] = str(variant.rank_feature)
    row["fragment_projection_comp_max_predictions"] = int(variant.max_predictions)
    row["fragment_projection_comp_second_min_score"] = float(variant.second_min_score)
    row["fragment_projection_comp_second_min_fragment_count"] = int(variant.second_min_fragment_count)
    row["fragment_projection_comp_second_min_length_ratio"] = float(variant.second_min_length_ratio)
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
    variants: tuple[ProjectionCompetitionVariant, ...] = DEFAULT_VARIANTS,
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
            predictions, stats = _projection_competition_predictions(candidate_rows, variant)
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
    _write_csv(output_dir / "fragment_projection_competition_variants.csv", rows)
    _write_csv(output_dir / "fragment_projection_competition_samples.csv", sample_rows)
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
        "[stopline_fragment_projection_competition] "
        f"best={best.get('variant')} "
        f"stop_f1={_float_value(best.get('stop_line_f1')):.4f} "
        f"tp/fp/fn={best.get('stop_line_tp')}/{best.get('stop_line_fp')}/{best.get('stop_line_fn')} "
        f"missing_gt={result.get('missing_gt_count')}"
    )
    print(f"[stopline_fragment_projection_competition] wrote {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
