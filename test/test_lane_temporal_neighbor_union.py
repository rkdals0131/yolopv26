from __future__ import annotations

import math

import cv2
import numpy as np

from tools.probe_pv26_lane_temporal_neighbor_union import (
    LANE_TEMPORAL_FEATURES,
    SCORE_KEY,
    _estimate_phase_translation_raw_shift,
    _lane_from_points_json,
    _lane_temporal_feature_vector,
    _nearest_lane_distance,
    _neighbor_sample_id,
    _parse_neighbor_offsets,
    _select_verified_temporal_lanes,
    _shift_lane_points,
    _split_temporal_sample_id,
)


def test_split_temporal_sample_id_preserves_prefix_and_width() -> None:
    assert _split_temporal_sample_id("aihub_lane_val_000123") == ("aihub_lane_val_", 123, 6)


def test_neighbor_sample_id_formats_adjacent_frame() -> None:
    assert _neighbor_sample_id("aihub_lane_val_000123", -1) == "aihub_lane_val_000122"
    assert _neighbor_sample_id("aihub_lane_val_000123", 2) == "aihub_lane_val_000125"


def test_neighbor_sample_id_rejects_non_temporal_id() -> None:
    assert _neighbor_sample_id("no_digits_here", 1) is None


def test_parse_neighbor_offsets_dedupes_and_rejects_zero() -> None:
    assert _parse_neighbor_offsets("-1,1,-1") == (-1, 1)
    try:
        _parse_neighbor_offsets("-1,0,1")
    except ValueError as exc:
        assert "offset 0" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_nearest_lane_distance_empty_is_infinite() -> None:
    lane = {"points_xy": [[0.0, 0.0], [10.0, 0.0]]}
    assert math.isinf(_nearest_lane_distance(lane, []))


def test_lane_from_points_json_requires_parseable_polyline() -> None:
    lane = _lane_from_points_json("[[1, 2], [3, 4]]", score=0.75)
    assert lane is not None
    assert lane["points_xy"] == [[1.0, 2.0], [3.0, 4.0]]
    assert lane["class_name"] == "white_lane"
    assert lane["lane_type"] == "solid"
    assert lane["temporal_mlp_score"] == 0.75
    assert _lane_from_points_json("[[1, 2]]") is None
    assert _lane_from_points_json("not-json") is None


def test_lane_temporal_feature_vector_is_fixed_and_finite() -> None:
    row = {
        "temporal_abs_offset": 0.125,
        "temporal_offset_sign": -1.0,
        "current_frame_center_point_mean": float("nan"),
        "pred_polyline_length_norm": 0.5,
    }
    vector = _lane_temporal_feature_vector(row)
    assert len(vector) == len(LANE_TEMPORAL_FEATURES)
    assert all(math.isfinite(value) for value in vector)
    assert vector[LANE_TEMPORAL_FEATURES.index("current_frame_center_point_mean")] == 0.0


def test_select_verified_temporal_lanes_respects_score_dedupe_and_cap() -> None:
    baseline_lanes = [{"points_xy": [[0.0, 0.0], [10.0, 0.0]]}]
    duplicate_row = {
        SCORE_KEY: 0.95,
        "candidate_points_json": "[[1, 0], [11, 0]]",
        "would_match_baseline_fn": True,
    }
    selected_row = {
        SCORE_KEY: 0.9,
        "candidate_points_json": "[[100, 0], [120, 0]]",
        "would_match_baseline_fn": True,
    }
    low_score_row = {
        SCORE_KEY: 0.2,
        "candidate_points_json": "[[200, 0], [220, 0]]",
        "would_match_baseline_fn": True,
    }
    selected = _select_verified_temporal_lanes(
        [duplicate_row, selected_row, low_score_row],
        threshold=0.5,
        top_k_per_sample=3,
        max_added_per_sample=1,
        dedupe_distance=40.0,
        baseline_lanes=baseline_lanes,
    )
    assert [lane["points_xy"] for lane in selected] == [[[100.0, 0.0], [120.0, 0.0]]]
    assert duplicate_row.get("verifier_selected") is not True
    assert selected_row.get("verifier_selected") is True
    assert low_score_row.get("verifier_selected") is not True


def test_phase_translation_alignment_estimates_neighbor_to_current_shift() -> None:
    target = np.zeros((64, 64), dtype=np.float32)
    target[20:30, 25:35] = 1.0
    transform = np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])
    neighbor = cv2.warpAffine(target, transform, (64, 64))

    shift = _estimate_phase_translation_raw_shift(
        target_image=target,
        neighbor_image=neighbor,
        target_meta={"raw_hw": (64, 64), "transform": {"scale": 1.0}},
    )

    assert shift["response"] > 0.5
    assert abs(shift["raw_dx"] + 5.0) < 0.5
    assert abs(shift["raw_dy"] + 3.0) < 0.5


def test_shift_lane_points_clips_to_raw_frame() -> None:
    lane = {"points_xy": [[2.0, 4.0], [20.0, 30.0]], "score": 0.5}

    shifted = _shift_lane_points(lane, dx=-5.0, dy=10.0, meta={"raw_hw": (40, 50)})

    assert shifted["points_xy"] == [[0.0, 14.0], [15.0, 39.0]]
    assert shifted["temporal_alignment_dx"] == -5.0
