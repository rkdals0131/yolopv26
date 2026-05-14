from __future__ import annotations

import math

from tools.probe_pv26_lane_temporal_neighbor_union import (
    _nearest_lane_distance,
    _neighbor_sample_id,
    _parse_neighbor_offsets,
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
