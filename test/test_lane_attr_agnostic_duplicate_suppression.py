from __future__ import annotations

from tools.probe_pv26_lane_attr_agnostic_duplicate_suppression import suppress_duplicate_lanes


def _lane(x_offset: float, *, class_name: str, lane_type: str, score: float) -> dict:
    return {
        "class_name": class_name,
        "lane_type": lane_type,
        "score": score,
        "points_xy": [[x_offset, 0.0], [x_offset + 10.0, 10.0]],
    }


def test_attr_agnostic_duplicate_suppresses_cross_schema_lane() -> None:
    lanes = [
        _lane(0.0, class_name="white_lane", lane_type="solid", score=0.8),
        _lane(1.0, class_name="yellow_lane", lane_type="dotted", score=0.7),
    ]
    kept, suppressed = suppress_duplicate_lanes(lanes, duplicate_distance=24.0, require_same_schema=False)
    assert len(kept) == 1
    assert len(suppressed) == 1
    assert suppressed[0]["same_schema"] is False


def test_same_schema_duplicate_keeps_cross_schema_lane() -> None:
    lanes = [
        _lane(0.0, class_name="white_lane", lane_type="solid", score=0.8),
        _lane(1.0, class_name="yellow_lane", lane_type="dotted", score=0.7),
    ]
    kept, suppressed = suppress_duplicate_lanes(lanes, duplicate_distance=24.0, require_same_schema=True)
    assert len(kept) == 2
    assert suppressed == []


def test_duplicate_suppression_keeps_higher_score_lane() -> None:
    lanes = [
        _lane(0.0, class_name="white_lane", lane_type="solid", score=0.2),
        _lane(1.0, class_name="white_lane", lane_type="solid", score=0.9),
    ]
    kept, suppressed = suppress_duplicate_lanes(lanes, duplicate_distance=24.0, require_same_schema=False)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9
    assert suppressed[0]["original_index"] == 0
