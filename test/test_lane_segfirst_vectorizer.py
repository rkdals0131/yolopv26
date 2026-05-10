from __future__ import annotations

import unittest

import torch

from common.pv26_schema import LANE_CLASSES, LANE_TYPES
from model.engine.lane_segfirst_vectorizer import LaneSegFirstVectorizerConfig, vectorize_lane_segfirst_maps


def _maps_with_vertical_gap() -> dict[str, torch.Tensor]:
    centerline = torch.zeros((1, 12, 12), dtype=torch.float32)
    centerline[0, 9:12, 5] = 0.9
    centerline[0, 4:7, 5] = 0.9
    return {
        "centerline_core": centerline,
        "color_map": torch.zeros((len(LANE_CLASSES), 12, 12), dtype=torch.float32),
        "lane_type_map": torch.zeros((len(LANE_TYPES), 12, 12), dtype=torch.float32),
    }


def _maps_with_zigzag_track() -> dict[str, torch.Tensor]:
    centerline = torch.zeros((1, 12, 12), dtype=torch.float32)
    centerline[0, 11, 2] = 0.9
    centerline[0, 10, 6] = 0.9
    centerline[0, 9, 2] = 0.9
    return {
        "centerline_core": centerline,
        "color_map": torch.zeros((len(LANE_CLASSES), 12, 12), dtype=torch.float32),
        "lane_type_map": torch.zeros((len(LANE_TYPES), 12, 12), dtype=torch.float32),
    }


class LaneSegFirstVectorizerTests(unittest.TestCase):
    def test_row_scan_track_mode_can_bridge_vertical_gaps(self) -> None:
        maps = _maps_with_vertical_gap()

        component_predictions = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=3,
                max_link_dx=1.0,
            ),
        )
        row_scan_predictions = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=3,
                max_link_dx=1.0,
            ),
        )

        self.assertEqual(len(component_predictions), 2)
        self.assertEqual(len(row_scan_predictions), 1)
        self.assertGreaterEqual(len(row_scan_predictions[0]["points_xy"]), 4)

    def test_row_scan_turn_guard_rejects_zigzag_tracks(self) -> None:
        maps = _maps_with_zigzag_track()

        unguarded = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=2,
                max_link_dx=5.0,
            ),
        )
        guarded = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=2,
                max_link_dx=5.0,
                max_turn_degrees=60.0,
            ),
        )

        self.assertEqual(len(unguarded), 1)
        self.assertEqual(len(guarded), 0)

    def test_rejects_unknown_track_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "track_mode"):
            vectorize_lane_segfirst_maps(
                _maps_with_vertical_gap(),
                config=LaneSegFirstVectorizerConfig(track_mode="unknown"),
            )


if __name__ == "__main__":
    unittest.main()
