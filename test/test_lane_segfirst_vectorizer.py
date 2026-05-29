from __future__ import annotations

import unittest

import torch

from common.pv26_schema import LANE_CLASSES, LANE_TYPES
from model.engine.lane_segfirst_vectorizer import (
    LaneSegFirstTargetConfig,
    LaneSegFirstVectorizerConfig,
    render_lane_segfirst_targets,
    vectorize_lane_segfirst_maps,
)


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


def _maps_with_tangent_track() -> dict[str, torch.Tensor]:
    centerline = torch.zeros((1, 12, 12), dtype=torch.float32)
    centerline[0, 11, 2] = 0.9
    centerline[0, 10, 4] = 0.9
    centerline[0, 9, 6] = 0.9
    tangent_axis = torch.zeros((2, 12, 12), dtype=torch.float32)
    tangent_axis[0, :, :] = 0.7
    tangent_axis[1, :, :] = -0.7
    return {
        "centerline_core": centerline,
        "tangent_axis": tangent_axis,
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

    def test_row_scan_tangent_track_mode_uses_tangent_map(self) -> None:
        predictions = vectorize_lane_segfirst_maps(
            _maps_with_tangent_track(),
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan_tangent",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=2,
                max_link_dx=4.0,
            ),
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]["points_xy"]), 3)

    def test_seed_trace_starts_from_seed_map_and_follows_centerline(self) -> None:
        maps = _maps_with_vertical_gap()
        seed_map = torch.zeros((1, 12, 12), dtype=torch.float32)
        seed_map[0, 11, 5] = 0.95
        maps["seed_map"] = seed_map

        predictions = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="seed_trace",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=3,
                max_link_dx=1.0,
                seed_threshold=0.5,
                seed_trace_max_seeds=4,
            ),
        )

        self.assertEqual(len(predictions), 1)
        self.assertGreaterEqual(len(predictions[0]["points_xy"]), 4)

    def test_row_scan_seed_trace_preserves_row_scan_without_seed(self) -> None:
        maps = _maps_with_vertical_gap()
        maps["seed_map"] = torch.zeros((1, 12, 12), dtype=torch.float32)

        predictions = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan_tangent_seed_trace",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=3,
                max_link_dx=1.0,
                seed_threshold=0.5,
                seed_trace_max_seeds=4,
            ),
        )

        self.assertEqual(len(predictions), 1)
        self.assertGreaterEqual(len(predictions[0]["points_xy"]), 4)

    def test_row_scan_seed_trace_suppresses_duplicate_seed_trace(self) -> None:
        maps = _maps_with_vertical_gap()
        seed_map = torch.zeros((1, 12, 12), dtype=torch.float32)
        seed_map[0, 11, 5] = 0.95
        maps["seed_map"] = seed_map

        predictions = vectorize_lane_segfirst_maps(
            maps,
            config=LaneSegFirstVectorizerConfig(
                track_mode="row_scan_tangent_seed_trace",
                centerline_threshold=0.5,
                row_stride=1,
                max_row_gap=3,
                max_link_dx=1.0,
                seed_threshold=0.5,
                seed_trace_max_seeds=4,
                lane_match_threshold=40.0,
            ),
        )

        self.assertEqual(len(predictions), 1)
        self.assertGreaterEqual(len(predictions[0]["points_xy"]), 4)

    def test_rejects_unknown_track_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "track_mode"):
            vectorize_lane_segfirst_maps(
                _maps_with_vertical_gap(),
                config=LaneSegFirstVectorizerConfig(track_mode="unknown"),
            )

    def test_residual_risk_targets_only_mark_bucketed_lanes(self) -> None:
        rows = [
            {"points_xy": torch.tensor([[320.0, 560.0], [330.0, 440.0]], dtype=torch.float32)},
            {"points_xy": torch.tensor([[40.0, 560.0], [60.0, 440.0]], dtype=torch.float32)},
            {"points_xy": torch.tensor([[320.0, 240.0], [330.0, 120.0]], dtype=torch.float32)},
        ]

        maps = render_lane_segfirst_targets(
            rows,
            lane_valid_mask=torch.tensor([True, True, True]),
            config=LaneSegFirstTargetConfig(output_hw=(80, 80)),
        )

        risk_core = maps["residual_risk_core"]
        risk_ring_negative = maps["residual_risk_ring_negative"]

        self.assertGreater(float(risk_core.sum()), 0.0)
        self.assertGreater(float(risk_ring_negative.sum()), 0.0)
        self.assertEqual(risk_core.shape, torch.Size([1, 80, 80]))
        self.assertEqual(risk_ring_negative.shape, torch.Size([1, 80, 80]))
        self.assertEqual(
            int(((risk_ring_negative > 0.0) & (maps["support"] > 0.0)).sum().item()),
            0,
        )


if __name__ == "__main__":
    unittest.main()
