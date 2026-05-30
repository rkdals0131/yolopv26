import unittest

import torch

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_lane_flip_tta import (
    _apply_lane_lateral_duplicate_variant,
    _apply_stop_line_source_mode,
    _merge_lane_outputs,
    _merge_lane_dense_predictions,
    _merge_stop_line_outputs,
    _unflip_lane_dense_outputs,
    _unflip_lane_tangent_axis,
)


def _identity_meta() -> dict[str, object]:
    transform = compute_letterbox_transform((608, 800), network_hw=(608, 800))
    return {
        "raw_hw": (608, 800),
        "network_hw": (608, 800),
        "transform": transform.as_meta(),
    }


class LaneFlipTTAProbeTest(unittest.TestCase):
    def test_unflip_tangent_axis_reverses_width_and_x_sign(self) -> None:
        tangent = torch.tensor(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]]]
        )

        unflipped = _unflip_lane_tangent_axis(tangent)

        expected = torch.tensor(
            [[[[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]], [[30.0, 20.0, 10.0], [60.0, 50.0, 40.0]]]]
        )
        self.assertTrue(torch.equal(unflipped, expected))

    def test_unflip_dense_outputs_only_changes_lane_dense_keys(self) -> None:
        predictions = {
            "lane_seg_centerline_logits": torch.tensor([[[[1.0, 2.0]]]]),
            "lane_seg_tangent_axis": torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),
            "stop_line_mask_logits": torch.tensor([[[[5.0, 6.0]]]]),
        }

        unflipped = _unflip_lane_dense_outputs(predictions)

        self.assertTrue(torch.equal(unflipped["lane_seg_centerline_logits"], torch.tensor([[[[2.0, 1.0]]]])))
        self.assertTrue(torch.equal(unflipped["lane_seg_tangent_axis"], torch.tensor([[[[-2.0, -1.0]], [[4.0, 3.0]]]])))
        self.assertIs(unflipped["stop_line_mask_logits"], predictions["stop_line_mask_logits"])

    def test_merge_centerline_avg_does_not_mutate_base(self) -> None:
        base = {"lane_seg_centerline_logits": torch.tensor([[[[1.0, 3.0]]]])}
        flipped = {"lane_seg_centerline_logits": torch.tensor([[[[5.0, 2.0]]]])}

        averaged = _merge_lane_dense_predictions(base, flipped, variant="flip_centerline_avg")

        self.assertTrue(torch.equal(averaged["lane_seg_centerline_logits"], torch.tensor([[[[3.0, 2.5]]]])))
        self.assertTrue(torch.equal(base["lane_seg_centerline_logits"], torch.tensor([[[[1.0, 3.0]]]])))

    def test_lane_task_mask_competition_suppresses_only_lane_centerline(self) -> None:
        base = {
            "lane_seg_centerline_logits": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            "crosswalk_mask_logits": torch.tensor([[[[0.0, 4.0]]]], dtype=torch.float32),
            "stop_line_mask_logits": torch.tensor([[[[9.0, 11.0]]]], dtype=torch.float32),
        }
        flipped = {"lane_seg_centerline_logits": torch.zeros((1, 1, 1, 2), dtype=torch.float32)}

        suppressed = _merge_lane_dense_predictions(
            base,
            flipped,
            variant="flip_centerline_avg_lane_cross_comp050",
        )

        self.assertLess(float(suppressed["lane_seg_centerline_logits"][0, 0, 0, 1]), -0.65)
        self.assertIs(suppressed["crosswalk_mask_logits"], base["crosswalk_mask_logits"])
        self.assertIs(suppressed["stop_line_mask_logits"], base["stop_line_mask_logits"])

    def test_lateral_duplicate_variant_adds_one_dense_supported_shifted_lane(self) -> None:
        centerline = torch.full((1, 1, 608, 800), -8.0)
        support = torch.full((1, 1, 608, 800), -8.0)
        for y in (120, 180, 240, 300):
            centerline[0, 0, y, 152] = 8.0
            support[0, 0, y, 152] = 8.0
        predictions = [
            {
                "lanes": [
                    {
                        "score": 0.9,
                        "class_name": "white_lane",
                        "lane_type": "solid",
                        "points_xy": [[100.0, 120.0], [100.0, 180.0], [100.0, 240.0], [100.0, 300.0]],
                    }
                ],
                "stop_lines": [],
                "crosswalks": [],
            }
        ]

        augmented, stats = _apply_lane_lateral_duplicate_variant(
            predictions,
            {
                "lane_seg_centerline_logits": centerline,
                "lane_seg_support_logits": support,
            },
            [_identity_meta()],
            variant="flip_centerline_avg_lane_cross_comp050_lateral_dup",
        )

        self.assertEqual(stats["added_duplicates"], 1)
        self.assertEqual(len(augmented[0]["lanes"]), 2)
        duplicate = augmented[0]["lanes"][1]
        self.assertTrue(duplicate["lateral_duplicate"])
        self.assertEqual([round(point[0]) for point in duplicate["points_xy"]], [152, 152, 152, 152])

    def test_lateral_duplicate_variant_rejects_low_dense_support(self) -> None:
        logits = torch.full((1, 1, 608, 800), -8.0)
        predictions = [
            {
                "lanes": [
                    {
                        "score": 0.9,
                        "class_name": "white_lane",
                        "lane_type": "solid",
                        "points_xy": [[100.0, 120.0], [100.0, 180.0], [100.0, 240.0], [100.0, 300.0]],
                    }
                ],
                "stop_lines": [],
                "crosswalks": [],
            }
        ]

        augmented, stats = _apply_lane_lateral_duplicate_variant(
            predictions,
            {
                "lane_seg_centerline_logits": logits,
                "lane_seg_support_logits": logits,
            },
            [_identity_meta()],
            variant="flip_centerline_avg_lane_cross_comp050_lateral_dup",
        )

        self.assertEqual(stats["added_duplicates"], 0)
        self.assertEqual(len(augmented[0]["lanes"]), 1)

    def test_merge_stop_line_outputs_replaces_only_stop_line_keys(self) -> None:
        base = {
            "lane_seg_centerline_logits": torch.tensor([1.0]),
            "crosswalk_mask_logits": torch.tensor([2.0]),
            "stop_line": torch.tensor([3.0]),
            "stop_line_mask_logits": torch.tensor([4.0]),
        }
        stop_line_outputs = {
            "lane_seg_centerline_logits": torch.tensor([10.0]),
            "crosswalk_mask_logits": torch.tensor([20.0]),
            "stop_line": torch.tensor([30.0]),
            "stop_line_mask_logits": torch.tensor([40.0]),
            "stop_line_center_logits": torch.tensor([50.0]),
        }

        merged = _merge_stop_line_outputs(base, stop_line_outputs)

        self.assertIs(merged["lane_seg_centerline_logits"], base["lane_seg_centerline_logits"])
        self.assertIs(merged["crosswalk_mask_logits"], base["crosswalk_mask_logits"])
        self.assertIs(merged["stop_line"], stop_line_outputs["stop_line"])
        self.assertIs(merged["stop_line_mask_logits"], stop_line_outputs["stop_line_mask_logits"])
        self.assertIs(merged["stop_line_center_logits"], stop_line_outputs["stop_line_center_logits"])

    def test_merge_lane_outputs_replaces_only_lane_keys(self) -> None:
        base = {
            "lane": torch.tensor([1.0]),
            "lane_seg_centerline_logits": torch.tensor([2.0]),
            "stop_line_mask_logits": torch.tensor([3.0]),
            "crosswalk_mask_logits": torch.tensor([4.0]),
        }
        lane_outputs = {
            "lane": torch.tensor([10.0]),
            "lane_seg_centerline_logits": torch.tensor([20.0]),
            "lane_feature": torch.tensor([30.0]),
            "stop_line_mask_logits": torch.tensor([40.0]),
            "crosswalk_mask_logits": torch.tensor([50.0]),
        }

        merged = _merge_lane_outputs(base, lane_outputs)

        self.assertIs(merged["lane"], lane_outputs["lane"])
        self.assertIs(merged["lane_seg_centerline_logits"], lane_outputs["lane_seg_centerline_logits"])
        self.assertIs(merged["lane_feature"], lane_outputs["lane_feature"])
        self.assertIs(merged["stop_line_mask_logits"], base["stop_line_mask_logits"])
        self.assertIs(merged["crosswalk_mask_logits"], base["crosswalk_mask_logits"])

    def test_stop_line_source_absent_fallback_uses_specialist_only_when_primary_empty(self) -> None:
        primary = [
            {"lanes": [], "stop_lines": [], "crosswalks": ["cw"]},
            {"lanes": [], "stop_lines": [{"points_xy": [[0.0, 0.0], [10.0, 0.0]], "score": 0.8}], "crosswalks": []},
        ]
        specialist = [
            {"lanes": [], "stop_lines": [{"points_xy": [[1.0, 0.0], [11.0, 0.0]], "score": 0.7}], "crosswalks": []},
            {"lanes": [], "stop_lines": [{"points_xy": [[100.0, 0.0], [110.0, 0.0]], "score": 0.9}], "crosswalks": []},
        ]

        routed = _apply_stop_line_source_mode(primary, specialist, mode="primary_absent_specialist")

        self.assertEqual(routed[0]["stop_lines"], specialist[0]["stop_lines"])
        self.assertEqual(routed[1]["stop_lines"], primary[1]["stop_lines"])
        self.assertEqual(routed[0]["crosswalks"], ["cw"])

    def test_stop_line_source_union_dedupes_close_lines_by_score(self) -> None:
        primary = [
            {
                "lanes": [],
                "stop_lines": [
                    {"points_xy": [[0.0, 0.0], [20.0, 0.0]], "score": 0.6},
                    {"points_xy": [[200.0, 0.0], [220.0, 0.0]], "score": 0.7},
                ],
                "crosswalks": [],
            }
        ]
        specialist = [
            {
                "lanes": [],
                "stop_lines": [
                    {"points_xy": [[2.0, 0.0], [22.0, 0.0]], "score": 0.9},
                    {"points_xy": [[400.0, 0.0], [420.0, 0.0]], "score": 0.5},
                ],
                "crosswalks": [],
            }
        ]

        routed = _apply_stop_line_source_mode(primary, specialist, mode="union_dedupe")

        self.assertEqual(len(routed[0]["stop_lines"]), 3)
        self.assertEqual(routed[0]["stop_lines"][0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
