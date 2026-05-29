import unittest

import torch

from tools.probe_pv26_lane_flip_tta import (
    _merge_lane_dense_predictions,
    _merge_stop_line_outputs,
    _unflip_lane_dense_outputs,
    _unflip_lane_tangent_axis,
)


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


if __name__ == "__main__":
    unittest.main()
