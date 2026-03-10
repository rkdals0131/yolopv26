import unittest

import torch

from pv26.eval.segmentation import lane_subclass_eval_valid_mask


class TestLaneSubclassEvalValidMask(unittest.TestCase):
    def test_eval_mask_requires_gt_lane_marker(self):
        rm_mask = torch.tensor(
            [
                [
                    [[1, 0], [1, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                ]
            ],
            dtype=torch.uint8,
        )
        lane_sub = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.uint8)
        has_rm = torch.tensor([[1, 0, 0]], dtype=torch.long)
        has_lane_sub = torch.tensor([1], dtype=torch.long)

        valid, supervised = lane_subclass_eval_valid_mask(
            rm_mask=rm_mask,
            rm_lane_subclass_mask=lane_sub,
            has_rm=has_rm,
            has_rm_lane_subclass=has_lane_sub,
        )

        self.assertEqual(valid.tolist(), [[[True, False], [True, False]]])
        self.assertEqual(supervised.tolist(), [True])

    def test_eval_mask_is_empty_without_lane_marker_supervision(self):
        rm_mask = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
        lane_sub = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.uint8)
        has_rm = torch.tensor([[0, 0, 0]], dtype=torch.long)
        has_lane_sub = torch.tensor([1], dtype=torch.long)

        valid, supervised = lane_subclass_eval_valid_mask(
            rm_mask=rm_mask,
            rm_lane_subclass_mask=lane_sub,
            has_rm=has_rm,
            has_rm_lane_subclass=has_lane_sub,
        )

        self.assertFalse(bool(valid.any()))
        self.assertEqual(supervised.tolist(), [False])


if __name__ == "__main__":
    unittest.main()
