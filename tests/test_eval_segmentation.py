import unittest

import torch

from pv26.eval.segmentation import lane_subclass_eval_valid_mask


class TestEvalSegmentation(unittest.TestCase):
    def test_lane_subclass_eval_mask_uses_gt_lane_and_positive_subclass_only(self):
        valid, supervised = lane_subclass_eval_valid_mask(
            rm_mask=torch.tensor(
                [
                    [
                        [[0, 1, 1], [1, 0, 1]],
                        [[255, 255, 255], [255, 255, 255]],
                        [[255, 255, 255], [255, 255, 255]],
                    ],
                    [
                        [[1, 1, 1], [1, 1, 1]],
                        [[255, 255, 255], [255, 255, 255]],
                        [[255, 255, 255], [255, 255, 255]],
                    ],
                ],
                dtype=torch.uint8,
            ),
            rm_lane_subclass_mask=torch.tensor(
                [
                    [[2, 1, 0], [255, 3, 4]],
                    [[1, 2, 3], [4, 0, 255]],
                ],
                dtype=torch.uint8,
            ),
            has_rm=torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.long),
            has_rm_lane_subclass=torch.tensor([1, 1], dtype=torch.long),
        )

        self.assertEqual(
            valid.to(dtype=torch.int64).tolist(),
            [
                [[0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0]],
            ],
        )
        self.assertEqual(supervised.to(dtype=torch.int64).tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
