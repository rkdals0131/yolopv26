import unittest

import torch

from pv26.criterion import PV26Criterion
from pv26.multitask_model import PV26MultiHeadOutput


class TestPV26CriterionMasking(unittest.TestCase):
    def setUp(self):
        self.criterion = PV26Criterion(num_det_classes=2)

    def _base_batch(self):
        return {
            "det_yolo": [torch.zeros((0, 5), dtype=torch.float32)],
            "da_mask": torch.full((1, 2, 2), 255, dtype=torch.uint8),
            "rm_mask": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
            "has_det": torch.tensor([0], dtype=torch.long),
            "has_da": torch.tensor([0], dtype=torch.long),
            "has_rm_lane_marker": torch.tensor([0], dtype=torch.long),
            "has_rm_road_marker_non_lane": torch.tensor([0], dtype=torch.long),
            "has_rm_stop_line": torch.tensor([0], dtype=torch.long),
            "det_label_scope": ["none"],
        }

    def _preds(self):
        return PV26MultiHeadOutput(
            det=torch.zeros(1, 7, 1, 1),
            da=torch.zeros(1, 1, 2, 2),
            rm=torch.zeros(1, 3, 2, 2),
        )

    def test_da_loss_ignores_255_pixels(self):
        preds_a = self._preds()
        preds_b = self._preds()

        # Change only ignored pixel logit.
        preds_b.da[0, 0, 0, 0] = 100.0

        batch = self._base_batch()
        batch["has_da"] = torch.tensor([1], dtype=torch.long)
        batch["da_mask"] = torch.tensor([[[255, 1], [0, 1]]], dtype=torch.uint8)

        da_a = self.criterion(preds_a, batch)["da"]
        da_b = self.criterion(preds_b, batch)["da"]
        self.assertAlmostEqual(float(da_a), float(da_b), places=6)

    def test_da_loss_is_fully_masked_when_has_da_is_zero(self):
        preds = self._preds()
        preds.da.fill_(10.0)

        batch = self._base_batch()
        # Even with non-ignore labels, has_da=0 should hard-mask to zero.
        batch["da_mask"] = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
        batch["has_da"] = torch.tensor([0], dtype=torch.long)

        da_loss = self.criterion(preds, batch)["da"]
        self.assertEqual(float(da_loss), 0.0)

    def test_rm_stop_line_channel_is_zeroed_when_has_rm_stop_line_is_zero(self):
        preds_a = self._preds()
        preds_b = self._preds()

        # Perturb only stop_line channel logits.
        preds_b.rm[:, 2, :, :] = 20.0

        batch = self._base_batch()
        batch["has_rm_lane_marker"] = torch.tensor([1], dtype=torch.long)
        batch["has_rm_road_marker_non_lane"] = torch.tensor([1], dtype=torch.long)
        batch["has_rm_stop_line"] = torch.tensor([0], dtype=torch.long)
        batch["rm_mask"] = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)

        rm_a = self.criterion(preds_a, batch)["rm"]
        rm_b = self.criterion(preds_b, batch)["rm"]
        self.assertAlmostEqual(float(rm_a), float(rm_b), places=6)

    def test_od_loss_is_zero_when_det_label_scope_is_none(self):
        preds = self._preds()
        preds.det[:, 4, :, :] = 3.0  # non-zero objectness logits

        batch = self._base_batch()
        batch["has_det"] = torch.tensor([1], dtype=torch.long)
        batch["det_label_scope"] = ["none"]

        od_loss = self.criterion(preds, batch)["od"]
        self.assertEqual(float(od_loss), 0.0)

    def test_od_empty_gt_subset_is_zero_but_full_is_positive(self):
        preds = self._preds()
        preds.det[:, 4, :, :] = 3.0  # non-zero objectness logits

        batch = self._base_batch()
        batch["has_det"] = torch.tensor([1], dtype=torch.long)

        batch["det_label_scope"] = ["subset"]
        od_subset = self.criterion(preds, batch)["od"]

        batch["det_label_scope"] = ["full"]
        od_full = self.criterion(preds, batch)["od"]

        self.assertEqual(float(od_subset), 0.0)
        self.assertGreater(float(od_full), 0.0)


if __name__ == "__main__":
    unittest.main()
