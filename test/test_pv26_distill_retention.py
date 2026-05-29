from __future__ import annotations

import unittest

import torch

from model.engine.loss import PV26MultiTaskLoss
from model.engine.trainer import DISTILL_TEACHER_CACHE_KEYS
from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW


class PV26DistillRetentionTests(unittest.TestCase):
    def test_teacher_cache_keys_include_segfirst_lane_outputs(self) -> None:
        expected = {
            "lane_seg_centerline_logits",
            "lane_seg_support_logits",
            "lane_seg_center_offset",
            "lane_seg_tangent_axis",
            "lane_seg_color_logits",
            "lane_seg_type_logits",
        }

        self.assertTrue(expected.issubset(set(DISTILL_TEACHER_CACHE_KEYS)))

    def test_segfirst_lane_distill_backprops_without_row_head_outputs(self) -> None:
        batch_size = 1
        height, width = ROADMARK_DENSE_OUTPUT_HW
        predictions = {
            "lane": torch.zeros((batch_size, 1, 1), requires_grad=True),
            "lane_seg_centerline_logits": torch.zeros((batch_size, 1, height, width), requires_grad=True),
            "lane_seg_support_logits": torch.zeros((batch_size, 1, height, width), requires_grad=True),
            "lane_seg_center_offset": torch.zeros((batch_size, 2, height, width), requires_grad=True),
            "lane_seg_tangent_axis": torch.zeros((batch_size, 2, height, width), requires_grad=True),
            "lane_seg_color_logits": torch.zeros((batch_size, 4, height, width), requires_grad=True),
            "lane_seg_type_logits": torch.zeros((batch_size, 3, height, width), requires_grad=True),
            "lane_feature": torch.zeros((batch_size, 8, height, width), requires_grad=True),
        }
        teacher_cache = {
            key: value.detach().clone() + 0.25
            for key, value in predictions.items()
            if key.startswith("lane_seg_") or key == "lane_feature"
        }
        encoded = {"teacher_cache": teacher_cache, "_distill_phase": "train"}

        criterion = PV26MultiTaskLoss(
            stage="stage_4_lane_family_finetune",
            task_mode="roadmark_joint",
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 0.0, "stop_line": 0.0, "crosswalk": 0.0},
            distill_enabled=True,
            distill_loss_weights={"lane": 1.0, "stop_line": 0.0, "crosswalk": 0.0},
        )

        loss, agreement = criterion._lane_distill_loss(predictions, encoded)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach().cpu()), 0.0)
        self.assertIsNotNone(agreement["logit_kl"])
        loss.backward()
        self.assertIsNotNone(predictions["lane_seg_centerline_logits"].grad)


if __name__ == "__main__":
    unittest.main()
