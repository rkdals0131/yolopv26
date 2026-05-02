from __future__ import annotations

import unittest

import torch

from model.data.target_encoder import encode_pv26_batch
from model.engine.loss import PV26MultiTaskLoss
from model.net.heads import PV26Heads
from model.net.trunk import (
    YOLO26_ROADMARK_SOURCE_INDICES,
    YOLO26_ROADMARK_SOURCE_STRIDES,
    expected_roadmark_pyramid_channels,
)


class RoadmarkNativeContractTest(unittest.TestCase):
    def test_heads_keep_od_tl_on_p3_p5_and_roadmark_on_p2_p5(self) -> None:
        heads = PV26Heads((128, 128, 256, 512))
        features = (
            torch.randn(2, 128, 152, 200),
            torch.randn(2, 128, 76, 100),
            torch.randn(2, 256, 38, 50),
            torch.randn(2, 512, 19, 25),
        )

        outputs = heads(features, encoded={})

        self.assertTrue(heads.supports_encoded_context)
        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["tl_attr"].shape), (2, 9975, 4))
        self.assertEqual(tuple(outputs["lane"].shape), (2, 24, 38))
        self.assertEqual(tuple(outputs["stop_line"].shape), (2, 8, 9))
        self.assertEqual(tuple(outputs["crosswalk"].shape), (2, 8, 33))
        self.assertEqual(outputs["det_feature_strides"], [8, 16, 32])
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])

    def test_trunk_roadmark_pyramid_contract(self) -> None:
        self.assertEqual(YOLO26_ROADMARK_SOURCE_INDICES, (2, 16, 19, 22))
        self.assertEqual(YOLO26_ROADMARK_SOURCE_STRIDES, (4, 8, 16, 32))
        self.assertEqual(expected_roadmark_pyramid_channels(variant="s"), (128, 128, 256, 512))

    def test_target_encoder_emits_native_and_dense_roadmark_payloads(self) -> None:
        batch = {
            "image": torch.zeros((1, 3, 608, 800), dtype=torch.float32),
            "det_targets": [{"boxes_xyxy": torch.zeros((0, 4)), "classes": torch.zeros((0,), dtype=torch.long)}],
            "tl_attr_targets": [{"bits": torch.zeros((0, 4))}],
            "lane_targets": [
                {
                    "lanes": [
                        {
                            "points_xy": [(100.0, 600.0), (120.0, 300.0), (150.0, 20.0)],
                            "visibility": [1.0, 1.0, 1.0],
                            "color": 0,
                            "lane_type": 0,
                        }
                    ],
                    "stop_lines": [{"points_xy": [(200.0, 500.0), (350.0, 500.0), (360.0, 510.0)]}],
                    "crosswalks": [
                        {"points_xy": [(400.0, 420.0), (520.0, 420.0), (530.0, 520.0), (390.0, 520.0)]}
                    ],
                }
            ],
            "source_mask": [
                {"det": False, "tl_attr": False, "lane": True, "stop_line": True, "crosswalk": True}
            ],
            "valid_mask": [
                {
                    "det": torch.zeros((0,), dtype=torch.bool),
                    "tl_attr": torch.zeros((0,), dtype=torch.bool),
                    "lane": torch.ones((1,), dtype=torch.bool),
                    "stop_line": torch.ones((1,), dtype=torch.bool),
                    "crosswalk": torch.ones((1,), dtype=torch.bool),
                }
            ],
            "meta": [{"dataset_key": "unit", "sample_id": "roadmark_contract"}],
        }

        encoded = encode_pv26_batch(batch)

        self.assertEqual(tuple(encoded["lane"].shape), (1, 24, 38))
        self.assertEqual(tuple(encoded["stop_line"].shape), (1, 8, 9))
        self.assertEqual(tuple(encoded["crosswalk"].shape), (1, 8, 33))
        self.assertIn("roadmark_v2", encoded)
        self.assertEqual(tuple(encoded["roadmark_v2"]["crosswalk_mask"].shape), (1, 1, 152, 200))
        self.assertGreaterEqual(int(encoded["mask"]["lane_supervised_count"][0]), 1)

    def test_loss_exports_dynamic_coverage_knob_and_keeps_unified_task_mode(self) -> None:
        criterion = PV26MultiTaskLoss(
            loss_weights={"det": 1.0, "tl_attr": 1.0, "lane": 0.0, "stop_line": 0.0, "crosswalk": 0.0},
            task_mode="roadmark_joint",
            lane_dynamic_coverage_weight=0.1,
        )
        config = criterion.export_config()
        self.assertEqual(config["task_mode"], "roadmark_joint")
        self.assertEqual(config["lane_dynamic_coverage_weight"], 0.1)
        self.assertEqual(config["loss_weights"]["det"], 1.0)
        self.assertEqual(config["loss_weights"]["tl_attr"], 1.0)

    def test_segfirst_joint_loss_is_finite_on_synthetic_roadmark_batch(self) -> None:
        batch = {
            "image": torch.zeros((1, 3, 608, 800), dtype=torch.float32),
            "det_targets": [{"boxes_xyxy": torch.zeros((0, 4)), "classes": torch.zeros((0,), dtype=torch.long)}],
            "tl_attr_targets": [{"bits": torch.zeros((0, 4))}],
            "lane_targets": [
                {
                    "lanes": [
                        {
                            "points_xy": [(120.0, 590.0), (170.0, 390.0), (230.0, 160.0)],
                            "visibility": [1.0, 1.0, 1.0],
                            "color": 0,
                            "lane_type": 0,
                        }
                    ],
                    "stop_lines": [{"points_xy": [(260.0, 500.0), (430.0, 500.0)]}],
                    "crosswalks": [
                        {"points_xy": [(450.0, 410.0), (590.0, 410.0), (600.0, 500.0), (440.0, 500.0)]}
                    ],
                }
            ],
            "source_mask": [
                {"det": False, "tl_attr": False, "lane": True, "stop_line": True, "crosswalk": True}
            ],
            "valid_mask": [
                {
                    "det": torch.zeros((0,), dtype=torch.bool),
                    "tl_attr": torch.zeros((0,), dtype=torch.bool),
                    "lane": torch.ones((1,), dtype=torch.bool),
                    "stop_line": torch.ones((1,), dtype=torch.bool),
                    "crosswalk": torch.ones((1,), dtype=torch.bool),
                }
            ],
            "meta": [{"dataset_key": "unit", "sample_id": "segfirst_numeric_contract"}],
        }
        encoded = encode_pv26_batch(batch, include_lane_segfirst_targets=True)
        heads = PV26Heads((64, 64, 128, 256), lane_head_mode="seg_first")
        features = (
            torch.randn(1, 64, 152, 200),
            torch.randn(1, 64, 76, 100),
            torch.randn(1, 128, 38, 50),
            torch.randn(1, 256, 19, 25),
        )
        predictions = heads(features, encoded=encoded)
        for key in (
            "lane_seg_centerline_logits",
            "lane_seg_support_logits",
            "lane_seg_tangent_axis",
            "lane_seg_color_logits",
            "lane_seg_type_logits",
        ):
            predictions[key] = predictions[key].half()
        criterion = PV26MultiTaskLoss(
            stage="stage_3_end_to_end_finetune",
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 1.0, "stop_line": 1.0, "crosswalk": 1.0},
            task_mode="roadmark_joint",
        )

        losses = criterion(predictions, encoded)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]).item())
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "seg_first_dense")


if __name__ == "__main__":
    unittest.main()
