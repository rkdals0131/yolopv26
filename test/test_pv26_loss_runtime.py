from __future__ import annotations

import unittest

import torch

from model.loss.spec import build_loss_spec
from runtime_support import has_yolo26_runtime


OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")
SIGN_CLASS_ID = OD_CLASSES.index("sign")


def _make_encoded_batch(batch_size: int, q_det: int) -> dict:
    del q_det
    det_boxes = torch.zeros((batch_size, 3, 4), dtype=torch.float32)
    det_classes = torch.full((batch_size, 3), -1, dtype=torch.long)
    det_valid = torch.zeros((batch_size, 3), dtype=torch.bool)
    tl_bits = torch.zeros((batch_size, 3, 4), dtype=torch.float32)
    tl_mask = torch.zeros((batch_size, 3), dtype=torch.bool)

    lane = torch.zeros((batch_size, 12, 54), dtype=torch.float32)
    stop_line = torch.zeros((batch_size, 6, 9), dtype=torch.float32)
    crosswalk = torch.zeros((batch_size, 4, 17), dtype=torch.float32)
    lane_valid = torch.zeros((batch_size, 12), dtype=torch.bool)
    stop_line_valid = torch.zeros((batch_size, 6), dtype=torch.bool)
    crosswalk_valid = torch.zeros((batch_size, 4), dtype=torch.bool)

    for batch_index in range(batch_size):
        det_boxes[batch_index, 0] = torch.tensor([40.0, 50.0, 120.0, 180.0])
        det_boxes[batch_index, 1] = torch.tensor([220.0, 80.0, 280.0, 160.0])
        det_classes[batch_index, 0] = TL_CLASS_ID
        det_classes[batch_index, 1] = 0
        det_valid[batch_index, :2] = True
        tl_bits[batch_index, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        tl_mask[batch_index, 0] = True

        lane[batch_index, 0, 0] = 1.0
        lane[batch_index, 0, 1] = 1.0
        lane[batch_index, 0, 4] = 1.0
        lane[batch_index, 0, 6:38] = torch.linspace(0.0, 31.0, 32)
        lane[batch_index, 0, 38:54] = 1.0
        lane_valid[batch_index, 0] = True

        stop_line[batch_index, 0, 0] = 1.0
        stop_line[batch_index, 0, 1:9] = torch.linspace(0.0, 7.0, 8)
        stop_line_valid[batch_index, 0] = True

        crosswalk[batch_index, 0, 0] = 1.0
        crosswalk[batch_index, 0, 1:17] = torch.linspace(0.0, 15.0, 16)
        crosswalk_valid[batch_index, 0] = True

    return {
        "image": torch.randn(batch_size, 3, 608, 800),
        "det_gt": {
            "boxes_xyxy": det_boxes,
            "classes": det_classes,
            "valid_mask": det_valid,
        },
        "tl_attr_gt_bits": tl_bits,
        "tl_attr_gt_mask": tl_mask,
        "lane": lane,
        "stop_line": stop_line,
        "crosswalk": crosswalk,
        "mask": {
            "det_source": torch.ones(batch_size, dtype=torch.bool),
            "det_supervised_class_mask": torch.ones((batch_size, len(OD_CLASSES)), dtype=torch.bool),
            "det_allow_objectness_negatives": torch.ones(batch_size, dtype=torch.bool),
            "det_allow_unmatched_class_negatives": torch.ones(batch_size, dtype=torch.bool),
            "tl_attr_source": torch.ones(batch_size, dtype=torch.bool),
            "lane_source": torch.ones(batch_size, dtype=torch.bool),
            "stop_line_source": torch.ones(batch_size, dtype=torch.bool),
            "crosswalk_source": torch.ones(batch_size, dtype=torch.bool),
            "lane_valid": lane_valid,
            "stop_line_valid": stop_line_valid,
            "crosswalk_valid": crosswalk_valid,
        },
        "meta": [{"sample_id": f"sample_{index}"} for index in range(batch_size)],
    }


def _zero_predictions(batch_size: int, q_det: int) -> dict[str, torch.Tensor]:
    return {
        "det": torch.zeros((batch_size, q_det, 12), dtype=torch.float32, requires_grad=True),
        "tl_attr": torch.zeros((batch_size, q_det, 4), dtype=torch.float32, requires_grad=True),
        "lane": torch.zeros((batch_size, 12, 54), dtype=torch.float32, requires_grad=True),
        "stop_line": torch.zeros((batch_size, 6, 9), dtype=torch.float32, requires_grad=True),
        "crosswalk": torch.zeros((batch_size, 4, 17), dtype=torch.float32, requires_grad=True),
    }


class PV26LossRuntimeTests(unittest.TestCase):
    def test_loss_returns_finite_components_and_supports_backward(self) -> None:
        from model.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=2, q_det=8)
        predictions = {
            "det": torch.randn(2, 8, 12, requires_grad=True),
            "tl_attr": torch.randn(2, 8, 4, requires_grad=True),
            "lane": torch.randn(2, 12, 54, requires_grad=True),
            "stop_line": torch.randn(2, 6, 9, requires_grad=True),
            "crosswalk": torch.randn(2, 4, 17, requires_grad=True),
        }

        criterion = PV26MultiTaskLoss(stage="stage_0_smoke", allow_test_det_fallback=True)
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "prefix_smoke")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "hungarian")

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(losses["det"]))
        self.assertTrue(torch.isfinite(losses["tl_attr"]))
        self.assertTrue(torch.isfinite(losses["lane"]))
        self.assertTrue(torch.isfinite(losses["stop_line"]))
        self.assertTrue(torch.isfinite(losses["crosswalk"]))

        losses["total"].backward()
        self.assertIsNotNone(predictions["det"].grad)
        self.assertIsNotNone(predictions["tl_attr"].grad)
        self.assertIsNotNone(predictions["lane"].grad)

    def test_loss_handles_no_source_batch_without_nan(self) -> None:
        from model.loss import PV26MultiTaskLoss

        encoded = _make_encoded_batch(batch_size=1, q_det=8)
        for key in ("det_source", "tl_attr_source", "lane_source", "stop_line_source", "crosswalk_source"):
            encoded["mask"][key] = torch.zeros(1, dtype=torch.bool)
        encoded["det_gt"]["valid_mask"].zero_()
        encoded["tl_attr_gt_mask"].zero_()
        encoded["mask"]["lane_valid"].zero_()
        encoded["mask"]["stop_line_valid"].zero_()
        encoded["mask"]["crosswalk_valid"].zero_()
        encoded["lane"].zero_()
        encoded["stop_line"].zero_()
        encoded["crosswalk"].zero_()

        predictions = {
            "det": torch.randn(1, 8, 12, requires_grad=True),
            "tl_attr": torch.randn(1, 8, 4, requires_grad=True),
            "lane": torch.randn(1, 12, 54, requires_grad=True),
            "stop_line": torch.randn(1, 6, 9, requires_grad=True),
            "crosswalk": torch.randn(1, 4, 17, requires_grad=True),
        }

        criterion = PV26MultiTaskLoss(stage="stage_0_smoke")
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "zero_positive")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "hungarian")

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["total"].detach().cpu()), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(predictions["det"].grad)

    def test_tl_attr_loss_only_binds_to_traffic_light_matches(self) -> None:
        from model.loss import PV26MultiTaskLoss

        for class_id, expected_non_zero in ((TL_CLASS_ID, True), (SIGN_CLASS_ID, False)):
            encoded = {
                "image": torch.randn(1, 3, 608, 800),
                "det_gt": {
                    "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                    "classes": torch.tensor([[class_id]], dtype=torch.long),
                    "valid_mask": torch.tensor([[True]], dtype=torch.bool),
                },
                "tl_attr_gt_bits": torch.tensor([[[1.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
                "tl_attr_gt_mask": torch.tensor([[True]], dtype=torch.bool),
                "lane": torch.zeros((1, 12, 54), dtype=torch.float32),
                "stop_line": torch.zeros((1, 6, 9), dtype=torch.float32),
                "crosswalk": torch.zeros((1, 4, 17), dtype=torch.float32),
                "mask": {
                    "det_source": torch.tensor([True], dtype=torch.bool),
                    "det_supervised_class_mask": torch.ones((1, len(OD_CLASSES)), dtype=torch.bool),
                    "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                    "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                    "tl_attr_source": torch.tensor([True], dtype=torch.bool),
                    "lane_source": torch.tensor([False], dtype=torch.bool),
                    "stop_line_source": torch.tensor([False], dtype=torch.bool),
                    "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                    "lane_valid": torch.zeros((1, 12), dtype=torch.bool),
                    "stop_line_valid": torch.zeros((1, 6), dtype=torch.bool),
                    "crosswalk_valid": torch.zeros((1, 4), dtype=torch.bool),
                },
                "meta": [{"sample_id": f"class_{class_id}"}],
            }
            predictions = _zero_predictions(batch_size=1, q_det=2)
            criterion = PV26MultiTaskLoss(stage="stage_0_smoke", allow_test_det_fallback=True)

            losses = criterion(predictions, encoded)
            losses["tl_attr"].backward()
            tl_grad_sum = float(predictions["tl_attr"].grad.abs().sum().item())

            with self.subTest(class_id=class_id):
                if expected_non_zero:
                    self.assertGreater(float(losses["tl_attr"].detach().cpu()), 0.0)
                    self.assertGreater(tl_grad_sum, 0.0)
                else:
                    self.assertEqual(float(losses["tl_attr"].detach().cpu()), 0.0)
                    self.assertEqual(tl_grad_sum, 0.0)

    def test_partial_det_samples_enable_class_negatives_without_objectness_negatives(self) -> None:
        from model.loss import PV26MultiTaskLoss

        encoded = {
            "image": torch.randn(1, 3, 608, 800),
            "det_gt": {
                "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                "classes": torch.tensor([[TL_CLASS_ID]], dtype=torch.long),
                "valid_mask": torch.tensor([[True]], dtype=torch.bool),
            },
            "tl_attr_gt_bits": torch.zeros((1, 1, 4), dtype=torch.float32),
            "tl_attr_gt_mask": torch.zeros((1, 1), dtype=torch.bool),
            "lane": torch.zeros((1, 12, 54), dtype=torch.float32),
            "stop_line": torch.zeros((1, 6, 9), dtype=torch.float32),
            "crosswalk": torch.zeros((1, 4, 17), dtype=torch.float32),
            "mask": {
                "det_source": torch.tensor([True], dtype=torch.bool),
                "det_supervised_class_mask": torch.tensor([[False, False, False, False, False, True, True]], dtype=torch.bool),
                "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                "tl_attr_source": torch.tensor([False], dtype=torch.bool),
                "lane_source": torch.tensor([False], dtype=torch.bool),
                "stop_line_source": torch.tensor([False], dtype=torch.bool),
                "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                "lane_valid": torch.zeros((1, 12), dtype=torch.bool),
                "stop_line_valid": torch.zeros((1, 6), dtype=torch.bool),
                "crosswalk_valid": torch.zeros((1, 4), dtype=torch.bool),
            },
            "meta": [{"sample_id": "partial_det"}],
        }
        predictions = _zero_predictions(batch_size=1, q_det=2)

        criterion = PV26MultiTaskLoss(stage="stage_0_smoke", allow_test_det_fallback=True)
        losses = criterion(predictions, encoded)
        losses["det"].backward()

        det_grad = predictions["det"].grad[0]
        self.assertNotEqual(float(det_grad[0, 4]), 0.0)
        self.assertEqual(float(det_grad[1, 4]), 0.0)
        self.assertEqual(float(det_grad[1, 5:10].abs().sum()), 0.0)
        self.assertGreater(float(det_grad[1, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[1, 11].abs()), 0.0)
        self.assertGreater(float(det_grad[0, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[0, 11].abs()), 0.0)

    def test_large_query_class_negative_scaling_stays_finite(self) -> None:
        from model.loss import PV26MultiTaskLoss

        encoded = {
            "image": torch.randn(1, 3, 608, 800),
            "det_gt": {
                "boxes_xyxy": torch.tensor([[[40.0, 50.0, 120.0, 180.0]]], dtype=torch.float32),
                "classes": torch.tensor([[TL_CLASS_ID]], dtype=torch.long),
                "valid_mask": torch.tensor([[True]], dtype=torch.bool),
            },
            "tl_attr_gt_bits": torch.zeros((1, 1, 4), dtype=torch.float32),
            "tl_attr_gt_mask": torch.zeros((1, 1), dtype=torch.bool),
            "lane": torch.zeros((1, 12, 54), dtype=torch.float32),
            "stop_line": torch.zeros((1, 6, 9), dtype=torch.float32),
            "crosswalk": torch.zeros((1, 4, 17), dtype=torch.float32),
            "mask": {
                "det_source": torch.tensor([True], dtype=torch.bool),
                "det_supervised_class_mask": torch.tensor([[False, False, False, False, False, True, True]], dtype=torch.bool),
                "det_allow_objectness_negatives": torch.tensor([False], dtype=torch.bool),
                "det_allow_unmatched_class_negatives": torch.tensor([True], dtype=torch.bool),
                "tl_attr_source": torch.tensor([False], dtype=torch.bool),
                "lane_source": torch.tensor([False], dtype=torch.bool),
                "stop_line_source": torch.tensor([False], dtype=torch.bool),
                "crosswalk_source": torch.tensor([False], dtype=torch.bool),
                "lane_valid": torch.zeros((1, 12), dtype=torch.bool),
                "stop_line_valid": torch.zeros((1, 6), dtype=torch.bool),
                "crosswalk_valid": torch.zeros((1, 4), dtype=torch.bool),
            },
            "meta": [{"sample_id": "stress_det"}],
        }
        predictions = _zero_predictions(batch_size=1, q_det=9975)

        criterion = PV26MultiTaskLoss(stage="stage_0_smoke", allow_test_det_fallback=True)
        losses = criterion(predictions, encoded)
        losses["det"].backward()

        det_grad = predictions["det"].grad[0]
        self.assertEqual(float(det_grad[100, 4]), 0.0)
        self.assertEqual(float(det_grad[100, 5:10].abs().sum()), 0.0)
        self.assertGreater(float(det_grad[100, 10].abs()), 0.0)
        self.assertGreater(float(det_grad[100, 11].abs()), 0.0)
        self.assertTrue(torch.isfinite(losses["det"]))
        self.assertTrue(torch.isfinite(torch.tensor(float(criterion.last_det_loss_breakdown["det_cls_matched_loss"]))))
        self.assertTrue(torch.isfinite(torch.tensor(float(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_loss"]))))
        self.assertGreater(int(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_count"]), 1000)
        self.assertGreater(int(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_count"]), int(criterion.last_det_loss_breakdown["det_cls_matched_count"]))
        self.assertLessEqual(
            float(criterion.last_det_loss_breakdown["det_cls_unmatched_neg_loss"]),
            float(criterion.last_det_loss_breakdown["det_cls_matched_loss"]) * 2.0 + 1e-6,
        )

    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_real_trunk_heads_and_loss_support_backward_smoke(self) -> None:
        from model.heads import PV26Heads
        from model.loss import PV26MultiTaskLoss
        from model.trunk import build_yolo26n_trunk
        from model.trunk import forward_pyramid_features

        encoded = _make_encoded_batch(batch_size=1, q_det=9975)
        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        image = encoded["image"].clone().requires_grad_(False)

        features = forward_pyramid_features(adapter, image)
        predictions = heads(features)
        criterion = PV26MultiTaskLoss(stage="stage_0_smoke")
        losses = criterion(predictions, encoded)
        self.assertEqual(criterion.last_det_assignment_mode, "task_aligned")
        self.assertEqual(criterion.last_lane_assignment_modes["lane"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["stop_line"], "hungarian")
        self.assertEqual(criterion.last_lane_assignment_modes["crosswalk"], "hungarian")

        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

        head_parameter = next(heads.parameters())
        trunk_parameter = next(adapter.trunk.parameters())
        self.assertIsNotNone(head_parameter.grad)
        self.assertIsNotNone(trunk_parameter.grad)


if __name__ == "__main__":
    unittest.main()
