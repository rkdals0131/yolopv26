from __future__ import annotations

import unittest

import torch.nn as nn
import torch
from model.loss.spec import build_loss_spec
from test_pv26_eval_metrics import make_raw_sample_batch
from runtime_support import has_yolo26_runtime


OD_CLASSES = tuple(build_loss_spec()["model_contract"]["od_classes"])
TL_CLASS_ID = OD_CLASSES.index("traffic_light")


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
        "meta": [
            {
                "sample_id": f"sample_{index}",
                "dataset_key": "synthetic",
                "split": "train",
                "image_path": f"/tmp/sample_{index}.jpg",
                "raw_hw": (608, 800),
                "network_hw": (608, 800),
                "transform": {
                    "scale": 1.0,
                    "pad_left": 0,
                    "pad_top": 0,
                    "pad_right": 0,
                    "pad_bottom": 0,
                    "resized_hw": (608, 800),
                },
            }
            for index in range(batch_size)
        ],
    }


class _StaticAdapter:
    def __init__(self) -> None:
        self.raw_model = nn.Identity()


class _StaticHeads(nn.Module):
    def forward(self, features):
        del features
        q_det = 76 * 100 + 38 * 50 + 19 * 25
        det = torch.zeros((1, q_det, 12), dtype=torch.float32)
        tl_attr = torch.zeros((1, q_det, 4), dtype=torch.float32)
        lane = torch.zeros((1, 12, 54), dtype=torch.float32)
        stop_line = torch.zeros((1, 6, 9), dtype=torch.float32)
        crosswalk = torch.zeros((1, 4, 17), dtype=torch.float32)

        det[0, 1020, :4] = torch.tensor([1.8, 1.4, 1.8, 1.4], dtype=torch.float32)
        det[0, 1020, 4] = 8.0
        det[0, 1020, 10] = 6.0
        tl_attr[0, 1020] = torch.tensor([5.0, -5.0, -5.0, 5.0], dtype=torch.float32)

        det[0, 1600, :4] = torch.tensor([1.6, 1.2, 1.6, 1.2], dtype=torch.float32)
        det[0, 1600, 4] = 7.0
        det[0, 1600, 11] = 6.0

        lane[0, 0, 0] = 8.0
        lane[0, 0, 2] = 6.0
        lane[0, 0, 5] = 6.0
        lane[0, 0, 6:38] = torch.linspace(120.0, 300.0, 32)
        lane[0, 0, 38:54] = 8.0

        stop_line[0, 0, 0] = 8.0
        stop_line[0, 0, 1:9] = torch.tensor([100.0, 500.0, 180.0, 500.0, 260.0, 500.0, 340.0, 500.0])

        crosswalk[0, 0, 0] = 8.0
        crosswalk[0, 0, 1:17] = torch.tensor(
            [200.0, 400.0, 260.0, 400.0, 320.0, 400.0, 380.0, 400.0, 380.0, 480.0, 320.0, 480.0, 260.0, 480.0, 200.0, 480.0]
        )

        return {
            "det": det,
            "tl_attr": tl_attr,
            "lane": lane,
            "stop_line": stop_line,
            "crosswalk": crosswalk,
            "det_feature_shapes": [(76, 100), (38, 50), (19, 25)],
            "det_feature_strides": [8, 16, 32],
        }


class _ExplodingCriterion(nn.Module):
    def forward(self, predictions, encoded):  # type: ignore[override]
        del predictions, encoded
        raise AssertionError("criterion should not be called")


class PV26EvaluatorTests(unittest.TestCase):
    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_evaluator_returns_loss_and_count_summary(self) -> None:
        from model.eval import PV26Evaluator
        from model.heads import PV26Heads
        from model.trunk import build_yolo26n_trunk

        adapter = build_yolo26n_trunk()
        heads = PV26Heads(in_channels=(64, 128, 256))
        evaluator = PV26Evaluator(adapter, heads, stage="stage_0_smoke")

        summary = evaluator.evaluate_batch(_make_encoded_batch(batch_size=1, q_det=9975))

        self.assertEqual(summary["batch_size"], 1)
        self.assertGreater(summary["losses"]["total"], 0.0)
        self.assertEqual(summary["counts"]["det_gt"], 2)
        self.assertEqual(summary["counts"]["tl_attr_gt"], 1)
        self.assertEqual(summary["counts"]["lane_rows"], 1)
        self.assertEqual(summary["prediction_shapes"]["det"], [1, 9975, 12])
        self.assertEqual(summary["prediction_shapes"]["tl_attr"], [1, 9975, 4])

    def test_predict_batch_returns_postprocessed_prediction_bundle(self) -> None:
        from model.eval.pv26_evaluator import PV26Evaluator

        evaluator = PV26Evaluator(
            _StaticAdapter(),
            _StaticHeads(),
            stage="stage_0_smoke",
            criterion=_ExplodingCriterion(),
        )
        evaluator.forward_encoded_batch = lambda encoded: evaluator.heads(None)  # type: ignore[method-assign]

        predictions = evaluator.predict_batch(_make_encoded_batch(batch_size=1, q_det=9975))

        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]["detections"]), 2)
        self.assertEqual(predictions[0]["detections"][0]["class_name"], "traffic_light")
        self.assertGreater(predictions[0]["detections"][0]["tl_attr_scores"]["red"], 0.9)
        self.assertEqual(predictions[0]["detections"][1]["class_name"], "sign")
        self.assertEqual(len(predictions[0]["lanes"]), 1)
        self.assertEqual(predictions[0]["lanes"][0]["class_name"], "yellow_lane")
        self.assertEqual(predictions[0]["lanes"][0]["lane_type"], "dotted")
        self.assertEqual(len(predictions[0]["stop_lines"]), 1)
        self.assertEqual(len(predictions[0]["crosswalks"]), 1)

    def test_evaluate_batch_without_loss_still_returns_metrics(self) -> None:
        from model.eval.pv26_evaluator import PV26Evaluator

        evaluator = PV26Evaluator(
            _StaticAdapter(),
            _StaticHeads(),
            stage="stage_0_smoke",
            criterion=_ExplodingCriterion(),
        )
        evaluator.forward_encoded_batch = lambda encoded: evaluator.heads(None)  # type: ignore[method-assign]

        summary = evaluator.evaluate_batch(make_raw_sample_batch(), compute_loss=False)

        self.assertEqual(summary["losses"], {})
        self.assertIn("metrics", summary)
        self.assertIn("detector", summary["metrics"])
        self.assertIn("traffic_light", summary["metrics"])

    def test_evaluate_batch_on_raw_sample_batch_includes_metrics(self) -> None:
        from model.eval.pv26_evaluator import PV26Evaluator

        evaluator = PV26Evaluator(
            _StaticAdapter(),
            _StaticHeads(),
            stage="stage_0_smoke",
            criterion=_ExplodingCriterion(),
        )
        evaluator.forward_encoded_batch = lambda encoded: evaluator.heads(None)  # type: ignore[method-assign]

        summary = evaluator.evaluate_batch(make_raw_sample_batch(), compute_loss=False)

        self.assertIn("metrics", summary)
        self.assertIn("detector", summary["metrics"])
        self.assertIn("traffic_light", summary["metrics"])
        self.assertIn("lane", summary["metrics"])
        self.assertIn("map50", summary["metrics"]["detector"])

    def test_evaluate_batch_on_encoded_eval_batch_with_raw_bundle_includes_metrics(self) -> None:
        from model.encoding import encode_pv26_batch
        from model.eval.pv26_evaluator import PV26Evaluator

        raw_batch = make_raw_sample_batch()
        encoded_batch = encode_pv26_batch(raw_batch)
        encoded_batch["_raw_batch"] = {
            "det_targets": list(raw_batch["det_targets"]),
            "tl_attr_targets": list(raw_batch["tl_attr_targets"]),
            "lane_targets": list(raw_batch["lane_targets"]),
            "source_mask": list(raw_batch["source_mask"]),
            "valid_mask": list(raw_batch["valid_mask"]),
            "meta": list(raw_batch["meta"]),
        }

        evaluator = PV26Evaluator(
            _StaticAdapter(),
            _StaticHeads(),
            stage="stage_0_smoke",
            criterion=_ExplodingCriterion(),
        )
        evaluator.forward_encoded_batch = lambda encoded: evaluator.heads(None)  # type: ignore[method-assign]

        summary = evaluator.evaluate_batch(encoded_batch, compute_loss=False)

        self.assertIn("metrics", summary)
        self.assertIn("detector", summary["metrics"])
        self.assertIn("lane", summary["metrics"])
        self.assertGreaterEqual(summary["metrics"]["lane"]["f1"], 0.0)

    def test_encoded_eval_collate_preserves_raw_supervision_for_metrics(self) -> None:
        from model.loading import collate_pv26_encoded_eval_batch

        raw_batch = make_raw_sample_batch()
        sample = {
            "image": raw_batch["image"][0],
            "det_targets": raw_batch["det_targets"][0],
            "tl_attr_targets": raw_batch["tl_attr_targets"][0],
            "lane_targets": raw_batch["lane_targets"][0],
            "source_mask": raw_batch["source_mask"][0],
            "valid_mask": raw_batch["valid_mask"][0],
            "meta": raw_batch["meta"][0],
        }

        encoded_batch = collate_pv26_encoded_eval_batch([sample])

        self.assertIn("_raw_batch", encoded_batch)
        self.assertIn("det_targets", encoded_batch["_raw_batch"])
        self.assertIn("lane_targets", encoded_batch["_raw_batch"])
        self.assertIn("meta", encoded_batch["_raw_batch"])
        self.assertNotIn("image", encoded_batch["_raw_batch"])


if __name__ == "__main__":
    unittest.main()
