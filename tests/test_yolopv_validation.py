import unittest

import torch

from pv26.dataset.loading.manifest_dataset import Pv26Sample
from pv26.eval.bdd_yolopv2_validation import (
    BDD_NATIVE_DET_NAME_TO_ID,
    COCO_TO_BDD_NATIVE_DET,
    _bdd_native_det_boxes_from_record,
)
from pv26.eval.common_validation import binary_metric_summary_from_confusion
from pv26.eval.pv26_validation import LANE_SUBCLASS_GROUPS, _accumulate_lane_subclass_stats
from pv26.eval.yolopv2_validation import collate_yolopv2_eval, decode_yolopv2_predictions


class TestYoloPvValidationHelpers(unittest.TestCase):
    def test_decode_yolopv2_predictions_decodes_xywh_grid(self):
        raw_pred = torch.zeros((1, 6, 1, 1), dtype=torch.float32)
        anchor_grid = torch.tensor([[[[[10.0, 20.0]]]]], dtype=torch.float32)

        decoded = decode_yolopv2_predictions(
            [raw_pred],
            [anchor_grid],
            input_h=32,
            input_w=32,
        )

        self.assertEqual(tuple(decoded.shape), (1, 1, 6))
        self.assertAlmostEqual(float(decoded[0, 0, 0].item()), 16.0, places=4)
        self.assertAlmostEqual(float(decoded[0, 0, 1].item()), 16.0, places=4)
        self.assertAlmostEqual(float(decoded[0, 0, 2].item()), 10.0, places=4)
        self.assertAlmostEqual(float(decoded[0, 0, 3].item()), 20.0, places=4)

    def test_binary_metric_summary_from_confusion_reports_expected_scores(self):
        metric = binary_metric_summary_from_confusion(
            supervised_samples=2,
            valid_pixels=8,
            true_positive=2,
            false_positive=1,
            false_negative=1,
            true_negative=4,
        )

        self.assertAlmostEqual(metric.iou, 0.5)
        self.assertAlmostEqual(metric.precision, 2.0 / 3.0)
        self.assertAlmostEqual(metric.recall, 2.0 / 3.0)
        self.assertAlmostEqual(metric.f1, 2.0 / 3.0)

    def test_collate_yolopv2_eval_keeps_lane_marker_channel(self):
        sample = Pv26Sample(
            sample_id="sample-1",
            split="val",
            image=torch.zeros((3, 4, 4), dtype=torch.uint8),
            det_yolo=torch.zeros((0, 5), dtype=torch.float32),
            da_mask=torch.zeros((4, 4), dtype=torch.uint8),
            rm_mask=torch.stack(
                [
                    torch.ones((4, 4), dtype=torch.uint8),
                    torch.zeros((4, 4), dtype=torch.uint8),
                    torch.full((4, 4), 255, dtype=torch.uint8),
                ],
                dim=0,
            ),
            rm_lane_subclass_mask=torch.zeros((4, 4), dtype=torch.uint8),
            has_det=0,
            has_da=1,
            has_rm_lane_marker=1,
            has_rm_road_marker_non_lane=0,
            has_rm_stop_line=0,
            has_rm_lane_subclass=0,
            det_label_scope="full",
            det_annotated_class_ids="",
        )

        batch = collate_yolopv2_eval([sample])

        self.assertEqual(tuple(batch.images.shape), (1, 3, 4, 4))
        self.assertEqual(tuple(batch.rm_lane_mask.shape), (1, 4, 4))
        self.assertTrue(bool((batch.rm_lane_mask == 1).all()))

    def test_lane_subclass_group_aggregation_tracks_group_iou(self):
        stats = {
            "white_solid": {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "white_dashed": {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "yellow_solid": {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "yellow_dashed": {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0},
            **{
                name: {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
                for name in LANE_SUBCLASS_GROUPS
            },
        }
        pred = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
        target = torch.tensor([[[1, 1], [3, 4]]], dtype=torch.uint8)
        valid = torch.ones((1, 2, 2), dtype=torch.bool)

        _accumulate_lane_subclass_stats(stats, pred_class=pred, target_class=target, valid_mask=valid)

        white_all = binary_metric_summary_from_confusion(
            supervised_samples=stats["white_all"]["supervised_samples"],
            valid_pixels=stats["white_all"]["valid_pixels"],
            true_positive=stats["white_all"]["tp"],
            false_positive=stats["white_all"]["fp"],
            false_negative=stats["white_all"]["fn"],
            true_negative=stats["white_all"]["tn"],
        )
        self.assertAlmostEqual(float(white_all.iou), 1.0)

    def test_bdd_native_detection_parser_keeps_bdd10_classes(self):
        record = {
            "frames": [
                {
                    "objects": [
                        {"category": "person", "box2d": {"x1": 10, "y1": 20, "x2": 30, "y2": 60}},
                        {"category": "traffic light", "box2d": {"x1": 5, "y1": 8, "x2": 15, "y2": 18}},
                        {"category": "lane/single white", "box2d": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}},
                    ]
                }
            ]
        }

        det = _bdd_native_det_boxes_from_record(record, width=100, height=80)

        self.assertEqual(tuple(det.shape), (2, 5))
        self.assertEqual(int(det[0, 0].item()), BDD_NATIVE_DET_NAME_TO_ID["pedestrian"])
        self.assertEqual(int(det[1, 0].item()), BDD_NATIVE_DET_NAME_TO_ID["traffic light"])

    def test_coco_to_bdd_mapping_covers_overlap_classes_only(self):
        self.assertEqual(COCO_TO_BDD_NATIVE_DET[0], BDD_NATIVE_DET_NAME_TO_ID["pedestrian"])
        self.assertEqual(COCO_TO_BDD_NATIVE_DET[11], BDD_NATIVE_DET_NAME_TO_ID["traffic sign"])
        self.assertNotIn(12, COCO_TO_BDD_NATIVE_DET)


if __name__ == "__main__":
    unittest.main()
