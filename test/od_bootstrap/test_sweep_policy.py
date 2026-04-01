from __future__ import annotations

import unittest

from tools.od_bootstrap.teacher.policy import apply_policy_to_predictions, row_passes_policy
from tools.od_bootstrap.data.sweep import ClassPolicy


class ODBootstrapPolicyTests(unittest.TestCase):
    def test_row_passes_policy_applies_source_and_geometry_priors(self) -> None:
        policy = ClassPolicy(
            score_threshold=0.4,
            nms_iou_threshold=0.5,
            min_box_size=8,
            allowed_source_datasets=("aihub_traffic_seoul",),
            center_y_range=(0.0, 0.7),
            aspect_ratio_range=(0.2, 1.5),
            area_ratio_range=(0.0001, 0.1),
        )
        allowed_row = {"class_name": "traffic_light", "confidence": 0.9, "xyxy": [10.0, 10.0, 30.0, 50.0]}
        disallowed_source_row = {"class_name": "traffic_light", "confidence": 0.9, "xyxy": [10.0, 10.0, 30.0, 50.0]}
        low_position_row = {"class_name": "traffic_light", "confidence": 0.9, "xyxy": [10.0, 80.0, 30.0, 100.0]}

        self.assertTrue(
            row_passes_policy(
                row=allowed_row,
                policy=policy,
                dataset_key="aihub_traffic_seoul",
                image_width=100,
                image_height=100,
            )
        )
        self.assertFalse(
            row_passes_policy(
                row=disallowed_source_row,
                policy=policy,
                dataset_key="aihub_obstacle_seoul",
                image_width=100,
                image_height=100,
            )
        )
        self.assertFalse(
            row_passes_policy(
                row=low_position_row,
                policy=policy,
                dataset_key="aihub_traffic_seoul",
                image_width=100,
                image_height=100,
            )
        )

    def test_apply_policy_to_predictions_suppresses_cross_class_overlaps(self) -> None:
        policies = {
            "traffic_light": ClassPolicy(
                score_threshold=0.3,
                nms_iou_threshold=0.5,
                min_box_size=4,
                suppress_with_classes=("sign",),
                cross_class_iou_threshold=0.3,
            ),
            "sign": ClassPolicy(
                score_threshold=0.3,
                nms_iou_threshold=0.5,
                min_box_size=4,
                suppress_with_classes=("traffic_light",),
                cross_class_iou_threshold=0.3,
            ),
        }
        rows = [
            {"class_name": "traffic_light", "confidence": 0.95, "xyxy": [10.0, 10.0, 30.0, 50.0]},
            {"class_name": "sign", "confidence": 0.80, "xyxy": [12.0, 12.0, 28.0, 48.0]},
        ]

        kept = apply_policy_to_predictions(
            rows=rows,
            class_policy=policies,
            dataset_key="aihub_traffic_seoul",
            image_width=100,
            image_height=100,
            raw_boxes_by_class={},
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["class_name"], "traffic_light")

    def test_apply_policy_to_predictions_suppresses_against_raw_other_class(self) -> None:
        policies = {
            "traffic_light": ClassPolicy(
                score_threshold=0.3,
                nms_iou_threshold=0.5,
                min_box_size=4,
                suppress_with_classes=("sign",),
                cross_class_iou_threshold=0.3,
            ),
            "sign": ClassPolicy(
                score_threshold=0.3,
                nms_iou_threshold=0.5,
                min_box_size=4,
                suppress_with_classes=("traffic_light",),
                cross_class_iou_threshold=0.3,
            ),
        }
        rows = [{"class_name": "sign", "confidence": 0.90, "xyxy": [10.0, 10.0, 30.0, 50.0]}]

        kept = apply_policy_to_predictions(
            rows=rows,
            class_policy=policies,
            dataset_key="aihub_traffic_seoul",
            image_width=100,
            image_height=100,
            raw_boxes_by_class={"traffic_light": [[12.0, 12.0, 28.0, 48.0]]},
        )

        self.assertEqual(kept, [])
