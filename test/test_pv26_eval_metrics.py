from __future__ import annotations

import unittest

import torch

from model.eval import PV26MetricConfig, summarize_pv26_metrics


def _identity_meta() -> dict:
    return {
        "sample_id": "sample_0",
        "dataset_key": "synthetic",
        "split": "val",
        "image_path": "/tmp/sample_0.jpg",
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


def make_raw_sample_batch() -> dict:
    return {
        "image": torch.zeros((1, 3, 608, 800), dtype=torch.float32),
        "det_targets": [
            {
                "boxes_xyxy": torch.tensor(
                    [
                        [100.0, 100.0, 160.0, 220.0],
                        [300.0, 80.0, 360.0, 180.0],
                    ],
                    dtype=torch.float32,
                ),
                "classes": torch.tensor([5, 6], dtype=torch.long),
            }
        ],
        "tl_attr_targets": [
            {
                "bits": torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "is_traffic_light": torch.tensor([True, False], dtype=torch.bool),
                "collapse_reason": ["valid", "not_traffic_light"],
            }
        ],
        "lane_targets": [
            {
                "lanes": [
                    {
                        "points_xy": torch.tensor(
                            [[120.0, 520.0], [160.0, 440.0], [200.0, 360.0], [240.0, 280.0]],
                            dtype=torch.float32,
                        ),
                        "color": 1,
                        "lane_type": 1,
                    }
                ],
                "stop_lines": [
                    {
                        "points_xy": torch.tensor(
                            [[100.0, 500.0], [180.0, 500.0], [260.0, 500.0], [340.0, 500.0]],
                            dtype=torch.float32,
                        )
                    }
                ],
                "crosswalks": [
                    {
                        "points_xy": torch.tensor(
                            [[200.0, 400.0], [380.0, 400.0], [380.0, 480.0], [200.0, 480.0]],
                            dtype=torch.float32,
                        )
                    }
                ],
            }
        ],
        "source_mask": [
            {
                "det": True,
                "tl_attr": True,
                "lane": True,
                "stop_line": True,
                "crosswalk": True,
            }
        ],
        "valid_mask": [
            {
                "det": torch.tensor([True, True], dtype=torch.bool),
                "tl_attr": torch.tensor([True, False], dtype=torch.bool),
                "lane": torch.tensor([True], dtype=torch.bool),
                "stop_line": torch.tensor([True], dtype=torch.bool),
                "crosswalk": torch.tensor([True], dtype=torch.bool),
            }
        ],
        "meta": [_identity_meta()],
    }


def make_prediction_bundle() -> list[dict]:
    return [
        {
            "meta": _identity_meta(),
            "detections": [
                {
                    "box_xyxy": [100.0, 100.0, 160.0, 220.0],
                    "score": 0.95,
                    "class_id": 5,
                    "class_name": "traffic_light",
                    "tl_attr_scores": {"red": 0.95, "yellow": 0.05, "green": 0.05, "arrow": 0.90},
                },
                {
                    "box_xyxy": [300.0, 80.0, 360.0, 180.0],
                    "score": 0.90,
                    "class_id": 6,
                    "class_name": "sign",
                    "tl_attr_scores": {"red": 0.0, "yellow": 0.0, "green": 0.0, "arrow": 0.0},
                },
                {
                    "box_xyxy": [500.0, 500.0, 560.0, 560.0],
                    "score": 0.70,
                    "class_id": 0,
                    "class_name": "vehicle",
                    "tl_attr_scores": {"red": 0.0, "yellow": 0.0, "green": 0.0, "arrow": 0.0},
                },
            ],
            "lanes": [
                {
                    "score": 0.9,
                    "class_name": "yellow_lane",
                    "lane_type": "dotted",
                    "points_xy": [[125.0, 520.0], [165.0, 440.0], [205.0, 360.0], [245.0, 280.0]],
                }
            ],
            "stop_lines": [
                {
                    "score": 0.9,
                    "points_xy": [[105.0, 500.0], [185.0, 500.0], [265.0, 500.0], [345.0, 500.0]],
                }
            ],
            "crosswalks": [
                {
                    "score": 0.9,
                    "points_xy": [[205.0, 405.0], [385.0, 405.0], [385.0, 485.0], [205.0, 485.0]],
                }
            ],
        }
    ]


class PV26EvalMetricsTests(unittest.TestCase):
    def test_metrics_runtime_summarizes_detector_tl_and_lane_family_scores(self) -> None:
        summary = summarize_pv26_metrics(
            make_prediction_bundle(),
            make_raw_sample_batch(),
            config=PV26MetricConfig(),
        )

        self.assertAlmostEqual(summary["detector"]["precision"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["detector"]["recall"], 1.0, places=6)
        self.assertAlmostEqual(summary["detector"]["map50"], 1.0, places=6)
        self.assertEqual(summary["detector"]["per_class"]["traffic_light"]["tp"], 1)
        self.assertEqual(summary["detector"]["per_class"]["sign"]["tp"], 1)
        self.assertEqual(summary["detector"]["per_class"]["vehicle"]["fp"], 1)

        self.assertEqual(summary["traffic_light"]["matched_pairs"], 1)
        self.assertAlmostEqual(summary["traffic_light"]["combo_accuracy"], 1.0, places=6)
        self.assertEqual(summary["traffic_light"]["per_bit"]["red"]["tp"], 1)
        self.assertEqual(summary["traffic_light"]["per_bit"]["arrow"]["tp"], 1)

        self.assertEqual(summary["lane"]["tp"], 1)
        self.assertAlmostEqual(summary["lane"]["color_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["lane"]["type_accuracy"], 1.0, places=6)
        self.assertLess(summary["lane"]["mean_point_distance"], 10.0)

        self.assertEqual(summary["stop_line"]["tp"], 1)
        self.assertAlmostEqual(summary["stop_line"]["mean_angle_error"], 0.0, places=6)

        self.assertEqual(summary["crosswalk"]["tp"], 1)
        self.assertGreater(summary["crosswalk"]["mean_polygon_iou"], 0.8)


if __name__ == "__main__":
    unittest.main()
