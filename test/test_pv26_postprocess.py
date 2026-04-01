from __future__ import annotations

import math
import unittest

import torch

from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def _meta_identity() -> list[dict]:
    return [
        {
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
    ]


def _meta_letterboxed() -> list[dict]:
    return [
        {
            "sample_id": "sample_letterbox",
            "dataset_key": "synthetic",
            "split": "val",
            "image_path": "/tmp/sample_letterbox.jpg",
            "raw_hw": (720, 1280),
            "network_hw": (608, 800),
            "transform": {
                "scale": 0.625,
                "pad_left": 0,
                "pad_top": 79,
                "pad_right": 0,
                "pad_bottom": 79,
                "resized_hw": (450, 800),
            },
        }
    ]


def _make_prediction_batch() -> dict[str, torch.Tensor | list]:
    q_det = 76 * 100 + 38 * 50 + 19 * 25
    det = torch.zeros((1, q_det, 12), dtype=torch.float32)
    tl_attr = torch.zeros((1, q_det, 4), dtype=torch.float32)
    lane = torch.zeros((1, 12, 54), dtype=torch.float32)
    stop_line = torch.zeros((1, 6, 9), dtype=torch.float32)
    crosswalk = torch.zeros((1, 4, 17), dtype=torch.float32)

    high = _inverse_softplus(5.0)
    det[0, 1020, :4] = torch.tensor([high, high, high, high], dtype=torch.float32)
    det[0, 1020, 4] = 8.0
    det[0, 1020, 10] = 6.0
    tl_attr[0, 1020] = torch.tensor([5.0, -5.0, -5.0, 5.0], dtype=torch.float32)

    det[0, 1021, :4] = torch.tensor([high, high, high, high], dtype=torch.float32)
    det[0, 1021, 4] = 6.0
    det[0, 1021, 10] = 5.0
    tl_attr[0, 1021] = torch.tensor([4.0, -4.0, -4.0, 4.0], dtype=torch.float32)

    det[0, 1600, :4] = torch.tensor([high, high, high, high], dtype=torch.float32)
    det[0, 1600, 4] = 7.0
    det[0, 1600, 11] = 6.0

    lane[0, 0, 0] = 8.0
    lane[0, 0, 2] = 6.0
    lane[0, 0, 5] = 6.0
    lane[0, 0, 6:38] = torch.tensor(
        [
            120.0,
            520.0,
            130.0,
            500.0,
            140.0,
            480.0,
            150.0,
            460.0,
            160.0,
            440.0,
            170.0,
            420.0,
            180.0,
            400.0,
            190.0,
            380.0,
            200.0,
            360.0,
            210.0,
            340.0,
            220.0,
            320.0,
            230.0,
            300.0,
            240.0,
            280.0,
            250.0,
            260.0,
            260.0,
            240.0,
            270.0,
            220.0,
        ],
        dtype=torch.float32,
    )
    lane[0, 0, 38:54] = 8.0

    stop_line[0, 0, 0] = 8.0
    stop_line[0, 0, 1:9] = torch.tensor([100.0, 500.0, 180.0, 500.0, 260.0, 500.0, 340.0, 500.0], dtype=torch.float32)

    crosswalk[0, 0, 0] = 8.0
    crosswalk[0, 0, 1:17] = torch.tensor(
        [200.0, 400.0, 260.0, 400.0, 320.0, 400.0, 380.0, 400.0, 380.0, 480.0, 320.0, 480.0, 260.0, 480.0, 200.0, 480.0],
        dtype=torch.float32,
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


class PV26PostprocessTests(unittest.TestCase):
    def test_postprocess_decodes_detection_tl_and_lane_family_outputs(self) -> None:
        predictions = postprocess_pv26_batch(
            _make_prediction_batch(),
            _meta_identity(),
            config=PV26PostprocessConfig(det_iou_threshold=0.5),
        )

        self.assertEqual(len(predictions), 1)
        sample = predictions[0]
        self.assertEqual(sample["meta"]["sample_id"], "sample_0")
        self.assertEqual(len(sample["detections"]), 2)
        self.assertEqual(sample["detections"][0]["class_name"], "traffic_light")
        self.assertGreater(sample["detections"][0]["tl_attr_scores"]["red"], 0.9)
        self.assertGreater(sample["detections"][0]["tl_attr_scores"]["arrow"], 0.9)
        self.assertEqual(sample["detections"][1]["class_name"], "sign")
        self.assertEqual(len(sample["lanes"]), 1)
        self.assertEqual(sample["lanes"][0]["class_name"], "yellow_lane")
        self.assertEqual(sample["lanes"][0]["lane_type"], "dotted")
        self.assertEqual(len(sample["stop_lines"]), 1)
        self.assertEqual(len(sample["crosswalks"]), 1)

    def test_postprocess_restores_raw_space_coordinates_from_letterbox_meta(self) -> None:
        predictions = postprocess_pv26_batch(
            _make_prediction_batch(),
            _meta_letterboxed(),
            config=PV26PostprocessConfig(det_iou_threshold=0.5),
        )

        traffic_light = predictions[0]["detections"][0]
        box = traffic_light["box_xyxy"]
        self.assertAlmostEqual(box[0], 198.4, places=1)
        self.assertAlmostEqual(box[1], 0.0, places=1)
        self.assertAlmostEqual(box[2], 326.4, places=1)
        self.assertAlmostEqual(box[3], 72.0, places=1)


if __name__ == "__main__":
    unittest.main()
