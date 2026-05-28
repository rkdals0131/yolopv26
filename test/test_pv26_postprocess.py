from __future__ import annotations

import math
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from model.engine.loss import build_loss_spec
from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
LANE_ANCHOR_COUNT = int(SPEC["heads"]["lane"]["target_encoding"]["anchor_rows"])
LANE_VECTOR_DIM = int(SPEC["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(SPEC["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(SPEC["heads"]["crosswalk"]["shape"].split(" x ")[-1])
CROSSWALK_POINT_COUNT = int(SPEC["heads"]["crosswalk"]["target_encoding"]["sequence_points"])
LANE_X_SLICE = slice(6, 6 + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


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
    lane = torch.zeros((1, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32)
    stop_line = torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32)
    crosswalk = torch.zeros((1, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32)

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
    lane[0, 0, LANE_X_SLICE] = torch.linspace(120.0, 270.0, LANE_ANCHOR_COUNT, dtype=torch.float32)
    lane[0, 0, LANE_VIS_SLICE] = 8.0
    lane[0, 1, 0] = 7.5
    lane[0, 1, 2] = 5.8
    lane[0, 1, 5] = 5.8
    lane[0, 1, LANE_X_SLICE] = torch.linspace(123.0, 273.0, LANE_ANCHOR_COUNT, dtype=torch.float32)
    lane[0, 1, LANE_VIS_SLICE] = 7.0

    stop_line[0, 0, 0] = 8.0
    stop_line[0, 0, 1:5] = torch.tensor([100.0, 500.0, 340.0, 500.0], dtype=torch.float32)
    stop_line[0, 0, 5] = 10.0
    stop_line[0, 1, 0] = 7.0
    stop_line[0, 1, 1:5] = torch.tensor([104.0, 500.0, 344.0, 500.0], dtype=torch.float32)
    stop_line[0, 1, 5] = 9.0

    crosswalk[0, 0, 0] = 8.0
    crosswalk[0, 0, 1:9] = torch.tensor([200.0, 400.0, 380.0, 400.0, 380.0, 480.0, 200.0, 480.0], dtype=torch.float32)
    crosswalk[0, 1, 0] = 7.0
    crosswalk[0, 1, 1:9] = torch.tensor([204.0, 404.0, 384.0, 404.0, 384.0, 484.0, 204.0, 484.0], dtype=torch.float32)

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

    def test_postprocess_thresholds_filter_detection_and_lane_predictions(self) -> None:
        predictions = _make_prediction_batch()

        suppressed = postprocess_pv26_batch(
            predictions,
            _meta_identity(),
            config=PV26PostprocessConfig(
                det_conf_threshold=0.999,
                lane_obj_threshold=0.9999,
                stop_line_obj_threshold=0.9999,
                crosswalk_obj_threshold=0.9999,
            ),
        )

        sample = suppressed[0]
        self.assertEqual(sample["detections"], [])
        self.assertEqual(sample["lanes"], [])
        self.assertEqual(sample["stop_lines"], [])
        self.assertEqual(sample["crosswalks"], [])

    def test_crosswalk_hull_polygon_mode_decodes_mask_component(self) -> None:
        predictions = _make_prediction_batch()
        predictions["det"] = torch.zeros_like(predictions["det"])
        predictions["lane"] = torch.zeros_like(predictions["lane"])
        predictions["stop_line"] = torch.zeros_like(predictions["stop_line"])
        predictions["crosswalk"] = torch.zeros_like(predictions["crosswalk"])
        crosswalk_mask = torch.full((1, 1, 20, 20), -8.0, dtype=torch.float32)
        crosswalk_mask[0, 0, 5:12, 6:15] = 8.0
        predictions["crosswalk_mask_logits"] = crosswalk_mask

        decoded = postprocess_pv26_batch(
            predictions,
            _meta_identity(),
            config=PV26PostprocessConfig(
                det_conf_threshold=0.999,
                lane_obj_threshold=0.999,
                stop_line_obj_threshold=0.999,
                crosswalk_obj_threshold=0.90,
                crosswalk_polygon_mode="hull",
                crosswalk_min_component_pixels=4,
                crosswalk_min_bbox_aspect=0.0,
                crosswalk_min_polygon_area_px=0.0,
            ),
        )

        self.assertEqual(len(decoded[0]["crosswalks"]), 1)
        self.assertEqual(len(decoded[0]["crosswalks"][0]["points_xy"]), CROSSWALK_POINT_COUNT)

    def test_stopline_haf_consensus_decodes_support_voted_segment(self) -> None:
        from model.data.roadmark_v2_targets import build_stopline_dense_targets

        targets = build_stopline_dense_targets(
            [{"points_xy": [[100.0, 500.0], [340.0, 500.0]]}],
            [True],
        )
        predictions = _make_prediction_batch()
        predictions["det"] = torch.zeros_like(predictions["det"])
        predictions["lane"] = torch.zeros_like(predictions["lane"])
        predictions["stop_line"] = torch.zeros_like(predictions["stop_line"])
        predictions["crosswalk"] = torch.zeros_like(predictions["crosswalk"])
        h, w = targets["stop_line_haf_valid"].shape[-2:]
        haf_valid_logits = torch.full((1, 1, h, w), -8.0, dtype=torch.float32)
        haf_valid_logits[0] = torch.where(
            targets["stop_line_haf_valid"] > 0.5,
            torch.full_like(targets["stop_line_haf_valid"], 8.0),
            haf_valid_logits[0],
        )
        predictions["stop_line_haf_endpoint"] = targets["stop_line_haf_endpoint"].unsqueeze(0)
        predictions["stop_line_haf_valid_logits"] = haf_valid_logits

        decoded = postprocess_pv26_batch(
            predictions,
            _meta_identity(),
            config=PV26PostprocessConfig(
                det_conf_threshold=0.999,
                lane_obj_threshold=0.999,
                crosswalk_obj_threshold=0.999,
                stop_line_haf_enabled=True,
                stop_line_haf_valid_threshold=0.90,
                stop_line_haf_min_votes=2,
                stop_line_haf_cluster_endpoint_tolerance=0.25,
                stop_line_haf_max_endpoint_covariance=0.01,
            ),
        )

        self.assertEqual(len(decoded[0]["stop_lines"]), 1)
        points = decoded[0]["stop_lines"][0]["points_xy"]
        self.assertAlmostEqual(points[0][0], 100.0, places=1)
        self.assertAlmostEqual(points[0][1], 500.0, places=1)
        self.assertAlmostEqual(points[-1][0], 340.0, places=1)
        self.assertAlmostEqual(points[-1][1], 500.0, places=1)

    def test_stopline_segment_set_decodes_normalized_segment(self) -> None:
        predictions = _make_prediction_batch()
        predictions["det"] = torch.zeros_like(predictions["det"])
        predictions["lane"] = torch.zeros_like(predictions["lane"])
        predictions["stop_line"] = torch.zeros_like(predictions["stop_line"])
        predictions["crosswalk"] = torch.zeros_like(predictions["crosswalk"])
        predictions["stop_line_segment_logits"] = torch.full((1, STOP_LINE_QUERY_COUNT), -8.0, dtype=torch.float32)
        predictions["stop_line_segment_logits"][0, 0] = 8.0
        predictions["stop_line_segment_points"] = torch.zeros((1, STOP_LINE_QUERY_COUNT, 2, 2), dtype=torch.float32)
        predictions["stop_line_segment_points"][0, 0, 0] = torch.tensor([100.0 / 800.0, 500.0 / 608.0])
        predictions["stop_line_segment_points"][0, 0, 1] = torch.tensor([340.0 / 800.0, 500.0 / 608.0])

        decoded = postprocess_pv26_batch(
            predictions,
            _meta_identity(),
            config=PV26PostprocessConfig(
                det_conf_threshold=0.999,
                lane_obj_threshold=0.999,
                crosswalk_obj_threshold=0.999,
                stop_line_segment_set_enabled=True,
                stop_line_segment_set_score_threshold=0.90,
            ),
        )

        self.assertEqual(len(decoded[0]["stop_lines"]), 1)
        points = decoded[0]["stop_lines"][0]["points_xy"]
        self.assertAlmostEqual(points[0][0], 100.0, places=1)
        self.assertAlmostEqual(points[0][1], 500.0, places=1)
        self.assertAlmostEqual(points[-1][0], 340.0, places=1)
        self.assertAlmostEqual(points[-1][1], 500.0, places=1)

    def test_postprocess_raises_when_torchvision_batched_nms_fails_by_default(self) -> None:
        predictions = _make_prediction_batch()
        torchvision_module = ModuleType("torchvision")
        ops_module = ModuleType("torchvision.ops")

        def _broken_batched_nms(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("nms backend missing")

        ops_module.batched_nms = _broken_batched_nms
        torchvision_module.ops = ops_module
        with patch.dict(sys.modules, {"torchvision": torchvision_module, "torchvision.ops": ops_module}):
            with self.assertRaisesRegex(RuntimeError, "nms backend missing"):
                postprocess_pv26_batch(
                    predictions,
                    _meta_identity(),
                    config=PV26PostprocessConfig(det_iou_threshold=0.5),
                )

    def test_postprocess_uses_python_nms_only_when_explicitly_enabled(self) -> None:
        predictions = _make_prediction_batch()
        torchvision_module = ModuleType("torchvision")
        ops_module = ModuleType("torchvision.ops")

        def _broken_batched_nms(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("nms backend missing")

        ops_module.batched_nms = _broken_batched_nms
        torchvision_module.ops = ops_module
        with patch.dict(sys.modules, {"torchvision": torchvision_module, "torchvision.ops": ops_module}):
            decoded = postprocess_pv26_batch(
                predictions,
                _meta_identity(),
                config=PV26PostprocessConfig(det_iou_threshold=0.5, allow_python_nms_fallback=True),
            )

        self.assertEqual(len(decoded[0]["detections"]), 2)


if __name__ == "__main__":
    unittest.main()
