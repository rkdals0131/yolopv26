from __future__ import annotations

import unittest

import torch
from model.engine.loss import build_loss_spec


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
LANE_VECTOR_DIM = int(SPEC["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(SPEC["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(SPEC["heads"]["crosswalk"]["shape"].split(" x ")[-1])


class PV26HeadsTests(unittest.TestCase):
    def test_heads_produce_documented_output_shapes(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        features = [
            torch.randn(2, 64, 76, 100),
            torch.randn(2, 128, 38, 50),
            torch.randn(2, 256, 19, 25),
        ]

        outputs = heads(features)

        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["tl_attr"].shape), (2, 9975, 4))
        self.assertEqual(tuple(outputs["lane"].shape), (2, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        self.assertEqual(tuple(outputs["stop_line"].shape), (2, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM))
        self.assertEqual(tuple(outputs["crosswalk"].shape), (2, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM))
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])
        self.assertEqual(outputs["det_feature_strides"], [8, 16, 32])

    def test_heads_expose_feature_contract_metadata(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        summary = heads.describe()

        self.assertEqual(summary["feature_channels"], [64, 128, 256])
        self.assertEqual(summary["feature_strides"], [8, 16, 32])
        self.assertEqual(summary["det_dim"], 12)
        self.assertEqual(summary["tl_attr_dim"], 4)
        self.assertEqual(summary["lane_queries"], LANE_QUERY_COUNT)
        self.assertEqual(summary["stop_line_queries"], STOP_LINE_QUERY_COUNT)
        self.assertEqual(summary["crosswalk_queries"], CROSSWALK_QUERY_COUNT)

    def test_heads_reject_wrong_feature_count(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        with self.assertRaisesRegex(ValueError, "3 feature maps"):
            heads([torch.randn(1, 64, 76, 100)])

    def test_heads_support_yolo26s_channel_contract(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(128, 256, 512))
        features = [
            torch.randn(2, 128, 76, 100),
            torch.randn(2, 256, 38, 50),
            torch.randn(2, 512, 19, 25),
        ]

        outputs = heads(features)

        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["lane"].shape), (2, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])


if __name__ == "__main__":
    unittest.main()
