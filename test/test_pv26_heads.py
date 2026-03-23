from __future__ import annotations

import unittest

import torch


class PV26HeadsTests(unittest.TestCase):
    def test_heads_produce_documented_output_shapes(self) -> None:
        from model.heads import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        features = [
            torch.randn(2, 64, 76, 100),
            torch.randn(2, 128, 38, 50),
            torch.randn(2, 256, 19, 25),
        ]

        outputs = heads(features)

        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["tl_attr"].shape), (2, 9975, 4))
        self.assertEqual(tuple(outputs["lane"].shape), (2, 12, 54))
        self.assertEqual(tuple(outputs["stop_line"].shape), (2, 6, 9))
        self.assertEqual(tuple(outputs["crosswalk"].shape), (2, 4, 17))
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])
        self.assertEqual(outputs["det_feature_strides"], [8, 16, 32])

    def test_heads_expose_feature_contract_metadata(self) -> None:
        from model.heads import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        summary = heads.describe()

        self.assertEqual(summary["feature_channels"], [64, 128, 256])
        self.assertEqual(summary["feature_strides"], [8, 16, 32])
        self.assertEqual(summary["det_dim"], 12)
        self.assertEqual(summary["tl_attr_dim"], 4)
        self.assertEqual(summary["lane_queries"], 12)
        self.assertEqual(summary["stop_line_queries"], 6)
        self.assertEqual(summary["crosswalk_queries"], 4)

    def test_heads_reject_wrong_feature_count(self) -> None:
        from model.heads import PV26Heads

        heads = PV26Heads(in_channels=(64, 128, 256))
        with self.assertRaisesRegex(ValueError, "3 feature maps"):
            heads([torch.randn(1, 64, 76, 100)])


if __name__ == "__main__":
    unittest.main()
