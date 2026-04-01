from __future__ import annotations

import unittest

import torch

from runtime_support import has_yolo26_runtime


class PV26TrunkFeatureTests(unittest.TestCase):
    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_real_trunk_feature_extractor_returns_p3_p4_p5(self) -> None:
        from model.net import PV26Heads
        from model.net import build_yolo26n_trunk
        from model.net.trunk import forward_pyramid_features

        adapter = build_yolo26n_trunk()
        with torch.no_grad():
            features = forward_pyramid_features(adapter, torch.zeros(1, 3, 608, 800))

        self.assertEqual(len(features), 3)
        self.assertEqual(tuple(features[0].shape), (1, 64, 76, 100))
        self.assertEqual(tuple(features[1].shape), (1, 128, 38, 50))
        self.assertEqual(tuple(features[2].shape), (1, 256, 19, 25))

        heads = PV26Heads(in_channels=(64, 128, 256))
        outputs = heads(features)
        self.assertEqual(tuple(outputs["det"].shape), (1, 9975, 12))
        self.assertEqual(tuple(outputs["tl_attr"].shape), (1, 9975, 4))


if __name__ == "__main__":
    unittest.main()
