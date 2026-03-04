import unittest

import torch

from pv26.multitask_model import PV26MultiHead


class TestPV26MultiHead(unittest.TestCase):
    def test_forward_shapes(self):
        model = PV26MultiHead(num_det_classes=11)
        x = torch.randn(2, 3, 256, 384)
        y = model(x)
        self.assertEqual(tuple(y.da.shape), (2, 1, 256, 384))
        self.assertEqual(tuple(y.rm.shape), (2, 3, 256, 384))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 256, 384))
        self.assertEqual(y.det.shape[1], 16)  # 4 bbox + 1 obj + 11 classes
        self.assertEqual(tuple(y.det.shape[-2:]), (32, 48))  # stride-8 grid


if __name__ == "__main__":
    unittest.main()
