import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from pv26.multitask_model import DrivableAreaHeadP3, PV26MultiHead, RoadMarkingHeadDeconv


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

    def test_da_head_skips_final_interpolate_when_size_already_matches(self):
        head = DrivableAreaHeadP3(in_ch=128)
        x = torch.randn(2, 128, 32, 48)

        with mock.patch("pv26.multitask_model.F.interpolate", wraps=F.interpolate) as interp:
            y = head(x, out_size=(256, 384))

        self.assertEqual(tuple(y.shape), (2, 1, 256, 384))
        self.assertEqual(interp.call_count, 3)

    def test_da_head_runs_final_interpolate_when_size_differs(self):
        head = DrivableAreaHeadP3(in_ch=128)
        x = torch.randn(2, 128, 32, 48)

        with mock.patch("pv26.multitask_model.F.interpolate", wraps=F.interpolate) as interp:
            y = head(x, out_size=(255, 383))

        self.assertEqual(tuple(y.shape), (2, 1, 255, 383))
        self.assertEqual(interp.call_count, 4)

    def test_rm_head_skips_final_interpolate_when_size_already_matches(self):
        head = RoadMarkingHeadDeconv(in_ch=160, out_ch=3)
        x = torch.randn(2, 160, 32, 48)

        with mock.patch("pv26.multitask_model.F.interpolate", wraps=F.interpolate) as interp:
            y = head(x, out_size=(256, 384))

        self.assertEqual(tuple(y.shape), (2, 3, 256, 384))
        self.assertEqual(interp.call_count, 0)

    def test_rm_head_runs_final_interpolate_when_size_differs(self):
        head = RoadMarkingHeadDeconv(in_ch=160, out_ch=3)
        x = torch.randn(2, 160, 32, 48)

        with mock.patch("pv26.multitask_model.F.interpolate", wraps=F.interpolate) as interp:
            y = head(x, out_size=(255, 383))

        self.assertEqual(tuple(y.shape), (2, 3, 255, 383))
        self.assertEqual(interp.call_count, 1)


if __name__ == "__main__":
    unittest.main()
