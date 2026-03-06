import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from pv26.multitask_model import (
    DrivableAreaHeadP3,
    PV26DetBackendOutput,
    PV26MultiHead,
    PV26MultiHeadYOLO26,
    RoadMarkingHeadDeconv,
)


class _FakeDetBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_model = nn.Identity()
        self.p3_backbone_proj = nn.Conv2d(3, 8, kernel_size=1, stride=8)
        self.p3_head_proj = nn.Conv2d(3, 16, kernel_size=1, stride=8)

    def forward(self, x):
        p3_backbone = self.p3_backbone_proj(x)
        p3_head = self.p3_head_proj(x)
        return PV26DetBackendOutput(
            det={"backend": "fake"},
            p3_backbone=p3_backbone,
            p3_head=p3_head,
        )


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

    def test_forward_shapes_with_half_res_seg_outputs(self):
        model = PV26MultiHead(num_det_classes=11, seg_output_stride=2)
        x = torch.randn(2, 3, 256, 384)
        y = model(x)

        self.assertEqual(tuple(y.da.shape), (2, 1, 128, 192))
        self.assertEqual(tuple(y.rm.shape), (2, 3, 128, 192))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 128, 192))
        self.assertEqual(tuple(y.det.shape[-2:]), (32, 48))

    def test_shared_rm_decoder_runs_once_for_both_heads(self):
        model = PV26MultiHead(num_det_classes=11)
        x = torch.randn(2, 3, 256, 384)

        with mock.patch.object(model.rm_decoder, "forward", wraps=model.rm_decoder.forward) as decoder_forward:
            y = model(x)

        self.assertEqual(decoder_forward.call_count, 1)
        self.assertEqual(tuple(y.rm.shape), (2, 3, 256, 384))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 256, 384))

    def test_shared_rm_decoder_backward_smoke(self):
        model = PV26MultiHead(num_det_classes=11)
        x = torch.randn(2, 3, 64, 96, requires_grad=True)

        y = model(x)
        loss = y.rm.mean() + y.rm_lane_subclass.mean()
        loss.backward()

        self.assertIsNotNone(model.rm_decoder.stem.conv.weight.grad)
        self.assertIsNotNone(model.rm_head.pred.weight.grad)
        self.assertIsNotNone(model.rm_lane_subclass_head.pred.weight.grad)

    def test_yolo26_wrapper_accepts_fake_backend_and_preserves_det_model_contract(self):
        backend = _FakeDetBackend()
        model = PV26MultiHeadYOLO26(
            num_det_classes=11,
            det_backend=backend,
            seg_output_stride=2,
        )
        x = torch.randn(2, 3, 256, 384)

        y = model(x)

        self.assertIs(model.det_backend, backend)
        self.assertIs(model.det_model, backend.det_model)
        self.assertEqual(y.det, {"backend": "fake"})
        self.assertEqual(tuple(y.da.shape), (2, 1, 128, 192))
        self.assertEqual(tuple(y.rm.shape), (2, 3, 128, 192))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 128, 192))

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
