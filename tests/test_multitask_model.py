import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from pv26.multitask_model import (
    DrivableAreaHeadP3,
    PV26DetBackendOutput,
    PV26LegacyMultiHeadYOLO26,
    PV26MultiHead,
    PV26MultiHeadYOLO26,
    RoadMarkingHeadDeconv,
    UltralyticsYOLO26DetBackend,
    build_pv26_inference_model_from_state_dict,
    infer_pv26_checkpoint_layout,
)


class _FakeDetBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, bias=False))
        self.det_loss_adapter = object()
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

    def build_det_loss_adapter(self):
        return self.det_loss_adapter


class _FakeIndexedConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, i: int, f: int = -1):
        super().__init__()
        self.i = i
        self.f = f
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _FakeDetectModule(nn.Module):
    def __init__(self, *, i: int, f: int = -1):
        super().__init__()
        self.i = i
        self.f = f

    def forward(self, x):
        return {
            "one2many": {"feats": [x]},
            "one2one": {"feats": [x]},
        }


class _FakeUltralyticsLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                _FakeIndexedConv(3, 4, i=0),
                _FakeIndexedConv(4, 5, i=1),
                _FakeIndexedConv(5, 6, i=2),
                _FakeIndexedConv(6, 7, i=3),
                _FakeIndexedConv(7, 8, i=4),
                _FakeDetectModule(i=5),
            ]
        )
        self.save = [4]

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def _rename_det_backend_state_as_legacy_direct_det_model(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("det_backend.det_model."):
            key = key.replace("det_backend.det_model.", "det_model.", 1)
        renamed[key] = value
    return renamed


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
        self.assertIs(model.build_det_loss_adapter(), backend.det_loss_adapter)
        self.assertEqual(y.det, {"backend": "fake"})
        self.assertEqual(tuple(y.da.shape), (2, 1, 128, 192))
        self.assertEqual(tuple(y.rm.shape), (2, 3, 128, 192))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 128, 192))

    def test_legacy_yolo26_wrapper_accepts_fake_det_model(self):
        det_model = _FakeUltralyticsLikeModel()
        model = PV26LegacyMultiHeadYOLO26(
            num_det_classes=11,
            det_model=det_model,
            seg_output_stride=2,
        )
        x = torch.randn(2, 3, 256, 384)

        y = model(x)

        self.assertIs(model.det_model, det_model)
        self.assertEqual(tuple(y.da.shape), (2, 1, 128, 192))
        self.assertEqual(tuple(y.rm.shape), (2, 3, 128, 192))
        self.assertEqual(tuple(y.rm_lane_subclass.shape), (2, 5, 128, 192))

    def test_infer_checkpoint_layout_detects_current_and_legacy_formats(self):
        current_model = PV26MultiHeadYOLO26(
            num_det_classes=11,
            det_backend=_FakeDetBackend(),
        )
        legacy_model = PV26LegacyMultiHeadYOLO26(
            num_det_classes=11,
            det_model=_FakeUltralyticsLikeModel(),
        )
        legacy_shared_state = _rename_det_backend_state_as_legacy_direct_det_model(current_model.state_dict())

        self.assertEqual(
            infer_pv26_checkpoint_layout(current_model.state_dict()),
            "current_shared_rm_decoder",
        )
        self.assertEqual(
            infer_pv26_checkpoint_layout(legacy_shared_state),
            "legacy_shared_rm_decoder",
        )
        self.assertEqual(
            infer_pv26_checkpoint_layout(legacy_model.state_dict()),
            "legacy_separate_rm_heads",
        )

    def test_inference_builder_loads_current_checkpoint_layout(self):
        source = PV26MultiHeadYOLO26(
            num_det_classes=11,
            det_backend=_FakeDetBackend(),
            seg_output_stride=2,
        )

        loaded, layout = build_pv26_inference_model_from_state_dict(
            source.state_dict(),
            num_det_classes=11,
            seg_output_stride=2,
            det_backend=_FakeDetBackend(),
        )

        self.assertIsInstance(loaded, PV26MultiHeadYOLO26)
        self.assertEqual(layout, "current_shared_rm_decoder")
        self.assertEqual(set(loaded.state_dict().keys()), set(source.state_dict().keys()))

    def test_inference_builder_loads_legacy_shared_rm_checkpoint_layout(self):
        source = PV26MultiHeadYOLO26(
            num_det_classes=11,
            det_backend=_FakeDetBackend(),
            seg_output_stride=2,
        )
        legacy_shared_state = _rename_det_backend_state_as_legacy_direct_det_model(source.state_dict())

        loaded, layout = build_pv26_inference_model_from_state_dict(
            legacy_shared_state,
            num_det_classes=11,
            seg_output_stride=2,
            det_backend=_FakeDetBackend(),
        )

        self.assertIsInstance(loaded, PV26MultiHeadYOLO26)
        self.assertEqual(layout, "legacy_shared_rm_decoder")
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)

    def test_inference_builder_loads_legacy_checkpoint_layout(self):
        source = PV26LegacyMultiHeadYOLO26(
            num_det_classes=11,
            det_model=_FakeUltralyticsLikeModel(),
            seg_output_stride=2,
        )

        loaded, layout = build_pv26_inference_model_from_state_dict(
            source.state_dict(),
            num_det_classes=11,
            seg_output_stride=2,
            legacy_det_model=_FakeUltralyticsLikeModel(),
        )

        self.assertIsInstance(loaded, PV26LegacyMultiHeadYOLO26)
        self.assertEqual(layout, "legacy_separate_rm_heads")
        self.assertEqual(set(loaded.state_dict().keys()), set(source.state_dict().keys()))

    def test_ultralytics_backend_adapter_can_capture_p3_without_hooks(self):
        backend = UltralyticsYOLO26DetBackend.__new__(UltralyticsYOLO26DetBackend)
        nn.Module.__init__(backend)
        backend.det_model = _FakeUltralyticsLikeModel()

        out = backend.forward(torch.randn(2, 3, 32, 32))

        self.assertFalse(hasattr(backend, "_feat"))
        self.assertEqual(tuple(out.p3_backbone.shape), (2, 8, 32, 32))
        self.assertIs(out.p3_head, out.det["one2many"]["feats"][0])

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
