from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_VALID_SEG_OUTPUT_STRIDES = {1, 2}


def _validate_seg_output_stride(seg_output_stride: int) -> int:
    stride = int(seg_output_stride)
    if stride not in _VALID_SEG_OUTPUT_STRIDES:
        raise ValueError(f"invalid seg_output_stride: {stride}")
    return stride


@dataclass(frozen=True)
class PV26MultiHeadOutput:
    # Detection output:
    # - stub: dense logits [B, (4 + 1 + num_classes), H/8, W/8]
    # - YOLO26: Ultralytics Detect output (train: dict(one2many/one2one), eval: (y, preds))
    det: Any
    # Drivable logits: [B, 1, H, W]
    da: Tensor
    # Road-marking logits: [B, 3, H, W]
    rm: Tensor
    # Lane-subclass logits: [B, 5, H, W] => [background + 4 subclasses]
    rm_lane_subclass: Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class TinyPV26Backbone(nn.Module):
    """
    Small backbone for integration/smoke tests.
    It is not a production YOLO26 backbone; it is a shape-compatible placeholder
    for multi-head integration until full YOLO26 modules are wired in.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.s2 = ConvBNAct(in_ch, 32, k=3, s=2)
        self.s4 = ConvBNAct(32, 64, k=3, s=2)
        self.s8 = ConvBNAct(64, 128, k=3, s=2)
        self.s16 = ConvBNAct(128, 192, k=3, s=2)
        self.s32 = ConvBNAct(192, 256, k=3, s=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.s2(x)
        x = self.s4(x)
        p3 = self.s8(x)   # stride 8
        p4 = self.s16(p3) # stride 16
        p5 = self.s32(p4) # stride 32
        return p3, p4, p5


class TinyFPN(nn.Module):
    """
    Lightweight neck: fuses P3/P4/P5 and returns one stride-8 feature map.
    """

    def __init__(self, ch_p3: int = 128, ch_p4: int = 192, ch_p5: int = 256, out_ch: int = 160):
        super().__init__()
        self.lat_p3 = ConvBNAct(ch_p3, out_ch, k=1, s=1)
        self.lat_p4 = ConvBNAct(ch_p4, out_ch, k=1, s=1)
        self.lat_p5 = ConvBNAct(ch_p5, out_ch, k=1, s=1)
        self.fuse = ConvBNAct(out_ch * 3, out_ch, k=3, s=1)

    def forward(self, p3: Tensor, p4: Tensor, p5: Tensor) -> Tensor:
        p3l = self.lat_p3(p3)
        p4l = F.interpolate(self.lat_p4(p4), size=p3.shape[-2:], mode="nearest")
        p5l = F.interpolate(self.lat_p5(p5), size=p3.shape[-2:], mode="nearest")
        fused = torch.cat([p3l, p4l, p5l], dim=1)
        return self.fuse(fused)


class DetectionHeadDense(nn.Module):
    """
    Dense detector head outputting channels for [bbox(4), obj(1), classes(C)].
    This defines interface/shape for early multi-head training wiring.
    """

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.out_ch = 5 + num_classes
        self.stem = ConvBNAct(in_ch, in_ch, k=3, s=1)
        self.pred = nn.Conv2d(in_ch, self.out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.pred(self.stem(x))


class DrivableAreaHeadP3(nn.Module):
    """
    DA branch from shallow stride-8 feature (pre-FPN), then explicit upsampling.
    """

    def __init__(self, in_ch: int, *, output_stride: int = 1):
        super().__init__()
        self.output_stride = _validate_seg_output_stride(output_stride)
        mid = max(32, in_ch // 2)
        self.stage0 = ConvBNAct(in_ch, mid, k=3, s=1)
        self.up8_to4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.stage1 = ConvBNAct(mid, mid, k=3, s=1)
        self.up4_to2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.stage2 = ConvBNAct(mid, mid // 2, k=3, s=1)
        self.up2_to1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.pred = nn.Conv2d(mid // 2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, out_size: Tuple[int, int]) -> Tensor:
        x = self.stage0(x)
        x = self.up8_to4(x)
        x = self.stage1(x)
        x = self.up4_to2(x)
        x = self.stage2(x)
        if self.output_stride == 1:
            x = self.up2_to1(x)
        logits = self.pred(x)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class RoadMarkingHeadDeconv(nn.Module):
    """
    RM branch from fused neck feature (post-FPN) with deconvolution blocks.
    """

    def __init__(self, in_ch: int, out_ch: int = 3, *, output_stride: int = 1):
        super().__init__()
        self.decoder = RoadMarkingDecoderDeconv(in_ch=in_ch, output_stride=output_stride)
        self.pred = RoadMarkingPredictionHead(in_ch=self.decoder.out_ch, out_ch=out_ch)

    def forward(self, x: Tensor, out_size: Tuple[int, int]) -> Tensor:
        return self.pred(self.decoder(x, out_size))


class RoadMarkingDecoderDeconv(nn.Module):
    """
    Shared RM decoder trunk up to full-resolution hidden features.
    """

    def __init__(self, in_ch: int, *, output_stride: int = 1):
        super().__init__()
        self.output_stride = _validate_seg_output_stride(output_stride)
        c1 = max(64, in_ch // 2)
        c2 = max(32, c1 // 2)
        c3 = max(16, c2 // 2)
        self.out_ch = c3
        self.stem = ConvBNAct(in_ch, c1, k=3, s=1)
        self.deconv1 = nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.deconv2 = nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c3)
        self.deconv3 = nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor, out_size: Tuple[int, int]) -> Tensor:
        x = self.stem(x)
        x = self.act(self.bn1(self.deconv1(x)))
        x = self.act(self.bn2(self.deconv2(x)))
        if self.output_stride == 1:
            x = self.act(self.bn3(self.deconv3(x)))
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class RoadMarkingPredictionHead(nn.Module):
    def __init__(self, *, in_ch: int, out_ch: int):
        super().__init__()
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.pred(x)


class PV26MultiHead(nn.Module):
    """
    YOLO-PV26 multi-head interface:
      shared trunk -> OD head + DA head + RM head
    """

    def __init__(
        self,
        num_det_classes: int = 11,
        num_lane_subclasses: int = 4,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        fused_ch: int = 160,
        seg_output_stride: int = 1,
    ):
        super().__init__()
        self.num_lane_subclasses = int(num_lane_subclasses)
        self.seg_output_stride = _validate_seg_output_stride(seg_output_stride)
        self.backbone = backbone if backbone is not None else TinyPV26Backbone()
        self.neck = neck if neck is not None else TinyFPN(out_ch=fused_ch)
        self.det_head = DetectionHeadDense(in_ch=fused_ch, num_classes=num_det_classes)
        # YOLOPv2-style split:
        # - DA from shallower feature (before neck/FPN)
        # - RM from deeper fused feature (after neck/FPN) with a shared decoder trunk
        self.da_head = DrivableAreaHeadP3(in_ch=128, output_stride=self.seg_output_stride)
        self.rm_decoder = RoadMarkingDecoderDeconv(in_ch=fused_ch, output_stride=self.seg_output_stride)
        self.rm_head = RoadMarkingPredictionHead(in_ch=self.rm_decoder.out_ch, out_ch=3)
        self.rm_lane_subclass_head = RoadMarkingPredictionHead(
            in_ch=self.rm_decoder.out_ch,
            out_ch=(self.num_lane_subclasses + 1),
        )

    def forward(self, x: Tensor) -> PV26MultiHeadOutput:
        in_h, in_w = x.shape[-2:]
        seg_out_size = (in_h // self.seg_output_stride, in_w // self.seg_output_stride)
        p3, p4, p5 = self.backbone(x)
        fused = self.neck(p3, p4, p5)
        det = self.det_head(fused)
        da = self.da_head(p3, out_size=seg_out_size)
        rm_hidden = self.rm_decoder(fused, out_size=seg_out_size)
        rm = self.rm_head(rm_hidden)
        rm_lane_subclass = self.rm_lane_subclass_head(rm_hidden)
        return PV26MultiHeadOutput(det=det, da=da, rm=rm, rm_lane_subclass=rm_lane_subclass)


class PV26MultiHeadYOLO26(nn.Module):
    """
    PV26 multi-task model using an Ultralytics YOLO26 detection trunk.

    - OD branch: Ultralytics Detect head output (train: dict(one2many/one2one), eval: (y, preds))
    - DA branch: from shallow backbone P3/8 feature (pre-neck)
    - RM branch: from fused head P3/8 feature (post-neck)

    Notes:
    - This class intentionally keeps the PV26 segmentation heads lightweight and independent.
    - YOLO26 internals are imported lazily to keep unit tests fast when only the stub model is used.
    """

    def __init__(
        self,
        *,
        num_det_classes: int = 11,
        num_lane_subclasses: int = 4,
        yolo26_cfg: str = "yolo26n.yaml",
        verbose: bool = False,
        seg_output_stride: int = 1,
    ):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
        self.num_lane_subclasses = int(num_lane_subclasses)
        self.yolo26_cfg = str(yolo26_cfg)
        self.seg_output_stride = _validate_seg_output_stride(seg_output_stride)

        # Lazy import: Ultralytics writes settings.json under a user config dir by default.
        # In this repo sandbox, $HOME may not be writable, so we default to /tmp.
        import os

        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

        from ultralytics.nn.tasks import DetectionModel  # type: ignore
        from ultralytics.utils import DEFAULT_CFG  # type: ignore

        self.det_model = DetectionModel(self.yolo26_cfg, ch=3, nc=self.num_det_classes, verbose=bool(verbose))
        # Loss classes expect model.args to be an IterableSimpleNamespace with .box/.cls/.dfl/.epochs, etc.
        self.det_model.args = DEFAULT_CFG

        self._feat: dict[str, Tensor] = {}

        def _save(name: str):
            def hook(_module, _inp, out):
                # out: Tensor
                self._feat[name] = out

            return hook

        # For Ultralytics YOLO26 (P3/8-P5/32) configs:
        # - backbone P3/8 feature corresponds to module index 4
        # - head P3/8 feature is available as preds["one2many"]["feats"][0]
        if len(self.det_model.model) <= 4:
            raise RuntimeError("unexpected yolo26 model layout: module count too small")
        self.det_model.model[4].register_forward_hook(_save("p3_backbone"))

        # Infer channels for segmentation heads with a small dry forward.
        was_training = bool(self.det_model.training)
        self.det_model.eval()  # avoid polluting BN running stats during init
        with torch.inference_mode():
            self._feat.clear()
            dummy = torch.zeros(1, 3, 256, 384)
            det_out = self.det_model(dummy)
            preds = self._extract_preds_dict(det_out, context="during init")
            if "p3_backbone" not in self._feat:
                raise RuntimeError("failed to capture backbone P3 feature (hook did not fire)")
            one2many = preds.get("one2many")
            if not isinstance(one2many, dict):
                raise RuntimeError("unexpected yolo26 one2many output during init")
            feats = one2many.get("feats")
            if not isinstance(feats, (list, tuple)) or len(feats) == 0 or not torch.is_tensor(feats[0]):
                raise RuntimeError("unexpected yolo26 one2many.feats output during init")
            p3_backbone = self._feat["p3_backbone"]
            p3_head = feats[0]
        self.det_model.train(was_training)

        self.da_head = DrivableAreaHeadP3(in_ch=int(p3_backbone.shape[1]), output_stride=self.seg_output_stride)
        self.rm_decoder = RoadMarkingDecoderDeconv(
            in_ch=int(p3_head.shape[1]),
            output_stride=self.seg_output_stride,
        )
        self.rm_head = RoadMarkingPredictionHead(in_ch=self.rm_decoder.out_ch, out_ch=3)
        self.rm_lane_subclass_head = RoadMarkingPredictionHead(
            in_ch=self.rm_decoder.out_ch,
            out_ch=(self.num_lane_subclasses + 1),
        )

    def forward(self, x: Tensor) -> PV26MultiHeadOutput:
        in_h, in_w = x.shape[-2:]
        seg_out_size = (in_h // self.seg_output_stride, in_w // self.seg_output_stride)
        self._feat.clear()
        det_out = self.det_model(x)

        preds = self._extract_preds_dict(det_out, context="")
        if "one2many" not in preds:
            raise RuntimeError("unexpected yolo26 forward output")
        one2many = preds.get("one2many")
        if not isinstance(one2many, dict):
            raise RuntimeError("unexpected yolo26 one2many output")
        feats = one2many.get("feats")
        if not isinstance(feats, (list, tuple)) or len(feats) == 0 or not torch.is_tensor(feats[0]):
            raise RuntimeError("unexpected yolo26 one2many.feats output")

        p3_backbone = self._feat.get("p3_backbone", None)
        if p3_backbone is None:
            raise RuntimeError("missing backbone P3 feature (hook did not fire)")
        p3_head = feats[0]

        da = self.da_head(p3_backbone, out_size=seg_out_size)
        rm_hidden = self.rm_decoder(p3_head, out_size=seg_out_size)
        rm = self.rm_head(rm_hidden)
        rm_lane_subclass = self.rm_lane_subclass_head(rm_hidden)
        return PV26MultiHeadOutput(det=det_out, da=da, rm=rm, rm_lane_subclass=rm_lane_subclass)

    @staticmethod
    def _extract_preds_dict(det_out: Any, *, context: str) -> dict[str, Any]:
        preds = det_out
        if isinstance(det_out, (tuple, list)):
            if len(det_out) >= 2 and isinstance(det_out[1], dict):
                preds = det_out[1]
            elif len(det_out) >= 1 and isinstance(det_out[0], dict):
                preds = det_out[0]
        if not isinstance(preds, dict):
            detail = f" {context}" if context else ""
            raise RuntimeError(f"unexpected yolo26 forward output{detail}")
        return preds
