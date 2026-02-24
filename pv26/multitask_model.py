from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PV26MultiHeadOutput:
    # Dense detection logits: [B, (4 + 1 + num_classes), H/8, W/8]
    det: Tensor
    # Drivable logits: [B, 1, H, W]
    da: Tensor
    # Road-marking logits: [B, 3, H, W]
    rm: Tensor


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

    def __init__(self, in_ch: int):
        super().__init__()
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
        x = self.up2_to1(x)
        logits = self.pred(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class RoadMarkingHeadDeconv(nn.Module):
    """
    RM branch from fused neck feature (post-FPN) with deconvolution blocks.
    """

    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        c1 = max(64, in_ch // 2)
        c2 = max(32, c1 // 2)
        c3 = max(16, c2 // 2)
        self.stem = ConvBNAct(in_ch, c1, k=3, s=1)
        self.deconv1 = nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.deconv2 = nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c3)
        self.deconv3 = nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.act = nn.SiLU(inplace=True)
        self.pred = nn.Conv2d(c3, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, out_size: Tuple[int, int]) -> Tensor:
        x = self.stem(x)
        x = self.act(self.bn1(self.deconv1(x)))
        x = self.act(self.bn2(self.deconv2(x)))
        x = self.act(self.bn3(self.deconv3(x)))
        logits = self.pred(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class PV26MultiHead(nn.Module):
    """
    YOLO-PV26 multi-head interface:
      shared trunk -> OD head + DA head + RM head
    """

    def __init__(
        self,
        num_det_classes: int = 11,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        fused_ch: int = 160,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyPV26Backbone()
        self.neck = neck if neck is not None else TinyFPN(out_ch=fused_ch)
        self.det_head = DetectionHeadDense(in_ch=fused_ch, num_classes=num_det_classes)
        # YOLOPv2-style split:
        # - DA from shallower feature (before neck/FPN)
        # - RM from deeper fused feature (after neck/FPN) with deconvolution
        self.da_head = DrivableAreaHeadP3(in_ch=128)
        self.rm_head = RoadMarkingHeadDeconv(in_ch=fused_ch, out_ch=3)

    def forward(self, x: Tensor) -> PV26MultiHeadOutput:
        in_h, in_w = x.shape[-2:]
        p3, p4, p5 = self.backbone(x)
        fused = self.neck(p3, p4, p5)
        det = self.det_head(fused)
        da = self.da_head(p3, out_size=(in_h, in_w))
        rm = self.rm_head(fused, out_size=(in_h, in_w))
        return PV26MultiHeadOutput(det=det, da=da, rm=rm)
