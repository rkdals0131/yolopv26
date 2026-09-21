"""Shared YOLO26-s signal detector and stride-4 road-marking decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

from common.schema import ROADMARK_CLASSES, SIGNAL_CLASSES


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FEATURE_INDICES = (2, 16, 19, 22)


class _RoadmarkDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], width: int) -> None:
        super().__init__()
        self.p4 = nn.Sequential(nn.Conv2d(channels[2], width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU())
        self.p3 = nn.Sequential(nn.Conv2d(channels[1], width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU())
        self.p2 = nn.Sequential(nn.Conv2d(channels[0], width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU())
        self.refine_p3 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.SiLU())
        self.refine_p2 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.SiLU())
        self.logits = nn.Conv2d(width, len(ROADMARK_CLASSES), 1)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        fused_p3 = self.p3(p3) + F.interpolate(self.p4(p4), size=p3.shape[-2:], mode="nearest")
        fused_p3 = self.refine_p3(fused_p3)
        fused_p2 = self.p2(p2) + F.interpolate(fused_p3, size=p2.shape[-2:], mode="nearest")
        return self.logits(self.refine_p2(fused_p2))


class PV26FocusedModel(nn.Module):
    """Two-class official YOLO26 detector plus a lightweight P2/P3/P4 decoder.

    ``weights=None`` constructs the same architecture without reading a checkpoint,
    for loading a self-contained fine-tuned ``state_dict``. The default weights path
    is resolved inside this package; it never triggers an Ultralytics download.
    """

    def __init__(
        self,
        weights: str | Path | None = _PROJECT_ROOT / "yolo26s.pt",
        *,
        variant: str = "s",
        roadmark_width: int = 64,
    ) -> None:
        super().__init__()
        if variant != "s":
            raise ValueError("The focused model currently supports the YOLO26-s architecture.")
        if roadmark_width <= 0:
            raise ValueError("roadmark_width must be positive")
        self.variant = variant
        self.roadmark_width = int(roadmark_width)
        pretrained = None
        if weights is not None:
            weight_path = Path(weights).expanduser().resolve()
            if not weight_path.is_file():
                raise FileNotFoundError(f"YOLO26-s checkpoint not found: {weight_path}")
            pretrained = YOLO(str(weight_path)).model
            if (
                pretrained.yaml.get("scale") != "s"
                or not pretrained.model[-1].end2end
                or pretrained.model[-1].reg_max != 1
            ):
                raise ValueError("Expected an end-to-end YOLO26-s checkpoint")

        # DetectionModel creates the official Detect head, stride metadata and bias
        # initialization. The pretrained backbone, neck and shape-compatible head
        # parameters are then transferred; class-logit layers are new when their
        # source class count differs from the focused two-class contract.
        self.detector = DetectionModel(cfg="yolo26s.yaml", nc=len(SIGNAL_CLASSES), verbose=False)
        self.detector.args = get_cfg(DEFAULT_CFG)
        self.detector.names = dict(enumerate(SIGNAL_CLASSES))
        if pretrained is not None:
            source_state = pretrained.float().state_dict()
            target_state = self.detector.state_dict()
            matched = {
                key: value for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            self.detector.load_state_dict(matched, strict=False)

        # Actual YOLO26-s layer outputs: index 2 is stride 4, 16 is stride 8,
        # 19 is stride 16, and 22 is stride 32. The decoder uses P2/P3/P4.
        self.roadmark_decoder = _RoadmarkDecoder((128, 128, 256), self.roadmark_width)
        self._train_stage = "joint"

    def model_config(self) -> dict[str, Any]:
        return {"weights": None, "variant": self.variant, "roadmark_width": self.roadmark_width}

    @property
    def train_stage(self) -> str:
        return self._train_stage

    def set_train_stage(self, stage: str) -> None:
        if stage not in {"detector", "roadmark", "joint"}:
            raise ValueError("stage must be detector, roadmark, or joint")
        self._train_stage = stage
        self.train(self.training)

    def train(self, mode: bool = True) -> PV26FocusedModel:
        super().train(mode)
        if self._train_stage == "roadmark":
            self.detector.requires_grad_(False)
            self.detector.eval()  # keep pretrained BN statistics fixed during decoder warmup
            self.roadmark_decoder.requires_grad_(True)
        elif self._train_stage == "detector":
            self.detector.requires_grad_(True)
            self.roadmark_decoder.requires_grad_(False)
            self.roadmark_decoder.eval()
        else:
            self.detector.requires_grad_(True)
            self.roadmark_decoder.requires_grad_(True)
        return self

    def _features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor | None] = []
        current = image
        needed = set(self.detector.save).union(_FEATURE_INDICES)
        for layer in self.detector.model[:-1]:
            source = layer.f
            if source != -1:
                layer_input = outputs[source] if isinstance(source, int) else [
                    current if index == -1 else outputs[index] for index in source
                ]
            else:
                layer_input = current
            current = layer(layer_input)
            outputs.append(current if layer.i in needed else None)
        p2, p3, p4, p5 = (outputs[index] for index in _FEATURE_INDICES)
        return p2, p3, p4, p5

    def _raw_detection(self, features: list[torch.Tensor]) -> dict[str, Any]:
        head = self.detector.model[-1]
        return {
            "one2many": head.forward_head(features, **head.one2many),
            "one2one": head.forward_head([feature.detach() for feature in features], **head.one2one),
        }

    def forward(self, image: torch.Tensor, *, return_raw: bool = False) -> dict[str, Any]:
        p2, p3, p4, p5 = self._features(image)
        detect_features = [p3, p4, p5]
        if return_raw:
            det = self._raw_detection(detect_features)
        else:
            det = self.detector.model[-1](detect_features)
            if isinstance(det, tuple):
                det = det[0]
        roadmark_logits = self.roadmark_decoder(p2, p3, p4)
        return {"det": det, "roadmark_logits": roadmark_logits}

    def forward_for_loss(self, image: torch.Tensor) -> dict[str, Any]:
        """Return raw official detection predictions even with BN in eval mode."""
        p2, p3, p4, p5 = self._features(image)
        det = None if self._train_stage == "roadmark" else self._raw_detection([p3, p4, p5])
        roadmark_logits = None if self._train_stage == "detector" else self.roadmark_decoder(p2, p3, p4)
        return {"det": det, "roadmark_logits": roadmark_logits}

    def decode_raw_detection(self, raw: dict[str, Any]) -> torch.Tensor:
        """Official end-to-end xyxy/conf/class decoding without a second trunk pass."""
        head = self.detector.model[-1]
        decoded = head._inference(raw["one2one"])
        return head.postprocess(decoded.permute(0, 2, 1))

    def forward_export(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed tensor view: network-pixel xyxy/conf/class and stride-4 logits."""
        output = self.forward(image)
        if not isinstance(output["det"], torch.Tensor):
            raise RuntimeError("forward_export requires eval mode")
        return output["det"], output["roadmark_logits"]
