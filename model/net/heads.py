from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .roadmark_joint_native import ROADMARK_JOINT_NATIVE_NAME, PV26RoadMarkNativeJointHeads
from .roadmark_v2_heads import ROADMARK_V2_FEATURE_STRIDES


DET_DIM = 12
TL_ATTR_DIM = 4
LANE_QUERY_COUNT = 24
STOP_LINE_QUERY_COUNT = 8
CROSSWALK_QUERY_COUNT = 8
LANE_VECTOR_DIM = 38
STOP_LINE_VECTOR_DIM = 9
CROSSWALK_VECTOR_DIM = 33
FEATURE_STRIDES = ROADMARK_V2_FEATURE_STRIDES
DETECT_FEATURE_STRIDES = (8, 16, 32)


class _ScalePredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        prediction = self.block(feature)
        return prediction.flatten(2).transpose(1, 2).contiguous()


class PV26Heads(nn.Module):
    """Unified PV26 heads: OD/TL on P3-P5, native roadmark on P2-P5."""

    supports_encoded_context = True

    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = FEATURE_STRIDES,
        *,
        lane_head_mode: str = "seg_first",
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = str(lane_head_mode).strip().lower()
        if len(self.in_channels) != 4:
            raise ValueError("PV26Heads expects exactly 4 pyramid levels (P2/P3/P4/P5).")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26Heads expects exactly 4 feature strides.")
        if self.feature_strides != FEATURE_STRIDES:
            raise ValueError(f"PV26Heads expects feature strides {FEATURE_STRIDES}, got {self.feature_strides}.")

        self.det_in_channels = self.in_channels[1:]
        self.det_feature_strides = self.feature_strides[1:]
        self.det_heads = nn.ModuleList(
            [_ScalePredictionHead(channel, DET_DIM) for channel in self.det_in_channels]
        )
        self.tl_attr_heads = nn.ModuleList(
            [_ScalePredictionHead(channel, TL_ATTR_DIM) for channel in self.det_in_channels]
        )
        self.roadmark_heads = PV26RoadMarkNativeJointHeads(
            self.in_channels,
            self.feature_strides,
            lane_head_mode=self.lane_head_mode,
        )
        self.lane_head_mode = self.roadmark_heads.lane_head_mode
        self.lane_head = self.roadmark_heads.lane_head
        self.stop_line_head = self.roadmark_heads.stop_line_head
        self.crosswalk_head = self.roadmark_heads.crosswalk_head

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return self.roadmark_heads.lane_family_modules()

    def describe(self) -> dict[str, object]:
        roadmark_payload = self.roadmark_heads.describe()
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "det_feature_channels": list(self.det_in_channels),
            "det_feature_strides": list(self.det_feature_strides),
            "det_dim": DET_DIM,
            "tl_attr_dim": TL_ATTR_DIM,
            "lane_queries": LANE_QUERY_COUNT,
            "stop_line_queries": STOP_LINE_QUERY_COUNT,
            "crosswalk_queries": CROSSWALK_QUERY_COUNT,
            "roadmark_architecture": ROADMARK_JOINT_NATIVE_NAME,
            "roadmark": roadmark_payload,
        }

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[tuple[int, int]] | list[int]]:
        if len(features) != 4:
            raise ValueError("PV26Heads expects 4 feature maps from the trunk pyramid.")

        for feature, channel_count in zip(features, self.in_channels):
            if feature.ndim != 4 or int(feature.shape[1]) != channel_count:
                raise ValueError(
                    f"Expected feature map with shape [B, {channel_count}, H, W], "
                    f"but received {tuple(feature.shape)}."
                )

        det_outputs: list[torch.Tensor] = []
        tl_attr_outputs: list[torch.Tensor] = []
        feature_shapes: list[tuple[int, int]] = []
        for feature, det_head, tl_attr_head in zip(features[1:], self.det_heads, self.tl_attr_heads):
            feature_shapes.append((int(feature.shape[2]), int(feature.shape[3])))
            det_outputs.append(det_head(feature))
            tl_attr_outputs.append(tl_attr_head(feature))

        roadmark_outputs = self.roadmark_heads(features, encoded=encoded)
        return {
            **roadmark_outputs,
            "det": torch.cat(det_outputs, dim=1),
            "tl_attr": torch.cat(tl_attr_outputs, dim=1),
            "det_feature_shapes": feature_shapes,
            "det_feature_strides": list(self.det_feature_strides),
        }


__all__ = ["PV26Heads"]
