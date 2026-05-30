from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .roadmark_joint_native import ROADMARK_JOINT_NATIVE_NAME, PV26RoadMarkNativeJointHeads
from .roadmark_current_family import CurrentFamilyRoadMarkHeads
from .roadmark_v2_heads import (
    LANE_ONLY_ROW_CLASSIFIER_NAME,
    ROADMARK_V2_FEATURE_STRIDES,
    ROADMARK_V3_JOINT_NAME,
    PV26LaneOnlyHeads,
    PV26RoadMarkV3JointHeads,
    PV26StopLineOnlyHeads,
)


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
STOPLINE_ONLY_MASK_FIRST_NAME = "stopline_only_mask_first"
CURRENT_FAMILY_NAME = "current_family"
CURRENT_FAMILY_SIGMOID_NAME = "current_family_sigmoid"
CURRENT_FAMILY_ANCHOR_SIGMOID_NAME = "current_family_anchor_sigmoid"
CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME = "current_family_anchor_denoise_sigmoid"


def _normalize_roadmark_architecture(value: str) -> str:
    architecture = str(value or ROADMARK_JOINT_NATIVE_NAME).strip().lower()
    if architecture in {"native", "joint_native", ROADMARK_JOINT_NATIVE_NAME}:
        return ROADMARK_JOINT_NATIVE_NAME
    if architecture in {"v3", "v3_stopline_isolated", ROADMARK_V3_JOINT_NAME}:
        return ROADMARK_V3_JOINT_NAME
    if architecture in {"lane_only", "lane_only_row_classifier", LANE_ONLY_ROW_CLASSIFIER_NAME}:
        return LANE_ONLY_ROW_CLASSIFIER_NAME
    if architecture in {"stopline_only", "stop_line_only", STOPLINE_ONLY_MASK_FIRST_NAME}:
        return STOPLINE_ONLY_MASK_FIRST_NAME
    if architecture in {"current", CURRENT_FAMILY_NAME}:
        return CURRENT_FAMILY_NAME
    if architecture in {"current_sigmoid", "current_family_normalized", CURRENT_FAMILY_SIGMOID_NAME}:
        return CURRENT_FAMILY_SIGMOID_NAME
    if architecture in {"current_anchor", "current_family_anchor", CURRENT_FAMILY_ANCHOR_SIGMOID_NAME}:
        return CURRENT_FAMILY_ANCHOR_SIGMOID_NAME
    if architecture in {
        "current_anchor_denoise",
        "current_family_anchor_denoise",
        CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME,
    }:
        return CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME
    raise ValueError(
        "roadmark_architecture must be one of: "
        f"{ROADMARK_JOINT_NATIVE_NAME}, {ROADMARK_V3_JOINT_NAME}, "
        f"v3_stopline_isolated, {LANE_ONLY_ROW_CLASSIFIER_NAME}, {STOPLINE_ONLY_MASK_FIRST_NAME}, "
        f"{CURRENT_FAMILY_NAME}, {CURRENT_FAMILY_SIGMOID_NAME}, {CURRENT_FAMILY_ANCHOR_SIGMOID_NAME}, "
        f"{CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME}"
    )


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
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        roadmark_architecture: str = ROADMARK_JOINT_NATIVE_NAME,
        lane_family_shared_adapter_enabled: bool = False,
        lane_family_task_adapter_enabled: bool = False,
        lane_family_cross_stitch_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = str(lane_head_mode).strip().lower()
        self.lane_conditional_row_coordinate_mode = str(lane_conditional_row_coordinate_mode).strip().lower()
        self.lane_conditional_row_max_delta_px = float(lane_conditional_row_max_delta_px)
        self.roadmark_architecture = _normalize_roadmark_architecture(roadmark_architecture)
        self.lane_family_shared_adapter_enabled = bool(lane_family_shared_adapter_enabled)
        self.lane_family_task_adapter_enabled = bool(lane_family_task_adapter_enabled)
        self.lane_family_cross_stitch_enabled = bool(lane_family_cross_stitch_enabled)
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
        if self.roadmark_architecture == ROADMARK_V3_JOINT_NAME:
            roadmark_head_cls = PV26RoadMarkV3JointHeads
        elif self.roadmark_architecture == LANE_ONLY_ROW_CLASSIFIER_NAME:
            roadmark_head_cls = PV26LaneOnlyHeads
        elif self.roadmark_architecture == STOPLINE_ONLY_MASK_FIRST_NAME:
            roadmark_head_cls = PV26StopLineOnlyHeads
        elif self.roadmark_architecture in {
            CURRENT_FAMILY_NAME,
            CURRENT_FAMILY_SIGMOID_NAME,
            CURRENT_FAMILY_ANCHOR_SIGMOID_NAME,
            CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME,
        }:
            roadmark_head_cls = CurrentFamilyRoadMarkHeads
        else:
            roadmark_head_cls = PV26RoadMarkNativeJointHeads
        if roadmark_head_cls is PV26StopLineOnlyHeads:
            self.roadmark_heads = roadmark_head_cls(self.in_channels, self.feature_strides)
        elif roadmark_head_cls is CurrentFamilyRoadMarkHeads:
            coordinate_mode = (
                "sigmoid_network"
                if self.roadmark_architecture
                in {
                    CURRENT_FAMILY_SIGMOID_NAME,
                    CURRENT_FAMILY_ANCHOR_SIGMOID_NAME,
                    CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME,
                }
                else "raw"
            )
            self.roadmark_heads = roadmark_head_cls(
                self.det_in_channels,
                self.det_feature_strides,
                coordinate_mode=coordinate_mode,
                anchor_template_enabled=self.roadmark_architecture
                in {CURRENT_FAMILY_ANCHOR_SIGMOID_NAME, CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME},
                denoise_enabled=self.roadmark_architecture == CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME,
            )
        elif roadmark_head_cls is PV26LaneOnlyHeads:
            self.roadmark_heads = roadmark_head_cls(
                self.in_channels,
                self.feature_strides,
                lane_head_mode=self.lane_head_mode,
                lane_conditional_row_coordinate_mode=self.lane_conditional_row_coordinate_mode,
                lane_conditional_row_max_delta_px=self.lane_conditional_row_max_delta_px,
            )
        else:
            self.roadmark_heads = roadmark_head_cls(
                self.in_channels,
                self.feature_strides,
                lane_head_mode=self.lane_head_mode,
                lane_conditional_row_coordinate_mode=self.lane_conditional_row_coordinate_mode,
                lane_conditional_row_max_delta_px=self.lane_conditional_row_max_delta_px,
                lane_family_shared_adapter_enabled=self.lane_family_shared_adapter_enabled,
                lane_family_task_adapter_enabled=self.lane_family_task_adapter_enabled,
                lane_family_cross_stitch_enabled=self.lane_family_cross_stitch_enabled,
            )
        self.lane_head_mode = str(getattr(self.roadmark_heads, "lane_head_mode", self.lane_head_mode))
        self.lane_head = getattr(self.roadmark_heads, "lane_head", None)
        self.stop_line_head = getattr(self.roadmark_heads, "stop_line_head", None)
        self.crosswalk_head = getattr(self.roadmark_heads, "crosswalk_head", None)

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return self.roadmark_heads.lane_family_modules()

    def stop_line_modules(self) -> tuple[nn.Module, ...]:
        getter = getattr(self.roadmark_heads, "stop_line_modules", None)
        if callable(getter):
            return tuple(module for module in getter() if isinstance(module, nn.Module))
        return (self.stop_line_head,)

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
            "roadmark_architecture": self.roadmark_architecture,
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

        current_family_architectures = {
            CURRENT_FAMILY_NAME,
            CURRENT_FAMILY_SIGMOID_NAME,
            CURRENT_FAMILY_ANCHOR_SIGMOID_NAME,
            CURRENT_FAMILY_ANCHOR_DENOISE_SIGMOID_NAME,
        }
        roadmark_features = features[1:] if self.roadmark_architecture in current_family_architectures else features
        roadmark_outputs = self.roadmark_heads(roadmark_features, encoded=encoded)
        return {
            **roadmark_outputs,
            "det": torch.cat(det_outputs, dim=1),
            "tl_attr": torch.cat(tl_attr_outputs, dim=1),
            "det_feature_shapes": feature_shapes,
            "det_feature_strides": list(self.det_feature_strides),
        }


__all__ = ["PV26Heads"]
