from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .crosswalk_head_mask import CrosswalkMaskFirstHead
from .lane_head_dense_row import LaneDenseRowSeedHead
from .lane_head_segfirst import LaneSegFirstHead
from .roadmark_blocks import ConvNormAct
from .stopline_head_line import StopLineDenseLocalHead


ROADMARK_V2_FEATURE_STRIDES = (4, 8, 16, 32)
ROADMARK_V2_BRANCH_A_NAME = "roadmark_v2_branch_a"
ROADMARK_V3_JOINT_NAME = "roadmark_v3_joint"
LANE_HEAD_ROW_NATIVE = "row_native"
LANE_HEAD_SEG_FIRST = "seg_first"


def _normalize_lane_head_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"", "row", "row_native", "native"}:
        return LANE_HEAD_ROW_NATIVE
    if mode in {"seg_first", "segfirst", "seg-first", "dense"}:
        return LANE_HEAD_SEG_FIRST
    raise ValueError("lane_head_mode must be one of: row_native, seg_first")


class _StoplineFeatureNeck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.adapter = nn.Sequential(
            ConvNormAct(self.channels, self.channels),
            ConvNormAct(self.channels, self.channels),
            ConvNormAct(self.channels, self.channels),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit).to(device=feature.device, dtype=feature.dtype)
        return feature + gate.view(1, 1, 1, 1) * self.adapter(feature)


class RoadMarkV2Heads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        if len(self.in_channels) != 4:
            raise ValueError("RoadMarkV2Heads expects exactly 4 pyramid levels (P2/P3/P4/P5).")
        if len(self.feature_strides) != 4:
            raise ValueError("RoadMarkV2Heads expects exactly 4 feature strides.")

        p2, p3, p4, _ = self.in_channels
        if self.lane_head_mode == LANE_HEAD_SEG_FIRST:
            self.lane_head = LaneSegFirstHead((p2, p3, p4))
        else:
            self.lane_head = LaneDenseRowSeedHead((p2, p3, p4))
        self.stop_line_head = StopLineDenseLocalHead((p2, p3))
        self.crosswalk_head = CrosswalkMaskFirstHead((p2, p3, p4))

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "roadmark_architecture": "roadmark_v2_scaffold",
            "lane_head": "seg_first_dense_centerline_tangent_color"
            if self.lane_head_mode == LANE_HEAD_SEG_FIRST
            else "row_classification_plus_dense_centerline_candidates",
            "lane_head_mode": self.lane_head_mode,
            "stop_line_head": "mask_first_line_decode",
            "crosswalk_head": "mask_first",
        }

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("RoadMarkV2Heads expects 4 feature maps from the trunk pyramid.")
        p2, p3, p4, p5 = features
        _ = p5  # reserved for future coarse context branches
        outputs: dict[str, torch.Tensor] = {}
        outputs.update(self.lane_head((p2, p3, p4), encoded=encoded))
        outputs.update(self.stop_line_head((p2, p3), encoded=encoded))
        outputs.update(self.crosswalk_head((p2, p3, p4)))
        return outputs


class PV26RoadMarkV2LaneFamilyHeads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        if len(self.in_channels) != 4:
            raise ValueError("PV26RoadMarkV2LaneFamilyHeads expects exactly 4 pyramid levels.")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26RoadMarkV2LaneFamilyHeads expects exactly 4 feature strides.")
        self.roadmark_heads = RoadMarkV2Heads(
            self.in_channels,
            self.feature_strides,
            lane_head_mode=self.lane_head_mode,
        )
        self.lane_head = self.roadmark_heads.lane_head
        self.stop_line_head = self.roadmark_heads.stop_line_head
        self.crosswalk_head = self.roadmark_heads.crosswalk_head

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.lane_head,
            self.stop_line_head,
            self.crosswalk_head,
        )

    def describe(self) -> dict[str, object]:
        payload = self.roadmark_heads.describe()
        payload["mode"] = "lane_family_only"
        return payload

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...], *, encoded: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor | list[int]]:
        if len(features) != 4:
            raise ValueError("PV26RoadMarkV2LaneFamilyHeads expects 4 feature maps from the trunk pyramid.")
        batch_size = int(features[0].shape[0])
        device = features[0].device
        dtype = features[0].dtype
        roadmark_outputs = self.roadmark_heads(features, encoded=encoded)
        return {
            "det": torch.zeros((batch_size, 0, 12), device=device, dtype=dtype),
            "tl_attr": torch.zeros((batch_size, 0, 4), device=device, dtype=dtype),
            **roadmark_outputs,
            "det_feature_shapes": [],
            "det_feature_strides": [],
        }


class PV26RoadMarkV3JointHeads(PV26RoadMarkV2LaneFamilyHeads):
    supports_encoded_context = True

    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
    ) -> None:
        super().__init__(in_channels, feature_strides=feature_strides, lane_head_mode=lane_head_mode)
        p2, p3, _, _ = self.in_channels
        self.stopline_p2_isolator = _StoplineFeatureNeck(p2)
        self.stopline_p3_isolator = _StoplineFeatureNeck(p3)

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.lane_head,
            self.stop_line_head,
            self.crosswalk_head,
            self.stopline_p2_isolator,
            self.stopline_p3_isolator,
        )

    def describe(self) -> dict[str, object]:
        payload = self.roadmark_heads.describe()
        payload["mode"] = "roadmark_joint"
        payload["roadmark_architecture"] = ROADMARK_V3_JOINT_NAME
        payload["stopline_feature_isolation"] = "gated_stopline_residual_isolator_p2_p3"
        return payload

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[int]]:
        if len(features) != 4:
            raise ValueError("PV26RoadMarkV3JointHeads expects 4 feature maps from the trunk pyramid.")
        batch_size = int(features[0].shape[0])
        device = features[0].device
        dtype = features[0].dtype
        p2, p3, p4, p5 = features
        _ = p5
        lane_outputs = self.lane_head((p2, p3, p4), encoded=encoded)
        stop_outputs = self.stop_line_head(
            (
                self.stopline_p2_isolator(p2),
                self.stopline_p3_isolator(p3),
            ),
            encoded=encoded,
        )
        crosswalk_outputs = self.crosswalk_head((p2, p3, p4))
        return {
            "det": torch.zeros((batch_size, 0, 12), device=device, dtype=dtype),
            "tl_attr": torch.zeros((batch_size, 0, 4), device=device, dtype=dtype),
            **lane_outputs,
            **stop_outputs,
            **crosswalk_outputs,
            "det_feature_shapes": [],
            "det_feature_strides": [],
        }


class PV26LaneOnlyHeads(nn.Module):
    supports_encoded_context = True

    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        if len(self.in_channels) != 4:
            raise ValueError("PV26LaneOnlyHeads expects exactly 4 pyramid levels.")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26LaneOnlyHeads expects exactly 4 feature strides.")
        p2, p3, p4, _ = self.in_channels
        if self.lane_head_mode == LANE_HEAD_SEG_FIRST:
            self.lane_head = LaneSegFirstHead((p2, p3, p4))
        else:
            self.lane_head = LaneDenseRowSeedHead((p2, p3, p4))

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (self.lane_head,)

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "mode": "lane_family_only",
            "roadmark_architecture": "lane_only_row_classifier",
            "lane_head": "seg_first_dense_centerline_tangent_color"
            if self.lane_head_mode == LANE_HEAD_SEG_FIRST
            else "row_classification_plus_dense_centerline_candidates",
            "lane_head_mode": self.lane_head_mode,
            "lane_supervised_row_slots": 8,
            "lane_dense_candidate_queries": 16,
        }

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[int]]:
        if len(features) != 4:
            raise ValueError("PV26LaneOnlyHeads expects 4 feature maps from the trunk pyramid.")
        batch_size = int(features[0].shape[0])
        device = features[0].device
        dtype = features[0].dtype
        p2, p3, p4, p5 = features
        _ = p5
        lane_outputs = self.lane_head((p2, p3, p4), encoded=encoded)
        return {
            "det": torch.zeros((batch_size, 0, 12), device=device, dtype=dtype),
            "tl_attr": torch.zeros((batch_size, 0, 4), device=device, dtype=dtype),
            **lane_outputs,
            "det_feature_shapes": [],
            "det_feature_strides": [],
        }


class PV26StopLineOnlyHeads(nn.Module):
    supports_encoded_context = True

    def __init__(self, in_channels: Iterable[int], feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        if len(self.in_channels) != 4:
            raise ValueError("PV26StopLineOnlyHeads expects exactly 4 pyramid levels.")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26StopLineOnlyHeads expects exactly 4 feature strides.")
        p2, p3, _, _ = self.in_channels
        self.stop_line_head = StopLineDenseLocalHead((p2, p3))

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (self.stop_line_head,)

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "mode": "lane_family_only",
            "roadmark_architecture": "stopline_only_mask_first",
            "stop_line_head": "mask_first_line_decode",
            "stopline_active_queries": 8,
        }

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[int]]:
        if len(features) != 4:
            raise ValueError("PV26StopLineOnlyHeads expects 4 feature maps from the trunk pyramid.")
        batch_size = int(features[0].shape[0])
        device = features[0].device
        dtype = features[0].dtype
        p2, p3, p4, p5 = features
        _ = p4, p5
        stop_outputs = self.stop_line_head((p2, p3), encoded=encoded)
        return {
            "det": torch.zeros((batch_size, 0, 12), device=device, dtype=dtype),
            "tl_attr": torch.zeros((batch_size, 0, 4), device=device, dtype=dtype),
            "lane": torch.zeros((batch_size, 24, 38), device=device, dtype=dtype),
            **stop_outputs,
            "crosswalk": torch.zeros((batch_size, 8, 33), device=device, dtype=dtype),
            "det_feature_shapes": [],
            "det_feature_strides": [],
        }


__all__ = [
    "ROADMARK_V2_BRANCH_A_NAME",
    "ROADMARK_V2_FEATURE_STRIDES",
    "ROADMARK_V3_JOINT_NAME",
    "PV26LaneOnlyHeads",
    "PV26RoadMarkV2LaneFamilyHeads",
    "PV26RoadMarkV3JointHeads",
    "PV26StopLineOnlyHeads",
    "RoadMarkV2Heads",
    "LANE_HEAD_ROW_NATIVE",
    "LANE_HEAD_SEG_FIRST",
]
