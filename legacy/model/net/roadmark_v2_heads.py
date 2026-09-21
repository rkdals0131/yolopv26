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
LANE_ONLY_ROW_CLASSIFIER_NAME = "lane_only_row_classifier"
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


class _ZeroInitResidualAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        hidden_channels = max(16, self.channels // 4)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.adapter = nn.Sequential(
            ConvNormAct(self.channels, hidden_channels, kernel_size=1),
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit).to(device=feature.device, dtype=feature.dtype)
        return feature + gate.view(1, 1, 1, 1) * self.adapter(feature)


class _LaneFamilySharedFeatureAdapters(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(_ZeroInitResidualAdapter(channel) for channel in channels)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("Lane-family shared adapters expect 4 feature maps.")
        p2, p3, p4, p5 = features
        adapted = [adapter(feature) for adapter, feature in zip(self.adapters, (p2, p3, p4))]
        return adapted[0], adapted[1], adapted[2], p5


class _LaneFamilyTaskFeatureAdapters(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        self.lane_adapters = _LaneFamilySharedFeatureAdapters(channels)
        self.stop_line_adapters = _LaneFamilySharedFeatureAdapters(channels)
        self.crosswalk_adapters = _LaneFamilySharedFeatureAdapters(channels)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return {
            "lane": self.lane_adapters(features),
            "stop_line": self.stop_line_adapters(features),
            "crosswalk": self.crosswalk_adapters(features),
        }


class _LaneFamilyCrossStitchMixer(nn.Module):
    TASKS = ("lane", "stop_line", "crosswalk")

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        self.levels = int(levels)
        if self.levels <= 0:
            raise ValueError("lane-family cross-stitch mixer requires at least one feature level.")
        self.mix = nn.Parameter(torch.eye(len(self.TASKS), dtype=torch.float32).repeat(self.levels, 1, 1))

    def forward(
        self,
        task_features: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        missing = [task for task in self.TASKS if task not in task_features]
        if missing:
            raise ValueError(f"cross-stitch task features missing tasks: {missing}")
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for out_index, out_task in enumerate(self.TASKS):
            mixed_levels: list[torch.Tensor] = []
            for level in range(self.levels):
                mixed: torch.Tensor | None = None
                for in_index, in_task in enumerate(self.TASKS):
                    source = task_features[in_task][level]
                    weight = self.mix[level, out_index, in_index].to(device=source.device, dtype=source.dtype)
                    value = weight * source
                    mixed = value if mixed is None else mixed + value
                if mixed is None:
                    raise RuntimeError("cross-stitch mixer produced an empty feature level")
                mixed_levels.append(mixed)
            mixed_levels.append(task_features[out_task][3])
            outputs[out_task] = (
                mixed_levels[0],
                mixed_levels[1],
                mixed_levels[2],
                mixed_levels[3],
            )
        return outputs


class RoadMarkV2Heads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        lane_conditional_denoise_hard_negative_count: int = 0,
        lane_conditional_denoise_hard_negative_offset_px: float = 80.0,
        lane_family_shared_adapter_enabled: bool = False,
        lane_family_task_adapter_enabled: bool = False,
        lane_family_cross_stitch_enabled: bool = False,
        stopline_lane_context_fusion_enabled: bool = False,
        stopline_lane_context_detach: bool = True,
        stopline_crosswalk_context_fusion_enabled: bool = False,
        stopline_crosswalk_context_detach: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        self.lane_conditional_row_coordinate_mode = str(lane_conditional_row_coordinate_mode).strip().lower()
        self.lane_conditional_row_max_delta_px = float(lane_conditional_row_max_delta_px)
        self.lane_conditional_denoise_hard_negative_count = int(
            lane_conditional_denoise_hard_negative_count
        )
        self.lane_conditional_denoise_hard_negative_offset_px = float(
            lane_conditional_denoise_hard_negative_offset_px
        )
        self.lane_family_shared_adapter_enabled = bool(lane_family_shared_adapter_enabled)
        self.lane_family_task_adapter_enabled = bool(lane_family_task_adapter_enabled)
        self.lane_family_cross_stitch_enabled = bool(lane_family_cross_stitch_enabled)
        self.stopline_lane_context_fusion_enabled = bool(stopline_lane_context_fusion_enabled)
        self.stopline_lane_context_detach = bool(stopline_lane_context_detach)
        self.stopline_crosswalk_context_fusion_enabled = bool(stopline_crosswalk_context_fusion_enabled)
        self.stopline_crosswalk_context_detach = bool(stopline_crosswalk_context_detach)
        if len(self.in_channels) != 4:
            raise ValueError("RoadMarkV2Heads expects exactly 4 pyramid levels (P2/P3/P4/P5).")
        if len(self.feature_strides) != 4:
            raise ValueError("RoadMarkV2Heads expects exactly 4 feature strides.")

        p2, p3, p4, _ = self.in_channels
        if self.lane_head_mode == LANE_HEAD_SEG_FIRST:
            self.lane_head = LaneSegFirstHead(
                (p2, p3, p4),
                conditional_row_coordinate_mode=self.lane_conditional_row_coordinate_mode,
                conditional_row_max_delta_px=self.lane_conditional_row_max_delta_px,
                conditional_denoise_hard_negative_count=(
                    self.lane_conditional_denoise_hard_negative_count
                ),
                conditional_denoise_hard_negative_offset_px=(
                    self.lane_conditional_denoise_hard_negative_offset_px
                ),
            )
        else:
            self.lane_head = LaneDenseRowSeedHead((p2, p3, p4))
        self.stop_line_head = StopLineDenseLocalHead(
            (p2, p3),
            lane_context_fusion_enabled=self.stopline_lane_context_fusion_enabled,
            lane_context_detach=self.stopline_lane_context_detach,
            crosswalk_context_fusion_enabled=self.stopline_crosswalk_context_fusion_enabled,
            crosswalk_context_detach=self.stopline_crosswalk_context_detach,
        )
        self.crosswalk_head = CrosswalkMaskFirstHead((p2, p3, p4))
        self.shared_feature_adapters = (
            _LaneFamilySharedFeatureAdapters((p2, p3, p4))
            if self.lane_family_shared_adapter_enabled
            else None
        )
        self.task_feature_adapters = (
            _LaneFamilyTaskFeatureAdapters((p2, p3, p4))
            if self.lane_family_task_adapter_enabled
            else None
        )
        self.cross_stitch_mixer = (
            _LaneFamilyCrossStitchMixer(levels=3)
            if self.lane_family_cross_stitch_enabled
            else None
        )

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "roadmark_architecture": "roadmark_v2_scaffold",
            "lane_head": "seg_first_dense_centerline_tangent_color"
            if self.lane_head_mode == LANE_HEAD_SEG_FIRST
            else "row_classification_plus_dense_centerline_candidates",
            "lane_head_mode": self.lane_head_mode,
            "lane_conditional_row_coordinate_mode": getattr(
                self.lane_head,
                "conditional_row_coordinate_mode",
                "disabled",
            ),
            "lane_conditional_row_max_delta_px": float(
                getattr(self.lane_head, "conditional_row_max_delta_px", 0.0)
            ),
            "lane_conditional_denoise_hard_negative_count": int(
                getattr(self.lane_head, "conditional_denoise_hard_negative_count", 0)
            ),
            "lane_conditional_denoise_hard_negative_offset_px": float(
                getattr(self.lane_head, "conditional_denoise_hard_negative_offset_px", 0.0)
            ),
            "stop_line_head": "mask_first_line_decode",
            "crosswalk_head": "mask_first",
            "lane_family_shared_adapter": "zero_init_residual_p2_p3_p4"
            if self.lane_family_shared_adapter_enabled
            else "disabled",
            "lane_family_task_adapter": "zero_init_residual_per_task_p2_p3_p4"
            if self.lane_family_task_adapter_enabled
            else "disabled",
            "lane_family_cross_stitch": "task_feature_cross_stitch_p2_p3_p4"
            if self.lane_family_cross_stitch_enabled
            else "disabled",
            "stopline_lane_context_fusion": "lane_dense_prob_residual"
            if self.stopline_lane_context_fusion_enabled
            else "disabled",
            "stopline_lane_context_detach": bool(self.stopline_lane_context_detach),
            "stopline_crosswalk_context_fusion": "crosswalk_dense_prob_residual"
            if self.stopline_crosswalk_context_fusion_enabled
            else "disabled",
            "stopline_crosswalk_context_detach": bool(self.stopline_crosswalk_context_detach),
        }

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("RoadMarkV2Heads expects 4 feature maps from the trunk pyramid.")
        if self.shared_feature_adapters is not None:
            features = self.shared_feature_adapters(features)
        task_features = None
        if self.task_feature_adapters is not None:
            task_features = self.task_feature_adapters(features)
        p2, p3, p4, p5 = features
        if self.cross_stitch_mixer is not None:
            if task_features is None:
                task_features = {
                    "lane": (p2, p3, p4, p5),
                    "stop_line": (p2, p3, p4, p5),
                    "crosswalk": (p2, p3, p4, p5),
                }
            task_features = self.cross_stitch_mixer(task_features)
        _ = p5  # reserved for future coarse context branches
        outputs: dict[str, torch.Tensor] = {}
        lane_p2, lane_p3, lane_p4, _lane_p5 = task_features["lane"] if task_features is not None else (p2, p3, p4, p5)
        stop_p2, stop_p3, _stop_p4, _stop_p5 = (
            task_features["stop_line"] if task_features is not None else (p2, p3, p4, p5)
        )
        cross_p2, cross_p3, cross_p4, _cross_p5 = (
            task_features["crosswalk"] if task_features is not None else (p2, p3, p4, p5)
        )
        lane_outputs = self.lane_head((lane_p2, lane_p3, lane_p4), encoded=encoded)
        crosswalk_outputs = self.crosswalk_head((cross_p2, cross_p3, cross_p4))
        outputs.update(lane_outputs)
        outputs.update(
            self.stop_line_head(
                (stop_p2, stop_p3),
                encoded=encoded,
                lane_context=lane_outputs,
                crosswalk_context=crosswalk_outputs,
            )
        )
        outputs.update(crosswalk_outputs)
        return outputs


class PV26RoadMarkV2LaneFamilyHeads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = LANE_HEAD_ROW_NATIVE,
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        lane_conditional_denoise_hard_negative_count: int = 0,
        lane_conditional_denoise_hard_negative_offset_px: float = 80.0,
        lane_family_shared_adapter_enabled: bool = False,
        lane_family_task_adapter_enabled: bool = False,
        lane_family_cross_stitch_enabled: bool = False,
        stopline_lane_context_fusion_enabled: bool = False,
        stopline_lane_context_detach: bool = True,
        stopline_crosswalk_context_fusion_enabled: bool = False,
        stopline_crosswalk_context_detach: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        self.lane_conditional_row_coordinate_mode = str(lane_conditional_row_coordinate_mode).strip().lower()
        self.lane_conditional_row_max_delta_px = float(lane_conditional_row_max_delta_px)
        self.lane_conditional_denoise_hard_negative_count = int(
            lane_conditional_denoise_hard_negative_count
        )
        self.lane_conditional_denoise_hard_negative_offset_px = float(
            lane_conditional_denoise_hard_negative_offset_px
        )
        self.lane_family_shared_adapter_enabled = bool(lane_family_shared_adapter_enabled)
        self.lane_family_task_adapter_enabled = bool(lane_family_task_adapter_enabled)
        self.lane_family_cross_stitch_enabled = bool(lane_family_cross_stitch_enabled)
        self.stopline_lane_context_fusion_enabled = bool(stopline_lane_context_fusion_enabled)
        self.stopline_lane_context_detach = bool(stopline_lane_context_detach)
        self.stopline_crosswalk_context_fusion_enabled = bool(stopline_crosswalk_context_fusion_enabled)
        self.stopline_crosswalk_context_detach = bool(stopline_crosswalk_context_detach)
        if len(self.in_channels) != 4:
            raise ValueError("PV26RoadMarkV2LaneFamilyHeads expects exactly 4 pyramid levels.")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26RoadMarkV2LaneFamilyHeads expects exactly 4 feature strides.")
        self.roadmark_heads = RoadMarkV2Heads(
            self.in_channels,
            self.feature_strides,
            lane_head_mode=self.lane_head_mode,
            lane_conditional_row_coordinate_mode=self.lane_conditional_row_coordinate_mode,
            lane_conditional_row_max_delta_px=self.lane_conditional_row_max_delta_px,
            lane_conditional_denoise_hard_negative_count=(
                self.lane_conditional_denoise_hard_negative_count
            ),
            lane_conditional_denoise_hard_negative_offset_px=(
                self.lane_conditional_denoise_hard_negative_offset_px
            ),
            lane_family_shared_adapter_enabled=self.lane_family_shared_adapter_enabled,
            lane_family_task_adapter_enabled=self.lane_family_task_adapter_enabled,
            lane_family_cross_stitch_enabled=self.lane_family_cross_stitch_enabled,
            stopline_lane_context_fusion_enabled=self.stopline_lane_context_fusion_enabled,
            stopline_lane_context_detach=self.stopline_lane_context_detach,
            stopline_crosswalk_context_fusion_enabled=self.stopline_crosswalk_context_fusion_enabled,
            stopline_crosswalk_context_detach=self.stopline_crosswalk_context_detach,
        )
        self.lane_head = self.roadmark_heads.lane_head
        self.stop_line_head = self.roadmark_heads.stop_line_head
        self.crosswalk_head = self.roadmark_heads.crosswalk_head
        self.shared_feature_adapters = self.roadmark_heads.shared_feature_adapters
        self.task_feature_adapters = self.roadmark_heads.task_feature_adapters
        self.cross_stitch_mixer = self.roadmark_heads.cross_stitch_mixer

    def lane_family_adapter_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = []
        if self.shared_feature_adapters is not None:
            modules.append(self.shared_feature_adapters)
        if self.task_feature_adapters is not None:
            modules.append(self.task_feature_adapters)
        if self.cross_stitch_mixer is not None:
            modules.append(self.cross_stitch_mixer)
        return tuple(modules)

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.lane_head,
            self.stop_line_head,
            self.crosswalk_head,
            *self.lane_family_adapter_modules(),
        )

    def lane_modules(self) -> tuple[nn.Module, ...]:
        return (self.lane_head,)

    def stop_line_modules(self) -> tuple[nn.Module, ...]:
        return (self.stop_line_head,)

    def describe(self) -> dict[str, object]:
        payload = self.roadmark_heads.describe()
        payload["mode"] = "lane_family_only"
        return payload

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[int]]:
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
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        lane_conditional_denoise_hard_negative_count: int = 0,
        lane_conditional_denoise_hard_negative_offset_px: float = 80.0,
        lane_family_shared_adapter_enabled: bool = False,
        lane_family_task_adapter_enabled: bool = False,
        lane_family_cross_stitch_enabled: bool = False,
        stopline_lane_context_fusion_enabled: bool = False,
        stopline_lane_context_detach: bool = True,
        stopline_crosswalk_context_fusion_enabled: bool = False,
        stopline_crosswalk_context_detach: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            feature_strides=feature_strides,
            lane_head_mode=lane_head_mode,
            lane_conditional_row_coordinate_mode=lane_conditional_row_coordinate_mode,
            lane_conditional_row_max_delta_px=lane_conditional_row_max_delta_px,
            lane_conditional_denoise_hard_negative_count=(
                lane_conditional_denoise_hard_negative_count
            ),
            lane_conditional_denoise_hard_negative_offset_px=(
                lane_conditional_denoise_hard_negative_offset_px
            ),
            lane_family_shared_adapter_enabled=lane_family_shared_adapter_enabled,
            lane_family_task_adapter_enabled=lane_family_task_adapter_enabled,
            lane_family_cross_stitch_enabled=lane_family_cross_stitch_enabled,
            stopline_lane_context_fusion_enabled=stopline_lane_context_fusion_enabled,
            stopline_lane_context_detach=stopline_lane_context_detach,
            stopline_crosswalk_context_fusion_enabled=stopline_crosswalk_context_fusion_enabled,
            stopline_crosswalk_context_detach=stopline_crosswalk_context_detach,
        )
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
            *self.lane_family_adapter_modules(),
        )

    def stop_line_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.stop_line_head,
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
        if self.shared_feature_adapters is not None:
            features = self.shared_feature_adapters(features)
        task_features = None
        if self.task_feature_adapters is not None:
            task_features = self.task_feature_adapters(features)
        p2, p3, p4, p5 = features
        if self.cross_stitch_mixer is not None:
            if task_features is None:
                task_features = {
                    "lane": (p2, p3, p4, p5),
                    "stop_line": (p2, p3, p4, p5),
                    "crosswalk": (p2, p3, p4, p5),
                }
            task_features = self.cross_stitch_mixer(task_features)
        _ = p5
        lane_p2, lane_p3, lane_p4, _lane_p5 = task_features["lane"] if task_features is not None else (p2, p3, p4, p5)
        stop_p2, stop_p3, _stop_p4, _stop_p5 = (
            task_features["stop_line"] if task_features is not None else (p2, p3, p4, p5)
        )
        cross_p2, cross_p3, cross_p4, _cross_p5 = (
            task_features["crosswalk"] if task_features is not None else (p2, p3, p4, p5)
        )
        lane_outputs = self.lane_head((lane_p2, lane_p3, lane_p4), encoded=encoded)
        crosswalk_outputs = self.crosswalk_head((cross_p2, cross_p3, cross_p4))
        stop_outputs = self.stop_line_head(
            (
                self.stopline_p2_isolator(stop_p2),
                self.stopline_p3_isolator(stop_p3),
            ),
            encoded=encoded,
            lane_context=lane_outputs,
            crosswalk_context=crosswalk_outputs,
        )
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
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        lane_conditional_denoise_hard_negative_count: int = 0,
        lane_conditional_denoise_hard_negative_offset_px: float = 80.0,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        self.lane_head_mode = _normalize_lane_head_mode(lane_head_mode)
        self.lane_conditional_row_coordinate_mode = str(lane_conditional_row_coordinate_mode).strip().lower()
        self.lane_conditional_row_max_delta_px = float(lane_conditional_row_max_delta_px)
        self.lane_conditional_denoise_hard_negative_count = int(
            lane_conditional_denoise_hard_negative_count
        )
        self.lane_conditional_denoise_hard_negative_offset_px = float(
            lane_conditional_denoise_hard_negative_offset_px
        )
        if len(self.in_channels) != 4:
            raise ValueError("PV26LaneOnlyHeads expects exactly 4 pyramid levels.")
        if len(self.feature_strides) != 4:
            raise ValueError("PV26LaneOnlyHeads expects exactly 4 feature strides.")
        p2, p3, p4, _ = self.in_channels
        if self.lane_head_mode == LANE_HEAD_SEG_FIRST:
            self.lane_head = LaneSegFirstHead(
                (p2, p3, p4),
                conditional_row_coordinate_mode=self.lane_conditional_row_coordinate_mode,
                conditional_row_max_delta_px=self.lane_conditional_row_max_delta_px,
                conditional_denoise_hard_negative_count=(
                    self.lane_conditional_denoise_hard_negative_count
                ),
                conditional_denoise_hard_negative_offset_px=(
                    self.lane_conditional_denoise_hard_negative_offset_px
                ),
            )
        else:
            self.lane_head = LaneDenseRowSeedHead((p2, p3, p4))

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (self.lane_head,)

    def lane_modules(self) -> tuple[nn.Module, ...]:
        return (self.lane_head,)

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "mode": "lane_family_only",
            "roadmark_architecture": LANE_ONLY_ROW_CLASSIFIER_NAME,
            "lane_head": "seg_first_dense_centerline_tangent_color"
            if self.lane_head_mode == LANE_HEAD_SEG_FIRST
            else "row_classification_plus_dense_centerline_candidates",
            "lane_head_mode": self.lane_head_mode,
            "lane_conditional_row_coordinate_mode": getattr(
                self.lane_head,
                "conditional_row_coordinate_mode",
                "disabled",
            ),
            "lane_conditional_row_max_delta_px": float(
                getattr(self.lane_head, "conditional_row_max_delta_px", 0.0)
            ),
            "lane_conditional_denoise_hard_negative_count": int(
                getattr(self.lane_head, "conditional_denoise_hard_negative_count", 0)
            ),
            "lane_conditional_denoise_hard_negative_offset_px": float(
                getattr(self.lane_head, "conditional_denoise_hard_negative_offset_px", 0.0)
            ),
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
            "stop_line": torch.zeros((batch_size, 8, 9), device=device, dtype=dtype),
            "crosswalk": torch.zeros((batch_size, 8, 33), device=device, dtype=dtype),
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
    "LANE_ONLY_ROW_CLASSIFIER_NAME",
    "PV26LaneOnlyHeads",
    "PV26RoadMarkV2LaneFamilyHeads",
    "PV26RoadMarkV3JointHeads",
    "PV26StopLineOnlyHeads",
    "RoadMarkV2Heads",
    "LANE_HEAD_ROW_NATIVE",
    "LANE_HEAD_SEG_FIRST",
]
