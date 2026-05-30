from __future__ import annotations

from typing import Iterable

from .roadmark_v2_heads import PV26RoadMarkV2LaneFamilyHeads, ROADMARK_V2_FEATURE_STRIDES


ROADMARK_JOINT_NATIVE_NAME = "roadmark_joint_native"


class PV26RoadMarkNativeJointHeads(PV26RoadMarkV2LaneFamilyHeads):
    supports_encoded_context = True

    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = ROADMARK_V2_FEATURE_STRIDES,
        *,
        lane_head_mode: str = "seg_first",
        lane_conditional_row_coordinate_mode: str = "absolute_sigmoid",
        lane_conditional_row_max_delta_px: float = 160.0,
        lane_conditional_denoise_hard_negative_count: int = 0,
        lane_conditional_denoise_hard_negative_offset_px: float = 80.0,
        lane_family_shared_adapter_enabled: bool = False,
        lane_family_task_adapter_enabled: bool = False,
        lane_family_cross_stitch_enabled: bool = False,
        stopline_lane_context_fusion_enabled: bool = False,
        stopline_lane_context_detach: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            feature_strides=feature_strides,
            lane_head_mode=lane_head_mode,
            lane_conditional_row_coordinate_mode=lane_conditional_row_coordinate_mode,
            lane_conditional_row_max_delta_px=lane_conditional_row_max_delta_px,
            lane_conditional_denoise_hard_negative_count=lane_conditional_denoise_hard_negative_count,
            lane_conditional_denoise_hard_negative_offset_px=(
                lane_conditional_denoise_hard_negative_offset_px
            ),
            lane_family_shared_adapter_enabled=lane_family_shared_adapter_enabled,
            lane_family_task_adapter_enabled=lane_family_task_adapter_enabled,
            lane_family_cross_stitch_enabled=lane_family_cross_stitch_enabled,
            stopline_lane_context_fusion_enabled=stopline_lane_context_fusion_enabled,
            stopline_lane_context_detach=stopline_lane_context_detach,
        )

    def describe(self) -> dict[str, object]:
        payload = self.roadmark_heads.describe()
        payload["mode"] = "roadmark_joint"
        payload["roadmark_architecture"] = ROADMARK_JOINT_NATIVE_NAME
        payload["joint_initialization"] = "pretrained_trunk_without_atomic_transplant"
        payload["task_native_proving"] = "lane_stopline_crosswalk_native_heads"
        return payload


__all__ = [
    "ROADMARK_JOINT_NATIVE_NAME",
    "PV26RoadMarkNativeJointHeads",
]
