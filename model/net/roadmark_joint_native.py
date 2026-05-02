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
    ) -> None:
        super().__init__(in_channels, feature_strides=feature_strides, lane_head_mode=lane_head_mode)

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
