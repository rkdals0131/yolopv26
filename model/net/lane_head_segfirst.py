from __future__ import annotations

import torch
import torch.nn as nn

from common.pv26_schema import LANE_CLASSES, LANE_TYPES
from ..data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW
from .roadmark_blocks import ConvNormAct, MultiScaleFusion
from .roadmark_current_family import LANE_QUERY_COUNT, LANE_VECTOR_DIM


class LaneSegFirstHead(nn.Module):
    """Dense centerline-first lane head.

    This head is opt-in only. It predicts the dense evidence maps consumed by
    the seg-first vectorizer path while keeping the generic ``lane`` tensor in
    the output dictionary as a zero placeholder until predicted-map vectorized
    evaluation is wired in.
    """

    def __init__(self, in_channels: tuple[int, int, int], *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        if len(self.in_channels) != 3:
            raise ValueError("LaneSegFirstHead expects P2/P3/P4 feature channels.")
        self.hidden_dim = int(hidden_dim)
        self.output_hw = ROADMARK_DENSE_OUTPUT_HW
        self.fusion = MultiScaleFusion(self.in_channels, self.hidden_dim, target_level=0, depth=2)
        self.stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.centerline_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.support_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.tangent_axis = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.color_logits = nn.Conv2d(self.hidden_dim, len(LANE_CLASSES), kernel_size=1)
        self.type_logits = nn.Conv2d(self.hidden_dim, len(LANE_TYPES), kernel_size=1)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del encoded
        if len(features) != 3:
            raise ValueError("LaneSegFirstHead expects exactly 3 feature maps (P2/P3/P4).")
        lane_feature = self.stem(self.fusion(features))
        batch_size = int(lane_feature.shape[0])
        dtype = lane_feature.dtype
        device = lane_feature.device
        lane_placeholder = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), device=device, dtype=dtype)
        return {
            "lane": lane_placeholder,
            "lane_seg_centerline_logits": self.centerline_logits(lane_feature),
            "lane_seg_support_logits": self.support_logits(lane_feature),
            "lane_seg_tangent_axis": self.tangent_axis(lane_feature),
            "lane_seg_color_logits": self.color_logits(lane_feature),
            "lane_seg_type_logits": self.type_logits(lane_feature),
            "lane_feature": lane_feature,
        }


__all__ = ["LaneSegFirstHead"]
