from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


DET_DIM = 12
TL_ATTR_DIM = 4
LANE_QUERY_COUNT = 12
STOP_LINE_QUERY_COUNT = 6
CROSSWALK_QUERY_COUNT = 4
LANE_VECTOR_DIM = 54
STOP_LINE_VECTOR_DIM = 9
CROSSWALK_VECTOR_DIM = 17
FEATURE_STRIDES = (8, 16, 32)


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


class _QueryMLPHead(nn.Module):
    def __init__(self, in_dim: int, query_count: int, vector_dim: int) -> None:
        super().__init__()
        self.query_count = query_count
        self.vector_dim = vector_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(inplace=True),
            nn.Linear(in_dim, query_count * vector_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        output = self.mlp(embedding)
        return output.view(embedding.shape[0], self.query_count, self.vector_dim)


class PV26Heads(nn.Module):
    def __init__(self, in_channels: Iterable[int], feature_strides: Iterable[int] = FEATURE_STRIDES) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        if len(self.in_channels) != 3:
            raise ValueError("PV26Heads expects exactly 3 pyramid levels.")
        if len(self.feature_strides) != 3:
            raise ValueError("PV26Heads expects exactly 3 feature strides.")

        self.det_heads = nn.ModuleList(
            [_ScalePredictionHead(channel, DET_DIM) for channel in self.in_channels]
        )
        self.tl_attr_heads = nn.ModuleList(
            [_ScalePredictionHead(channel, TL_ATTR_DIM) for channel in self.in_channels]
        )
        fused_dim = sum(self.in_channels)
        self.lane_head = _QueryMLPHead(fused_dim, LANE_QUERY_COUNT, LANE_VECTOR_DIM)
        self.stop_line_head = _QueryMLPHead(fused_dim, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM)
        self.crosswalk_head = _QueryMLPHead(fused_dim, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM)

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "det_dim": DET_DIM,
            "tl_attr_dim": TL_ATTR_DIM,
            "lane_queries": LANE_QUERY_COUNT,
            "stop_line_queries": STOP_LINE_QUERY_COUNT,
            "crosswalk_queries": CROSSWALK_QUERY_COUNT,
        }

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        if len(features) != 3:
            raise ValueError("PV26Heads expects 3 feature maps from the trunk pyramid.")

        det_outputs: list[torch.Tensor] = []
        tl_attr_outputs: list[torch.Tensor] = []
        pooled_features: list[torch.Tensor] = []
        for feature, channel_count, det_head, tl_attr_head in zip(
            features,
            self.in_channels,
            self.det_heads,
            self.tl_attr_heads,
        ):
            if feature.ndim != 4 or int(feature.shape[1]) != channel_count:
                raise ValueError(
                    f"Expected feature map with shape [B, {channel_count}, H, W], "
                    f"but received {tuple(feature.shape)}."
                )
            det_outputs.append(det_head(feature))
            tl_attr_outputs.append(tl_attr_head(feature))
            pooled_features.append(feature.mean(dim=(2, 3)))

        fused_embedding = torch.cat(pooled_features, dim=1)
        return {
            "det": torch.cat(det_outputs, dim=1),
            "tl_attr": torch.cat(tl_attr_outputs, dim=1),
            "lane": self.lane_head(fused_embedding),
            "stop_line": self.stop_line_head(fused_embedding),
            "crosswalk": self.crosswalk_head(fused_embedding),
        }


__all__ = ["PV26Heads"]
