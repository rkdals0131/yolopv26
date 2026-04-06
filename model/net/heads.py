from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


DET_DIM = 12
TL_ATTR_DIM = 4
LANE_QUERY_COUNT = 24
STOP_LINE_QUERY_COUNT = 8
CROSSWALK_QUERY_COUNT = 8
LANE_VECTOR_DIM = 38
STOP_LINE_VECTOR_DIM = 6
CROSSWALK_VECTOR_DIM = 9
FEATURE_STRIDES = (8, 16, 32)
LANE_HIDDEN_DIM = 256
LANE_DECODER_LAYERS = 2
LANE_DECODER_HEADS = 8
STOP_LINE_DECODER_LAYERS = 1
STOP_LINE_DECODER_HEADS = 8
CROSSWALK_DECODER_LAYERS = 1
CROSSWALK_DECODER_HEADS = 8


def _build_2d_sincos_position_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError("2D sine-cosine position encoding requires channels divisible by 4.")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(channels // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(float(channels // 4), 1.0)))
    x_encoding = x.reshape(-1, 1) * omega.reshape(1, -1)
    y_encoding = y.reshape(-1, 1) * omega.reshape(1, -1)
    position = torch.cat(
        [
            torch.sin(x_encoding),
            torch.cos(x_encoding),
            torch.sin(y_encoding),
            torch.cos(y_encoding),
        ],
        dim=1,
    )
    return position.to(dtype=dtype)


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


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class _SpatialFusionStem(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for channel in in_channels
            ]
        )
        self.fusion = nn.Sequential(
            _ConvNormAct(hidden_dim * len(in_channels), hidden_dim),
            _ConvNormAct(hidden_dim, hidden_dim),
        )

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        target_hw = (int(features[0].shape[2]), int(features[0].shape[3]))
        projected: list[torch.Tensor] = []
        for feature, projector in zip(features, self.projectors):
            projected_feature = projector(feature)
            if tuple(projected_feature.shape[-2:]) != target_hw:
                projected_feature = F.interpolate(
                    projected_feature,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(projected_feature)
        fused = torch.cat(projected, dim=1)
        return self.fusion(fused)


class _LaneDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        self_attended, _ = self.self_attn(query, query, query, need_weights=False)
        query = self.norm1(query + self_attended)
        cross_attended, _ = self.cross_attn(query, memory + memory_pos, memory, need_weights=False)
        query = self.norm2(query + cross_attended)
        return self.norm3(query + self.ffn(query))


class _LaneRowAnchorHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        query_count: int,
        vector_dim: int,
        decoder_layers: int,
        decoder_heads: int,
    ) -> None:
        super().__init__()
        self.query_count = query_count
        self.hidden_dim = hidden_dim
        self.memory = nn.Sequential(
            _ConvNormAct(hidden_dim, hidden_dim),
            _ConvNormAct(hidden_dim, hidden_dim),
        )
        self.query_embed = nn.Embedding(query_count, hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [_LaneDecoderBlock(hidden_dim, decoder_heads) for _ in range(decoder_layers)]
        )
        self.predictor = nn.Linear(hidden_dim, vector_dim)

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        lane_memory = self.memory(fused_feature)
        batch_size, channels, height, width = lane_memory.shape
        memory_tokens = lane_memory.flatten(2).transpose(1, 2).contiguous()
        memory_pos = _build_2d_sincos_position_encoding(
            height,
            width,
            channels,
            device=lane_memory.device,
            dtype=lane_memory.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        for decoder_layer in self.decoder_layers:
            query = decoder_layer(query, memory_tokens, memory_pos=memory_pos)
        return self.predictor(query)


class _SpatialQueryDecoderHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        query_count: int,
        vector_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        force_float32: bool = False,
    ) -> None:
        super().__init__()
        self.force_float32 = bool(force_float32)
        self.query_embed = nn.Embedding(query_count, hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [_LaneDecoderBlock(hidden_dim, decoder_heads) for _ in range(decoder_layers)]
        )
        self.predictor = nn.Linear(hidden_dim, vector_dim)

    def forward(self, memory_feature: torch.Tensor) -> torch.Tensor:
        if not self.force_float32:
            batch_size, channels, height, width = memory_feature.shape
            memory_tokens = memory_feature.flatten(2).transpose(1, 2).contiguous()
            memory_pos = _build_2d_sincos_position_encoding(
                height,
                width,
                channels,
                device=memory_feature.device,
                dtype=memory_feature.dtype,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            for decoder_layer in self.decoder_layers:
                query = decoder_layer(query, memory_tokens, memory_pos=memory_pos)
            return self.predictor(query)

        autocast_guard = (
            torch.autocast(device_type=memory_feature.device.type, enabled=False)
            if memory_feature.device.type == "cuda"
            else nullcontext()
        )
        with autocast_guard:
            stable_memory = memory_feature.to(dtype=torch.float32)
            batch_size, channels, height, width = stable_memory.shape
            memory_tokens = stable_memory.flatten(2).transpose(1, 2).contiguous()
            memory_pos = _build_2d_sincos_position_encoding(
                height,
                width,
                channels,
                device=stable_memory.device,
                dtype=stable_memory.dtype,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            query = self.query_embed.weight.to(dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
            for decoder_layer in self.decoder_layers:
                query = decoder_layer(query, memory_tokens, memory_pos=memory_pos)
            return self.predictor(query)


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
        self.spatial_fusion_stem = _SpatialFusionStem(self.in_channels, LANE_HIDDEN_DIM)
        self.lane_head = _LaneRowAnchorHead(
            hidden_dim=LANE_HIDDEN_DIM,
            query_count=LANE_QUERY_COUNT,
            vector_dim=LANE_VECTOR_DIM,
            decoder_layers=LANE_DECODER_LAYERS,
            decoder_heads=LANE_DECODER_HEADS,
        )
        self.geometry_memory = nn.Sequential(
            _ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            _ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
        )
        self.stop_line_head = _SpatialQueryDecoderHead(
            hidden_dim=LANE_HIDDEN_DIM,
            query_count=STOP_LINE_QUERY_COUNT,
            vector_dim=STOP_LINE_VECTOR_DIM,
            decoder_layers=STOP_LINE_DECODER_LAYERS,
            decoder_heads=STOP_LINE_DECODER_HEADS,
            force_float32=True,
        )
        self.crosswalk_head = _SpatialQueryDecoderHead(
            hidden_dim=LANE_HIDDEN_DIM,
            query_count=CROSSWALK_QUERY_COUNT,
            vector_dim=CROSSWALK_VECTOR_DIM,
            decoder_layers=CROSSWALK_DECODER_LAYERS,
            decoder_heads=CROSSWALK_DECODER_HEADS,
        )

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.spatial_fusion_stem,
            self.lane_head,
            self.geometry_memory,
            self.stop_line_head,
            self.crosswalk_head,
        )

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
        feature_shapes: list[tuple[int, int]] = []
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
            feature_shapes.append((int(feature.shape[2]), int(feature.shape[3])))
            det_outputs.append(det_head(feature))
            tl_attr_outputs.append(tl_attr_head(feature))
            pooled_features.append(feature.mean(dim=(2, 3)))

        fused_embedding = torch.cat(pooled_features, dim=1)
        fused_feature = self.spatial_fusion_stem(features)
        geometry_memory = self.geometry_memory(fused_feature)
        return {
            "det": torch.cat(det_outputs, dim=1),
            "tl_attr": torch.cat(tl_attr_outputs, dim=1),
            "lane": self.lane_head(fused_feature),
            "stop_line": self.stop_line_head(geometry_memory),
            "crosswalk": self.crosswalk_head(geometry_memory),
            "det_feature_shapes": feature_shapes,
            "det_feature_strides": list(self.feature_strides),
        }


__all__ = ["PV26Heads"]
