from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Iterable

import torch
import torch.nn as nn

from ..data.transform import NETWORK_HW
from .roadmark_blocks import ConvNormAct, MultiScaleFusion


FEATURE_STRIDES = (8, 16, 32)
LANE_QUERY_COUNT = 24
STOP_LINE_QUERY_COUNT = 8
CROSSWALK_QUERY_COUNT = 8
LANE_VECTOR_DIM = 38
STOP_LINE_VECTOR_DIM = 9
CROSSWALK_VECTOR_DIM = 33
LANE_HIDDEN_DIM = 256
LANE_DECODER_LAYERS = 2
LANE_DECODER_HEADS = 8
STOP_LINE_DECODER_LAYERS = 2
STOP_LINE_DECODER_HEADS = 8
CROSSWALK_DECODER_LAYERS = 2
CROSSWALK_DECODER_HEADS = 8
LANE_ANCHOR_COUNT = (LANE_VECTOR_DIM - (1 + 3 + 2)) // 2
LANE_X_SLICE = slice(1 + 3 + 2, 1 + 3 + 2 + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1.0e-4), 1.0 - 1.0e-4)
    return float(math.log(clipped / (1.0 - clipped)))


def _network_point_logit(x: float, y: float) -> tuple[float, float]:
    return (
        _logit(float(x) / max(float(NETWORK_HW[1] - 1), 1.0)),
        _logit(float(y) / max(float(NETWORK_HW[0] - 1), 1.0)),
    )


def _build_lane_anchor_template() -> torch.Tensor:
    template = torch.zeros((LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32)
    # Keep objectness neutral so training, not the prior, decides emission.
    template[:, 0] = 0.0
    x_min = float(NETWORK_HW[1]) * 0.08
    x_max = float(NETWORK_HW[1]) * 0.92
    centers = torch.linspace(x_min, x_max, steps=LANE_QUERY_COUNT)
    for query_index, center_x in enumerate(centers.tolist()):
        # Add a small deterministic fan so the vector loss starts from lane-like,
        # not fully collapsed, row coordinates.
        fan = (float(query_index % 5) - 2.0) * 2.0
        row_values = torch.linspace(center_x + fan, center_x - fan, steps=LANE_ANCHOR_COUNT)
        normalized = row_values.clamp(0.0, float(NETWORK_HW[1] - 1)) / float(NETWORK_HW[1] - 1)
        template[query_index, LANE_X_SLICE] = torch.logit(normalized.clamp(1.0e-4, 1.0 - 1.0e-4))
        template[query_index, LANE_VIS_SLICE] = 1.5
    return template


def _build_stop_line_anchor_template() -> torch.Tensor:
    template = torch.zeros((STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32)
    y_values = torch.linspace(float(NETWORK_HW[0]) * 0.42, float(NETWORK_HW[0]) * 0.88, steps=STOP_LINE_QUERY_COUNT)
    x_start_values = torch.linspace(float(NETWORK_HW[1]) * 0.10, float(NETWORK_HW[1]) * 0.28, steps=STOP_LINE_QUERY_COUNT)
    x_end_values = torch.linspace(float(NETWORK_HW[1]) * 0.72, float(NETWORK_HW[1]) * 0.92, steps=STOP_LINE_QUERY_COUNT)
    for query_index, (y_value, x_start, x_end) in enumerate(
        zip(y_values.tolist(), x_start_values.tolist(), x_end_values.tolist())
    ):
        points = [
            _network_point_logit(x_start, y_value),
            _network_point_logit(0.66 * x_start + 0.34 * x_end, y_value),
            _network_point_logit(0.34 * x_start + 0.66 * x_end, y_value),
            _network_point_logit(x_end, y_value),
        ]
        template[query_index, 1:] = torch.tensor([value for point in points for value in point], dtype=torch.float32)
    return template


def _build_crosswalk_anchor_template() -> torch.Tensor:
    template = torch.zeros((CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32)
    centers_y = torch.linspace(float(NETWORK_HW[0]) * 0.46, float(NETWORK_HW[0]) * 0.84, steps=CROSSWALK_QUERY_COUNT)
    centers_x = torch.linspace(float(NETWORK_HW[1]) * 0.38, float(NETWORK_HW[1]) * 0.62, steps=CROSSWALK_QUERY_COUNT)
    for query_index, (center_x, center_y) in enumerate(zip(centers_x.tolist(), centers_y.tolist())):
        half_w = float(NETWORK_HW[1]) * (0.18 + 0.02 * float(query_index % 3))
        half_h = float(NETWORK_HW[0]) * 0.035
        corners = [
            (center_x - half_w, center_y - half_h),
            (center_x + half_w, center_y - half_h),
            (center_x + half_w, center_y + half_h),
            (center_x - half_w, center_y + half_h),
        ]
        contour: list[tuple[float, float]] = []
        for start, end in zip(corners, corners[1:] + corners[:1]):
            for step in range(4):
                alpha = float(step) / 4.0
                contour.append(
                    (
                        (1.0 - alpha) * start[0] + alpha * end[0],
                        (1.0 - alpha) * start[1] + alpha * end[1],
                    )
                )
        template[query_index, 1:] = torch.tensor(
            [value for point in contour for value in _network_point_logit(point[0], point[1])],
            dtype=torch.float32,
        )
    return template


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
        self.query_embed = nn.Embedding(query_count, hidden_dim)
        self.memory = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim),
            ConvNormAct(hidden_dim, hidden_dim),
        )
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


class CurrentFamilyRoadMarkHeads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = FEATURE_STRIDES,
        *,
        coordinate_mode: str = "raw",
        anchor_template_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.coordinate_mode = str(coordinate_mode or "raw").strip().lower()
        if self.coordinate_mode not in {"raw", "sigmoid_network"}:
            raise ValueError("CurrentFamilyRoadMarkHeads coordinate_mode must be one of: raw, sigmoid_network")
        self.anchor_template_enabled = bool(anchor_template_enabled)
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.feature_strides = tuple(int(stride) for stride in feature_strides)
        if len(self.in_channels) != 3:
            raise ValueError("CurrentFamilyRoadMarkHeads expects exactly 3 pyramid levels.")
        if len(self.feature_strides) != 3:
            raise ValueError("CurrentFamilyRoadMarkHeads expects exactly 3 feature strides.")

        self.spatial_fusion_stem = MultiScaleFusion(
            self.in_channels,
            LANE_HIDDEN_DIM,
            target_level=0,
            depth=2,
        )
        self.lane_head = _LaneRowAnchorHead(
            hidden_dim=LANE_HIDDEN_DIM,
            query_count=LANE_QUERY_COUNT,
            vector_dim=LANE_VECTOR_DIM,
            decoder_layers=LANE_DECODER_LAYERS,
            decoder_heads=LANE_DECODER_HEADS,
        )
        self.stop_line_memory = nn.Sequential(
            ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
        )
        self.crosswalk_memory = nn.Sequential(
            ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            ConvNormAct(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
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
        if self.anchor_template_enabled:
            if self.coordinate_mode != "sigmoid_network":
                raise ValueError("current-family anchor templates require sigmoid_network coordinate mode")
            self.lane_anchor_template = nn.Parameter(_build_lane_anchor_template())
            self.stop_line_anchor_template = nn.Parameter(_build_stop_line_anchor_template())
            self.crosswalk_anchor_template = nn.Parameter(_build_crosswalk_anchor_template())
        else:
            self.register_parameter("lane_anchor_template", None)
            self.register_parameter("stop_line_anchor_template", None)
            self.register_parameter("crosswalk_anchor_template", None)

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.spatial_fusion_stem,
            self.lane_head,
            self.stop_line_memory,
            self.crosswalk_memory,
            self.stop_line_head,
            self.crosswalk_head,
        )

    def describe(self) -> dict[str, object]:
        return {
            "feature_channels": list(self.in_channels),
            "feature_strides": list(self.feature_strides),
            "lane_queries": LANE_QUERY_COUNT,
            "stop_line_queries": STOP_LINE_QUERY_COUNT,
            "crosswalk_queries": CROSSWALK_QUERY_COUNT,
            "roadmark_architecture": "current_family",
            "coordinate_mode": self.coordinate_mode,
            "anchor_template": "network_geometry_prior" if self.anchor_template_enabled else "disabled",
        }

    def _add_template(self, rows: torch.Tensor, template: torch.Tensor | None) -> torch.Tensor:
        if template is None:
            return rows
        return rows + template.to(device=rows.device, dtype=rows.dtype).unsqueeze(0)

    def _scale_lane_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.coordinate_mode != "sigmoid_network":
            return rows
        scaled = rows.clone()
        scaled[..., LANE_X_SLICE] = scaled[..., LANE_X_SLICE].sigmoid() * float(NETWORK_HW[1] - 1)
        return scaled

    def _scale_point_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.coordinate_mode != "sigmoid_network":
            return rows
        scaled = rows.clone()
        point_shape = scaled[..., 1:].shape
        points = scaled[..., 1:].reshape(*point_shape[:-1], -1, 2)
        scale = torch.tensor(
            [float(NETWORK_HW[1] - 1), float(NETWORK_HW[0] - 1)],
            device=scaled.device,
            dtype=scaled.dtype,
        )
        scaled[..., 1:] = (points.sigmoid() * scale).reshape_as(scaled[..., 1:])
        return scaled

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = encoded
        if len(features) != 3:
            raise ValueError("CurrentFamilyRoadMarkHeads expects 3 feature maps from the trunk pyramid.")
        for feature, channel_count in zip(features, self.in_channels):
            if feature.ndim != 4 or int(feature.shape[1]) != channel_count:
                raise ValueError(
                    f"Expected feature map with shape [B, {channel_count}, H, W], "
                    f"but received {tuple(feature.shape)}."
                )
        fused_feature = self.spatial_fusion_stem(features)
        stop_line_memory = self.stop_line_memory(fused_feature)
        crosswalk_memory = self.crosswalk_memory(fused_feature)
        return {
            "lane": self._scale_lane_rows(self._add_template(self.lane_head(fused_feature), self.lane_anchor_template)),
            "stop_line": self._scale_point_rows(
                self._add_template(self.stop_line_head(stop_line_memory), self.stop_line_anchor_template)
            ),
            "crosswalk": self._scale_point_rows(
                self._add_template(self.crosswalk_head(crosswalk_memory), self.crosswalk_anchor_template)
            ),
        }


__all__ = [
    "CROSSWALK_QUERY_COUNT",
    "CROSSWALK_VECTOR_DIM",
    "CurrentFamilyRoadMarkHeads",
    "FEATURE_STRIDES",
    "LANE_QUERY_COUNT",
    "LANE_VECTOR_DIM",
    "STOP_LINE_QUERY_COUNT",
    "STOP_LINE_VECTOR_DIM",
]
