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
STOP_LINE_POINT_COUNT = (STOP_LINE_VECTOR_DIM - 1) // 2
CROSSWALK_POINT_COUNT = (CROSSWALK_VECTOR_DIM - 1) // 2
LANE_ANCHOR_COUNT = (LANE_VECTOR_DIM - (1 + 3 + 2)) // 2
LANE_X_SLICE = slice(1 + 3 + 2, 1 + 3 + 2 + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1.0e-4), 1.0 - 1.0e-4)
    return float(math.log(clipped / (1.0 - clipped)))


def _logit_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


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
        batch_size = int(lane_memory.shape[0])
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return self.forward_with_query(lane_memory, query)

    def forward_with_query(self, lane_memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = lane_memory.shape
        memory_tokens = lane_memory.flatten(2).transpose(1, 2).contiguous()
        memory_pos = _build_2d_sincos_position_encoding(
            height,
            width,
            channels,
            device=lane_memory.device,
            dtype=lane_memory.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
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

    def forward(self, memory_feature: torch.Tensor, query_seed: torch.Tensor | None = None) -> torch.Tensor:
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
            query = (
                query_seed.to(device=memory_feature.device, dtype=memory_feature.dtype)
                if isinstance(query_seed, torch.Tensor)
                else self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            )
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
            query = (
                query_seed.to(device=stable_memory.device, dtype=torch.float32)
                if isinstance(query_seed, torch.Tensor)
                else self.query_embed.weight.to(dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
            )
            for decoder_layer in self.decoder_layers:
                query = decoder_layer(query, memory_tokens, memory_pos=memory_pos)
            return self.predictor(query)


class _DenseSeedQueryHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.seed_logits = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, memory_feature: torch.Tensor, *, query_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.seed_logits(memory_feature)
        batch_size, channels, height, width = memory_feature.shape
        flat_logits = logits.flatten(2).squeeze(1)
        topk = min(max(int(query_count), 1), int(flat_logits.shape[1]))
        scores, indices = torch.topk(flat_logits, k=topk, dim=1)
        if topk < int(query_count):
            pad_count = int(query_count) - topk
            indices = torch.cat([indices, indices[:, -1:].expand(batch_size, pad_count)], dim=1)
            scores = torch.cat([scores, scores[:, -1:].expand(batch_size, pad_count)], dim=1)
        memory_tokens = memory_feature.flatten(2).transpose(1, 2).contiguous()
        gather_index = indices[:, :, None].expand(batch_size, int(query_count), channels)
        selected = torch.gather(memory_tokens, dim=1, index=gather_index)
        y = torch.div(indices, width, rounding_mode="floor").to(dtype=memory_feature.dtype)
        x = (indices % width).to(dtype=memory_feature.dtype)
        coords = torch.stack(
            [
                x / max(float(width - 1), 1.0),
                y / max(float(height - 1), 1.0),
                scores.sigmoid(),
            ],
            dim=-1,
        )
        query = selected + self.position_mlp(coords.to(dtype=memory_feature.dtype))
        return query, logits, coords.to(dtype=memory_feature.dtype)


class CurrentFamilyRoadMarkHeads(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        feature_strides: Iterable[int] = FEATURE_STRIDES,
        *,
        coordinate_mode: str = "raw",
        anchor_template_enabled: bool = False,
        denoise_enabled: bool = False,
        anchor_query_seed_enabled: bool = False,
        dense_query_seed_enabled: bool = False,
        dense_seed_geometry_prior_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.coordinate_mode = str(coordinate_mode or "raw").strip().lower()
        if self.coordinate_mode not in {"raw", "sigmoid_network"}:
            raise ValueError("CurrentFamilyRoadMarkHeads coordinate_mode must be one of: raw, sigmoid_network")
        self.anchor_template_enabled = bool(anchor_template_enabled)
        self.denoise_enabled = bool(denoise_enabled)
        self.anchor_query_seed_enabled = bool(anchor_query_seed_enabled)
        self.dense_query_seed_enabled = bool(dense_query_seed_enabled)
        self.dense_seed_geometry_prior_enabled = bool(dense_seed_geometry_prior_enabled)
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
        if self.dense_query_seed_enabled:
            if self.coordinate_mode != "sigmoid_network":
                raise ValueError("current-family dense query seeds require sigmoid_network coordinate mode")
            self.lane_dense_query = _DenseSeedQueryHead(LANE_HIDDEN_DIM)
            self.stop_line_dense_query = _DenseSeedQueryHead(LANE_HIDDEN_DIM)
            self.crosswalk_dense_query = _DenseSeedQueryHead(LANE_HIDDEN_DIM)
        else:
            self.lane_dense_query = None
            self.stop_line_dense_query = None
            self.crosswalk_dense_query = None
        if self.dense_seed_geometry_prior_enabled and not self.dense_query_seed_enabled:
            raise ValueError("current-family dense seed geometry priors require dense query seeds")
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
        if self.anchor_query_seed_enabled:
            if not self.anchor_template_enabled or self.coordinate_mode != "sigmoid_network":
                raise ValueError("current-family anchor query seeds require sigmoid_network anchor templates")
        geometry_query_input_enabled = self.denoise_enabled or self.anchor_query_seed_enabled
        if geometry_query_input_enabled:
            if self.coordinate_mode != "sigmoid_network":
                raise ValueError("current-family geometry query inputs require sigmoid_network coordinate mode")
            self.lane_geometry_query_input = nn.Sequential(
                nn.Linear(LANE_VECTOR_DIM, LANE_HIDDEN_DIM),
                nn.LayerNorm(LANE_HIDDEN_DIM),
                nn.SiLU(inplace=True),
                nn.Linear(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            )
            self.stop_line_geometry_query_input = nn.Sequential(
                nn.Linear(STOP_LINE_VECTOR_DIM, LANE_HIDDEN_DIM),
                nn.LayerNorm(LANE_HIDDEN_DIM),
                nn.SiLU(inplace=True),
                nn.Linear(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            )
            self.crosswalk_geometry_query_input = nn.Sequential(
                nn.Linear(CROSSWALK_VECTOR_DIM, LANE_HIDDEN_DIM),
                nn.LayerNorm(LANE_HIDDEN_DIM),
                nn.SiLU(inplace=True),
                nn.Linear(LANE_HIDDEN_DIM, LANE_HIDDEN_DIM),
            )
        else:
            self.lane_geometry_query_input = None
            self.stop_line_geometry_query_input = None
            self.crosswalk_geometry_query_input = None

    def lane_family_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [
            self.spatial_fusion_stem,
            self.lane_head,
            self.stop_line_memory,
            self.crosswalk_memory,
            self.stop_line_head,
            self.crosswalk_head,
        ]
        for module in (
            self.lane_geometry_query_input,
            self.stop_line_geometry_query_input,
            self.crosswalk_geometry_query_input,
            self.lane_dense_query,
            self.stop_line_dense_query,
            self.crosswalk_dense_query,
        ):
            if isinstance(module, nn.Module):
                modules.append(module)
        return tuple(modules)

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
            "denoise": "gt_noised_query_aux" if self.denoise_enabled else "disabled",
            "anchor_query_seed": "shared_geometry_query_input" if self.anchor_query_seed_enabled else "disabled",
            "dense_query_seed": "topk_seed_heatmap" if self.dense_query_seed_enabled else "disabled",
            "dense_seed_geometry_prior": "seed_centered_shape_template"
            if self.dense_seed_geometry_prior_enabled
            else "disabled",
        }

    def _add_template(self, rows: torch.Tensor, template: torch.Tensor | None) -> torch.Tensor:
        if template is None:
            return rows
        return rows + template.to(device=rows.device, dtype=rows.dtype).unsqueeze(0)

    def _add_dynamic_template(self, rows: torch.Tensor, template: torch.Tensor | None) -> torch.Tensor:
        if template is None:
            return rows
        return rows + template.to(device=rows.device, dtype=rows.dtype)

    def _lane_dense_seed_template(self, seed_coords: torch.Tensor | None) -> torch.Tensor | None:
        if not self.dense_seed_geometry_prior_enabled or not isinstance(seed_coords, torch.Tensor):
            return None
        template = torch.zeros(
            (*seed_coords.shape[:2], LANE_VECTOR_DIM),
            device=seed_coords.device,
            dtype=seed_coords.dtype,
        )
        x_logits = _logit_tensor(seed_coords[..., 0])
        template[..., LANE_X_SLICE] = x_logits.unsqueeze(-1).expand(*seed_coords.shape[:2], LANE_ANCHOR_COUNT)
        template[..., LANE_VIS_SLICE] = 1.5
        return template

    def _stop_line_dense_seed_template(self, seed_coords: torch.Tensor | None) -> torch.Tensor | None:
        if not self.dense_seed_geometry_prior_enabled or not isinstance(seed_coords, torch.Tensor):
            return None
        template = torch.zeros(
            (*seed_coords.shape[:2], STOP_LINE_VECTOR_DIM),
            device=seed_coords.device,
            dtype=seed_coords.dtype,
        )
        offsets = torch.linspace(
            -0.28,
            0.28,
            STOP_LINE_POINT_COUNT,
            device=seed_coords.device,
            dtype=seed_coords.dtype,
        )
        x = (seed_coords[..., 0:1] + offsets).clamp(0.0, 1.0)
        y = seed_coords[..., 1:2].expand_as(x).clamp(0.0, 1.0)
        points = torch.stack((x, y), dim=-1)
        template[..., 1:] = _logit_tensor(points).reshape(*seed_coords.shape[:2], STOP_LINE_VECTOR_DIM - 1)
        return template

    def _crosswalk_dense_seed_template(self, seed_coords: torch.Tensor | None) -> torch.Tensor | None:
        if not self.dense_seed_geometry_prior_enabled or not isinstance(seed_coords, torch.Tensor):
            return None
        template = torch.zeros(
            (*seed_coords.shape[:2], CROSSWALK_VECTOR_DIM),
            device=seed_coords.device,
            dtype=seed_coords.dtype,
        )
        corners = [
            (-0.18, -0.035),
            (0.18, -0.035),
            (0.18, 0.035),
            (-0.18, 0.035),
        ]
        offsets: list[tuple[float, float]] = []
        for start, end in zip(corners, corners[1:] + corners[:1]):
            for step in range(4):
                alpha = float(step) / 4.0
                offsets.append(
                    (
                        (1.0 - alpha) * start[0] + alpha * end[0],
                        (1.0 - alpha) * start[1] + alpha * end[1],
                    )
                )
        offset_tensor = torch.tensor(offsets, device=seed_coords.device, dtype=seed_coords.dtype)
        x = (seed_coords[..., 0:1] + offset_tensor[:, 0]).clamp(0.0, 1.0)
        y = (seed_coords[..., 1:2] + offset_tensor[:, 1]).clamp(0.0, 1.0)
        points = torch.stack((x, y), dim=-1)
        template[..., 1:] = _logit_tensor(points).reshape(*seed_coords.shape[:2], CROSSWALK_VECTOR_DIM - 1)
        return template

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

    def _denoise_targets(
        self,
        encoded: dict[str, torch.Tensor] | None,
        *,
        task_name: str,
        query_count: int,
        vector_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros((0, query_count, vector_dim), device=device, dtype=dtype)
        valid = torch.zeros((0, query_count), device=device, dtype=torch.bool)
        if not isinstance(encoded, dict):
            return targets, valid
        raw_targets = encoded.get(task_name)
        mask = encoded.get("mask")
        if not isinstance(raw_targets, torch.Tensor) or not isinstance(mask, dict):
            return targets, valid
        source = mask.get(f"{task_name}_source")
        valid_rows = mask.get(f"{task_name}_valid")
        if not isinstance(source, torch.Tensor) or not isinstance(valid_rows, torch.Tensor):
            return targets, valid
        batch_size = int(raw_targets.shape[0])
        targets = torch.zeros((batch_size, query_count, vector_dim), device=device, dtype=dtype)
        valid = torch.zeros((batch_size, query_count), device=device, dtype=torch.bool)
        raw_targets = raw_targets.to(device=device, dtype=dtype)
        source = source.to(device=device, dtype=torch.bool)
        valid_rows = valid_rows.to(device=device, dtype=torch.bool)
        for batch_index in range(batch_size):
            if not bool(source[batch_index]):
                continue
            indices = torch.nonzero(valid_rows[batch_index], as_tuple=False).flatten()[:query_count]
            if int(indices.numel()) == 0:
                continue
            count = int(indices.numel())
            targets[batch_index, :count] = raw_targets[batch_index, indices]
            targets[batch_index, :count, 0] = 1.0
            valid[batch_index, :count] = True
        return targets, valid

    def _lane_geometry_query_features(
        self,
        targets: torch.Tensor,
        valid: torch.Tensor,
        *,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        normalized = targets.clone()
        normalized[..., 0] = valid.to(dtype=normalized.dtype)
        normalized[..., LANE_X_SLICE] = normalized[..., LANE_X_SLICE] / max(float(NETWORK_HW[1] - 1), 1.0)
        normalized[..., LANE_X_SLICE] = normalized[..., LANE_X_SLICE].clamp(0.0, 1.0)
        normalized[..., LANE_VIS_SLICE] = normalized[..., LANE_VIS_SLICE].clamp(0.0, 1.0)
        if self.training and float(noise_std) > 0.0 and bool(valid.any()):
            noise = torch.zeros_like(normalized)
            noise[..., LANE_X_SLICE] = torch.randn_like(normalized[..., LANE_X_SLICE]) * float(noise_std)
            normalized = normalized + noise * valid[:, :, None].to(dtype=normalized.dtype)
            normalized[..., LANE_X_SLICE] = normalized[..., LANE_X_SLICE].clamp(0.0, 1.0)
        return normalized

    def _lane_denoise_query_input(self, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self._lane_geometry_query_features(targets, valid, noise_std=0.03)

    def _point_geometry_query_features(
        self,
        targets: torch.Tensor,
        valid: torch.Tensor,
        *,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        normalized = targets.clone()
        normalized[..., 0] = valid.to(dtype=normalized.dtype)
        point_shape = normalized[..., 1:].shape
        points = normalized[..., 1:].reshape(*point_shape[:-1], -1, 2)
        scale = targets.new_tensor([max(float(NETWORK_HW[1] - 1), 1.0), max(float(NETWORK_HW[0] - 1), 1.0)])
        points = (points / scale).clamp(0.0, 1.0)
        if self.training and float(noise_std) > 0.0 and bool(valid.any()):
            points = (
                points + torch.randn_like(points) * float(noise_std) * valid[:, :, None, None].to(dtype=points.dtype)
            ).clamp(0.0, 1.0)
        normalized[..., 1:] = points.reshape_as(normalized[..., 1:])
        return normalized

    def _point_denoise_query_input(self, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self._point_geometry_query_features(targets, valid, noise_std=0.025)

    def _lane_anchor_query_seed(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.anchor_query_seed_enabled or not isinstance(self.lane_geometry_query_input, nn.Module):
            return None
        if not isinstance(self.lane_anchor_template, torch.Tensor):
            return None
        anchors = self._scale_lane_rows(
            self.lane_anchor_template.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1, -1)
        )
        valid = torch.ones((int(batch_size), LANE_QUERY_COUNT), device=device, dtype=torch.bool)
        return self.lane_geometry_query_input(self._lane_geometry_query_features(anchors, valid, noise_std=0.0))

    def _point_anchor_query_seed(
        self,
        template: torch.Tensor | None,
        query_count: int,
        query_input: nn.Module | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.anchor_query_seed_enabled or not isinstance(query_input, nn.Module):
            return None
        if not isinstance(template, torch.Tensor):
            return None
        anchors = self._scale_point_rows(
            template.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1, -1)
        )
        valid = torch.ones((int(batch_size), int(query_count)), device=device, dtype=torch.bool)
        return query_input(self._point_geometry_query_features(anchors, valid, noise_std=0.0))

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
        batch_size = int(fused_feature.shape[0])
        lane_memory = None
        lane_seed_logits = None
        lane_seed_coords = None
        if isinstance(self.lane_dense_query, _DenseSeedQueryHead):
            lane_memory = self.lane_head.memory(fused_feature)
            lane_query_seed, lane_seed_logits, lane_seed_coords = self.lane_dense_query(
                lane_memory,
                query_count=LANE_QUERY_COUNT,
            )
        else:
            lane_query_seed = self._lane_anchor_query_seed(
                batch_size,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
        if isinstance(lane_query_seed, torch.Tensor):
            if lane_memory is None:
                lane_memory = self.lane_head.memory(fused_feature)
            lane_rows = self.lane_head.forward_with_query(lane_memory, lane_query_seed)
        else:
            lane_rows = self.lane_head(fused_feature)
        stop_seed_logits = None
        stop_seed_coords = None
        if isinstance(self.stop_line_dense_query, _DenseSeedQueryHead):
            stop_line_query_seed, stop_seed_logits, stop_seed_coords = self.stop_line_dense_query(
                stop_line_memory,
                query_count=STOP_LINE_QUERY_COUNT,
            )
        else:
            stop_line_query_seed = self._point_anchor_query_seed(
                self.stop_line_anchor_template,
                STOP_LINE_QUERY_COUNT,
                self.stop_line_geometry_query_input,
                batch_size,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
        cross_seed_logits = None
        cross_seed_coords = None
        if isinstance(self.crosswalk_dense_query, _DenseSeedQueryHead):
            crosswalk_query_seed, cross_seed_logits, cross_seed_coords = self.crosswalk_dense_query(
                crosswalk_memory,
                query_count=CROSSWALK_QUERY_COUNT,
            )
        else:
            crosswalk_query_seed = self._point_anchor_query_seed(
                self.crosswalk_anchor_template,
                CROSSWALK_QUERY_COUNT,
                self.crosswalk_geometry_query_input,
                batch_size,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
        lane_rows = self._add_dynamic_template(lane_rows, self._lane_dense_seed_template(lane_seed_coords))
        stop_rows = self._add_dynamic_template(
            self.stop_line_head(stop_line_memory, query_seed=stop_line_query_seed),
            self._stop_line_dense_seed_template(stop_seed_coords),
        )
        cross_rows = self._add_dynamic_template(
            self.crosswalk_head(crosswalk_memory, query_seed=crosswalk_query_seed),
            self._crosswalk_dense_seed_template(cross_seed_coords),
        )
        outputs = {
            "lane": self._scale_lane_rows(self._add_template(lane_rows, self.lane_anchor_template)),
            "stop_line": self._scale_point_rows(
                self._add_template(
                    stop_rows,
                    self.stop_line_anchor_template,
                )
            ),
            "crosswalk": self._scale_point_rows(
                self._add_template(
                    cross_rows,
                    self.crosswalk_anchor_template,
                )
            ),
        }
        if isinstance(lane_seed_logits, torch.Tensor):
            outputs["lane_dense_seed_logits"] = lane_seed_logits
        if isinstance(stop_seed_logits, torch.Tensor):
            outputs["stop_line_dense_seed_logits"] = stop_seed_logits
        if isinstance(cross_seed_logits, torch.Tensor):
            outputs["crosswalk_dense_seed_logits"] = cross_seed_logits
        if self.denoise_enabled and self.training:
            lane_targets, lane_valid = self._denoise_targets(
                encoded,
                task_name="lane",
                query_count=LANE_QUERY_COUNT,
                vector_dim=LANE_VECTOR_DIM,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
            stop_targets, stop_valid = self._denoise_targets(
                encoded,
                task_name="stop_line",
                query_count=STOP_LINE_QUERY_COUNT,
                vector_dim=STOP_LINE_VECTOR_DIM,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
            cross_targets, cross_valid = self._denoise_targets(
                encoded,
                task_name="crosswalk",
                query_count=CROSSWALK_QUERY_COUNT,
                vector_dim=CROSSWALK_VECTOR_DIM,
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
            if isinstance(self.lane_geometry_query_input, nn.Module) and lane_targets.numel() > 0:
                if lane_memory is None:
                    lane_memory = self.lane_head.memory(fused_feature)
                lane_query = self.lane_geometry_query_input(self._lane_denoise_query_input(lane_targets, lane_valid))
                outputs["lane_denoise"] = self._scale_lane_rows(
                    self.lane_head.forward_with_query(lane_memory, lane_query)
                )
                outputs["lane_denoise_target"] = lane_targets
                outputs["lane_denoise_valid"] = lane_valid
            if isinstance(self.stop_line_geometry_query_input, nn.Module) and stop_targets.numel() > 0:
                stop_query = self.stop_line_geometry_query_input(
                    self._point_denoise_query_input(stop_targets, stop_valid)
                )
                outputs["stop_line_denoise"] = self._scale_point_rows(
                    self.stop_line_head(stop_line_memory, query_seed=stop_query)
                )
                outputs["stop_line_denoise_target"] = stop_targets
                outputs["stop_line_denoise_valid"] = stop_valid
            if isinstance(self.crosswalk_geometry_query_input, nn.Module) and cross_targets.numel() > 0:
                cross_query = self.crosswalk_geometry_query_input(
                    self._point_denoise_query_input(cross_targets, cross_valid)
                )
                outputs["crosswalk_denoise"] = self._scale_point_rows(
                    self.crosswalk_head(crosswalk_memory, query_seed=cross_query)
                )
                outputs["crosswalk_denoise_target"] = cross_targets
                outputs["crosswalk_denoise_valid"] = cross_valid
        return outputs


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
