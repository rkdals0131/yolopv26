from __future__ import annotations

import torch
import torch.nn as nn

from common.pv26_schema import LANE_CLASSES, LANE_TYPES
from ..data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW
from ..data.transform import NETWORK_HW
from .roadmark_blocks import ConvNormAct, MultiScaleFusion
from .roadmark_current_family import LANE_QUERY_COUNT, LANE_VECTOR_DIM

LANE_COLOR_SLICE = slice(1, 1 + len(LANE_CLASSES))
LANE_TYPE_SLICE = slice(LANE_COLOR_SLICE.stop, LANE_COLOR_SLICE.stop + len(LANE_TYPES))
LANE_ANCHOR_COUNT = (LANE_VECTOR_DIM - LANE_TYPE_SLICE.stop) // 2
LANE_X_SLICE = slice(LANE_TYPE_SLICE.stop, LANE_TYPE_SLICE.stop + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


LANE_CONDITIONAL_ROW_COORDINATE_MODES = ("absolute_sigmoid", "seed_relative")


def _normalize_conditional_row_coordinate_mode(value: str) -> str:
    mode = str(value or "absolute_sigmoid").strip().lower()
    if mode in {"absolute", "absolute_sigmoid", "sigmoid"}:
        return "absolute_sigmoid"
    if mode in {"seed_relative", "seed-local", "seed_local", "relative"}:
        return "seed_relative"
    raise ValueError("lane_conditional_row_coordinate_mode must be one of: absolute_sigmoid, seed_relative")


class LaneSegFirstHead(nn.Module):
    """Dense centerline-first lane head.

    This head is opt-in only. It predicts the dense evidence maps consumed by
    the seg-first vectorizer path while keeping the generic ``lane`` tensor in
    the output dictionary as a zero placeholder until predicted-map vectorized
    evaluation is wired in.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        *,
        hidden_dim: int = 128,
        conditional_row_coordinate_mode: str = "absolute_sigmoid",
        conditional_row_max_delta_px: float = 160.0,
        conditional_denoise_hard_negative_count: int = 0,
        conditional_denoise_hard_negative_offset_px: float = 80.0,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        if len(self.in_channels) != 3:
            raise ValueError("LaneSegFirstHead expects P2/P3/P4 feature channels.")
        self.hidden_dim = int(hidden_dim)
        self.conditional_row_coordinate_mode = _normalize_conditional_row_coordinate_mode(
            conditional_row_coordinate_mode
        )
        self.conditional_row_max_delta_px = float(conditional_row_max_delta_px)
        if self.conditional_row_max_delta_px <= 0.0:
            raise ValueError("conditional_row_max_delta_px must be positive")
        self.conditional_denoise_hard_negative_count = int(conditional_denoise_hard_negative_count)
        if self.conditional_denoise_hard_negative_count < 0:
            raise ValueError("conditional_denoise_hard_negative_count must be >= 0")
        self.conditional_denoise_hard_negative_offset_px = float(conditional_denoise_hard_negative_offset_px)
        if self.conditional_denoise_hard_negative_offset_px <= 0.0:
            raise ValueError("conditional_denoise_hard_negative_offset_px must be positive")
        self.output_hw = ROADMARK_DENSE_OUTPUT_HW
        self.fusion = MultiScaleFusion(self.in_channels, self.hidden_dim, target_level=0, depth=2)
        self.stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.centerline_refine_gate_logit = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.centerline_refine = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.centerline_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.support_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.center_offset = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.tangent_axis = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.instance_embedding = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.color_logits = nn.Conv2d(self.hidden_dim, len(LANE_CLASSES), kernel_size=1)
        self.type_logits = nn.Conv2d(self.hidden_dim, len(LANE_TYPES), kernel_size=1)
        self.conditional_seed_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.conditional_query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, LANE_VECTOR_DIM),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(features) != 3:
            raise ValueError("LaneSegFirstHead expects exactly 3 feature maps (P2/P3/P4).")
        lane_feature = self.stem(self.fusion(features))
        batch_size = int(lane_feature.shape[0])
        dtype = lane_feature.dtype
        device = lane_feature.device
        centerline_gate = torch.sigmoid(self.centerline_refine_gate_logit).to(device=device, dtype=dtype)
        centerline_feature = lane_feature + centerline_gate.view(1, 1, 1, 1) * self.centerline_refine(lane_feature)
        lane_placeholder = torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), device=device, dtype=dtype)
        conditional_seed_logits = self.conditional_seed_logits(lane_feature)
        conditional_rows = self._decode_conditional_rows(lane_feature, conditional_seed_logits)
        denoise_rows, denoise_targets, denoise_valid, denoise_hard_negative = self._decode_denoised_conditional_rows(
            lane_feature,
            encoded=encoded,
        )
        return {
            "lane": lane_placeholder,
            "lane_seg_centerline_logits": self.centerline_logits(centerline_feature),
            "lane_seg_support_logits": self.support_logits(lane_feature),
            "lane_seg_center_offset": self.center_offset(lane_feature),
            "lane_seg_tangent_axis": self.tangent_axis(lane_feature),
            "lane_seg_instance_embedding": self.instance_embedding(lane_feature),
            "lane_seg_color_logits": self.color_logits(lane_feature),
            "lane_seg_type_logits": self.type_logits(lane_feature),
            "lane_conditional_seed_logits": conditional_seed_logits,
            "lane_conditional_rows": conditional_rows,
            "lane_conditional_denoise_rows": denoise_rows,
            "lane_conditional_denoise_targets": denoise_targets,
            "lane_conditional_denoise_valid": denoise_valid,
            "lane_conditional_denoise_hard_negative": denoise_hard_negative,
            "lane_feature": lane_feature,
        }

    def _rows_from_seed_inputs(
        self,
        seed_features: torch.Tensor,
        seed_xy: torch.Tensor,
        seed_cols: torch.Tensor,
        *,
        seed_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.conditional_query_mlp(torch.cat([seed_features, seed_xy], dim=-1))
        batch_size, query_count, _ = raw.shape
        rows = raw.new_zeros((batch_size, query_count, LANE_VECTOR_DIM))
        rows[..., 0] = raw[..., 0] if seed_scores is None else raw[..., 0] + seed_scores
        rows[..., LANE_COLOR_SLICE] = raw[..., LANE_COLOR_SLICE]
        rows[..., LANE_TYPE_SLICE] = raw[..., LANE_TYPE_SLICE]
        if self.conditional_row_coordinate_mode == "seed_relative":
            seed_x = seed_cols.to(dtype=raw.dtype) / max(float(self.output_hw[1] - 1), 1.0) * float(NETWORK_HW[1] - 1)
            delta = torch.tanh(raw[..., LANE_X_SLICE]) * float(self.conditional_row_max_delta_px)
            rows[..., LANE_X_SLICE] = (seed_x.unsqueeze(-1) + delta).clamp(0.0, float(NETWORK_HW[1] - 1))
        else:
            rows[..., LANE_X_SLICE] = torch.sigmoid(raw[..., LANE_X_SLICE]) * float(NETWORK_HW[1] - 1)
        rows[..., LANE_VIS_SLICE] = raw[..., LANE_VIS_SLICE]
        return rows

    def _decode_conditional_rows(self, feature_map: torch.Tensor, seed_logits: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        flat_seed_logits = seed_logits.flatten(2).squeeze(1)
        query_count = min(LANE_QUERY_COUNT, int(flat_seed_logits.shape[1]))
        top_scores, top_indices = torch.topk(flat_seed_logits, k=query_count, dim=1)
        flat_features = feature_map.flatten(2).transpose(1, 2)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, channels)
        seed_features = torch.gather(flat_features, dim=1, index=gather_index)
        seed_rows = torch.div(top_indices, width, rounding_mode="floor")
        seed_cols = top_indices.remainder(width)
        denom_x = max(float(width - 1), 1.0)
        denom_y = max(float(height - 1), 1.0)
        seed_xy = torch.stack(
            [
                seed_cols.to(dtype=feature_map.dtype) / denom_x,
                seed_rows.to(dtype=feature_map.dtype) / denom_y,
            ],
            dim=-1,
        )
        rows = self._rows_from_seed_inputs(seed_features, seed_xy, seed_cols, seed_scores=top_scores)
        if query_count == LANE_QUERY_COUNT:
            return rows
        padded = rows.new_zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        padded[..., 0] = -10.0
        padded[:, :query_count] = rows
        return padded

    def _decode_denoised_conditional_rows(
        self,
        feature_map: torch.Tensor,
        *,
        encoded: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        rows = feature_map.new_zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        rows[..., 0] = -10.0
        targets = rows.new_zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        valid = torch.zeros((batch_size, LANE_QUERY_COUNT), device=feature_map.device, dtype=torch.bool)
        hard_negative = torch.zeros((batch_size, LANE_QUERY_COUNT), device=feature_map.device, dtype=torch.bool)
        if not isinstance(encoded, dict):
            return rows, targets, valid, hard_negative
        lane_target = encoded.get("lane")
        mask_payload = encoded.get("mask")
        if not isinstance(lane_target, torch.Tensor) or not isinstance(mask_payload, dict):
            return rows, targets, valid, hard_negative
        lane_valid = mask_payload.get("lane_valid")
        lane_source = mask_payload.get("lane_source")
        if not isinstance(lane_valid, torch.Tensor) or not isinstance(lane_source, torch.Tensor):
            return rows, targets, valid, hard_negative

        lane_target = lane_target.to(device=feature_map.device, dtype=feature_map.dtype)
        lane_valid = lane_valid.to(device=feature_map.device, dtype=torch.bool)
        lane_source = lane_source.to(device=feature_map.device, dtype=torch.bool)
        seed_features = feature_map.new_zeros((batch_size, LANE_QUERY_COUNT, channels))
        seed_xy = feature_map.new_zeros((batch_size, LANE_QUERY_COUNT, 2))
        seed_cols = feature_map.new_zeros((batch_size, LANE_QUERY_COUNT))
        anchor_rows = torch.linspace(
            float(NETWORK_HW[0] - 1),
            0.0,
            LANE_ANCHOR_COUNT,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        x_scale = float(width - 1) / max(float(NETWORK_HW[1] - 1), 1.0)
        y_scale = float(height - 1) / max(float(NETWORK_HW[0] - 1), 1.0)
        denom_x = max(float(width - 1), 1.0)
        denom_y = max(float(height - 1), 1.0)

        def _write_seed(
            batch_index: int,
            query_index: int,
            x_value: torch.Tensor,
            y_value: torch.Tensor,
            *,
            col_jitter: float = 0.0,
            row_jitter: float = 0.0,
        ) -> None:
            col_value = (x_value * x_scale + float(col_jitter)).clamp(0.0, float(width - 1))
            row_value = (y_value * y_scale + float(row_jitter)).clamp(0.0, float(height - 1))
            row_index = int(torch.round(row_value).clamp(0, height - 1).item())
            col_index = int(torch.round(col_value).clamp(0, width - 1).item())
            seed_features[batch_index, query_index] = feature_map[batch_index, :, row_index, col_index]
            seed_xy[batch_index, query_index, 0] = col_value / denom_x
            seed_xy[batch_index, query_index, 1] = row_value / denom_y
            seed_cols[batch_index, query_index] = col_value

        for batch_index in range(batch_size):
            if not bool(lane_source[batch_index]):
                continue
            gt_rows = lane_target[batch_index, lane_valid[batch_index]]
            query_index = 0
            positive_anchors: list[tuple[torch.Tensor, torch.Tensor]] = []
            bottom_x_values: list[torch.Tensor] = []
            for gt_index, gt_row in enumerate(gt_rows[:LANE_QUERY_COUNT]):
                visible = gt_row[LANE_VIS_SLICE] > 0.5
                visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
                if int(visible_indices.numel()) == 0:
                    continue
                if query_index >= LANE_QUERY_COUNT:
                    break
                bottom_index = int(visible_indices[0].item())
                x_value = gt_row[LANE_X_SLICE][bottom_index].clamp(0.0, float(NETWORK_HW[1] - 1))
                y_value = anchor_rows[bottom_index].clamp(0.0, float(NETWORK_HW[0] - 1))
                # Fixed sub-grid jitter makes the train-only query robust to imperfect runtime seeds.
                _write_seed(
                    batch_index,
                    query_index,
                    x_value,
                    y_value,
                    col_jitter=float((gt_index % 3) - 1) * 1.5,
                    row_jitter=float(((gt_index // 3) % 3) - 1) * 1.0,
                )
                targets[batch_index, query_index] = gt_row
                valid[batch_index, query_index] = True
                positive_anchors.append((x_value.detach(), y_value.detach()))
                bottom_x_values.append(x_value.detach())
                query_index += 1

            if self.conditional_denoise_hard_negative_count <= 0 or not positive_anchors:
                continue
            bottom_x_tensor = torch.stack(bottom_x_values) if bottom_x_values else None
            min_lane_gap = max(24.0, 0.35 * self.conditional_denoise_hard_negative_offset_px)
            min_boundary_shift = min(32.0, 0.5 * self.conditional_denoise_hard_negative_offset_px)
            for positive_index, (x_value, y_value) in enumerate(positive_anchors):
                if query_index >= LANE_QUERY_COUNT:
                    break
                for negative_index in range(self.conditional_denoise_hard_negative_count):
                    if query_index >= LANE_QUERY_COUNT:
                        break
                    direction = -1.0 if (positive_index + negative_index) % 2 == 0 else 1.0
                    multiplier = 1 + negative_index // 2
                    offset = direction * float(multiplier) * self.conditional_denoise_hard_negative_offset_px
                    neg_x = (x_value + offset).clamp(0.0, float(NETWORK_HW[1] - 1))
                    if abs(float((neg_x - x_value).item())) < min_boundary_shift:
                        continue
                    if bottom_x_tensor is not None:
                        nearest_lane_gap = float(torch.min(torch.abs(bottom_x_tensor - neg_x)).item())
                        if nearest_lane_gap < min_lane_gap:
                            continue
                    _write_seed(batch_index, query_index, neg_x, y_value)
                    hard_negative[batch_index, query_index] = True
                    query_index += 1
        rows = self._rows_from_seed_inputs(seed_features, seed_xy, seed_cols)
        return rows, targets, valid, hard_negative


__all__ = ["LaneSegFirstHead"]
