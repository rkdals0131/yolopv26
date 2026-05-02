from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.transform import NETWORK_HW
from .roadmark_blocks import ConvNormAct, MultiScaleFusion
from .roadmark_current_family import LANE_QUERY_COUNT, LANE_VECTOR_DIM


LANE_ROW_COUNT = 16
LANE_COLUMN_COUNT = 200
LANE_SLOT_COUNT = 8
LANE_COLOR_DIM = 3
LANE_TYPE_DIM = 2
LANE_ANCHOR_COUNT = 16
LANE_COLOR_SLICE = slice(1, 1 + LANE_COLOR_DIM)
LANE_TYPE_SLICE = slice(LANE_COLOR_SLICE.stop, LANE_COLOR_SLICE.stop + LANE_TYPE_DIM)
LANE_X_SLICE = slice(LANE_TYPE_SLICE.stop, LANE_TYPE_SLICE.stop + LANE_ANCHOR_COUNT)
LANE_VIS_SLICE = slice(LANE_X_SLICE.stop, LANE_X_SLICE.stop + LANE_ANCHOR_COUNT)


class LaneDenseRowSeedHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int],
        *,
        hidden_dim: int = 128,
        slot_count: int = LANE_SLOT_COUNT,
        output_queries: int = LANE_QUERY_COUNT,
        column_count: int = LANE_COLUMN_COUNT,
        refine_delta_limit: float = 8.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.slot_count = int(slot_count)
        self.output_queries = int(output_queries)
        self.column_count = int(column_count)
        self.refine_delta_limit = float(refine_delta_limit)
        self.fusion = MultiScaleFusion(in_channels, self.hidden_dim, target_level=1, depth=2)
        self.row_stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.centerline_stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.centerline_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.location_head = nn.Conv2d(self.hidden_dim, self.slot_count, kernel_size=1)
        # Change A: learnable positional anchors for left-to-right slot ordering.
        self.slot_column_anchor = nn.Parameter(
            torch.linspace(0.0, 1.0, self.slot_count)
        )
        self.anchor_temperature = nn.Parameter(torch.tensor(5.0))
        self.refine_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.existence_head = nn.Conv2d(self.hidden_dim, self.slot_count, kernel_size=1)
        self.color_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, LANE_COLOR_DIM),
        )
        self.slot_obj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, LANE_TYPE_DIM),
        )

    def _slot_tokens(self, row_features: torch.Tensor, row_probs: torch.Tensor) -> torch.Tensor:
        weights = row_probs / row_probs.sum(dim=(2, 3), keepdim=True).clamp(min=1.0e-6)
        feature_grid = row_features.permute(0, 2, 3, 1).unsqueeze(1)
        return (weights.unsqueeze(-1) * feature_grid).sum(dim=(2, 3))

    def _build_lane_vectors(
        self,
        *,
        col_expectation: torch.Tensor,
        existence_logits: torch.Tensor,
        slot_obj_logits: torch.Tensor,
        color_logits: torch.Tensor,
        type_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(col_expectation.shape[0])
        x_scale = float(NETWORK_HW[1] - 1) / max(float(self.column_count - 1), 1.0)
        x_pixels = col_expectation * x_scale
        objectness_logits = slot_obj_logits

        lane_vectors = torch.zeros(
            (batch_size, self.output_queries, LANE_VECTOR_DIM),
            device=col_expectation.device,
            dtype=col_expectation.dtype,
        )
        lane_vectors[..., 0] = -10.0
        lane_vectors[..., LANE_VIS_SLICE] = -10.0

        active = min(self.slot_count, self.output_queries)
        lane_vectors[:, :active, 0] = objectness_logits[:, :active]
        lane_vectors[:, :active, LANE_COLOR_SLICE] = color_logits[:, :active]
        lane_vectors[:, :active, LANE_TYPE_SLICE] = type_logits[:, :active]
        lane_vectors[:, :active, LANE_X_SLICE] = x_pixels[:, :active]
        lane_vectors[:, :active, LANE_VIS_SLICE] = existence_logits[:, :active]
        return lane_vectors

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del encoded
        lane_feat = self.fusion(features)
        lane_centerline_logits = self.centerline_logits(self.centerline_stem(lane_feat))
        row_features = F.adaptive_avg_pool2d(lane_feat, output_size=(LANE_ROW_COUNT, self.column_count))
        row_features = self.row_stem(row_features)

        row_logits = self.location_head(row_features).view(-1, self.slot_count, LANE_ROW_COUNT, self.column_count)
        # Change A: apply positional anchor bias to encourage left-to-right ordering.
        col_positions = torch.linspace(
            0.0, 1.0, self.column_count, device=row_logits.device, dtype=row_logits.dtype,
        )  # [C]
        anchor_bias = -self.anchor_temperature.abs() * (
            col_positions[None, None, :] - self.slot_column_anchor[:, None, None]  # [S, 1, C]
        ).pow(2)
        row_logits = row_logits + anchor_bias.unsqueeze(0)  # [B, S, R, C]
        row_probs = torch.softmax(row_logits, dim=-1)
        existence_map = self.existence_head(row_features).view(-1, self.slot_count, LANE_ROW_COUNT, self.column_count)
        # Couple visibility evidence to the predicted lane position instead of averaging over all columns.
        existence_logits = (row_probs.detach() * existence_map).sum(dim=-1)
        feature_grid = row_features.permute(0, 2, 3, 1).unsqueeze(1)
        # Keep the coarse row-classification path stable; refine operates on detached coarse assignments.
        row_context = (row_probs.detach().unsqueeze(-1) * feature_grid).sum(dim=3)
        refine_delta_raw = self.refine_head(row_context).squeeze(-1)
        if self.refine_delta_limit > 0.0:
            limit = float(self.refine_delta_limit)
            refine_delta = limit * torch.tanh(refine_delta_raw / limit)
        else:
            refine_delta = refine_delta_raw
        col_bins = torch.linspace(
            0.0,
            float(self.column_count - 1),
            self.column_count,
            device=row_logits.device,
            dtype=row_logits.dtype,
        ).view(1, 1, 1, -1)
        col_expectation = (row_probs * col_bins).sum(dim=-1)
        refined_col_expectation = (col_expectation + refine_delta).clamp(min=0.0, max=float(self.column_count - 1))
        slot_tokens = self._slot_tokens(row_features, row_probs)
        slot_obj_logits = self.slot_obj_head(slot_tokens).squeeze(-1)
        color_logits = self.color_head(slot_tokens)
        type_logits = self.type_head(slot_tokens)
        lane_vectors = self._build_lane_vectors(
            col_expectation=refined_col_expectation,
            existence_logits=existence_logits,
            slot_obj_logits=slot_obj_logits,
            color_logits=color_logits,
            type_logits=type_logits,
        )

        return {
            "lane": lane_vectors,
            "lane_row_logits": row_logits,
            "lane_exist_logits": existence_logits,
            "lane_slot_obj_logits": slot_obj_logits,
            "lane_slot_color_logits": color_logits,
            "lane_slot_type_logits": type_logits,
            "lane_centerline_logits": lane_centerline_logits,
            "lane_row_col_expectation": refined_col_expectation,
            "lane_row_col_coarse": col_expectation,
            "lane_row_refine_delta": refine_delta,
            "lane_row_refine_delta_raw": refine_delta_raw,
            "lane_feature": lane_feat,
        }


__all__ = ["LaneDenseRowSeedHead", "LANE_COLUMN_COUNT", "LANE_ROW_COUNT", "LANE_SLOT_COUNT"]
