from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .roadmark_blocks import ConvNormAct, MultiScaleFusion
from .roadmark_current_family import STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM


def _row_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def _col_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class StopLineDenseLocalHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int],
        *,
        hidden_dim: int = 128,
        output_queries: int = STOP_LINE_QUERY_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_queries = int(output_queries)
        self.fusion = MultiScaleFusion(in_channels, self.hidden_dim, target_level=0, depth=2)
        self.mask_stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.center_stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.row_pool_project = nn.Sequential(
            _row_conv_block(self.hidden_dim * 2 + 1, self.hidden_dim),
            _row_conv_block(self.hidden_dim, self.hidden_dim),
        )
        self.row_stem = nn.Sequential(
            _row_conv_block(self.hidden_dim, self.hidden_dim),
            _row_conv_block(self.hidden_dim, self.hidden_dim),
        )
        self.col_pool_project = nn.Sequential(
            _col_conv_block(self.hidden_dim * 2 + 1, self.hidden_dim),
            _col_conv_block(self.hidden_dim, self.hidden_dim),
        )
        self.col_stem = nn.Sequential(
            _col_conv_block(self.hidden_dim, self.hidden_dim),
            _col_conv_block(self.hidden_dim, self.hidden_dim),
        )
        self.selector_map_fuse = nn.Sequential(
            ConvNormAct(self.hidden_dim * 3, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.mask_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.row_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.x_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.selector_map_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.center_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.center_offset = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.angle = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.half_length = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor],
        *,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del encoded
        line_feat = self.fusion(features)
        dense_feat = self.mask_stem(line_feat)
        row_source = line_feat
        row_mean = row_source.mean(dim=-1, keepdim=True)
        row_max = row_source.amax(dim=-1, keepdim=True)
        row_coord = torch.linspace(
            0.0,
            1.0,
            steps=int(row_source.shape[2]),
            device=row_source.device,
            dtype=row_source.dtype,
        ).view(1, 1, int(row_source.shape[2]), 1)
        row_coord = row_coord.expand(int(row_source.shape[0]), 1, int(row_source.shape[2]), 1)
        row_feat = self.row_stem(self.row_pool_project(torch.cat([row_mean, row_max, row_coord], dim=1)))
        col_mean = line_feat.mean(dim=2, keepdim=True)
        col_max = line_feat.amax(dim=2, keepdim=True)
        col_coord = torch.linspace(
            0.0,
            1.0,
            steps=int(line_feat.shape[3]),
            device=line_feat.device,
            dtype=line_feat.dtype,
        ).view(1, 1, 1, int(line_feat.shape[3]))
        col_coord = col_coord.expand(int(line_feat.shape[0]), 1, 1, int(line_feat.shape[3]))
        col_feat = self.col_stem(self.col_pool_project(torch.cat([col_mean, col_max, col_coord], dim=1)))
        row_feat_2d = row_feat.expand(-1, -1, int(dense_feat.shape[2]), int(dense_feat.shape[3]))
        col_feat_2d = col_feat.expand(-1, -1, int(dense_feat.shape[2]), int(dense_feat.shape[3]))
        selector_feat = self.selector_map_fuse(torch.cat([dense_feat, row_feat_2d, col_feat_2d], dim=1))
        mask_logits = self.mask_logits(dense_feat)
        row_logits = self.row_logits(row_feat)
        x_logits = self.x_logits(col_feat)
        selector_map_logits = self.selector_map_logits(selector_feat)
        center_logits = self.center_logits(dense_feat)
        center_offset = torch.sigmoid(self.center_offset(dense_feat))
        angle = torch.tanh(self.angle(dense_feat))
        angle = F.normalize(angle, dim=1, eps=1.0e-6)
        half_length = F.softplus(self.half_length(dense_feat))
        batch_size = int(mask_logits.shape[0])

        stop_line_vectors = torch.zeros(
            (batch_size, self.output_queries, STOP_LINE_VECTOR_DIM),
            device=mask_logits.device,
            dtype=mask_logits.dtype,
        )
        stop_line_vectors[..., 0] = -10.0

        return {
            "stop_line": stop_line_vectors,
            "stop_line_mask_logits": mask_logits,
            "stop_line_row_logits": row_logits,
            "stop_line_x_logits": x_logits,
            "stop_line_selector_map_logits": selector_map_logits,
            "stop_line_center_logits": center_logits,
            "stop_line_center_offset": center_offset,
            "stop_line_angle": angle,
            "stop_line_half_length": half_length,
            "stop_line_feature": line_feat,
        }


__all__ = ["StopLineDenseLocalHead"]
