from __future__ import annotations

import torch
import torch.nn as nn

from .roadmark_blocks import ConvNormAct, MultiScaleFusion
from .roadmark_current_family import CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM


class CrosswalkMaskFirstHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int],
        *,
        hidden_dim: int = 96,
        topk: int = CROSSWALK_QUERY_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.fusion = MultiScaleFusion(in_channels, self.hidden_dim, target_level=0, depth=2)
        self.mask_stem = nn.Sequential(
            ConvNormAct(self.hidden_dim, self.hidden_dim),
            ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.mask_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.boundary_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.center_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.vector_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, CROSSWALK_VECTOR_DIM),
        )

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        crosswalk_feat = self.fusion(features)
        dense_feat = self.mask_stem(crosswalk_feat)
        center_logits = self.center_logits(dense_feat)
        batch_size, _, height, width = center_logits.shape
        flat_scores = center_logits.flatten(1)
        topk = min(self.topk, int(flat_scores.shape[1]))
        topk_scores, topk_indices = torch.topk(flat_scores, k=topk, dim=1)
        row_indices = torch.div(topk_indices, width, rounding_mode="floor")
        col_indices = topk_indices % width
        tokens = dense_feat.permute(0, 2, 3, 1).reshape(batch_size, -1, self.hidden_dim)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        selected_tokens = torch.gather(tokens, 1, gather_index)
        return {
            "crosswalk": self.vector_head(selected_tokens),
            "crosswalk_mask_logits": self.mask_logits(dense_feat),
            "crosswalk_boundary_logits": self.boundary_logits(dense_feat),
            "crosswalk_center_logits": center_logits,
            "crosswalk_seed_scores": topk_scores,
            "crosswalk_seed_rows": row_indices,
            "crosswalk_seed_cols": col_indices,
            "crosswalk_feature": crosswalk_feat,
        }


__all__ = ["CrosswalkMaskFirstHead"]
