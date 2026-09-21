from __future__ import annotations

import torch
import torch.nn.functional as F


def make_anchor_grid(
    feature_shapes: list[tuple[int, int]],
    feature_strides: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_points: list[torch.Tensor] = []
    stride_tensors: list[torch.Tensor] = []
    for (height, width), stride in zip(feature_shapes, feature_strides):
        sy = (torch.arange(height, device=device, dtype=dtype) + 0.5) * float(stride)
        sx = (torch.arange(width, device=device, dtype=dtype) + 0.5) * float(stride)
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        stride_tensors.append(torch.full((height * width, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensors, dim=0)


def decode_anchor_relative_boxes(
    pred_ltrb_logits: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    distances = F.softplus(pred_ltrb_logits) * stride_tensor.unsqueeze(0)
    x1 = anchor_points[:, 0].unsqueeze(0) - distances[..., 0]
    y1 = anchor_points[:, 1].unsqueeze(0) - distances[..., 1]
    x2 = anchor_points[:, 0].unsqueeze(0) + distances[..., 2]
    y2 = anchor_points[:, 1].unsqueeze(0) + distances[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


__all__ = ["decode_anchor_relative_boxes", "make_anchor_grid"]
