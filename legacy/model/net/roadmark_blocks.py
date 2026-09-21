from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class MultiScaleFusion(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        hidden_dim: int,
        *,
        target_level: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.target_level = int(target_level)
        if not self.in_channels:
            raise ValueError("MultiScaleFusion expects at least one input level.")
        if self.target_level < 0 or self.target_level >= len(self.in_channels):
            raise ValueError("target_level is out of range for the provided pyramid.")
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for channel in self.in_channels
            ]
        )
        layers: list[nn.Module] = [ConvNormAct(hidden_dim * len(self.in_channels), hidden_dim)]
        for _ in range(max(0, int(depth) - 1)):
            layers.append(ConvNormAct(hidden_dim, hidden_dim))
        self.fusion = nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"MultiScaleFusion expected {len(self.in_channels)} features but received {len(features)}."
            )
        target_hw = (int(features[self.target_level].shape[2]), int(features[self.target_level].shape[3]))
        projected: list[torch.Tensor] = []
        for feature, projector, channel_count in zip(features, self.projectors, self.in_channels):
            if feature.ndim != 4 or int(feature.shape[1]) != channel_count:
                raise ValueError(
                    f"Expected feature map [B, {channel_count}, H, W], got {tuple(feature.shape)}."
                )
            value = projector(feature)
            if tuple(value.shape[-2:]) != target_hw:
                value = F.interpolate(value, size=target_hw, mode="bilinear", align_corners=False)
            projected.append(value)
        return self.fusion(torch.cat(projected, dim=1))


__all__ = ["ConvNormAct", "MultiScaleFusion"]
