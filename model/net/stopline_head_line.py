from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.transform import NETWORK_HW
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
        self.haf_endpoint = nn.Conv2d(self.hidden_dim, 4, kernel_size=1)
        self.haf_valid_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.endpoint_logits = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.endpoint_offset = nn.Conv2d(self.hidden_dim, 4, kernel_size=1)
        self.segment_seed_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.segment_query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 5),
        )
        self.segment_verifier_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 5, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.axis_profile_sample_count = 33
        self.axis_profile_radius_px = 240.0
        self.axis_profile_normal_radius_px = 48.0
        self.axis_profile_encoder = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.axis_segment_query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 5, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor],
        *,
        encoded: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
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
        segment_seed_logits = self.segment_seed_logits(selector_feat)
        segment_logits, segment_points, segment_verifier_logits = self._decode_seeded_segment_set(
            selector_feat,
            segment_seed_logits,
        )
        axis_segment_logits, axis_segment_points, axis_segment_verifier_logits = self._decode_axis_profile_segment_set(
            selector_feat,
            segment_seed_logits,
            angle,
        )
        denoise_logits, denoise_points, denoise_targets, denoise_valid = self._decode_denoised_segment_set(
            selector_feat,
            encoded=encoded,
        )

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
            "stop_line_haf_endpoint": self.haf_endpoint(dense_feat),
            "stop_line_haf_valid_logits": self.haf_valid_logits(dense_feat),
            "stop_line_endpoint_logits": self.endpoint_logits(dense_feat),
            "stop_line_endpoint_offset": self.endpoint_offset(dense_feat),
            "stop_line_segment_seed_logits": segment_seed_logits,
            "stop_line_segment_logits": segment_logits,
            "stop_line_segment_points": segment_points,
            "stop_line_segment_verifier_logits": segment_verifier_logits,
            "stop_line_axis_segment_seed_logits": segment_seed_logits,
            "stop_line_axis_segment_logits": axis_segment_logits,
            "stop_line_axis_segment_points": axis_segment_points,
            "stop_line_axis_segment_verifier_logits": axis_segment_verifier_logits,
            "stop_line_segment_denoise_logits": denoise_logits,
            "stop_line_segment_denoise_points": denoise_points,
            "stop_line_segment_denoise_targets": denoise_targets,
            "stop_line_segment_denoise_valid": denoise_valid,
            "stop_line_feature": line_feat,
        }

    def _decode_seeded_segment_set(
        self,
        feature_map: torch.Tensor,
        seed_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        flat_seed_logits = seed_logits.flatten(2).squeeze(1)
        query_count = min(int(self.output_queries), int(flat_seed_logits.shape[1]))
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
        query_raw = self.segment_query_mlp(torch.cat([seed_features, seed_xy], dim=-1))
        segment_logits = query_raw[..., 0] + top_scores
        segment_points = torch.sigmoid(query_raw[..., 1:5]).view(batch_size, query_count, 2, 2)
        segment_verifier_logits = self._verify_segments(feature_map, seed_features, segment_points)
        if query_count == int(self.output_queries):
            return segment_logits, segment_points, segment_verifier_logits
        padded_logits = segment_logits.new_full((batch_size, int(self.output_queries)), -10.0)
        padded_verifier_logits = segment_verifier_logits.new_full((batch_size, int(self.output_queries)), -10.0)
        padded_points = segment_points.new_zeros((batch_size, int(self.output_queries), 2, 2))
        padded_logits[:, :query_count] = segment_logits
        padded_verifier_logits[:, :query_count] = segment_verifier_logits
        padded_points[:, :query_count] = segment_points
        return padded_logits, padded_points, padded_verifier_logits

    def _top_seed_features(
        self,
        feature_map: torch.Tensor,
        seed_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        flat_seed_logits = seed_logits.flatten(2).squeeze(1)
        query_count = min(int(self.output_queries), int(flat_seed_logits.shape[1]))
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
        return top_scores, top_indices, seed_features, seed_xy

    def _decode_axis_profile_segment_set(
        self,
        feature_map: torch.Tensor,
        seed_logits: torch.Tensor,
        angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, _height, _width = feature_map.shape
        top_scores, top_indices, seed_features, seed_xy = self._top_seed_features(feature_map, seed_logits)
        query_count = int(seed_xy.shape[1])
        if query_count == 0:
            logits = feature_map.new_full((batch_size, int(self.output_queries)), -10.0)
            points = feature_map.new_zeros((batch_size, int(self.output_queries), 2, 2))
            verifier = feature_map.new_full((batch_size, int(self.output_queries)), -10.0)
            return logits, points, verifier
        flat_angle = angle.flatten(2).transpose(1, 2)
        angle_gather_index = top_indices.unsqueeze(-1).expand(-1, -1, 2)
        seed_axis = torch.gather(flat_angle, dim=1, index=angle_gather_index)
        seed_axis = F.normalize(seed_axis, dim=-1, eps=1.0e-6)
        offsets = torch.linspace(
            -float(self.axis_profile_radius_px),
            float(self.axis_profile_radius_px),
            steps=int(self.axis_profile_sample_count),
            device=feature_map.device,
            dtype=feature_map.dtype,
        ).view(1, 1, int(self.axis_profile_sample_count), 1)
        network_scale = feature_map.new_tensor([float(NETWORK_HW[1]), float(NETWORK_HW[0])]).view(1, 1, 1, 2)
        profile_xy = seed_xy.unsqueeze(2) + seed_axis.unsqueeze(2) * offsets / network_scale
        profile_grid = profile_xy.clamp(0.0, 1.0).mul(2.0).sub(1.0).view(
            batch_size,
            query_count * int(self.axis_profile_sample_count),
            1,
            2,
        )
        sampled = F.grid_sample(
            feature_map,
            profile_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(batch_size, channels, query_count, int(self.axis_profile_sample_count))
        profile = sampled.permute(0, 2, 1, 3).reshape(
            batch_size * query_count,
            channels,
            int(self.axis_profile_sample_count),
        )
        encoded_profile = self.axis_profile_encoder(profile)
        encoded_profile = encoded_profile.view(batch_size, query_count, channels, int(self.axis_profile_sample_count))
        profile_mean = encoded_profile.mean(dim=-1)
        profile_max = encoded_profile.amax(dim=-1)
        query_input = torch.cat(
            [seed_features, profile_mean, profile_max, seed_xy, seed_axis, top_scores.unsqueeze(-1)],
            dim=-1,
        )
        query_raw = self.axis_segment_query_mlp(query_input)
        segment_logits = query_raw[..., 0] + top_scores
        along_delta_px = torch.tanh(query_raw[..., 1:2]) * float(self.axis_profile_radius_px)
        normal_delta_px = torch.tanh(query_raw[..., 2:3]) * float(self.axis_profile_normal_radius_px)
        half_length_px = F.softplus(query_raw[..., 3:4]) * 160.0 + 8.0
        normal_axis = torch.stack([-seed_axis[..., 1], seed_axis[..., 0]], dim=-1)
        network_xy = feature_map.new_tensor([float(NETWORK_HW[1]), float(NETWORK_HW[0])]).view(1, 1, 2)
        center_xy = seed_xy + (seed_axis * along_delta_px + normal_axis * normal_delta_px) / network_xy
        start_xy = center_xy - seed_axis * half_length_px / network_xy
        end_xy = center_xy + seed_axis * half_length_px / network_xy
        segment_points = torch.stack([start_xy, end_xy], dim=2).clamp(0.0, 1.0)
        segment_verifier_logits = self._verify_segments(feature_map, seed_features, segment_points)
        if query_count == int(self.output_queries):
            return segment_logits, segment_points, segment_verifier_logits
        padded_logits = segment_logits.new_full((batch_size, int(self.output_queries)), -10.0)
        padded_verifier_logits = segment_verifier_logits.new_full((batch_size, int(self.output_queries)), -10.0)
        padded_points = segment_points.new_zeros((batch_size, int(self.output_queries), 2, 2))
        padded_logits[:, :query_count] = segment_logits
        padded_verifier_logits[:, :query_count] = segment_verifier_logits
        padded_points[:, :query_count] = segment_points
        return padded_logits, padded_points, padded_verifier_logits

    def _verify_segments(
        self,
        feature_map: torch.Tensor,
        seed_features: torch.Tensor,
        segment_points: torch.Tensor,
        *,
        sample_count: int = 8,
    ) -> torch.Tensor:
        batch_size, channels, _height, _width = feature_map.shape
        query_count = int(segment_points.shape[1])
        steps = torch.linspace(
            0.0,
            1.0,
            steps=int(sample_count),
            device=feature_map.device,
            dtype=feature_map.dtype,
        ).view(1, 1, int(sample_count), 1)
        start = segment_points[:, :, 0:1, :]
        end = segment_points[:, :, 1:2, :]
        sample_xy = start * (1.0 - steps) + end * steps
        grid = sample_xy.mul(2.0).sub(1.0).view(batch_size, query_count * int(sample_count), 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(batch_size, channels, query_count, int(sample_count)).permute(0, 2, 3, 1)
        aligned_mean = sampled.mean(dim=2)
        aligned_max = sampled.amax(dim=2)
        flattened_points = segment_points.flatten(start_dim=2)
        segment_delta = segment_points[:, :, 1, :] - segment_points[:, :, 0, :]
        segment_length = torch.linalg.norm(segment_delta, dim=-1, keepdim=True)
        verifier_input = torch.cat(
            [seed_features, aligned_mean, aligned_max, flattened_points, segment_length],
            dim=-1,
        )
        return self.segment_verifier_mlp(verifier_input).squeeze(-1)

    def _decode_denoised_segment_set(
        self,
        feature_map: torch.Tensor,
        *,
        encoded: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, _height, _width = feature_map.shape
        query_count = int(self.output_queries)
        logits = feature_map.new_full((batch_size, query_count), -10.0)
        points = feature_map.new_zeros((batch_size, query_count, 2, 2))
        targets = feature_map.new_zeros((batch_size, query_count, 2, 2))
        valid = torch.zeros((batch_size, query_count), device=feature_map.device, dtype=torch.bool)
        if not self.training or not isinstance(encoded, dict):
            return logits, points, targets, valid
        stop_target = encoded.get("stop_line")
        mask_dict = encoded.get("mask")
        if not isinstance(stop_target, torch.Tensor) or not isinstance(mask_dict, dict):
            return logits, points, targets, valid
        stop_valid = mask_dict.get("stop_line_valid")
        stop_source = mask_dict.get("stop_line_source")
        if not isinstance(stop_valid, torch.Tensor) or not isinstance(stop_source, torch.Tensor):
            return logits, points, targets, valid
        stop_target = stop_target.to(device=feature_map.device, dtype=feature_map.dtype)
        stop_valid = stop_valid.to(device=feature_map.device, dtype=torch.bool)
        stop_source = stop_source.to(device=feature_map.device, dtype=torch.bool)
        denom = feature_map.new_tensor([float(NETWORK_HW[1]), float(NETWORK_HW[0])]).view(1, 1, 2)
        for batch_index in range(batch_size):
            if not bool(stop_source[batch_index]):
                continue
            valid_indices = torch.nonzero(stop_valid[batch_index], as_tuple=False).flatten()
            if int(valid_indices.numel()) == 0:
                continue
            seed_xy_rows: list[torch.Tensor] = []
            target_rows: list[torch.Tensor] = []
            for target_index in valid_indices.tolist():
                target_points = stop_target[batch_index, int(target_index), 1:].view(-1, 2)
                target_segment = torch.stack([target_points[0], target_points[-1]], dim=0)
                target_segment = (target_segment / denom.squeeze(0)).clamp(0.0, 1.0)
                midpoint = target_segment.mean(dim=0)
                axis = target_segment[1] - target_segment[0]
                axis = axis / torch.linalg.norm(axis).clamp(min=1.0e-6)
                jittered_midpoints = [
                    midpoint,
                    (midpoint + 0.04 * axis).clamp(0.0, 1.0),
                    (midpoint - 0.04 * axis).clamp(0.0, 1.0),
                ]
                for seed_xy in jittered_midpoints:
                    if len(seed_xy_rows) >= query_count:
                        break
                    seed_xy_rows.append(seed_xy)
                    target_rows.append(target_segment)
                if len(seed_xy_rows) >= query_count:
                    break
            if not seed_xy_rows:
                continue
            seed_xy = torch.stack(seed_xy_rows, dim=0)
            grid = seed_xy.mul(2.0).sub(1.0).view(1, int(seed_xy.shape[0]), 1, 2)
            sampled = F.grid_sample(
                feature_map[batch_index : batch_index + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            seed_features = sampled.view(1, channels, int(seed_xy.shape[0])).transpose(1, 2).squeeze(0)
            query_raw = self.segment_query_mlp(torch.cat([seed_features, seed_xy], dim=-1))
            current_count = int(seed_xy.shape[0])
            logits[batch_index, :current_count] = query_raw[..., 0]
            points[batch_index, :current_count] = torch.sigmoid(query_raw[..., 1:5]).view(current_count, 2, 2)
            targets[batch_index, :current_count] = torch.stack(target_rows, dim=0)
            valid[batch_index, :current_count] = True
        return logits, points, targets, valid


__all__ = ["StopLineDenseLocalHead"]
