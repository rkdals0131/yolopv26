from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw
import numpy as np
import torch

from .transform import NETWORK_HW


LANE_SEED_OUTPUT_HW = (16, 100)
LANE_CENTERLINE_OUTPUT_HW = (76, 100)
LANE_ROW_CLASS_OUTPUT_HW = (16, 200)
LANE_ROW_SLOT_COUNT = 8
ROADMARK_DENSE_OUTPUT_HW = (152, 200)


def _scale_points_to_output(points: torch.Tensor, *, output_hw: tuple[int, int]) -> torch.Tensor:
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    scaled = points.clone().to(dtype=torch.float32)
    scaled[:, 0] = scaled[:, 0] * float(output_w) / float(NETWORK_HW[1])
    scaled[:, 1] = scaled[:, 1] * float(output_h) / float(NETWORK_HW[0])
    return scaled


def build_lane_dense_row_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    output_rows: int = LANE_SEED_OUTPUT_HW[0],
    output_cols: int = LANE_SEED_OUTPUT_HW[1],
) -> dict[str, torch.Tensor]:
    from .target_encoder import LANE_QUERY_COUNT, LANE_VIS_SLICE, LANE_X_SLICE, _encode_lane_rows

    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    encoded, row_valid = _encode_lane_rows(rows, valid_tensor, source_enabled)
    seed_heatmap = torch.zeros((output_rows, output_cols), dtype=torch.float32)
    seed_offset = torch.zeros((output_rows, output_cols), dtype=torch.float32)
    seed_positive_rows = torch.full((LANE_QUERY_COUNT,), -1, dtype=torch.long)
    seed_positive_cols = torch.full((LANE_QUERY_COUNT,), -1, dtype=torch.long)

    if source_enabled:
        for query_index, row in enumerate(encoded[row_valid]):
            anchor_visibility = row[LANE_VIS_SLICE] > 0.5
            visible_indices = torch.nonzero(anchor_visibility, as_tuple=False).flatten()
            if visible_indices.numel() == 0:
                continue
            bottom_row = int(visible_indices.min().item())
            x_value = float(row[LANE_X_SLICE][bottom_row].item())
            normalized_x = max(0.0, min(1.0, x_value / max(float(NETWORK_HW[1] - 1), 1.0)))
            scaled_x = normalized_x * float(output_cols - 1)
            col = int(round(scaled_x))
            col = max(0, min(output_cols - 1, col))
            seed_heatmap[bottom_row, col] = 1.0
            seed_offset[bottom_row, col] = scaled_x - float(col)
            seed_positive_rows[query_index] = bottom_row
            seed_positive_cols[query_index] = col

    return {
        "lane_vector_targets": encoded,
        "lane_vector_valid": row_valid,
        "lane_seed_heatmap": seed_heatmap,
        "lane_seed_offset": seed_offset,
        "lane_seed_positive_rows": seed_positive_rows,
        "lane_seed_positive_cols": seed_positive_cols,
    }


def build_lane_row_class_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    slot_count: int = LANE_ROW_SLOT_COUNT,
    output_rows: int = LANE_ROW_CLASS_OUTPUT_HW[0],
    output_cols: int = LANE_ROW_CLASS_OUTPUT_HW[1],
) -> dict[str, torch.Tensor]:
    from .target_encoder import (
        LANE_COLOR_SLICE,
        LANE_TYPE_SLICE,
        LANE_VIS_SLICE,
        LANE_X_SLICE,
        _encode_lane_rows,
    )

    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    encoded, row_valid = _encode_lane_rows(rows, valid_tensor, source_enabled)
    slot_valid = torch.zeros((slot_count,), dtype=torch.bool)
    row_exists = torch.zeros((slot_count, output_rows), dtype=torch.float32)
    row_col_index = torch.full((slot_count, output_rows), -1, dtype=torch.long)
    row_col_target = torch.full((slot_count, output_rows), -1.0, dtype=torch.float32)
    # Change C: Gaussian-smoothed soft location targets.
    row_soft_target = torch.zeros((slot_count, output_rows, output_cols), dtype=torch.float32)
    slot_color = torch.zeros((slot_count,), dtype=torch.long)
    slot_type = torch.zeros((slot_count,), dtype=torch.long)

    if source_enabled:
        sortable_rows: list[tuple[float, torch.Tensor]] = []
        for row in encoded[row_valid]:
            visible = row[LANE_VIS_SLICE] > 0.5
            visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
            if visible_indices.numel() == 0:
                continue
            bottom_index = int(visible_indices[0].item())
            bottom_x = float(row[LANE_X_SLICE][bottom_index].item())
            sortable_rows.append((bottom_x, row))
        sortable_rows.sort(key=lambda item: item[0])

        x_scale = float(output_cols - 1) / max(float(NETWORK_HW[1] - 1), 1.0)
        col_positions = torch.arange(output_cols, dtype=torch.float32)  # [C]
        soft_sigma = 2.0  # bins
        for slot_index, (_, row) in enumerate(sortable_rows[:slot_count]):
            slot_valid[slot_index] = True
            slot_color[slot_index] = int(row[LANE_COLOR_SLICE].argmax().item())
            slot_type[slot_index] = int(row[LANE_TYPE_SLICE].argmax().item())
            visible = row[LANE_VIS_SLICE] > 0.5
            row_exists[slot_index] = visible.to(dtype=torch.float32)
            scaled_x = torch.round(row[LANE_X_SLICE] * x_scale).to(dtype=torch.long)
            scaled_x = scaled_x.clamp(min=0, max=output_cols - 1)
            row_col_index[slot_index, visible] = scaled_x[visible]
            # Build soft targets for each visible row.
            scaled_x_float = row[LANE_X_SLICE] * x_scale  # float positions [R]
            row_col_target[slot_index, visible] = scaled_x_float[visible]
            for row_idx in range(output_rows):
                if not bool(visible[row_idx]):
                    continue
                target_col = float(scaled_x_float[row_idx].item())
                soft = torch.exp(-0.5 * ((col_positions - target_col) / soft_sigma) ** 2)
                soft = soft / soft.sum().clamp(min=1.0e-6)
                row_soft_target[slot_index, row_idx] = soft

    return {
        "lane_slot_valid": slot_valid,
        "lane_row_exists": row_exists,
        "lane_row_col_index": row_col_index,
        "lane_row_col_target": row_col_target,
        "lane_row_soft_target": row_soft_target,
        "lane_slot_color": slot_color,
        "lane_slot_type": slot_type,
    }


def build_lane_centerline_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    output_hw: tuple[int, int] = LANE_CENTERLINE_OUTPUT_HW,
    line_width: int = 3,
) -> dict[str, torch.Tensor]:
    from .target_encoder import _sort_lane_points

    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    centerline = Image.new("L", (output_w, output_h), 0)
    centerline_draw = ImageDraw.Draw(centerline)

    if source_enabled:
        for source_index, row in enumerate(rows):
            valid = bool(valid_tensor.reshape(-1)[source_index]) if source_index < len(rows) else False
            if not valid:
                continue
            points = torch.as_tensor(row.get("points_xy", []), dtype=torch.float32).reshape(-1, 2)
            visibility = row.get("visibility")
            sorted_points, sorted_visibility = _sort_lane_points(points, visibility)
            visible_points = sorted_points[sorted_visibility > 0.5]
            if visible_points.shape[0] < 2:
                continue
            scaled = _scale_points_to_output(visible_points, output_hw=output_hw)
            polyline = [(float(point[0]), float(point[1])) for point in scaled]
            centerline_draw.line(polyline, fill=1, width=int(line_width))

    centerline_tensor = torch.from_numpy(np.array(centerline, dtype=np.float32)).unsqueeze(0)
    return {"lane_centerline": centerline_tensor}


def build_lane_segfirst_targets(
    lane_rows: list[dict[str, Any]],
    lane_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    stop_line_rows: list[dict[str, Any]] | None = None,
    stop_line_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...] | None = None,
    crosswalk_rows: list[dict[str, Any]] | None = None,
    crosswalk_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...] | None = None,
    source_enabled: bool = True,
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW,
) -> dict[str, torch.Tensor]:
    """Build opt-in dense lane targets for the seg-first lane head."""

    from .target_encoder import lane_supervised_valid_mask
    from ..engine.lane_segfirst_vectorizer import LaneSegFirstTargetConfig, render_lane_segfirst_targets

    supervised_mask = lane_supervised_valid_mask(lane_rows, lane_valid_mask)
    maps = render_lane_segfirst_targets(
        lane_rows,
        supervised_mask,
        stop_line_rows=stop_line_rows,
        stop_line_valid_mask=stop_line_valid_mask,
        crosswalk_rows=crosswalk_rows,
        crosswalk_valid_mask=crosswalk_valid_mask,
        source_enabled=bool(source_enabled),
        config=LaneSegFirstTargetConfig(output_hw=output_hw),
    )
    return {
        "lane_seg_centerline_core": maps["centerline_core"],
        "lane_seg_centerline_soft": maps["centerline_soft"],
        "lane_seg_support": maps["support"],
        "lane_seg_tangent_axis": maps["tangent_axis"],
        "lane_seg_color": maps["color_map"],
        "lane_seg_type": maps["lane_type_map"],
        "lane_seg_ignore": maps["lane_ignore"],
        "lane_seg_negative": maps["lane_negative"],
        "lane_seg_stop_line_ignore": maps["stop_line_ignore"],
        "lane_seg_crosswalk_ignore": maps["crosswalk_ignore"],
        "lane_seg_tangent_count": maps["tangent_count"],
    }


def build_stopline_dense_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW,
    center_radius: int = 2,
    center_span: int = 6,
) -> dict[str, torch.Tensor]:
    from .target_encoder import _encode_stop_line_rows, _sample_stop_line_points

    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    encoded, row_valid = _encode_stop_line_rows(rows, valid_tensor, source_enabled)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    center_heatmap = torch.zeros((1, output_h, output_w), dtype=torch.float32)
    center_offset = torch.zeros((2, output_h, output_w), dtype=torch.float32)
    angle = torch.zeros((2, output_h, output_w), dtype=torch.float32)
    half_length = torch.zeros((1, output_h, output_w), dtype=torch.float32)

    if source_enabled:
        for row in rows:
            points = _sample_stop_line_points(row.get("points_xy", []))
            if points.shape[0] < 2:
                continue
            start = points[0]
            end = points[-1]
            center = (start + end) * 0.5
            scaled_center = _scale_points_to_output(center.unsqueeze(0), output_hw=output_hw)[0]
            col = max(0, min(output_w - 1, int(torch.floor(scaled_center[0]).item())))
            row_index = max(0, min(output_h - 1, int(torch.floor(scaled_center[1]).item())))
            sigma = max(1.0, float(center_radius))
            support_radius = max(1, int(round(sigma * 3.0)))
            for d_row in range(-support_radius, support_radius + 1):
                for d_col in range(-support_radius, support_radius + 1):
                    target_row = row_index + d_row
                    target_col = col + d_col
                    if not (0 <= target_row < output_h and 0 <= target_col < output_w):
                        continue
                    distance_sq = float(d_row * d_row + d_col * d_col)
                    value = float(np.exp(-distance_sq / (2.0 * sigma * sigma)))
                    center_heatmap[0, target_row, target_col] = max(float(center_heatmap[0, target_row, target_col].item()), value)
            center_offset[0, row_index, col] = float(scaled_center[0] - col)
            center_offset[1, row_index, col] = float(scaled_center[1] - row_index)
            delta = end - start
            norm = max(float(torch.linalg.norm(delta).item()), 1.0e-6)
            angle[0, row_index, col] = float(delta[0] / norm)
            angle[1, row_index, col] = float(delta[1] / norm)
            half_length[0, row_index, col] = float(norm * 0.5 * float(output_w) / float(NETWORK_HW[1]))

    return {
        "stop_line_vector_targets": encoded,
        "stop_line_vector_valid": row_valid,
        "stop_line_center_heatmap": center_heatmap,
        "stop_line_center_offset": center_offset,
        "stop_line_angle": angle,
        "stop_line_half_length": half_length,
    }


def build_stopline_mask_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW,
    mask_width: int = 7,
    center_width: int = 1,
) -> dict[str, torch.Tensor]:
    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    mask = Image.new("L", (output_w, output_h), 0)
    center = Image.new("L", (output_w, output_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    center_draw = ImageDraw.Draw(center)

    if source_enabled:
        for source_index, row in enumerate(rows):
            valid = bool(valid_tensor.reshape(-1)[source_index]) if source_index < len(rows) else False
            if not valid:
                continue
            points = torch.as_tensor(row.get("points_xy", []), dtype=torch.float32).reshape(-1, 2)
            if points.shape[0] < 2:
                continue
            scaled = _scale_points_to_output(points[[0, -1]], output_hw=output_hw)
            start = (float(scaled[0, 0]), float(scaled[0, 1]))
            end = (float(scaled[-1, 0]), float(scaled[-1, 1]))
            mask_draw.line((start, end), fill=1, width=int(mask_width))
            center_draw.line((start, end), fill=1, width=int(center_width))

    mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
    center_tensor = torch.from_numpy(np.array(center, dtype=np.float32)).unsqueeze(0)
    return {
        "stop_line_mask": mask_tensor,
        "stop_line_centerline": center_tensor,
    }


def build_crosswalk_mask_targets(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    source_enabled: bool = True,
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW,
    boundary_width: int = 2,
    center_radius: int = 2,
) -> dict[str, torch.Tensor]:
    from .target_encoder import _encode_crosswalk_rows, _sample_crosswalk_points

    valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    encoded, row_valid = _encode_crosswalk_rows(rows, valid_tensor, source_enabled)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    mask = Image.new("L", (output_w, output_h), 0)
    boundary = Image.new("L", (output_w, output_h), 0)
    center = Image.new("L", (output_w, output_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    boundary_draw = ImageDraw.Draw(boundary)
    center_draw = ImageDraw.Draw(center)

    if source_enabled:
        for source_index, row in enumerate(rows):
            valid = bool(valid_tensor.reshape(-1)[source_index]) if source_index < len(rows) else False
            if not valid:
                continue
            points = _sample_crosswalk_points(row.get("points_xy", []))
            if points.shape[0] < 3:
                continue
            scaled = _scale_points_to_output(points, output_hw=output_hw)
            polygon = [(float(point[0]), float(point[1])) for point in scaled]
            mask_draw.polygon(polygon, outline=1, fill=1)
            boundary_draw.line(polygon + [polygon[0]], fill=1, width=int(boundary_width))
            centroid = scaled.mean(dim=0)
            cx = float(centroid[0])
            cy = float(centroid[1])
            center_draw.ellipse(
                (
                    cx - float(center_radius),
                    cy - float(center_radius),
                    cx + float(center_radius),
                    cy + float(center_radius),
                ),
                fill=1,
            )

    mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
    boundary_tensor = torch.from_numpy(np.array(boundary, dtype=np.float32)).unsqueeze(0)
    center_tensor = torch.from_numpy(np.array(center, dtype=np.float32)).unsqueeze(0)
    return {
        "crosswalk_vector_targets": encoded,
        "crosswalk_vector_valid": row_valid,
        "crosswalk_mask": mask_tensor,
        "crosswalk_boundary": boundary_tensor,
        "crosswalk_center": center_tensor,
    }


__all__ = [
    "build_crosswalk_mask_targets",
    "build_lane_centerline_targets",
    "build_lane_dense_row_targets",
    "build_lane_row_class_targets",
    "build_lane_segfirst_targets",
    "LANE_CENTERLINE_OUTPUT_HW",
    "build_stopline_mask_targets",
    "build_stopline_dense_targets",
    "LANE_ROW_CLASS_OUTPUT_HW",
    "LANE_ROW_SLOT_COUNT",
    "LANE_SEED_OUTPUT_HW",
    "ROADMARK_DENSE_OUTPUT_HW",
]
