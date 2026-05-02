from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

from common.pv26_schema import LANE_CLASSES, LANE_TYPES
from ..data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW
from ..data.transform import NETWORK_HW, clip_points, inverse_transform_points, transform_from_meta, unique_point_count
from .metrics import _hungarian_from_cost, _lane_family_metrics, _mean_point_distance


@dataclass(frozen=True)
class LaneSegFirstTargetConfig:
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW
    centerline_core_width: int = 1
    centerline_soft_sigma: float = 1.5
    centerline_soft_radius: int = 5
    support_width: int = 9
    tangent_width: int = 5
    stopline_ignore_width: int = 11
    crosswalk_boundary_width: int = 3


@dataclass(frozen=True)
class LaneSegFirstVectorizerConfig:
    centerline_threshold: float = 0.5
    min_component_pixels: int = 2
    max_component_pixels: int | None = 1000
    max_components: int | None = 256
    max_predictions: int | None = 64
    min_polyline_points: int = 2
    min_polyline_length_px: float = 0.0
    min_polyline_bottom_y_fraction: float = 0.0
    semantic_vote_mode: str = "component"
    row_stride: int = 2
    max_row_gap: int = 12
    max_link_dx: float = 8.0
    lane_match_threshold: float = 40.0


def _valid_mask_value(valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...], index: int) -> bool:
    values = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1)
    return index < int(values.numel()) and bool(values[index].item())


def _scale_points(
    points: torch.Tensor,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> torch.Tensor:
    scaled = points.clone().to(dtype=torch.float32)
    scaled[:, 0] = scaled[:, 0] * float(target_hw[1]) / max(float(source_hw[1]), 1.0)
    scaled[:, 1] = scaled[:, 1] * float(target_hw[0]) / max(float(source_hw[0]), 1.0)
    return scaled


def _scale_points_list(
    points_xy: list[list[float]],
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> list[list[float]]:
    if not points_xy:
        return []
    points = torch.tensor(points_xy, dtype=torch.float32).reshape(-1, 2)
    scaled = _scale_points(points, source_hw=source_hw, target_hw=target_hw)
    return [[float(x), float(y)] for x, y in scaled.tolist()]


def _visible_lane_points(row: dict[str, Any]) -> torch.Tensor:
    points = torch.as_tensor(row.get("points_xy", []), dtype=torch.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return points
    visibility = row.get("visibility")
    if isinstance(visibility, torch.Tensor):
        visible = visibility.to(dtype=torch.float32).reshape(-1)
    elif isinstance(visibility, (list, tuple)):
        visible = torch.tensor(visibility, dtype=torch.float32).reshape(-1)
    else:
        visible = torch.ones(points.shape[0], dtype=torch.float32)
    if int(visible.numel()) != int(points.shape[0]):
        visible = torch.ones(points.shape[0], dtype=torch.float32)
    points = points[visible >= 0.5]
    if points.shape[0] <= 1:
        return points
    order = torch.argsort(points[:, 1], descending=True)
    return points[order]


def _polyline_pairs(points: torch.Tensor) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if points.shape[0] < 2:
        return pairs
    values = points.detach().cpu().numpy().astype(np.float32)
    for start, end in zip(values[:-1], values[1:]):
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        if float(np.linalg.norm(end - start)) <= 1.0e-6:
            continue
        pairs.append(((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))
    return pairs


def _draw_line(canvas: Image.Image, points: torch.Tensor, *, fill: int | float, width: int) -> None:
    pairs = _polyline_pairs(points)
    if not pairs:
        return
    draw = ImageDraw.Draw(canvas)
    for start, end in pairs:
        draw.line((start, end), fill=fill, width=max(1, int(width)))


def _draw_poly_mask(output_hw: tuple[int, int], points: torch.Tensor, *, width: int) -> np.ndarray:
    h, w = int(output_hw[0]), int(output_hw[1])
    image = Image.new("L", (w, h), 0)
    _draw_line(image, points, fill=1, width=width)
    return np.asarray(image, dtype=bool)


def _render_tangent_and_attrs(
    lane_rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    output_hw: tuple[int, int],
    tangent_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = int(output_hw[0]), int(output_hw[1])
    tangent_sum = np.zeros((2, h, w), dtype=np.float32)
    tangent_count = np.zeros((h, w), dtype=np.float32)
    color_map = np.zeros((len(LANE_CLASSES), h, w), dtype=np.float32)
    lane_type_map = np.zeros((len(LANE_TYPES), h, w), dtype=np.float32)

    for lane_index, row in enumerate(lane_rows):
        if not _valid_mask_value(valid_mask, lane_index):
            continue
        points = _scale_points(
            _visible_lane_points(row),
            source_hw=NETWORK_HW,
            target_hw=output_hw,
        )
        color_index = int(row.get("color", -1))
        type_index = int(row.get("lane_type", -1))
        for start, end in _polyline_pairs(points):
            delta = np.asarray([end[0] - start[0], end[1] - start[1]], dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm <= 1.0e-6:
                continue
            axis = delta / norm
            # Store the axis with a deterministic bottom-to-top sign convention.
            if float(axis[1]) > 0.0:
                axis = -axis
            segment = torch.tensor([start, end], dtype=torch.float32)
            mask = _draw_poly_mask(output_hw, segment, width=tangent_width)
            tangent_sum[:, mask] += axis.reshape(2, 1)
            tangent_count[mask] += 1.0
            if 0 <= color_index < len(LANE_CLASSES):
                color_map[color_index, mask] += 1.0
            if 0 <= type_index < len(LANE_TYPES):
                lane_type_map[type_index, mask] += 1.0

    nonzero = tangent_count > 0.0
    tangent_axis = np.zeros_like(tangent_sum)
    tangent_axis[:, nonzero] = tangent_sum[:, nonzero] / tangent_count[nonzero]
    norms = np.linalg.norm(tangent_axis, axis=0)
    norm_mask = norms > 1.0e-6
    tangent_axis[:, norm_mask] /= norms[norm_mask]
    return tangent_axis, color_map, lane_type_map, tangent_count


def _binary_polyline_mask(
    rows: list[dict[str, Any]],
    valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    output_hw: tuple[int, int],
    width: int,
    polygon_fill: bool = False,
) -> np.ndarray:
    h, w = int(output_hw[0]), int(output_hw[1])
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    for row_index, row in enumerate(rows):
        if not _valid_mask_value(valid_mask, row_index):
            continue
        points = torch.as_tensor(row.get("points_xy", []), dtype=torch.float32).reshape(-1, 2)
        if points.shape[0] < (3 if polygon_fill else 2):
            continue
        points = _scale_points(points, source_hw=NETWORK_HW, target_hw=output_hw)
        coords = [(float(point[0]), float(point[1])) for point in points]
        if polygon_fill:
            draw.polygon(coords, outline=1, fill=1)
            draw.line(coords + [coords[0]], fill=1, width=max(1, int(width)))
        else:
            draw.line(coords, fill=1, width=max(1, int(width)))
    return np.asarray(image, dtype=np.float32)


def render_lane_segfirst_targets(
    lane_rows: list[dict[str, Any]],
    lane_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    stop_line_rows: list[dict[str, Any]] | None = None,
    stop_line_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...] | None = None,
    crosswalk_rows: list[dict[str, Any]] | None = None,
    crosswalk_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...] | None = None,
    source_enabled: bool = True,
    config: LaneSegFirstTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Render oracle dense lane maps for the seg-first vectorizer gate.

    This is an opt-in analysis renderer. It does not feed the training path and
    intentionally emits stop/cross masks as lane-ignore candidates rather than
    lane hard negatives.
    """
    cfg = config or LaneSegFirstTargetConfig()
    h, w = int(cfg.output_hw[0]), int(cfg.output_hw[1])
    core_image = Image.new("L", (w, h), 0)
    support_image = Image.new("L", (w, h), 0)

    if source_enabled:
        for lane_index, row in enumerate(lane_rows):
            if not _valid_mask_value(lane_valid_mask, lane_index):
                continue
            points = _scale_points(
                _visible_lane_points(row),
                source_hw=NETWORK_HW,
                target_hw=cfg.output_hw,
            )
            if points.shape[0] < 2:
                continue
            _draw_line(core_image, points, fill=1, width=cfg.centerline_core_width)
            _draw_line(support_image, points, fill=1, width=cfg.support_width)

    core = np.asarray(core_image, dtype=np.float32)
    support = np.asarray(support_image, dtype=np.float32)
    if core.any():
        distance = ndimage.distance_transform_edt(core <= 0.0).astype(np.float32)
        soft = np.exp(-0.5 * (distance / max(float(cfg.centerline_soft_sigma), 1.0e-6)) ** 2)
        soft[distance > float(cfg.centerline_soft_radius)] = 0.0
        soft = np.maximum(soft, core)
    else:
        soft = np.zeros_like(core, dtype=np.float32)

    tangent_axis, color_map, lane_type_map, tangent_count = _render_tangent_and_attrs(
        lane_rows,
        lane_valid_mask,
        output_hw=cfg.output_hw,
        tangent_width=cfg.tangent_width,
    ) if source_enabled else (
        np.zeros((2, h, w), dtype=np.float32),
        np.zeros((len(LANE_CLASSES), h, w), dtype=np.float32),
        np.zeros((len(LANE_TYPES), h, w), dtype=np.float32),
        np.zeros((h, w), dtype=np.float32),
    )

    color_den = np.maximum(color_map.sum(axis=0, keepdims=True), 1.0)
    color_map = color_map / color_den
    type_den = np.maximum(lane_type_map.sum(axis=0, keepdims=True), 1.0)
    lane_type_map = lane_type_map / type_den

    stop_rows = stop_line_rows or []
    stop_valid = stop_line_valid_mask if stop_line_valid_mask is not None else []
    cross_rows = crosswalk_rows or []
    cross_valid = crosswalk_valid_mask if crosswalk_valid_mask is not None else []
    stop_mask = _binary_polyline_mask(
        stop_rows,
        stop_valid,
        output_hw=cfg.output_hw,
        width=cfg.stopline_ignore_width,
        polygon_fill=False,
    )
    cross_mask = _binary_polyline_mask(
        cross_rows,
        cross_valid,
        output_hw=cfg.output_hw,
        width=cfg.crosswalk_boundary_width,
        polygon_fill=True,
    )
    ignore = np.clip(np.maximum(stop_mask, cross_mask) - core, 0.0, 1.0)
    negative = ((support <= 0.0) & (ignore <= 0.0)).astype(np.float32)

    return {
        "centerline_core": torch.from_numpy(core).unsqueeze(0),
        "centerline_soft": torch.from_numpy(soft.astype(np.float32)).unsqueeze(0),
        "support": torch.from_numpy(support).unsqueeze(0),
        "tangent_axis": torch.from_numpy(tangent_axis.astype(np.float32)),
        "color_map": torch.from_numpy(color_map.astype(np.float32)),
        "lane_type_map": torch.from_numpy(lane_type_map.astype(np.float32)),
        "stop_line_ignore": torch.from_numpy(stop_mask.astype(np.float32)).unsqueeze(0),
        "crosswalk_ignore": torch.from_numpy(cross_mask.astype(np.float32)).unsqueeze(0),
        "lane_ignore": torch.from_numpy(ignore.astype(np.float32)).unsqueeze(0),
        "lane_negative": torch.from_numpy(negative.astype(np.float32)).unsqueeze(0),
        "tangent_count": torch.from_numpy(tangent_count.astype(np.float32)).unsqueeze(0),
    }


def _as_numpy_channel(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"expected single-channel map, got shape {array.shape}")


def _as_numpy_chw(value: torch.Tensor | np.ndarray, *, channels: int) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim != 3 or array.shape[0] != channels:
        raise ValueError(f"expected CHW map with {channels} channels, got shape {array.shape}")
    return array.astype(np.float32)


def _row_clusters(row_mask: np.ndarray) -> list[float]:
    xs = np.nonzero(row_mask)[0]
    if xs.size == 0:
        return []
    clusters: list[np.ndarray] = []
    start = 0
    for index in range(1, int(xs.size)):
        if int(xs[index]) > int(xs[index - 1]) + 1:
            clusters.append(xs[start:index])
            start = index
    clusters.append(xs[start:])
    return [float(np.median(cluster)) for cluster in clusters if int(cluster.size) > 0]


def _component_to_polylines(
    component_mask: np.ndarray,
    *,
    row_stride: int,
    max_row_gap: int,
    max_link_dx: float,
) -> list[list[list[float]]]:
    ys, _ = np.nonzero(component_mask)
    if ys.size == 0:
        return []
    tracks: list[list[list[float]]] = []
    last_y_by_track: list[int] = []
    last_x_by_track: list[float] = []

    for y in sorted(np.unique(ys).tolist(), reverse=True):
        clusters = _row_clusters(component_mask[int(y)])
        if not clusters:
            continue
        assigned_tracks: set[int] = set()
        for x in sorted(clusters):
            best_track: int | None = None
            best_cost = float("inf")
            for track_index, (last_y, last_x) in enumerate(zip(last_y_by_track, last_x_by_track)):
                if track_index in assigned_tracks:
                    continue
                y_gap = abs(int(last_y) - int(y))
                if y_gap > int(max_row_gap):
                    continue
                dx = abs(float(x) - float(last_x))
                if dx > float(max_link_dx):
                    continue
                cost = dx + 0.1 * float(y_gap)
                if cost < best_cost:
                    best_track = track_index
                    best_cost = cost
            if best_track is None:
                tracks.append([[float(x), float(y)]])
                last_y_by_track.append(int(y))
                last_x_by_track.append(float(x))
                assigned_tracks.add(len(tracks) - 1)
            else:
                tracks[best_track].append([float(x), float(y)])
                last_y_by_track[best_track] = int(y)
                last_x_by_track[best_track] = float(x)
                assigned_tracks.add(best_track)
    stride = max(1, int(row_stride))
    downsampled = [track[::stride] if len(track) > stride else track for track in tracks]
    return [track for track in downsampled if len(track) >= 2]


def _vote_index(chw: np.ndarray, component_mask: np.ndarray, *, weights: np.ndarray | None = None) -> int:
    if chw.shape[0] == 0 or not bool(component_mask.any()):
        return 0
    values = chw[:, component_mask]
    if weights is not None:
        vote_weights = np.asarray(weights[component_mask], dtype=np.float32).reshape(1, -1)
        values = values * vote_weights
    votes = values.sum(axis=1)
    if not np.isfinite(votes).all() or float(votes.sum()) <= 0.0:
        return 0
    return int(np.argmax(votes))


def _semantic_vote_weights(
    centerline: np.ndarray,
    component: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> np.ndarray | None:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"component", "uniform", "none"}:
        return None
    if normalized_mode in {"centerline", "centerline_prob", "prob"}:
        return np.clip(centerline, 0.0, 1.0).astype(np.float32)
    if normalized_mode in {"centerline_excess", "excess"}:
        return np.clip(centerline - float(threshold), 0.0, 1.0).astype(np.float32)
    if normalized_mode in {"component_core", "core"}:
        eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool), border_value=0)
        core = eroded if bool(eroded.any()) else component
        return core.astype(np.float32)
    raise ValueError("semantic_vote_mode must be one of: component, centerline, centerline_excess, component_core")


def _polyline_length(points_xy: list[list[float]]) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _passes_bottom_y_filter(
    points_xy: list[list[float]],
    *,
    min_fraction: float,
    target_hw: tuple[int, int],
) -> bool:
    if float(min_fraction) <= 0.0:
        return True
    if not points_xy:
        return False
    bottom_y = max(float(point[1]) for point in points_xy)
    return bottom_y >= float(target_hw[0]) * float(min_fraction)


def vectorize_lane_segfirst_maps(
    maps: dict[str, torch.Tensor | np.ndarray],
    *,
    meta: dict[str, Any] | None = None,
    config: LaneSegFirstVectorizerConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or LaneSegFirstVectorizerConfig()
    centerline = _as_numpy_channel(maps["centerline_core"])
    binary = np.asarray(centerline >= float(cfg.centerline_threshold), dtype=bool)
    map_hw = (int(binary.shape[0]), int(binary.shape[1]))
    labels, component_count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.int8))
    color_map = _as_numpy_chw(maps.get("color_map", np.zeros((len(LANE_CLASSES), *binary.shape), dtype=np.float32)), channels=len(LANE_CLASSES))
    type_map = _as_numpy_chw(maps.get("lane_type_map", np.zeros((len(LANE_TYPES), *binary.shape), dtype=np.float32)), channels=len(LANE_TYPES))
    transform = transform_from_meta(meta) if meta is not None else None

    predictions: list[dict[str, Any]] = []
    component_ids: list[int] = []
    component_sizes = ndimage.sum(
        binary.astype(np.float32),
        labels,
        index=list(range(1, int(component_count) + 1)),
    )
    for component_id, component_size in zip(range(1, int(component_count) + 1), component_sizes):
        component_pixels = int(component_size)
        if component_pixels < int(cfg.min_component_pixels):
            continue
        if cfg.max_component_pixels is not None and component_pixels > int(cfg.max_component_pixels):
            continue
        component_ids.append(component_id)
    if cfg.max_components is not None and len(component_ids) > int(cfg.max_components):
        sizes = ndimage.sum(binary.astype(np.float32), labels, index=component_ids)
        ranked = sorted(zip(component_ids, sizes), key=lambda item: float(item[1]), reverse=True)
        component_ids = [component_id for component_id, _ in ranked[: int(cfg.max_components)]]
    for component_id in component_ids:
        if cfg.max_predictions is not None and len(predictions) >= int(cfg.max_predictions):
            break
        component = labels == component_id
        map_polylines = _component_to_polylines(
            component,
            row_stride=max(1, int(cfg.row_stride)),
            max_row_gap=max(1, int(cfg.max_row_gap)),
            max_link_dx=float(cfg.max_link_dx),
        )
        for map_points in map_polylines:
            if cfg.max_predictions is not None and len(predictions) >= int(cfg.max_predictions):
                break
            if len(map_points) < int(cfg.min_polyline_points):
                continue
            if transform is not None:
                network_points = _scale_points_list(
                    map_points,
                    source_hw=map_hw,
                    target_hw=transform.network_hw,
                )
            else:
                network_points = map_points
            network_points = clip_points(network_points, map_hw if transform is None else transform.network_hw)
            if unique_point_count(network_points) < 2:
                continue
            output_points = inverse_transform_points(network_points, transform) if transform is not None else network_points
            if unique_point_count(output_points) < 2:
                continue
            if float(cfg.min_polyline_length_px) > 0.0 and _polyline_length(output_points) < float(cfg.min_polyline_length_px):
                continue
            output_hw = transform.raw_hw if transform is not None else map_hw
            if not _passes_bottom_y_filter(
                output_points,
                min_fraction=float(cfg.min_polyline_bottom_y_fraction),
                target_hw=output_hw,
            ):
                continue
            semantic_weights = _semantic_vote_weights(
                centerline,
                component,
                mode=cfg.semantic_vote_mode,
                threshold=float(cfg.centerline_threshold),
            )
            color_index = _vote_index(color_map, component, weights=semantic_weights)
            type_index = _vote_index(type_map, component, weights=semantic_weights)
            predictions.append(
                {
                    "score": 1.0,
                    "class_name": LANE_CLASSES[color_index],
                    "lane_type": LANE_TYPES[type_index],
                    "points_xy": [[float(x), float(y)] for x, y in output_points],
                }
            )
    predictions.sort(key=lambda item: (max(point[1] for point in item["points_xy"]), -np.mean([point[0] for point in item["points_xy"]])), reverse=True)
    return predictions


def lane_segfirst_prediction_maps(
    predictions: dict[str, torch.Tensor],
    *,
    batch_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Convert seg-first head logits into the map schema consumed by the vectorizer."""

    centerline_logits = predictions["lane_seg_centerline_logits"][batch_index].detach().cpu()
    support_logits = predictions["lane_seg_support_logits"][batch_index].detach().cpu()
    tangent_axis = predictions["lane_seg_tangent_axis"][batch_index].detach().cpu()
    color_logits = predictions["lane_seg_color_logits"][batch_index].detach().cpu()
    type_logits = predictions["lane_seg_type_logits"][batch_index].detach().cpu()
    tangent_norm = torch.linalg.norm(tangent_axis, dim=0, keepdim=True).clamp(min=1.0e-6)
    tangent_axis = tangent_axis / tangent_norm
    centerline_prob = centerline_logits.sigmoid()
    support_prob = support_logits.sigmoid()
    h, w = int(centerline_prob.shape[-2]), int(centerline_prob.shape[-1])
    return {
        "centerline_core": centerline_prob,
        "centerline_soft": centerline_prob,
        "support": support_prob,
        "tangent_axis": tangent_axis,
        "color_map": color_logits.softmax(dim=0),
        "lane_type_map": type_logits.softmax(dim=0),
        "stop_line_ignore": torch.zeros((1, h, w), dtype=torch.float32),
        "crosswalk_ignore": torch.zeros((1, h, w), dtype=torch.float32),
        "lane_ignore": torch.zeros((1, h, w), dtype=torch.float32),
        "lane_negative": (support_prob <= 0.5).to(dtype=torch.float32),
        "tangent_count": support_prob,
    }


def summarize_lane_segfirst_target_maps(maps: dict[str, torch.Tensor | np.ndarray]) -> dict[str, float]:
    h, w = _as_numpy_channel(maps["centerline_core"]).shape
    total = float(max(1, h * w))
    summary: dict[str, float] = {"pixels_total": total}
    for key in ("centerline_core", "centerline_soft", "support", "stop_line_ignore", "crosswalk_ignore", "lane_ignore", "lane_negative"):
        if key not in maps:
            continue
        array = _as_numpy_channel(maps[key])
        if key == "centerline_soft":
            count = float((array > 0.05).sum())
        else:
            count = float((array > 0.5).sum())
        summary[f"{key}_pixels"] = count
        summary[f"{key}_frequency"] = count / total
    return summary


def lane_segfirst_overlap_counts(maps: dict[str, torch.Tensor | np.ndarray]) -> dict[str, int]:
    core = _as_numpy_channel(maps["centerline_core"]) > 0.5
    support = _as_numpy_channel(maps["support"]) > 0.5
    stop = _as_numpy_channel(maps["stop_line_ignore"]) > 0.5
    cross = _as_numpy_channel(maps["crosswalk_ignore"]) > 0.5
    ignore = _as_numpy_channel(maps["lane_ignore"]) > 0.5
    negative = _as_numpy_channel(maps["lane_negative"]) > 0.5
    return {
        "centerline_x_stop": int(np.logical_and(core, stop).sum()),
        "centerline_x_crosswalk": int(np.logical_and(core, cross).sum()),
        "support_x_stop": int(np.logical_and(support, stop).sum()),
        "support_x_crosswalk": int(np.logical_and(support, cross).sum()),
        "ignore_x_stop_or_crosswalk": int(np.logical_and(ignore, np.logical_or(stop, cross)).sum()),
        "negative_x_stop_or_crosswalk": int(np.logical_and(negative, np.logical_or(stop, cross)).sum()),
    }


def lane_adjacent_separations(
    lane_rows: list[dict[str, Any]],
    lane_valid_mask: torch.BoolTensor | list[bool] | tuple[bool, ...],
    *,
    output_hw: tuple[int, int] = ROADMARK_DENSE_OUTPUT_HW,
    sample_rows: int = 64,
) -> list[float]:
    y_values = np.linspace(0.0, float(output_hw[0] - 1), max(2, int(sample_rows)), dtype=np.float32)
    separations: list[float] = []
    polylines = []
    for lane_index, row in enumerate(lane_rows):
        if not _valid_mask_value(lane_valid_mask, lane_index):
            continue
        points = _scale_points(
            _visible_lane_points(row),
            source_hw=NETWORK_HW,
            target_hw=output_hw,
        ).detach().cpu().numpy().astype(np.float32)
        if points.shape[0] >= 2:
            polylines.append(points)
    for y in y_values:
        xs_at_y: list[float] = []
        for points in polylines:
            for p0, p1 in zip(points[:-1], points[1:]):
                y0, y1 = float(p0[1]), float(p1[1])
                if y < min(y0, y1) - 1.0e-6 or y > max(y0, y1) + 1.0e-6:
                    continue
                if abs(y1 - y0) <= 1.0e-6:
                    xs_at_y.append(float((p0[0] + p1[0]) * 0.5))
                    break
                ratio = (float(y) - y0) / (y1 - y0)
                xs_at_y.append(float(p0[0] + ratio * (p1[0] - p0[0])))
                break
        xs_at_y.sort()
        separations.extend(float(b - a) for a, b in zip(xs_at_y[:-1], xs_at_y[1:]) if b >= a)
    return separations


def strict_lane_schema_metrics(
    predictions: list[dict[str, Any]],
    gt_lanes: list[dict[str, Any]],
    *,
    target_count: int = 16,
    match_threshold: float = 40.0,
) -> dict[str, float | int]:
    cost = np.zeros((len(predictions), len(gt_lanes)), dtype=np.float32)
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(gt_lanes):
            if pred.get("class_name") != gt.get("class_name") or pred.get("lane_type") != gt.get("lane_type"):
                cost[pred_index, gt_index] = float("inf")
            else:
                cost[pred_index, gt_index] = _mean_point_distance(pred["points_xy"], gt["points_xy"], target_count)
    matches = _hungarian_from_cost(cost, max_cost=float(match_threshold))
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(gt_lanes) - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def lane_segfirst_upper_bound_metrics(
    prediction_samples: list[dict[str, Any]],
    gt_samples: list[dict[str, Any]],
    *,
    match_threshold: float = 40.0,
) -> dict[str, Any]:
    geometry = _lane_family_metrics(
        prediction_samples,
        gt_samples,
        field_name="lanes",
        target_count=16,
        match_threshold=float(match_threshold),
    )
    strict_totals = {"tp": 0, "fp": 0, "fn": 0}
    for pred_sample, gt_sample in zip(prediction_samples, gt_samples):
        sample_strict = strict_lane_schema_metrics(
            list(pred_sample.get("lanes", [])),
            list(gt_sample.get("lanes", [])),
            match_threshold=float(match_threshold),
        )
        strict_totals["tp"] += int(sample_strict["tp"])
        strict_totals["fp"] += int(sample_strict["fp"])
        strict_totals["fn"] += int(sample_strict["fn"])
    tp = strict_totals["tp"]
    fp = strict_totals["fp"]
    fn = strict_totals["fn"]
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    full_schema = {**strict_totals, "precision": precision, "recall": recall, "f1": f1}
    return {"geometry": geometry, "full_schema": full_schema}


__all__ = [
    "LaneSegFirstTargetConfig",
    "LaneSegFirstVectorizerConfig",
    "lane_adjacent_separations",
    "lane_segfirst_prediction_maps",
    "lane_segfirst_overlap_counts",
    "lane_segfirst_upper_bound_metrics",
    "render_lane_segfirst_targets",
    "strict_lane_schema_metrics",
    "summarize_lane_segfirst_target_maps",
    "vectorize_lane_segfirst_maps",
]
