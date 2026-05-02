from __future__ import annotations

from typing import Any

import numpy as np


def _contiguous_float32(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float32)


def _as_numpy_points(points_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return _contiguous_float32(points)


def _dedupe_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return _contiguous_float32(points)
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    return _contiguous_float32(points[np.sort(unique_indices)])


def _stable_endpoint_order(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return _contiguous_float32(points)
    first = points[0]
    second = points[1]
    if float(first[0]) > float(second[0]) or (
        np.isclose(float(first[0]), float(second[0])) and float(first[1]) > float(second[1])
    ):
        return _contiguous_float32(points[::-1])
    return _contiguous_float32(points)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    if points.shape[0] == 1 or float(np.linalg.norm(centered)) <= 1.0e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float32, copy=False)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    axis = axis / norm
    if abs(float(axis[0])) >= abs(float(axis[1])):
        if float(axis[0]) < 0.0:
            axis = -axis
    elif float(axis[1]) < 0.0:
        axis = -axis
    return _contiguous_float32(axis)


def _resample_open_polyline(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = _contiguous_float32(points).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((target_count, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, target_count, axis=0)

    deltas = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float32)))
    total_length = float(cumulative[-1])
    if total_length <= 1.0e-6:
        return np.repeat(points[:1], target_count, axis=0)

    targets = np.linspace(0.0, total_length, target_count, dtype=np.float32)
    resampled: list[np.ndarray] = []
    for target in targets:
        upper = int(np.searchsorted(cumulative, target, side="left"))
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        left_distance = float(cumulative[lower])
        right_distance = float(cumulative[upper])
        interval = right_distance - left_distance
        if interval <= 1.0e-6:
            resampled.append(points[lower])
            continue
        ratio = float((target - left_distance) / interval)
        resampled.append(points[lower] + ratio * (points[upper] - points[lower]))
    return _contiguous_float32(np.stack(resampled, axis=0))


def _resample_closed_polygon(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = _contiguous_float32(points).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((target_count, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, target_count, axis=0)

    next_points = np.roll(points, shift=-1, axis=0)
    deltas = next_points - points
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float32)))
    total_length = float(cumulative[-1])
    if total_length <= 1.0e-6:
        return np.repeat(points[:1], target_count, axis=0)

    targets = np.linspace(0.0, total_length, target_count + 1, dtype=np.float32)[:-1]
    resampled: list[np.ndarray] = []
    for target in targets:
        upper = int(np.searchsorted(cumulative, target, side="right"))
        upper = min(max(upper, 1), points.shape[0])
        lower = upper - 1
        start = points[lower]
        end = points[upper % points.shape[0]]
        left_distance = float(cumulative[lower])
        right_distance = float(cumulative[upper])
        interval = right_distance - left_distance
        if interval <= 1.0e-6:
            resampled.append(start)
            continue
        ratio = float((target - left_distance) / interval)
        resampled.append(start + ratio * (end - start))
    return _contiguous_float32(np.stack(resampled, axis=0))


def _densify_closed_polygon(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = _contiguous_float32(points).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((target_count, 2), dtype=np.float32)
    if points.shape[0] >= target_count:
        return _resample_closed_polygon(points, target_count)
    if points.shape[0] <= 2:
        return _resample_open_polyline(points, target_count)

    next_points = np.roll(points, shift=-1, axis=0)
    edge_lengths = np.linalg.norm(next_points - points, axis=1)
    total_length = float(edge_lengths.sum())
    if total_length <= 1.0e-6:
        return np.repeat(points[:1], target_count, axis=0)

    extra_points = target_count - points.shape[0]
    raw_allocation = edge_lengths / total_length * float(extra_points)
    extra_per_edge = np.floor(raw_allocation).astype(np.int64)
    remainder = int(extra_points - int(extra_per_edge.sum()))
    if remainder > 0:
        fractional = raw_allocation - extra_per_edge.astype(np.float32)
        order = np.argsort(-fractional, kind="stable")
        extra_per_edge[order[:remainder]] += 1

    densified: list[np.ndarray] = []
    for edge_index, start in enumerate(points):
        end = next_points[edge_index]
        densified.append(start)
        inserts = int(extra_per_edge[edge_index])
        if inserts <= 0:
            continue
        ratios = (np.arange(inserts, dtype=np.float32) + 1.0) / float(inserts + 1)
        inserted = start[None, :] + ratios[:, None] * (end - start)[None, :]
        densified.extend(inserted)
    return _contiguous_float32(np.stack(densified[:target_count], axis=0))


def canonicalize_stop_line_points(points_xy: Any) -> np.ndarray:
    points = _dedupe_points(_as_numpy_points(points_xy))
    if points.shape[0] <= 1:
        return _contiguous_float32(points)
    if points.shape[0] == 2:
        return _stable_endpoint_order(points)
    center = points.mean(axis=0)
    axis = _principal_axis(points)
    projections = (points - center) @ axis
    start = center + axis * float(projections.min())
    end = center + axis * float(projections.max())
    return _stable_endpoint_order(np.stack((start, end), axis=0))


def estimate_stop_line_width(points_xy: Any) -> tuple[float, bool]:
    points = _dedupe_points(_as_numpy_points(points_xy))
    if points.shape[0] < 3:
        return -1.0, False
    center = points.mean(axis=0)
    axis = _principal_axis(points)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    offsets = (points - center) @ normal
    width = float(offsets.max() - offsets.min())
    if width <= 1.0e-6:
        return -1.0, False
    return width, True


def sample_stop_line_centerline(points_xy: Any, target_count: int) -> np.ndarray:
    return _resample_open_polyline(canonicalize_stop_line_points(points_xy), target_count)


def canonicalize_crosswalk_points(points_xy: Any) -> np.ndarray:
    points = _dedupe_points(_as_numpy_points(points_xy))
    if points.shape[0] <= 2:
        return _contiguous_float32(points)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles, kind="stable")[::-1]
    ordered = points[order]
    top_left_score = ordered[:, 1] * 10000.0 + ordered[:, 0]
    start = int(np.argmin(top_left_score))
    return _contiguous_float32(np.roll(ordered, -start, axis=0))


def sample_crosswalk_contour(points_xy: Any, target_count: int) -> np.ndarray:
    points = canonicalize_crosswalk_points(points_xy)
    if points.shape[0] < 3:
        return _resample_open_polyline(points, target_count)
    if points.shape[0] < target_count:
        return _densify_closed_polygon(points, target_count)
    if points.shape[0] == target_count:
        return points
    return _resample_closed_polygon(points, target_count)


__all__ = [
    "canonicalize_crosswalk_points",
    "canonicalize_stop_line_points",
    "estimate_stop_line_width",
    "sample_crosswalk_contour",
    "sample_stop_line_centerline",
]
