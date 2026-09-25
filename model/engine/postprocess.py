"""Network-coordinate detections and centerline maps to raw-image observations."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
import torch

from common.schema import ROADMARK_CLASSES, SIGNAL_CLASSES
from model.data.geometry import inverse_transform_box_xyxy, transform_from_meta


def decode_focused_detections(
    decoded: torch.Tensor,
    meta: list[dict[str, Any]],
    *,
    conf_threshold: float = 0.25,
) -> list[list[dict[str, Any]]]:
    """Decode official end-to-end Detect [B,N,6] xyxy/conf/class output."""
    if decoded.ndim != 3 or decoded.shape[-1] != 6 or decoded.shape[0] != len(meta):
        raise ValueError("detections must be [B,N,6] and align with meta")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("nonfinite detection output")
    array = decoded.detach().float().cpu().numpy()
    result: list[list[dict[str, Any]]] = []
    for image_index, rows in enumerate(array):
        transform = transform_from_meta(meta[image_index])
        flipped = bool(meta[image_index].get("flipped", False))
        detections = []
        for row in rows:
            score = float(row[4])
            if score < conf_threshold:
                continue
            class_id = int(row[5])
            if class_id < 0 or class_id >= len(SIGNAL_CLASSES):
                raise ValueError(f"invalid signal class ID: {class_id}")
            box = row[:4].astype(np.float64)
            if flipped:
                x1, x2 = float(box[0]), float(box[2])
                box[0] = transform.network_hw[1] - 1.0 - x2
                box[2] = transform.network_hw[1] - 1.0 - x1
            raw_box = inverse_transform_box_xyxy(box, transform)
            if raw_box is None:
                continue
            detections.append({
                "class_id": class_id,
                "class_name": SIGNAL_CLASSES[class_id],
                "score": score,
                "bbox_xyxy": raw_box,
            })
        result.append(detections)
    return result


def _trace_ridges(
    probability: np.ndarray,
    *,
    along_rows: bool,
    threshold: float,
    max_link: float,
    max_gap: int,
    min_points: int,
) -> list[tuple[list[tuple[float, float]], float]]:
    """Link transverse ridge maxima, not whole connected components.

    A connected mask can contain several distinct lanes or stop lines; tracing
    one peak per line and scan step keeps those observations separate.
    """
    axis_size = probability.shape[0] if along_rows else probability.shape[1]
    positions = range(axis_size - 1, -1, -1) if along_rows else range(axis_size)
    transverse_axis = 1 if along_rows else 0
    neighborhood_max = maximum_filter1d(probability, size=5, axis=transverse_axis, mode="constant", cval=-1.0)
    right = np.full_like(probability, -1.0)
    if along_rows:
        right[:, :-1] = probability[:, 1:]
    else:
        right[:-1, :] = probability[1:, :]
    # The strict right comparison chooses one point from a flat ridge plateau.
    ridge_mask = (probability > threshold) & (probability >= neighborhood_max) & (probability > right)
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    for axis in positions:
        profile = probability[axis, :] if along_rows else probability[:, axis]
        peaks = np.flatnonzero(ridge_mask[axis, :] if along_rows else ridge_mask[:, axis])
        candidates = [(float(peak), float(profile[peak])) for peak in peaks]
        if active and peaks.size:
            last_transverse = np.array([track["points"][-1][0] for track in active], dtype=np.float64)
            last_axis = np.array([track["points"][-1][1] for track in active], dtype=np.float64)
            previous_transverse = np.array(
                [track["points"][-2][0] if len(track["points"]) > 1 else track["points"][-1][0]
                 for track in active], dtype=np.float64,
            )
            previous_axis = np.array(
                [track["points"][-2][1] if len(track["points"]) > 1 else track["points"][-1][1]
                 for track in active], dtype=np.float64,
            )
            axis_delta = last_axis - previous_axis
            slope = np.divide(
                last_transverse - previous_transverse,
                axis_delta,
                out=np.zeros_like(last_transverse),
                where=axis_delta != 0,
            )
            predicted = last_transverse + slope * (float(axis) - last_axis)
            distance = np.abs(predicted[:, None] - peaks[None, :])
            gap = np.array([track["gap"] for track in active], dtype=np.int32)
            track_indices, peak_indices = np.nonzero(distance <= max_link * (gap[:, None] + 1))
            edge_order = np.lexsort((peak_indices, track_indices, distance[track_indices, peak_indices]))
        else:
            track_indices = peak_indices = edge_order = ()
        used_tracks: set[int] = set()
        used_peaks: set[int] = set()
        for edge_index in edge_order:
            track_index = int(track_indices[edge_index])
            peak_index = int(peak_indices[edge_index])
            if track_index in used_tracks or peak_index in used_peaks:
                continue
            transverse, score = candidates[peak_index]
            active[track_index]["points"].append((transverse, float(axis)))
            active[track_index]["scores"].append(score)
            active[track_index]["gap"] = 0
            used_tracks.add(track_index)
            used_peaks.add(peak_index)
        remaining = []
        for track_index, track in enumerate(active):
            if track_index not in used_tracks:
                track["gap"] += 1
            if track["gap"] > max_gap:
                finished.append(track)
            else:
                remaining.append(track)
        active = remaining
        for peak_index, (transverse, score) in enumerate(candidates):
            if peak_index not in used_peaks:
                active.append({"points": [(transverse, float(axis))], "scores": [score], "gap": 0})
    finished.extend(active)
    traces = []
    for track in finished:
        points = track["points"]
        if len(points) < min_points or abs(points[-1][1] - points[0][1]) < min_points - 1:
            continue
        if along_rows:
            grid_points = [(transverse, axis) for transverse, axis in points]
        else:
            grid_points = [(axis, transverse) for transverse, axis in points]
        traces.append((grid_points, float(np.mean(track["scores"]))))
    return traces


def _subpixel_peak(probability: np.ndarray, x: float, y: float,
                   *, along_rows: bool) -> tuple[float, float]:
    """Fit the local ridge maximum without changing its trace or support."""
    column, row = int(x), int(y)
    if along_rows:
        if column <= 0 or column + 1 >= probability.shape[1]:
            return x, y
        before, center, after = probability[row, column - 1:column + 2]
    else:
        if row <= 0 or row + 1 >= probability.shape[0]:
            return x, y
        before, center, after = probability[row - 1:row + 2, column]
    curvature = float(before - 2 * center + after)
    offset = float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5)) if curvature < 0 else 0.0
    return (x + offset, y) if along_rows else (x, y + offset)


def _smooth_trace(points: list[tuple[float, float]], *, along_rows: bool) -> list[tuple[float, float]]:
    """Smooth only contiguous transverse coordinates, preserving segment ends."""
    transverse_axis = 0 if along_rows else 1
    longitudinal_axis = 1 - transverse_axis
    coordinates = np.asarray(points, dtype=np.float64)
    longitudinal = coordinates[:, longitudinal_axis]
    boundaries = [0, *(np.flatnonzero(np.abs(np.diff(longitudinal)) != 1) + 1), len(points)]
    for start, stop in zip(boundaries, boundaries[1:]):
        if stop - start < 5:
            continue
        transverse = coordinates[start:stop, transverse_axis]
        fitted = savgol_filter(transverse, window_length=5, polyorder=2, mode="interp")
        coordinates[start + 1:stop - 1, transverse_axis] = fitted[1:-1]
    return [tuple(point) for point in coordinates]


def _merge_traces(
    traces: list[tuple[list[tuple[float, float]], float, bool]],
    *,
    overlap_cells: float = 1.5,
    join_cells: float = 3.0,
) -> list[tuple[list[tuple[float, float]], float, bool]]:
    """Merge the two scans' traces of one class into one set of lines.

    A curved line crossing the diagonal is traced partly by each scan, and a
    near-diagonal line can survive in both. Drop traces lying mostly on a longer
    kept trace, then join traces whose ends meet into one polyline.
    """
    ordered = sorted(traces, key=lambda item: -len(item[0]))
    kept: list[tuple[list[tuple[float, float]], float, bool]] = []
    for points, score, along_rows in ordered:
        candidate = np.asarray(points, dtype=np.float64)
        if kept:
            reference = np.concatenate([np.asarray(item[0], dtype=np.float64) for item in kept])
            distance = np.sqrt(((candidate[:, None, :] - reference[None, :, :]) ** 2).sum(-1)).min(1)
            if float(np.mean(distance <= overlap_cells)) > 0.5:
                continue
        kept.append((points, score, along_rows))
    merged = True
    while merged and len(kept) > 1:
        merged = False
        for first in range(len(kept)):
            for second in range(first + 1, len(kept)):
                a, score_a, rows_a = kept[first]
                b, score_b, _ = kept[second]
                best = None
                for a_points in (a, a[::-1]):
                    for b_points in (b, b[::-1]):
                        gap = math.dist(a_points[-1], b_points[0])
                        if gap <= join_cells and (best is None or gap < best[0]):
                            best = (gap, list(a_points) + list(b_points))
                if best is None:
                    continue
                score = (score_a * len(a) + score_b * len(b)) / (len(a) + len(b))
                kept[first] = (best[1], score, rows_a)
                del kept[second]
                merged = True
                break
            if merged:
                break
    return kept


def _per_class(value: float | Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.repeat(values, len(ROADMARK_CLASSES))
    if values.shape != (len(ROADMARK_CLASSES),) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be one value or one per white, yellow and stop line")
    return values


def decode_roadmark_points(
    logits: torch.Tensor,
    meta: list[dict[str, Any]],
    *,
    threshold: float | Sequence[float] = 0.5,
    min_points: int = 6,
    localization: str = "grid",
    orientation: str = "fixed",
    max_gap: int | Sequence[int] = 3,
    min_score: float | Sequence[float] = 0.0,
    min_length_px: float | Sequence[float] = 0.0,
) -> list[list[dict[str, Any]]]:
    """Return each lane/stop line as raw-image center points with class and score.

    ``orientation="fixed"`` scans lanes row by row and stop lines column by
    column. A scan cannot represent a line running along it: a lane near
    horizontal collapses to a plateau of one peak per row. ``"auto"`` scans
    every class both ways and keeps each trace only from the scan transverse to
    its chord, so every line is traced by the direction that can represent it.
    ``min_score`` (mean ridge probability) and ``min_length_px`` (raw-image arc
    length) reject whole traced lines, per class.
    """
    if logits.ndim != 4 or logits.shape[1] != len(ROADMARK_CLASSES) or logits.shape[0] != len(meta):
        raise ValueError("roadmark logits must be [B,3,H/4,W/4] and align with meta")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite roadmark output")
    return decode_roadmark_probabilities(
        logits.detach().float().sigmoid().cpu().numpy(), meta, threshold=threshold,
        min_points=min_points, localization=localization, orientation=orientation,
        max_gap=max_gap, min_score=min_score, min_length_px=min_length_px)


def decode_roadmark_probabilities(
    probability: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    threshold: float | Sequence[float] = 0.5,
    min_points: int = 6,
    localization: str = "grid",
    orientation: str = "fixed",
    max_gap: int | Sequence[int] = 3,
    min_score: float | Sequence[float] = 0.0,
    min_length_px: float | Sequence[float] = 0.0,
) -> list[list[dict[str, Any]]]:
    """``decode_roadmark_points`` on float32 probabilities [B,3,H/4,W/4].

    Deployments that compute the sigmoid themselves decode through this entry,
    so reference results do not depend on one library's sigmoid rounding.
    """
    probability = np.array(probability, dtype=np.float32, copy=True)
    if probability.ndim != 4 or probability.shape[1] != len(ROADMARK_CLASSES) \
            or probability.shape[0] != len(meta):
        raise ValueError("roadmark probabilities must be [B,3,H/4,W/4] and align with meta")
    if not np.isfinite(probability).all():
        raise ValueError("nonfinite roadmark probability")
    if localization not in {"grid", "subpixel", "smooth"}:
        raise ValueError("localization must be grid, subpixel or smooth")
    thresholds = np.asarray(threshold, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds, len(ROADMARK_CLASSES))
    if thresholds.shape != (len(ROADMARK_CLASSES),) or not np.isfinite(thresholds).all() \
            or (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("roadmark thresholds must be probabilities for white, yellow and stop lines")
    if orientation not in {"fixed", "auto"}:
        raise ValueError("orientation must be fixed or auto")
    gaps = _per_class(max_gap, "max_gap")
    min_scores = _per_class(min_score, "min_score")
    min_lengths = _per_class(min_length_px, "min_length_px")
    if (gaps < 0).any() or (gaps != np.round(gaps)).any():
        raise ValueError("max_gap must be non-negative integers")
    if (min_scores < 0).any() or (min_scores > 1).any() or (min_lengths < 0).any():
        raise ValueError("min_score must be a probability and min_length_px non-negative")
    result: list[list[dict[str, Any]]] = []
    for image_index, image_probability in enumerate(probability):
        transform = transform_from_meta(meta[image_index])
        stride_y = transform.network_hw[0] / image_probability.shape[1]
        stride_x = transform.network_hw[1] / image_probability.shape[2]
        flipped = bool(meta[image_index].get("flipped", False))
        x_centers = (np.arange(image_probability.shape[2]) + 0.5) * stride_x
        y_centers = (np.arange(image_probability.shape[1]) + 0.5) * stride_y
        if flipped:
            x_centers = transform.network_hw[1] - 1.0 - x_centers
        content = ((y_centers[:, None] >= transform.pad_top)
                   & (y_centers[:, None] < transform.pad_top + transform.resized_hw[0])
                   & (x_centers[None, :] >= transform.pad_left)
                   & (x_centers[None, :] < transform.pad_left + transform.resized_hw[1]))
        image_probability *= content[None, :, :]
        lines = []
        for class_id, class_name in enumerate(ROADMARK_CLASSES):
            directions = ((True, False) if orientation == "auto"
                          else (class_name != "stop_line",))
            traces = []
            for along_rows in directions:
                for grid_points, score in _trace_ridges(
                    image_probability[class_id],
                    along_rows=along_rows,
                    threshold=float(thresholds[class_id]),
                    max_link=3.0,
                    max_gap=int(gaps[class_id]),
                    min_points=min_points,
                ):
                    if orientation == "auto":
                        dx = abs(grid_points[-1][0] - grid_points[0][0])
                        dy = abs(grid_points[-1][1] - grid_points[0][1])
                        # Diagonal chords belong to the row scan alone.
                        if (dy >= dx) != along_rows:
                            continue
                    traces.append((grid_points, score, along_rows))
            if orientation == "auto":
                traces = _merge_traces(traces)
            for grid_points, score, along_rows in traces:
                if score < min_scores[class_id]:
                    continue
                if localization in {"subpixel", "smooth"}:
                    grid_points = [_subpixel_peak(image_probability[class_id], x, y,
                                                  along_rows=along_rows)
                                   for x, y in grid_points]
                if localization == "smooth":
                    grid_points = _smooth_trace(grid_points, along_rows=along_rows)
                raw_points = []
                for grid_x, grid_y in grid_points:
                    x = (grid_x + 0.5) * stride_x
                    y = (grid_y + 0.5) * stride_y
                    if flipped:
                        x = transform.network_hw[1] - 1.0 - x
                    if not (
                        transform.pad_left <= x < transform.pad_left + transform.resized_hw[1]
                        and transform.pad_top <= y < transform.pad_top + transform.resized_hw[0]
                    ):
                        continue
                    raw_points.append([
                        max(0.0, min((x - transform.pad_left) / transform.scale, transform.raw_hw[1] - 1.0)),
                        max(0.0, min((y - transform.pad_top) / transform.scale, transform.raw_hw[0] - 1.0)),
                    ])
                if len(raw_points) < min_points:
                    continue
                if min_lengths[class_id] > 0 and float(np.linalg.norm(
                        np.diff(np.asarray(raw_points), axis=0), axis=1).sum()) < min_lengths[class_id]:
                    continue
                lines.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "points_xy": raw_points,
                })
        result.append(lines)
    return result
