"""Network-coordinate detections and centerline maps to raw-image observations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d
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


def decode_roadmark_points(
    logits: torch.Tensor,
    meta: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
    min_points: int = 6,
) -> list[list[dict[str, Any]]]:
    """Return each lane/stop line as raw-image center points with class and score."""
    if logits.ndim != 4 or logits.shape[1] != len(ROADMARK_CLASSES) or logits.shape[0] != len(meta):
        raise ValueError("roadmark logits must be [B,3,H/4,W/4] and align with meta")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite roadmark output")
    probability = logits.detach().float().sigmoid().cpu().numpy()
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
            traces = _trace_ridges(
                image_probability[class_id],
                along_rows=(class_name != "stop_line"),
                threshold=threshold,
                max_link=3.0,
                max_gap=3,
                min_points=min_points,
            )
            for grid_points, score in traces:
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
                lines.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "points_xy": raw_points,
                })
        result.append(lines)
    return result
