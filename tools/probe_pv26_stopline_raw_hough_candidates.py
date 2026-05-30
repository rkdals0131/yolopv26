from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import site
import sys
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.data.transform import transform_from_meta, transform_points
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import STOP_LINE_POINT_COUNT, _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.probe_pv26_lane_feature_roi_repair import DEFAULT_CHECKPOINT
from tools.probe_pv26_stopline_angle_mask_extent import (
    _as_2d_array,
    _detach_to_cpu,
    _row_from_metrics,
    _sample_tensor,
    _write_csv,
)
from tools.probe_pv26_stopline_candidate_pool import (
    _best_task_threshold,
    _bilinear_sample_gray,
    _fit_raw_patch_mlp,
    _load_raw_gray_image,
    _nearest_gt,
    _predict_raw_patch_mlp,
    _standardize_from_train,
    _write_candidate_features_csv,
)
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SCORE_KEY = "raw_hough_mlp_score"
HOUGH_FEATURES = (
    "raw_hough_score",
    "raw_hough_length",
    "raw_hough_length_norm",
    "raw_hough_abs_cos",
    "raw_hough_abs_sin",
    "raw_hough_center_x_norm",
    "raw_hough_center_y_norm",
    "raw_hough_edge_mean",
    "raw_hough_edge_max",
    "raw_hough_mask_mean",
    "raw_hough_mask_max",
    "raw_hough_center_mean",
    "raw_hough_center_max",
    "raw_hough_selector_mean",
    "raw_hough_selector_max",
    "raw_hough_proposal_mean",
    "raw_hough_proposal_max",
    "raw_hough_endpoint_proposal_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stop-line candidates from raw-image line segments, train a small "
            "candidate verifier on canonical train batches, and replay it on validation."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--train-record-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default="", help="Override dataset root for detached worktrees.")
    parser.add_argument(
        "--candidate-generator",
        choices=("hough", "lsd", "support_pca"),
        default="hough",
        help="Raw-image line segment generator. Defaults to the original Canny/Hough path.",
    )
    parser.add_argument("--hough-max-candidates", type=int, default=16)
    parser.add_argument("--hough-top-k", type=int, default=8)
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--verifier-epochs", type=int, default=60)
    parser.add_argument("--verifier-lr", type=float, default=0.001)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_device(requested: str, scenario_device: str) -> str:
    value = str(requested or "auto").strip().lower()
    candidate = str(scenario_device) if value == "auto" else str(requested)
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        print("[stopline_raw_hough] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return candidate


def _scenario_with_dataset_root(scenario: Any, dataset_root: str) -> Any:
    dataset_root = str(dataset_root or "").strip()
    if not dataset_root:
        return scenario
    dataset = train_config_api.DatasetConfig(root=Path(dataset_root).expanduser().resolve())
    return replace(scenario, dataset=dataset)


def _train_config_with_runtime_defaults(train_config: Any, *, device: str, max_val_batches: int) -> Any:
    return replace(
        train_config,
        device=str(device),
        val_batches=int(max_val_batches),
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
        lane_segfirst_track_mode="row_scan_tangent",
        stop_line_projection_comp_enabled=True,
        stop_line_projection_comp_min_gap=4.0,
        stop_line_projection_comp_topk=50,
        stop_line_projection_comp_union_min_score=0.80,
        stop_line_projection_comp_single_min_score=0.90,
        stop_line_projection_comp_angle_threshold_deg=16.0,
        stop_line_projection_comp_offset_threshold_px=48.0,
        stop_line_projection_comp_min_cluster_count=2,
        stop_line_projection_comp_projection_gap_px=320.0,
        stop_line_projection_comp_max_predictions=2,
        stop_line_projection_comp_second_min_fragment_count=5,
        crosswalk_polygon_mode="hull",
    )


def _slice_raw_batch_sample(raw_batch: dict[str, Any], sample_index: int) -> dict[str, Any]:
    keys = ("det_targets", "tl_attr_targets", "lane_targets", "source_mask", "valid_mask", "meta")
    return {key: [raw_batch[key][sample_index]] for key in keys}


def _line_points(start: np.ndarray, end: np.ndarray, *, count: int = STOP_LINE_POINT_COUNT) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32).reshape(2)
    end = np.asarray(end, dtype=np.float32).reshape(2)
    xs = np.linspace(float(start[0]), float(end[0]), int(count), dtype=np.float32)
    ys = np.linspace(float(start[1]), float(end[1]), int(count), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _raw_points_to_output(points_xy: np.ndarray, meta: dict[str, Any], output_hw: tuple[int, int]) -> np.ndarray:
    transform = transform_from_meta(meta)
    network = np.asarray(transform_points(points_xy.tolist(), transform), dtype=np.float32)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_h, network_w = int(meta["network_hw"][0]), int(meta["network_hw"][1])
    output = network.copy()
    output[:, 0] *= float(output_w) / float(network_w)
    output[:, 1] *= float(output_h) / float(network_h)
    return output.astype(np.float32)


def _output_points_to_raw(points_xy: np.ndarray, meta: dict[str, Any], output_hw: tuple[int, int]) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    network_h, network_w = int(meta["network_hw"][0]), int(meta["network_hw"][1])
    transform = transform_from_meta(meta)
    network = points.copy()
    network[:, 0] *= float(network_w) / max(float(output_w), 1.0)
    network[:, 1] *= float(network_h) / max(float(output_h), 1.0)
    raw = network.copy()
    raw[:, 0] = (raw[:, 0] - float(transform.pad_left)) / max(float(transform.scale), 1.0e-6)
    raw[:, 1] = (raw[:, 1] - float(transform.pad_top)) / max(float(transform.scale), 1.0e-6)
    raw_h, raw_w = int(meta["raw_hw"][0]), int(meta["raw_hw"][1])
    raw[:, 0] = np.clip(raw[:, 0], 0.0, max(float(raw_w - 1), 0.0))
    raw[:, 1] = np.clip(raw[:, 1], 0.0, max(float(raw_h - 1), 0.0))
    return raw.astype(np.float32)


def _line_stats(map_array: np.ndarray | None, output_points: np.ndarray) -> tuple[float, float]:
    if map_array is None or output_points.size == 0:
        return 0.0, 0.0
    values = _bilinear_sample_gray(
        np.asarray(map_array, dtype=np.float32),
        output_points[:, 0].astype(np.float32),
        output_points[:, 1].astype(np.float32),
    )
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.max())


def _endpoint_proposal_mean(proposal_map: np.ndarray | None, output_points: np.ndarray) -> float:
    if proposal_map is None or output_points.shape[0] < 2:
        return 0.0
    endpoints = np.stack([output_points[0], output_points[-1]], axis=0)
    values = _bilinear_sample_gray(
        np.asarray(proposal_map, dtype=np.float32),
        endpoints[:, 0].astype(np.float32),
        endpoints[:, 1].astype(np.float32),
    )
    return float(values.mean()) if values.size else 0.0


def _candidate_row(candidate: dict[str, Any], *, batch_index: int, sample_index: int, meta: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    return {
        "batch_index": int(batch_index),
        "sample_index": int(sample_index),
        "sample_id": str(meta.get("sample_id", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "image_path": str(meta.get("image_path", "")),
        "proposal_source": str(candidate.get("proposal_source", "")),
        "proposal_rank": int(candidate.get("proposal_rank", 0)),
        "candidate_points_json": json.dumps([[float(x), float(y)] for x, y in points.tolist()], separators=(",", ":")),
        "score": float(candidate.get("score", 0.0)),
        "length": float(candidate.get("length", 0.0)),
        "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
        "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", float("inf"))),
        "is_oracle_positive": int(bool(candidate.get("is_oracle_positive", False))),
        **{name: float(candidate.get(name, 0.0)) for name in HOUGH_FEATURES},
    }


def _detect_raw_line_segments(raw_uint8: np.ndarray, *, candidate_generator: str) -> tuple[np.ndarray, np.ndarray]:
    generator = str(candidate_generator or "hough").strip().lower()
    if generator == "hough":
        blurred = cv2.GaussianBlur(raw_uint8, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)
        lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, threshold=36, minLineLength=24, maxLineGap=12)
        if lines is None:
            return np.zeros((0, 4), dtype=np.float32), edges.astype(np.float32) / 255.0
        return lines.reshape(-1, 4).astype(np.float32), edges.astype(np.float32) / 255.0
    if generator == "lsd":
        blurred = cv2.GaussianBlur(raw_uint8, (3, 3), 0)
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(blurred)
        lines = detected[0] if detected else None
        edges = cv2.Canny(blurred, 50, 140)
        if lines is None:
            return np.zeros((0, 4), dtype=np.float32), edges.astype(np.float32) / 255.0
        return lines.reshape(-1, 4).astype(np.float32), edges.astype(np.float32) / 255.0
    raise ValueError(f"unsupported candidate generator: {candidate_generator}")


def _raw_support_maps_at_output(
    image: np.ndarray,
    meta: dict[str, Any],
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    raw_uint8 = np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    edge_map = cv2.Canny(cv2.GaussianBlur(raw_uint8, (3, 3), 0), 50, 140).astype(np.float32) / 255.0
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    ys, xs = np.mgrid[0:output_h, 0:output_w].astype(np.float32)
    raw_points = _output_points_to_raw(np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1), meta, output_hw)
    brightness = _bilinear_sample_gray(
        np.asarray(image, dtype=np.float32),
        raw_points[:, 0],
        raw_points[:, 1],
    ).reshape(output_h, output_w)
    edge = _bilinear_sample_gray(edge_map, raw_points[:, 0], raw_points[:, 1]).reshape(output_h, output_w)
    return brightness.astype(np.float32), edge.astype(np.float32)


def _fit_support_pca_segments(
    *,
    image: np.ndarray,
    meta: dict[str, Any],
    mask_probs: np.ndarray,
    proposal_map: np.ndarray,
    max_candidates: int,
) -> list[dict[str, Any]]:
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    brightness, edge = _raw_support_maps_at_output(image, meta, output_hw)
    raw_support = np.maximum(edge, np.maximum(brightness - 0.55, 0.0) / 0.45)
    support_score = np.asarray(mask_probs, dtype=np.float32) * np.asarray(proposal_map, dtype=np.float32) * raw_support
    positive = (
        (np.asarray(mask_probs, dtype=np.float32) >= 0.35)
        & (np.asarray(proposal_map, dtype=np.float32) >= 0.20)
        & (raw_support >= 0.15)
    ).astype(np.uint8)
    positive = cv2.morphologyEx(positive, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(positive, connectivity=8)
    raw_h, raw_w = int(image.shape[0]), int(image.shape[1])
    candidates: list[dict[str, Any]] = []
    for component_id in range(1, int(component_count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        yy, xx = np.nonzero(labels == component_id)
        if xx.size < 6:
            continue
        coords = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)
        weights = support_score[yy, xx].astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 1.0e-6:
            weights = np.ones_like(weights, dtype=np.float32)
            weight_sum = float(weights.sum())
        center = (coords * weights[:, None]).sum(axis=0) / max(weight_sum, 1.0e-6)
        centered = coords - center[None, :]
        cov = (centered * weights[:, None]).T @ centered / max(weight_sum, 1.0e-6)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-6:
            continue
        axis = axis / axis_norm
        projection = centered @ axis
        if projection.size < 2:
            continue
        lo, hi = np.percentile(projection, [5.0, 95.0]).astype(np.float32)
        if float(hi - lo) < 3.0:
            continue
        start_output = center + axis * float(lo)
        end_output = center + axis * float(hi)
        output_points = _line_points(start_output, end_output)
        raw_points = _output_points_to_raw(output_points, meta, output_hw)
        delta = raw_points[-1] - raw_points[0]
        length = float(np.linalg.norm(delta))
        if length < 12.0 or not math.isfinite(length):
            continue
        mask_mean, mask_max = _line_stats(mask_probs, output_points)
        proposal_mean, proposal_max = _line_stats(proposal_map, output_points)
        raw_edge_mean, raw_edge_max = _line_stats(edge, output_points)
        raw_bright_mean, raw_bright_max = _line_stats(brightness, output_points)
        length_norm = min(length / max(float(raw_w), 1.0), 1.0)
        score = (
            0.30 * proposal_max
            + 0.22 * proposal_mean
            + 0.18 * mask_mean
            + 0.14 * raw_edge_mean
            + 0.10 * raw_bright_mean
            + 0.06 * length_norm
        )
        raw_axis = delta / max(length, 1.0e-6)
        candidates.append(
            {
                "score": float(score),
                "center_score": float(score),
                "length": float(length),
                "points_xy": [[float(x), float(y)] for x, y in raw_points.tolist()],
                "proposal_source": "raw_support_pca",
                "raw_hough_score": float(score),
                "raw_hough_length": float(length),
                "raw_hough_length_norm": float(length_norm),
                "raw_hough_abs_cos": float(abs(float(raw_axis[0]))),
                "raw_hough_abs_sin": float(abs(float(raw_axis[1]))),
                "raw_hough_center_x_norm": float(np.clip(float(raw_points[:, 0].mean()) / max(float(raw_w), 1.0), 0.0, 1.0)),
                "raw_hough_center_y_norm": float(np.clip(float(raw_points[:, 1].mean()) / max(float(raw_h), 1.0), 0.0, 1.0)),
                "raw_hough_edge_mean": raw_edge_mean,
                "raw_hough_edge_max": raw_edge_max,
                "raw_hough_mask_mean": mask_mean,
                "raw_hough_mask_max": mask_max,
                "raw_hough_center_mean": proposal_mean,
                "raw_hough_center_max": proposal_max,
                "raw_hough_selector_mean": raw_bright_mean,
                "raw_hough_selector_max": raw_bright_max,
                "raw_hough_proposal_mean": proposal_mean,
                "raw_hough_proposal_max": proposal_max,
                "raw_hough_endpoint_proposal_mean": _endpoint_proposal_mean(proposal_map, output_points),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            float(candidate.get("raw_hough_score", 0.0)),
            float(candidate.get("raw_hough_length", 0.0)),
        ),
        reverse=True,
    )
    kept = candidates[: max(1, int(max_candidates))]
    for rank, candidate in enumerate(kept, start=1):
        candidate["proposal_rank"] = int(rank)
    return kept


def _generate_raw_hough_candidates(
    *,
    image: np.ndarray | None,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    gt_stop_lines: list[dict[str, Any]],
    max_candidates: int,
    candidate_generator: str = "hough",
) -> list[dict[str, Any]]:
    if image is None or image.ndim != 2:
        return []
    proposal_map = None
    if center_probs is not None and selector_probs is not None:
        proposal_map = np.maximum(center_probs, selector_probs)
    elif center_probs is not None:
        proposal_map = center_probs
    elif selector_probs is not None:
        proposal_map = selector_probs
    if mask_probs is None or proposal_map is None:
        return []
    if str(candidate_generator).strip().lower() == "support_pca":
        candidates = _fit_support_pca_segments(
            image=np.asarray(image, dtype=np.float32),
            meta=meta,
            mask_probs=np.asarray(mask_probs, dtype=np.float32),
            proposal_map=np.asarray(proposal_map, dtype=np.float32),
            max_candidates=int(max_candidates),
        )
        for candidate in candidates:
            nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
            candidate["nearest_gt_distance"] = float(nearest_distance)
            candidate["nearest_gt_angle_error"] = float(nearest_angle)
            candidate["nearest_gt_index"] = int(nearest_index)
            candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
        return candidates
    output_hw = (int(mask_probs.shape[0]), int(mask_probs.shape[1]))
    raw_h, raw_w = int(image.shape[0]), int(image.shape[1])
    raw_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    lines, edge_float = _detect_raw_line_segments(raw_uint8, candidate_generator=str(candidate_generator))
    if lines.shape[0] == 0:
        return []
    candidates: list[dict[str, Any]] = []
    proposal_source = "raw_lsd" if str(candidate_generator).strip().lower() == "lsd" else "raw_hough"
    for line in lines.reshape(-1, 4).tolist():
        x1, y1, x2, y2 = [float(value) for value in line]
        start = np.asarray([x1, y1], dtype=np.float32)
        end = np.asarray([x2, y2], dtype=np.float32)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 24.0 or not math.isfinite(length):
            continue
        raw_points = _line_points(start, end)
        output_points = _raw_points_to_output(raw_points, meta, output_hw)
        inside = (
            (output_points[:, 0] >= 0.0)
            & (output_points[:, 1] >= 0.0)
            & (output_points[:, 0] <= float(output_hw[1] - 1))
            & (output_points[:, 1] <= float(output_hw[0] - 1))
        )
        if int(inside.sum()) < max(2, STOP_LINE_POINT_COUNT // 2):
            continue
        edge_values = _bilinear_sample_gray(edge_float, raw_points[:, 0], raw_points[:, 1])
        edge_mean = float(edge_values.mean()) if edge_values.size else 0.0
        edge_max = float(edge_values.max()) if edge_values.size else 0.0
        mask_mean, mask_max = _line_stats(mask_probs, output_points)
        center_mean, center_max = _line_stats(center_probs, output_points)
        selector_mean, selector_max = _line_stats(selector_probs, output_points)
        proposal_mean, proposal_max = _line_stats(proposal_map, output_points)
        endpoint_mean = _endpoint_proposal_mean(proposal_map, output_points)
        axis = delta / max(length, 1.0e-6)
        length_norm = min(length / max(float(raw_w), 1.0), 1.0)
        score = (
            0.34 * proposal_max
            + 0.22 * proposal_mean
            + 0.18 * mask_mean
            + 0.14 * edge_mean
            + 0.08 * endpoint_mean
            + 0.04 * length_norm
        )
        candidate = {
            "score": float(score),
            "center_score": float(score),
            "length": float(length),
            "points_xy": [[float(x), float(y)] for x, y in raw_points.tolist()],
            "proposal_source": proposal_source,
            "raw_hough_score": float(score),
            "raw_hough_length": float(length),
            "raw_hough_length_norm": float(length_norm),
            "raw_hough_abs_cos": float(abs(float(axis[0]))),
            "raw_hough_abs_sin": float(abs(float(axis[1]))),
            "raw_hough_center_x_norm": float(np.clip(((x1 + x2) * 0.5) / max(float(raw_w), 1.0), 0.0, 1.0)),
            "raw_hough_center_y_norm": float(np.clip(((y1 + y2) * 0.5) / max(float(raw_h), 1.0), 0.0, 1.0)),
            "raw_hough_edge_mean": edge_mean,
            "raw_hough_edge_max": edge_max,
            "raw_hough_mask_mean": mask_mean,
            "raw_hough_mask_max": mask_max,
            "raw_hough_center_mean": center_mean,
            "raw_hough_center_max": center_max,
            "raw_hough_selector_mean": selector_mean,
            "raw_hough_selector_max": selector_max,
            "raw_hough_proposal_mean": proposal_mean,
            "raw_hough_proposal_max": proposal_max,
            "raw_hough_endpoint_proposal_mean": endpoint_mean,
        }
        nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
        candidate["nearest_gt_distance"] = float(nearest_distance)
        candidate["nearest_gt_angle_error"] = float(nearest_angle)
        candidate["nearest_gt_index"] = int(nearest_index)
        candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
        candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            float(candidate.get("raw_hough_score", 0.0)),
            float(candidate.get("raw_hough_length", 0.0)),
        ),
        reverse=True,
    )
    kept = candidates[: max(1, int(max_candidates))]
    for rank, candidate in enumerate(kept, start=1):
        candidate["proposal_rank"] = int(rank)
    return kept


def _collect_records(
    *,
    loader: Any,
    evaluator: Any,
    postprocess_config: Any,
    max_batches: int,
    hough_max_candidates: int,
    candidate_generator: str,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    image_cache: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_raw_hough] collect {split} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("raw-Hough probe requires raw batches for metrics")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                gt_stop_lines = list(gt_sample.get("stop_lines", []))
                mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
                center_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", sample_index), sigmoid=True)
                selector_probs = _as_2d_array(
                    _sample_tensor(outputs, "stop_line_selector_map_logits", sample_index),
                    sigmoid=True,
                )
                image = _load_raw_gray_image(str(meta.get("image_path", "")), image_cache)
                candidates = _generate_raw_hough_candidates(
                    image=image,
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    gt_stop_lines=gt_stop_lines,
                    max_candidates=int(hough_max_candidates),
                    candidate_generator=str(candidate_generator),
                )
                candidate_rows = [
                    _candidate_row(candidate, batch_index=batch_index, sample_index=sample_index, meta=meta)
                    for candidate in candidates
                ]
                for row in candidate_rows:
                    row["split"] = str(split)
                rows.extend(candidate_rows)
                records.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": str(meta.get("sample_id", "")),
                        "raw_batch": _slice_raw_batch_sample(raw_batch, sample_index),
                        "baseline_prediction": dict(baseline_prediction),
                        "gt_stop_lines": gt_stop_lines,
                        "candidates": candidates,
                        "candidate_feature_rows": candidate_rows,
                    }
                )
    return records, rows


def _feature_matrix(records: list[dict[str, Any]], *, top_k: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    candidates: list[dict[str, Any]] = []
    features: list[list[float]] = []
    labels: list[float] = []
    for record in records:
        for candidate in record.get("candidates", []):
            if int(candidate.get("proposal_rank", 10**6)) > int(top_k):
                continue
            candidates.append(candidate)
            features.append([float(candidate.get(name, 0.0)) for name in HOUGH_FEATURES])
            labels.append(float(bool(candidate.get("is_oracle_positive", False))))
    if not features:
        return candidates, np.zeros((0, len(HOUGH_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return candidates, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _attach_scores(records: list[dict[str, Any]], scores: np.ndarray, *, top_k: int, score_key: str = SCORE_KEY) -> None:
    index = 0
    for record in records:
        for candidate, row in zip(record.get("candidates", []), record.get("candidate_feature_rows", [])):
            if int(candidate.get("proposal_rank", 10**6)) > int(top_k):
                continue
            score = float(scores[index])
            candidate[str(score_key)] = score
            row[str(score_key)] = score
            index += 1
    if index != int(scores.shape[0]):
        raise ValueError(f"score length mismatch: attached {index}, got {int(scores.shape[0])}")


def _select_hough_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if int(candidate.get("proposal_rank", 10**6)) <= int(top_k)
        and float(candidate.get(score_key, 0.0)) >= float(threshold)
    ]
    selected.sort(
        key=lambda candidate: (
            float(candidate.get(score_key, 0.0)),
            float(candidate.get("raw_hough_score", 0.0)),
            float(candidate.get("length", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        {
            "score": float(candidate.get(score_key, 0.0)),
            "center_score": float(candidate.get(score_key, 0.0)),
            "length": float(candidate.get("length", 0.0)),
            "points_xy": candidate.get("points_xy", []),
        }
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _metrics_row(
    records: list[dict[str, Any]],
    *,
    name: str,
    split: str,
    score_key: str = "",
    threshold: float = 0.0,
    top_k: int = 0,
    max_components: int = 2,
    union_baseline: bool = False,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    for record in records:
        raw_batches.append(record["raw_batch"])
        if score_key:
            stop_lines = _select_hough_stop_lines(
                list(record.get("candidates", [])),
                score_key=score_key,
                threshold=float(threshold),
                top_k=int(top_k),
                max_components=int(max_components),
            )
            if bool(union_baseline):
                stop_lines = list(record["baseline_prediction"].get("stop_lines", [])) + stop_lines
                stop_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
                stop_lines = _dedupe_stop_line_predictions(stop_lines)[: max(1, int(max_components))]
            predictions.append({**record["baseline_prediction"], "stop_lines": stop_lines})
        else:
            predictions.append(dict(record["baseline_prediction"]))
    merged_raw = _merge_raw_batches(raw_batches)
    metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
    prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
    row = _row_from_metrics(name, metrics, prediction_count=prediction_count, stats={})
    row.update(
        {
            "split": str(split),
            "score_key": str(score_key),
            "threshold": "" if not score_key else float(threshold),
            "top_k": "" if not score_key else int(top_k),
            "union_baseline": int(bool(union_baseline)),
            "sample_count": int(len(records)),
        }
    )
    return row


def _score_records(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    top_k: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    train_candidates, train_x, train_y = _feature_matrix(train_records, top_k=int(top_k))
    val_candidates, val_x, val_y = _feature_matrix(val_records, top_k=int(top_k))
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("raw-Hough verifier requires non-empty train and validation candidates")
    combined_x = np.concatenate([train_x, val_x], axis=0)
    combined_std, mean, std = _standardize_from_train(train_x.astype(np.float64), combined_x.astype(np.float64))
    train_std = combined_std[: train_x.shape[0]].astype(np.float32)
    val_std = combined_std[train_x.shape[0] :].astype(np.float32)
    model = _fit_raw_patch_mlp(train_std, train_y.astype(np.float32), epochs=int(epochs), lr=float(lr))
    train_scores = _predict_raw_patch_mlp(model, train_std)
    val_scores = _predict_raw_patch_mlp(model, val_std)
    _attach_scores(train_records, train_scores, top_k=int(top_k))
    _attach_scores(val_records, val_scores, top_k=int(top_k))
    return {
        "train_candidate_count": int(len(train_candidates)),
        "train_positive_count": int(train_y.sum()),
        "val_candidate_count": int(len(val_candidates)),
        "val_positive_count": int(val_y.sum()),
        "feature_dim": int(train_x.shape[1]),
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    scenario = train_cli.load_meta_train_scenario(args.preset)
    scenario = _scenario_with_dataset_root(scenario, str(args.dataset_root))
    phase = scenario.phases[int(args.phase_index) - 1]
    base_train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    device = _resolve_device(str(args.device), str(base_train_config.device))
    train_config = _train_config_with_runtime_defaults(
        base_train_config,
        device=device,
        max_val_batches=int(args.max_val_batches),
    )
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_raw_hough] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("raw-Hough probe requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = train_cli._build_postprocess_config(train_config)

    train_records, train_rows = _collect_records(
        loader=train_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.train_record_batches),
        hough_max_candidates=int(args.hough_max_candidates),
        candidate_generator=str(args.candidate_generator),
        split="train",
    )
    val_records, val_rows = _collect_records(
        loader=val_loader,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
        max_batches=int(args.max_val_batches),
        hough_max_candidates=int(args.hough_max_candidates),
        candidate_generator=str(args.candidate_generator),
        split="val",
    )
    score_summary = _score_records(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.hough_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
    )
    threshold_row = _best_task_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.hough_top_k),
        max_components=2,
        grid_size=int(args.threshold_grid),
    )
    threshold = float(threshold_row["threshold"])
    rows = [
        _metrics_row(train_records, name="baseline", split="train"),
        _metrics_row(val_records, name="baseline", split="val"),
        _metrics_row(
            train_records,
            name=f"raw_{args.candidate_generator}_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.hough_top_k),
            max_components=2,
        ),
        _metrics_row(
            val_records,
            name=f"raw_{args.candidate_generator}_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.hough_top_k),
            max_components=2,
        ),
        _metrics_row(
            train_records,
            name=f"baseline_plus_raw_{args.candidate_generator}_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.hough_top_k),
            max_components=2,
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name=f"baseline_plus_raw_{args.candidate_generator}_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.hough_top_k),
            max_components=2,
            union_baseline=True,
        ),
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "train_record_batches": int(args.train_record_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "candidate_generator": str(args.candidate_generator),
        "hough_max_candidates": int(args.hough_max_candidates),
        "hough_top_k": int(args.hough_top_k),
        "threshold_grid": int(args.threshold_grid),
        "verifier_epochs": int(args.verifier_epochs),
        "verifier_lr": float(args.verifier_lr),
        "threshold": threshold,
        "score_summary": score_summary,
        "rows": rows,
        "interpretation": (
            "Raw-image line stop-line candidate-generation probe. Line candidates are generated "
            "from existing images and scored with dense stop-line maps, then a train-split MLP "
            "verifier is replayed on validation. Runtime selection uses no GT; GT labels are used "
            "only for verifier training and audit metrics."
        ),
    }
    return {"summary": summary, "rows": rows, "train_rows": train_rows, "val_rows": val_rows}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    _write_csv(output_dir / f"raw_{args.candidate_generator}_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_rows"])
    (output_dir / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
