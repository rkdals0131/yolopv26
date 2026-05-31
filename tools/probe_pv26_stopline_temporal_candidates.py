from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.data.dataset import collate_pv26_samples
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.metrics import _extract_gt_samples, summarize_pv26_metrics
from model.engine.postprocess import _dedupe_stop_line_predictions, _stopline_prediction_sort_key, postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler, _postprocess_override_config
from tools.probe_pv26_lane60_postprocess_thresholds import _detach_to_cpu
from tools.probe_pv26_lane_temporal_neighbor_union import (
    _build_scenario,
    _neighbor_sample_id,
    _parse_neighbor_offsets,
)
from tools.probe_pv26_stopline_angle_mask_extent import _as_2d_array, _row_from_metrics, _sample_tensor, _write_csv
from tools.probe_pv26_stopline_candidate_pool import (
    _fit_raw_patch_mlp,
    _nearest_gt,
    _predict_raw_patch_mlp,
    _standardize_from_train,
    _write_candidate_features_csv,
)
from tools.probe_pv26_stopline_retained_suppressor import (
    _apply_suppressor as _apply_retained_suppressor,
    _assign_stopline_labels as _assign_retained_stopline_labels,
    _decision_audit as _retained_suppressor_decision_audit,
    _line_feature_vector as _retained_line_feature_vector,
    _train_suppressor as _train_retained_suppressor,
)
from tools.probe_pv26_stopline_raw_hough_candidates import (
    _endpoint_proposal_mean,
    _line_points,
    _line_stats,
    _output_points_to_raw,
    _raw_points_to_output,
    _slice_raw_batch_sample,
)
from tools.probe_pv26_lane_flip_tta import _stop_line_distance
from tools.pv26_train import cli as train_cli


SCORE_KEY = "temporal_mlp_score"
UNION_SCORE_KEY = "temporal_union_mlp_score"
TEMPORAL_DENSE_COMPONENT_AUDIT_FIELDS = (
    "temporal_dense_component_area",
    "temporal_dense_component_support_sum",
    "temporal_dense_component_length_norm",
    "temporal_dense_component_abs_cos",
    "temporal_dense_component_abs_sin",
    "temporal_dense_component_center_x_norm",
    "temporal_dense_component_center_y_norm",
    "temporal_dense_component_mask_mean",
    "temporal_dense_component_mask_max",
    "temporal_dense_component_center_mean",
    "temporal_dense_component_center_max",
    "temporal_dense_component_selector_mean",
    "temporal_dense_component_selector_max",
    "temporal_dense_component_proposal_mean",
    "temporal_dense_component_proposal_max",
    "temporal_dense_component_endpoint_proposal_mean",
)
TEMPORAL_FEATURES = (
    "temporal_source_neighbor",
    "temporal_source_envelope",
    "temporal_source_dense_component",
    *TEMPORAL_DENSE_COMPONENT_AUDIT_FIELDS,
    "temporal_neighbor_score",
    "temporal_neighbor_length",
    "temporal_neighbor_length_norm",
    "temporal_envelope_partner_distance_norm",
    "temporal_envelope_normal_distance_norm",
    "temporal_envelope_axis_error",
    "temporal_envelope_length_gain_norm",
    "temporal_abs_offset",
    "temporal_offset_sign",
    "temporal_neighbor_rank_norm",
    "temporal_current_stopline_count",
    "temporal_nearest_current_distance_norm",
    "temporal_mask_mean",
    "temporal_mask_max",
    "temporal_center_mean",
    "temporal_center_max",
    "temporal_selector_mean",
    "temporal_selector_max",
    "temporal_proposal_mean",
    "temporal_proposal_max",
    "temporal_endpoint_proposal_mean",
    "temporal_alignment_dx_norm",
    "temporal_alignment_dy_norm",
    "temporal_alignment_response",
    "temporal_center_x_norm",
    "temporal_center_y_norm",
    "temporal_abs_cos",
    "temporal_abs_sin",
)
TEMPORAL_UNION_FEATURES = (
    *TEMPORAL_FEATURES,
    "union_source_retained",
    "union_source_temporal",
    "union_candidate_rank_norm",
    "union_candidate_count_norm",
)


class _StopLineNeighborPredictor:
    def __init__(self, *, dataset: Any, evaluator: Any, postprocess_config: Any) -> None:
        self.dataset = dataset
        self.evaluator = evaluator
        self.postprocess_config = postprocess_config
        self.cache: dict[int, dict[str, Any]] = {}

    def predict_index(self, dataset_index: int) -> dict[str, Any]:
        dataset_index = int(dataset_index)
        cached = self.cache.get(dataset_index)
        if cached is not None:
            return cached
        batch = collate_pv26_samples([self.dataset[dataset_index]])
        encoded = self.evaluator.prepare_batch(batch)
        outputs = _detach_to_cpu(self.evaluator.forward_encoded_batch(encoded))
        meta = _detach_to_cpu(encoded["meta"])
        prediction = postprocess_pv26_batch(outputs, meta, config=self.postprocess_config)[0]
        dense = {
            "mask_probs": _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", 0), sigmoid=True),
            "center_probs": _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", 0), sigmoid=True),
            "selector_probs": _as_2d_array(
                _sample_tensor(outputs, "stop_line_selector_map_logits", 0),
                sigmoid=True,
            ),
        }
        payload = {"prediction": prediction, "meta": meta[0], "dense": dense}
        self.cache[dataset_index] = payload
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-GT stop-line temporal candidate verifier. Neighbor-frame stop-line "
            "predictions become candidates, current-frame dense stop-line maps provide quality "
            "features, and a train-split MLP verifier is replayed on validation."
        )
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--lane60-experiment", default="stopline_projection_comp_runtime")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--train-record-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--neighbor-offsets", default="-1,1")
    parser.add_argument(
        "--temporal-alignment-mode",
        choices=("none", "phase_corr", "sparse_affine", "orb_homography"),
        default="none",
    )
    parser.add_argument("--temporal-alignment-size", default="160x120")
    parser.add_argument("--temporal-alignment-max-shift-frac", type=float, default=0.15)
    parser.add_argument("--temporal-top-k", type=int, default=8)
    parser.add_argument("--max-temporal-candidates", type=int, default=16)
    parser.add_argument("--temporal-envelope-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--temporal-dense-component-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-temporal-dense-components", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--union-selector-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--union-top-k", type=int, default=12)
    parser.add_argument("--retained-suppressor-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--retained-suppressor-hidden-dim", type=int, default=64)
    parser.add_argument("--retained-suppressor-epochs", type=int, default=80)
    parser.add_argument("--retained-suppressor-batch-size", type=int, default=128)
    parser.add_argument("--retained-suppressor-lr", type=float, default=1.0e-3)
    parser.add_argument("--retained-suppressor-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--retained-suppressor-keep-threshold", type=float, default=0.50)
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--verifier-epochs", type=int, default=60)
    parser.add_argument("--verifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--stop-line-mask-binary-threshold", type=float, default=None)
    parser.add_argument("--stop-line-min-instance-score", type=float, default=None)
    parser.add_argument("--stop-line-presence-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-line-component-gate-source",
        choices=("center", "selector", "max", "validator", "max_validator", "product_validator"),
        default=None,
    )
    parser.add_argument("--crosswalk-polygon-mode", choices=("rect", "hull"), default="hull")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_alignment_size(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(",", "x")
    parts = [part for part in text.split("x") if part]
    if len(parts) != 2:
        raise ValueError(f"alignment size must be WIDTHxHEIGHT, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"alignment size must be positive, got {value!r}")
    return width, height


def _read_alignment_gray(path: str | Path, *, size: tuple[int, int], windowed: bool = True) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(path) as image:
        gray = image.convert("L").resize(size, resampling)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    if not bool(windowed):
        return array
    array -= float(array.mean())
    window_y = np.hanning(array.shape[0]).astype(np.float32)
    window_x = np.hanning(array.shape[1]).astype(np.float32)
    return array * window_y[:, None] * window_x[None, :]


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    if ref.shape != mov.shape or ref.ndim != 2:
        raise ValueError("phase correlation inputs must be 2D arrays with matching shape")
    if ref.size == 0:
        return 0.0, 0.0, 0.0
    ref_fft = np.fft.fft2(ref)
    mov_fft = np.fft.fft2(mov)
    cross = ref_fft * np.conj(mov_fft)
    cross /= np.maximum(np.abs(cross), 1.0e-9)
    corr = np.abs(np.fft.ifft2(cross))
    peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    height, width = corr.shape
    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width
    response = float(corr.max() / max(float(corr.sum()), 1.0e-9))
    return float(peak_x), float(peak_y), response


def _raw_affine_from_small(
    matrix: np.ndarray,
    *,
    raw_hw: tuple[int, int],
    size: tuple[int, int],
) -> tuple[float, float, float, float, float, float]:
    small = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    width, height = int(size[0]), int(size[1])
    scale_x = float(raw_w) / max(float(width), 1.0)
    scale_y = float(raw_h) / max(float(height), 1.0)
    a, b, c = (float(value) for value in small[0])
    d, e, f = (float(value) for value in small[1])
    return (
        a,
        b * scale_x / max(scale_y, 1.0e-6),
        c * scale_x,
        d * scale_y / max(scale_x, 1.0e-6),
        e,
        f * scale_y,
    )


def _corner_max_displacement(
    matrix: tuple[float, float, float, float, float, float],
    *,
    raw_hw: tuple[int, int],
) -> float:
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    corners = np.asarray(
        [
            [0.0, 0.0],
            [max(float(raw_w - 1), 0.0), 0.0],
            [0.0, max(float(raw_h - 1), 0.0)],
            [max(float(raw_w - 1), 0.0), max(float(raw_h - 1), 0.0)],
        ],
        dtype=np.float32,
    )
    a, b, c, d, e, f = matrix
    warped = np.stack(
        [
            a * corners[:, 0] + b * corners[:, 1] + c,
            d * corners[:, 0] + e * corners[:, 1] + f,
        ],
        axis=1,
    )
    return float(np.linalg.norm(warped - corners, axis=1).max())


def _raw_homography_from_small(
    matrix: np.ndarray,
    *,
    raw_hw: tuple[int, int],
    size: tuple[int, int],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    small = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    width, height = int(size[0]), int(size[1])
    scale_x = float(raw_w) / max(float(width), 1.0)
    scale_y = float(raw_h) / max(float(height), 1.0)
    small_to_raw = np.diag([scale_x, scale_y, 1.0])
    raw_to_small = np.diag([1.0 / max(scale_x, 1.0e-6), 1.0 / max(scale_y, 1.0e-6), 1.0])
    raw_matrix = small_to_raw @ small @ raw_to_small
    if abs(float(raw_matrix[2, 2])) > 1.0e-9:
        raw_matrix = raw_matrix / float(raw_matrix[2, 2])
    return tuple(float(value) for value in raw_matrix.reshape(-1))


def _corner_max_displacement_homography(
    matrix: tuple[float, float, float, float, float, float, float, float, float],
    *,
    raw_hw: tuple[int, int],
) -> float:
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    corners = np.asarray(
        [
            [0.0, 0.0],
            [max(float(raw_w - 1), 0.0), 0.0],
            [0.0, max(float(raw_h - 1), 0.0)],
            [max(float(raw_w - 1), 0.0), max(float(raw_h - 1), 0.0)],
        ],
        dtype=np.float64,
    )
    h00, h01, h02, h10, h11, h12, h20, h21, h22 = matrix
    denom = h20 * corners[:, 0] + h21 * corners[:, 1] + h22
    valid = np.abs(denom) > 1.0e-9
    if not bool(valid.all()):
        return float("inf")
    warped = np.stack(
        [
            (h00 * corners[:, 0] + h01 * corners[:, 1] + h02) / denom,
            (h10 * corners[:, 0] + h11 * corners[:, 1] + h12) / denom,
        ],
        axis=1,
    )
    if not bool(np.isfinite(warped).all()):
        return float("inf")
    return float(np.linalg.norm(warped - corners, axis=1).max())


def _sparse_affine_alignment_from_arrays(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    raw_hw: tuple[int, int],
    size: tuple[int, int],
    max_shift_frac: float,
) -> dict[str, float]:
    try:
        import cv2
    except ImportError:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    if ref.shape != mov.shape or ref.ndim != 2 or ref.size == 0:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref_u8 = np.clip(ref * 255.0, 0.0, 255.0).astype(np.uint8)
    mov_u8 = np.clip(mov * 255.0, 0.0, 255.0).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(
        mov_u8,
        maxCorners=240,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if points is None or int(points.shape[0]) < 8:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        mov_u8,
        ref_u8,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    valid = status.reshape(-1).astype(bool)
    src = points.reshape(-1, 2)[valid]
    dst = next_points.reshape(-1, 2)[valid]
    if int(src.shape[0]) < 8:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=500,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 8:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    transformed = (src @ matrix[:, :2].T) + matrix[:, 2][None, :]
    errors = np.linalg.norm(transformed - dst, axis=1)
    median_error = float(np.median(errors[inlier_mask])) if inlier_count else float("inf")
    inlier_fraction = float(inlier_count) / max(float(src.shape[0]), 1.0)
    response = float(np.clip(inlier_fraction / (1.0 + median_error / 4.0), 0.0, 1.0))
    raw_matrix = _raw_affine_from_small(matrix, raw_hw=raw_hw, size=size)
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    max_displacement = _corner_max_displacement(raw_matrix, raw_hw=raw_hw)
    max_allowed = float(max(max(raw_w, raw_h), 1)) * float(max_shift_frac)
    if max_displacement > max_allowed:
        return {"dx": 0.0, "dy": 0.0, "response": response, "applied": 0.0}
    a, b, c, d, e, f = raw_matrix
    return {
        "dx": float(c),
        "dy": float(f),
        "response": response,
        "applied": 1.0,
        "m00": float(a),
        "m01": float(b),
        "m02": float(c),
        "m10": float(d),
        "m11": float(e),
        "m12": float(f),
    }


def _orb_homography_alignment_from_arrays(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    raw_hw: tuple[int, int],
    size: tuple[int, int],
    max_shift_frac: float,
) -> dict[str, float]:
    try:
        import cv2
    except ImportError:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    if ref.shape != mov.shape or ref.ndim != 2 or ref.size == 0:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref_u8 = np.clip(ref * 255.0, 0.0, 255.0).astype(np.uint8)
    mov_u8 = np.clip(mov * 255.0, 0.0, 255.0).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=7)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_u8, None)
    mov_keypoints, mov_descriptors = orb.detectAndCompute(mov_u8, None)
    if (
        ref_descriptors is None
        or mov_descriptors is None
        or len(ref_keypoints) < 12
        or len(mov_keypoints) < 12
    ):
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(mov_descriptors, ref_descriptors, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if float(first.distance) <= 0.78 * float(second.distance):
            good_matches.append(first)
    if len(good_matches) < 10:
        matcher_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = list(matcher_cross.match(mov_descriptors, ref_descriptors))
        good_matches.sort(key=lambda item: float(item.distance))
        good_matches = good_matches[: min(len(good_matches), 120)]
    if len(good_matches) < 10:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    src = np.asarray([mov_keypoints[match.queryIdx].pt for match in good_matches], dtype=np.float32)
    dst = np.asarray([ref_keypoints[match.trainIdx].pt for match in good_matches], dtype=np.float32)
    matrix, inliers = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=1000,
        confidence=0.995,
    )
    if matrix is None or inliers is None:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 8:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    src_h = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float32)], axis=1)
    projected_h = (np.asarray(matrix, dtype=np.float64) @ src_h.T).T
    denom = projected_h[:, 2:3]
    valid = np.abs(denom[:, 0]) > 1.0e-9
    if not bool(valid.any()):
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    projected = np.full((projected_h.shape[0], 2), np.nan, dtype=np.float64)
    projected[valid] = projected_h[valid, :2] / denom[valid]
    valid_inliers = inlier_mask & np.isfinite(projected).all(axis=1)
    if int(valid_inliers.sum()) < 8:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    errors = np.linalg.norm(projected - dst, axis=1)
    median_error = float(np.median(errors[valid_inliers]))
    inlier_fraction = float(inlier_count) / max(float(src.shape[0]), 1.0)
    response = float(np.clip(inlier_fraction / (1.0 + median_error / 4.0), 0.0, 1.0))
    raw_matrix = _raw_homography_from_small(matrix, raw_hw=raw_hw, size=size)
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    max_displacement = _corner_max_displacement_homography(raw_matrix, raw_hw=raw_hw)
    max_allowed = float(max(max(raw_w, raw_h), 1)) * float(max_shift_frac)
    if max_displacement > max_allowed:
        return {"dx": 0.0, "dy": 0.0, "response": response, "applied": 0.0}
    h00, h01, h02, h10, h11, h12, h20, h21, h22 = raw_matrix
    return {
        "dx": float(h02),
        "dy": float(h12),
        "response": response,
        "applied": 1.0,
        "h00": float(h00),
        "h01": float(h01),
        "h02": float(h02),
        "h10": float(h10),
        "h11": float(h11),
        "h12": float(h12),
        "h20": float(h20),
        "h21": float(h21),
        "h22": float(h22),
    }


def _temporal_alignment(
    *,
    current_meta: dict[str, Any],
    neighbor_meta: dict[str, Any],
    mode: str,
    size: tuple[int, int],
    max_shift_frac: float,
) -> dict[str, float]:
    if str(mode) == "none":
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    if str(mode) not in {"phase_corr", "sparse_affine", "orb_homography"}:
        raise ValueError(f"unknown temporal alignment mode: {mode}")
    current_path = str(current_meta.get("image_path", ""))
    neighbor_path = str(neighbor_meta.get("image_path", ""))
    if not current_path or not neighbor_path:
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    try:
        if str(mode) in {"sparse_affine", "orb_homography"}:
            current_gray = _read_alignment_gray(current_path, size=size, windowed=False)
            neighbor_gray = _read_alignment_gray(neighbor_path, size=size, windowed=False)
            raw_h, raw_w = int(current_meta.get("raw_hw", (1, 1))[0]), int(current_meta.get("raw_hw", (1, 1))[1])
            if str(mode) == "orb_homography":
                return _orb_homography_alignment_from_arrays(
                    current_gray,
                    neighbor_gray,
                    raw_hw=(raw_h, raw_w),
                    size=size,
                    max_shift_frac=float(max_shift_frac),
                )
            return _sparse_affine_alignment_from_arrays(
                current_gray,
                neighbor_gray,
                raw_hw=(raw_h, raw_w),
                size=size,
                max_shift_frac=float(max_shift_frac),
            )
        current_gray = _read_alignment_gray(current_path, size=size)
        neighbor_gray = _read_alignment_gray(neighbor_path, size=size)
        dx_small, dy_small, response = _phase_correlation_shift(current_gray, neighbor_gray)
    except (OSError, ValueError):
        return {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    raw_h, raw_w = int(current_meta.get("raw_hw", (1, 1))[0]), int(current_meta.get("raw_hw", (1, 1))[1])
    width, height = int(size[0]), int(size[1])
    dx = float(dx_small) * float(raw_w) / max(float(width), 1.0)
    dy = float(dy_small) * float(raw_h) / max(float(height), 1.0)
    max_dx = max(1.0, float(raw_w) * float(max_shift_frac))
    max_dy = max(1.0, float(raw_h) * float(max_shift_frac))
    if abs(dx) > max_dx or abs(dy) > max_dy:
        return {"dx": 0.0, "dy": 0.0, "response": float(response), "applied": 0.0}
    return {"dx": float(dx), "dy": float(dy), "response": float(response), "applied": 1.0}


def _warp_stop_line_points(line: dict[str, Any], *, alignment: dict[str, float], meta: dict[str, Any]) -> list[list[float]]:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return []
    homography_required = ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")
    if all(key in alignment for key in homography_required):
        h00, h01, h02, h10, h11, h12, h20, h21, h22 = (float(alignment[key]) for key in homography_required)
        denom = h20 * points[:, 0] + h21 * points[:, 1] + h22
        valid = np.abs(denom) > 1.0e-9
        if not bool(valid.all()):
            return _translate_stop_line_points(
                line,
                dx=float(alignment.get("dx", 0.0)),
                dy=float(alignment.get("dy", 0.0)),
                meta=meta,
            )
        warped = np.stack(
            [
                (h00 * points[:, 0] + h01 * points[:, 1] + h02) / denom,
                (h10 * points[:, 0] + h11 * points[:, 1] + h12) / denom,
            ],
            axis=1,
        )
        if not bool(np.isfinite(warped).all()):
            return _translate_stop_line_points(
                line,
                dx=float(alignment.get("dx", 0.0)),
                dy=float(alignment.get("dy", 0.0)),
                meta=meta,
            )
        raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
        warped[:, 0] = np.clip(warped[:, 0], 0.0, max(float(raw_w - 1), 0.0))
        warped[:, 1] = np.clip(warped[:, 1], 0.0, max(float(raw_h - 1), 0.0))
        return [[float(point[0]), float(point[1])] for point in warped.tolist()]
    required = ("m00", "m01", "m02", "m10", "m11", "m12")
    if not all(key in alignment for key in required):
        return _translate_stop_line_points(
            line,
            dx=float(alignment.get("dx", 0.0)),
            dy=float(alignment.get("dy", 0.0)),
            meta=meta,
        )
    a, b, c, d, e, f = (float(alignment[key]) for key in required)
    warped = np.stack(
        [
            a * points[:, 0] + b * points[:, 1] + c,
            d * points[:, 0] + e * points[:, 1] + f,
        ],
        axis=1,
    )
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    warped[:, 0] = np.clip(warped[:, 0], 0.0, max(float(raw_w - 1), 0.0))
    warped[:, 1] = np.clip(warped[:, 1], 0.0, max(float(raw_h - 1), 0.0))
    return [[float(point[0]), float(point[1])] for point in warped.tolist()]


def _translate_stop_line_points(line: dict[str, Any], *, dx: float, dy: float, meta: dict[str, Any]) -> list[list[float]]:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return []
    shifted = points + np.asarray([float(dx), float(dy)], dtype=np.float32).reshape(1, 2)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    shifted[:, 0] = np.clip(shifted[:, 0], 0.0, max(float(raw_w - 1), 0.0))
    shifted[:, 1] = np.clip(shifted[:, 1], 0.0, max(float(raw_h - 1), 0.0))
    return [[float(point[0]), float(point[1])] for point in shifted.tolist()]


def _line_score(line: dict[str, Any]) -> float:
    candidates = (
        line.get("score"),
        line.get("center_score"),
        line.get("instance_score"),
        line.get("orientation_score"),
    )
    values = [float(value) for value in candidates if isinstance(value, (int, float))]
    return max(values) if values else 0.0


def _line_length(line: dict[str, Any]) -> float:
    value = line.get("length")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def _copy_stop_line(line: dict[str, Any], *, score: float, source: str) -> dict[str, Any]:
    copied = dict(line)
    copied["points_xy"] = [[float(point[0]), float(point[1])] for point in line.get("points_xy", [])]
    copied["score"] = float(score)
    copied["center_score"] = float(score)
    copied["source"] = str(source)
    copied["proposal_source"] = str(source)
    return copied


def _nearest_current_distance(candidate: dict[str, Any], current_stop_lines: list[dict[str, Any]]) -> float:
    if not current_stop_lines:
        return 1.0e6
    return float(min(_stop_line_distance(candidate, current) for current in current_stop_lines))


def _is_temporal_source(source: str) -> bool:
    return str(source) in {"temporal_neighbor", "temporal_endpoint_envelope", "temporal_dense_component"}


def _build_dense_component_stop_lines(
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    max_candidates: int,
    source: str = "temporal_dense_component",
) -> list[dict[str, Any]]:
    reference_map = next(
        (array for array in (mask_probs, center_probs, selector_probs) if isinstance(array, np.ndarray)),
        None,
    )
    if reference_map is None:
        return []
    try:
        import cv2
    except ImportError:
        return []
    output_h, output_w = int(reference_map.shape[0]), int(reference_map.shape[1])
    if output_h <= 0 or output_w <= 0:
        return []
    mask = np.asarray(mask_probs if mask_probs is not None else np.zeros_like(reference_map), dtype=np.float32)
    center = np.asarray(center_probs if center_probs is not None else np.zeros_like(reference_map), dtype=np.float32)
    selector = np.asarray(selector_probs if selector_probs is not None else np.zeros_like(reference_map), dtype=np.float32)
    proposal = np.maximum(center, selector)
    support = mask * proposal
    positive = (((mask >= 0.30) & (proposal >= 0.20)) | ((support >= 0.12) & (proposal >= 0.15))).astype(np.uint8)
    positive = cv2.morphologyEx(positive, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(positive, connectivity=8)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    output_hw = (output_h, output_w)
    candidates: list[dict[str, Any]] = []
    for component_id in range(1, int(component_count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        yy, xx = np.nonzero(labels == component_id)
        if int(xx.size) < 5:
            continue
        coords = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)
        weights = support[yy, xx].astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 1.0e-6:
            weights = np.ones_like(weights, dtype=np.float32)
            weight_sum = float(weights.sum())
        center_xy = (coords * weights[:, None]).sum(axis=0) / max(weight_sum, 1.0e-6)
        centered = coords - center_xy[None, :]
        cov = (centered * weights[:, None]).T @ centered / max(weight_sum, 1.0e-6)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-6:
            continue
        axis /= axis_norm
        projection = centered @ axis
        if projection.size < 2:
            continue
        lo, hi = np.percentile(projection, [4.0, 96.0]).astype(np.float32)
        if float(hi - lo) < 2.5:
            continue
        start_output = center_xy + axis * float(lo)
        end_output = center_xy + axis * float(hi)
        output_points = _line_points(start_output, end_output)
        raw_points = _output_points_to_raw(output_points, meta, output_hw)
        if (
            float(raw_points[0, 0]) > float(raw_points[-1, 0])
            or (
                abs(float(raw_points[0, 0]) - float(raw_points[-1, 0])) <= 1.0e-6
                and float(raw_points[0, 1]) > float(raw_points[-1, 1])
            )
        ):
            raw_points = raw_points[::-1].copy()
            output_points = output_points[::-1].copy()
        delta = raw_points[-1] - raw_points[0]
        length = float(np.linalg.norm(delta))
        if length < 12.0 or not math.isfinite(length):
            continue
        mask_mean, mask_max = _line_stats(mask, output_points)
        center_mean, center_max = _line_stats(center, output_points)
        selector_mean, selector_max = _line_stats(selector, output_points)
        proposal_mean, proposal_max = _line_stats(proposal, output_points)
        endpoint_mean = _endpoint_proposal_mean(proposal, output_points)
        length_norm = float(min(length / max(float(raw_w), 1.0), 1.0))
        axis_raw = delta / max(length, 1.0e-6)
        score = (
            0.30 * float(proposal_max)
            + 0.22 * float(proposal_mean)
            + 0.20 * float(mask_mean)
            + 0.12 * float(endpoint_mean)
            + 0.10 * float(mask_max)
            + 0.06 * length_norm
        )
        candidate = {
            "score": float(score),
            "center_score": float(score),
            "length": float(length),
            "points_xy": [[float(x), float(y)] for x, y in raw_points.tolist()],
            "source": str(source),
            "proposal_source": str(source),
            "temporal_dense_component_area": float(area),
            "temporal_dense_component_support_sum": float(weight_sum),
            "temporal_dense_component_length_norm": length_norm,
            "temporal_dense_component_abs_cos": float(abs(float(axis_raw[0]))),
            "temporal_dense_component_abs_sin": float(abs(float(axis_raw[1]))),
            "temporal_dense_component_center_x_norm": float(
                np.clip(float(raw_points[:, 0].mean()) / max(float(raw_w), 1.0), 0.0, 1.0)
            ),
            "temporal_dense_component_center_y_norm": float(
                np.clip(float(raw_points[:, 1].mean()) / max(float(raw_h), 1.0), 0.0, 1.0)
            ),
            "temporal_dense_component_mask_mean": float(mask_mean),
            "temporal_dense_component_mask_max": float(mask_max),
            "temporal_dense_component_center_mean": float(center_mean),
            "temporal_dense_component_center_max": float(center_max),
            "temporal_dense_component_selector_mean": float(selector_mean),
            "temporal_dense_component_selector_max": float(selector_max),
            "temporal_dense_component_proposal_mean": float(proposal_mean),
            "temporal_dense_component_proposal_max": float(proposal_max),
            "temporal_dense_component_endpoint_proposal_mean": float(endpoint_mean),
        }
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("temporal_dense_component_support_sum", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(max_candidates))]


def _stopline_temporal_features(
    candidate: dict[str, Any],
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    neighbor_offset: int,
    neighbor_rank: int,
    alignment: dict[str, float] | None,
    current_stop_lines: list[dict[str, Any]],
) -> dict[str, float]:
    points_raw = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    reference_map = next(
        (array for array in (mask_probs, center_probs, selector_probs) if isinstance(array, np.ndarray)),
        None,
    )
    if points_raw.shape[0] < 2 or reference_map is None:
        return {name: 0.0 for name in TEMPORAL_FEATURES}
    output_hw = (int(reference_map.shape[0]), int(reference_map.shape[1]))
    output_points = _raw_points_to_output(points_raw, meta, output_hw)
    proposal_map = None
    if center_probs is not None and selector_probs is not None:
        proposal_map = np.maximum(center_probs, selector_probs)
    elif center_probs is not None:
        proposal_map = center_probs
    elif selector_probs is not None:
        proposal_map = selector_probs
    mask_mean, mask_max = _line_stats(mask_probs, output_points)
    center_mean, center_max = _line_stats(center_probs, output_points)
    selector_mean, selector_max = _line_stats(selector_probs, output_points)
    proposal_mean, proposal_max = _line_stats(proposal_map, output_points)
    endpoint_mean = _endpoint_proposal_mean(proposal_map, output_points)
    length = _line_length(candidate)
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    center = points_raw.mean(axis=0)
    delta = points_raw[-1] - points_raw[0]
    norm = float(np.linalg.norm(delta))
    axis = delta / max(norm, 1.0e-6)
    nearest_distance = _nearest_current_distance(candidate, current_stop_lines)
    alignment_payload = alignment or {}
    alignment_dx = float(alignment_payload.get("dx", 0.0))
    alignment_dy = float(alignment_payload.get("dy", 0.0))
    alignment_response = float(alignment_payload.get("response", 0.0))
    source = str(candidate.get("source", ""))
    features = {
        "temporal_source_neighbor": float(1.0 if source == "temporal_neighbor" else 0.0),
        "temporal_source_envelope": float(1.0 if source == "temporal_endpoint_envelope" else 0.0),
        "temporal_source_dense_component": float(1.0 if source == "temporal_dense_component" else 0.0),
        **{name: float(candidate.get(name, 0.0)) for name in TEMPORAL_DENSE_COMPONENT_AUDIT_FIELDS},
        "temporal_neighbor_score": float(_line_score(candidate)),
        "temporal_neighbor_length": float(length),
        "temporal_neighbor_length_norm": float(min(length / max(float(raw_w), 1.0), 1.0)),
        "temporal_envelope_partner_distance_norm": float(
            min(float(candidate.get("temporal_envelope_partner_distance", 0.0)) / max(float(raw_w), 1.0), 4.0)
        ),
        "temporal_envelope_normal_distance_norm": float(
            min(float(candidate.get("temporal_envelope_normal_distance", 0.0)) / max(float(raw_w), 1.0), 4.0)
        ),
        "temporal_envelope_axis_error": float(min(float(candidate.get("temporal_envelope_axis_error", 0.0)), math.pi)),
        "temporal_envelope_length_gain_norm": float(
            np.clip(float(candidate.get("temporal_envelope_length_gain", 0.0)) / max(float(raw_w), 1.0), -1.0, 1.0)
        ),
        "temporal_abs_offset": float(abs(int(neighbor_offset))),
        "temporal_offset_sign": float(1.0 if int(neighbor_offset) > 0 else -1.0),
        "temporal_neighbor_rank_norm": float(1.0 / max(int(neighbor_rank), 1)),
        "temporal_current_stopline_count": float(len(current_stop_lines)),
        "temporal_nearest_current_distance_norm": float(min(nearest_distance / max(float(raw_w), 1.0), 4.0)),
        "temporal_mask_mean": float(mask_mean),
        "temporal_mask_max": float(mask_max),
        "temporal_center_mean": float(center_mean),
        "temporal_center_max": float(center_max),
        "temporal_selector_mean": float(selector_mean),
        "temporal_selector_max": float(selector_max),
        "temporal_proposal_mean": float(proposal_mean),
        "temporal_proposal_max": float(proposal_max),
        "temporal_endpoint_proposal_mean": float(endpoint_mean),
        "temporal_alignment_dx_norm": float(np.clip(alignment_dx / max(float(raw_w), 1.0), -1.0, 1.0)),
        "temporal_alignment_dy_norm": float(np.clip(alignment_dy / max(float(raw_h), 1.0), -1.0, 1.0)),
        "temporal_alignment_response": float(np.clip(alignment_response, 0.0, 1.0)),
        "temporal_center_x_norm": float(np.clip(float(center[0]) / max(float(raw_w), 1.0), 0.0, 1.0)),
        "temporal_center_y_norm": float(np.clip(float(center[1]) / max(float(raw_h), 1.0), 0.0, 1.0)),
        "temporal_abs_cos": float(abs(float(axis[0]))),
        "temporal_abs_sin": float(abs(float(axis[1]))),
    }
    return {key: 0.0 if not math.isfinite(float(value)) else float(value) for key, value in features.items()}


def _line_points_array(line: dict[str, Any]) -> np.ndarray:
    points = np.asarray(line.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([points[0], points[-1]], dtype=np.float32)


def _line_axis(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 2:
        return None
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-6:
        return None
    return (delta / norm).astype(np.float32)


def _temporal_endpoint_envelope(
    current_line: dict[str, Any],
    temporal_line: dict[str, Any],
    *,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    current_points = _line_points_array(current_line)
    temporal_points = _line_points_array(temporal_line)
    current_axis = _line_axis(current_points)
    temporal_axis = _line_axis(temporal_points)
    if current_axis is None or temporal_axis is None:
        return None
    axis_dot = float(np.clip(float(np.dot(current_axis, temporal_axis)), -1.0, 1.0))
    if axis_dot < 0.0:
        temporal_points = temporal_points[::-1].copy()
        temporal_axis = -temporal_axis
        axis_dot = -axis_dot
    axis_error = float(math.acos(np.clip(axis_dot, -1.0, 1.0)))
    if axis_error > math.radians(25.0):
        return None
    current_center = current_points.mean(axis=0)
    temporal_center = temporal_points.mean(axis=0)
    normal = np.asarray([-current_axis[1], current_axis[0]], dtype=np.float32)
    center_delta = temporal_center - current_center
    normal_distance = float(abs(float(np.dot(center_delta, normal))))
    if normal_distance > 48.0:
        return None
    current_length = float(np.linalg.norm(current_points[-1] - current_points[0]))
    temporal_length = float(np.linalg.norm(temporal_points[-1] - temporal_points[0]))
    along_distance = float(abs(float(np.dot(center_delta, current_axis))))
    if along_distance > max(180.0, 0.75 * max(current_length + temporal_length, 1.0)):
        return None
    all_points = np.concatenate([current_points, temporal_points], axis=0)
    origin = all_points.mean(axis=0)
    projected_axis = (all_points - origin[None, :]) @ current_axis
    projected_normal = (all_points - origin[None, :]) @ normal
    start_t = float(projected_axis.min())
    end_t = float(projected_axis.max())
    normal_t = float(projected_normal.mean())
    start = origin + current_axis * start_t + normal * normal_t
    end = origin + current_axis * end_t + normal * normal_t
    raw_h, raw_w = int(meta.get("raw_hw", (1, 1))[0]), int(meta.get("raw_hw", (1, 1))[1])
    start[0] = np.clip(start[0], 0.0, max(float(raw_w - 1), 0.0))
    start[1] = np.clip(start[1], 0.0, max(float(raw_h - 1), 0.0))
    end[0] = np.clip(end[0], 0.0, max(float(raw_w - 1), 0.0))
    end[1] = np.clip(end[1], 0.0, max(float(raw_h - 1), 0.0))
    envelope_length = float(np.linalg.norm(end - start))
    length_gain = envelope_length - current_length
    if envelope_length < 12.0 or length_gain < 4.0:
        return None
    partner_distance = float(_stop_line_distance(current_line, temporal_line))
    score = max(float(_line_score(current_line)), float(_line_score(temporal_line)))
    envelope = _copy_stop_line(
        {
            "points_xy": [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
            "length": envelope_length,
        },
        score=score,
        source="temporal_endpoint_envelope",
    )
    envelope["length"] = envelope_length
    envelope["temporal_envelope_partner_distance"] = partner_distance
    envelope["temporal_envelope_normal_distance"] = normal_distance
    envelope["temporal_envelope_axis_error"] = axis_error
    envelope["temporal_envelope_length_gain"] = length_gain
    return envelope


def _build_temporal_endpoint_envelopes(
    *,
    meta: dict[str, Any],
    temporal_candidates: list[dict[str, Any]],
    current_stop_lines: list[dict[str, Any]],
    gt_stop_lines: list[dict[str, Any]],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
) -> list[dict[str, Any]]:
    envelopes: list[dict[str, Any]] = []
    for temporal_candidate in temporal_candidates:
        best_envelope: dict[str, Any] | None = None
        best_key: tuple[float, float] | None = None
        for current_rank, current_line in enumerate(current_stop_lines, start=1):
            envelope = _temporal_endpoint_envelope(current_line, temporal_candidate, meta=meta)
            if envelope is None:
                continue
            key = (
                -float(envelope.get("temporal_envelope_normal_distance", 0.0)),
                float(envelope.get("temporal_envelope_length_gain", 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_envelope = envelope
                best_envelope["temporal_envelope_current_rank"] = int(current_rank)
        if best_envelope is None:
            continue
        best_envelope["neighbor_offset"] = int(temporal_candidate.get("neighbor_offset", 0))
        best_envelope["neighbor_dataset_index"] = int(temporal_candidate.get("neighbor_dataset_index", -1))
        best_envelope["neighbor_rank"] = int(temporal_candidate.get("neighbor_rank", 0))
        best_envelope["temporal_alignment_dx"] = float(temporal_candidate.get("temporal_alignment_dx", 0.0))
        best_envelope["temporal_alignment_dy"] = float(temporal_candidate.get("temporal_alignment_dy", 0.0))
        best_envelope["temporal_alignment_response_raw"] = float(
            temporal_candidate.get("temporal_alignment_response_raw", 0.0)
        )
        best_envelope["temporal_alignment_applied"] = float(temporal_candidate.get("temporal_alignment_applied", 0.0))
        alignment = {
            "dx": float(best_envelope.get("temporal_alignment_dx", 0.0)),
            "dy": float(best_envelope.get("temporal_alignment_dy", 0.0)),
            "response": float(best_envelope.get("temporal_alignment_response_raw", 0.0)),
        }
        best_envelope.update(
            _stopline_temporal_features(
                best_envelope,
                meta=meta,
                mask_probs=mask_probs,
                center_probs=center_probs,
                selector_probs=selector_probs,
                neighbor_offset=int(best_envelope.get("neighbor_offset", 0)),
                neighbor_rank=int(best_envelope.get("neighbor_rank", 0)),
                alignment=alignment,
                current_stop_lines=current_stop_lines,
            )
        )
        nearest_distance, nearest_angle, nearest_index = _nearest_gt(best_envelope, gt_stop_lines)
        best_envelope["nearest_gt_distance"] = float(nearest_distance)
        best_envelope["nearest_gt_angle_error"] = float(nearest_angle)
        best_envelope["nearest_gt_index"] = int(nearest_index)
        best_envelope["is_oracle_positive"] = bool(nearest_distance <= 40.0)
        best_envelope["temporal_rank_score"] = (
            float(best_envelope.get("temporal_proposal_max", 0.0))
            + 0.5 * float(best_envelope.get("temporal_mask_mean", 0.0))
            + 0.25 * float(best_envelope.get("temporal_neighbor_score", 0.0))
            + 0.10 * float(best_envelope.get("temporal_source_envelope", 0.0))
            - 0.05 * float(abs(int(best_envelope.get("neighbor_offset", 0))))
        )
        envelopes.append(best_envelope)
    return envelopes


def _candidate_row(candidate: dict[str, Any], *, batch_index: int, sample_index: int, meta: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(candidate.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    return {
        "batch_index": int(batch_index),
        "sample_index": int(sample_index),
        "sample_id": str(meta.get("sample_id", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "image_path": str(meta.get("image_path", "")),
        "source": str(candidate.get("source", "")),
        "neighbor_offset": int(candidate.get("neighbor_offset", 0)),
        "neighbor_dataset_index": int(candidate.get("neighbor_dataset_index", -1)),
        "neighbor_rank": int(candidate.get("neighbor_rank", 0)),
        "temporal_alignment_dx": float(candidate.get("temporal_alignment_dx", 0.0)),
        "temporal_alignment_dy": float(candidate.get("temporal_alignment_dy", 0.0)),
        "temporal_alignment_response_raw": float(candidate.get("temporal_alignment_response_raw", 0.0)),
        "temporal_alignment_applied": float(candidate.get("temporal_alignment_applied", 0.0)),
        "candidate_points_json": json.dumps([[float(x), float(y)] for x, y in points.tolist()], separators=(",", ":")),
        "score": float(candidate.get("score", 0.0)),
        "length": float(candidate.get("length", 0.0)),
        "nearest_gt_distance": float(candidate.get("nearest_gt_distance", float("inf"))),
        "nearest_gt_angle_error": float(candidate.get("nearest_gt_angle_error", float("inf"))),
        "is_oracle_positive": int(bool(candidate.get("is_oracle_positive", False))),
        **{name: float(candidate.get(name, 0.0)) for name in TEMPORAL_FEATURES},
    }


def _assign_union_match_labels(candidates: list[dict[str, Any]], gt_stop_lines: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        candidate["is_union_positive"] = False
    if not candidates or not gt_stop_lines:
        return
    pairs: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        for gt_index, gt_line in enumerate(gt_stop_lines):
            distance = _stop_line_distance(candidate, gt_line)
            if math.isfinite(distance) and float(distance) <= 40.0:
                pairs.append((float(distance), int(candidate_index), int(gt_index)))
    pairs.sort(key=lambda item: item[0])
    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    for _distance, candidate_index, gt_index in pairs:
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        candidates[candidate_index]["is_union_positive"] = True
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)


def _build_baseline_union_candidates(
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    baseline_prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    current_stop_lines = list(baseline_prediction.get("stop_lines", []))
    current_stop_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
    candidates: list[dict[str, Any]] = []
    for rank, line in enumerate(current_stop_lines, start=1):
        candidate = _copy_stop_line(line, score=_line_score(line), source="retained_projection_comp")
        candidate["neighbor_offset"] = 0
        candidate["neighbor_dataset_index"] = -1
        candidate["neighbor_rank"] = int(rank)
        candidate["temporal_alignment_dx"] = 0.0
        candidate["temporal_alignment_dy"] = 0.0
        candidate["temporal_alignment_response_raw"] = 0.0
        candidate["temporal_alignment_applied"] = 0.0
        candidate["length"] = _line_length(candidate)
        candidate.update(
            _stopline_temporal_features(
                candidate,
                meta=meta,
                mask_probs=mask_probs,
                center_probs=center_probs,
                selector_probs=selector_probs,
                neighbor_offset=0,
                neighbor_rank=int(rank),
                alignment={},
                current_stop_lines=current_stop_lines,
            )
        )
        candidate["temporal_rank_score"] = (
            float(candidate.get("temporal_proposal_max", 0.0))
            + 0.5 * float(candidate.get("temporal_mask_mean", 0.0))
            + 0.25 * float(candidate.get("temporal_neighbor_score", 0.0))
        )
        candidates.append(candidate)
    return candidates


def _build_union_candidates(
    *,
    baseline_candidates: list[dict[str, Any]],
    temporal_candidates: list[dict[str, Any]],
    gt_stop_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in baseline_candidates]
    for candidate in temporal_candidates:
        copied = dict(candidate)
        source = str(copied.get("source", "temporal_neighbor"))
        copied["source"] = source
        copied["proposal_source"] = str(copied.get("proposal_source", source))
        candidates.append(copied)
    for candidate in candidates:
        source = str(candidate.get("source", ""))
        candidate["union_source_retained"] = float(1.0 if source == "retained_projection_comp" else 0.0)
        candidate["union_source_temporal"] = float(1.0 if _is_temporal_source(source) else 0.0)
    candidates.sort(
        key=lambda item: (
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("temporal_neighbor_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    _assign_union_match_labels(candidates, gt_stop_lines)
    return candidates


def _retained_suppressor_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_dim=int(args.retained_suppressor_hidden_dim),
        suppressor_epochs=int(args.retained_suppressor_epochs),
        suppressor_batch_size=int(args.retained_suppressor_batch_size),
        suppressor_lr=float(args.retained_suppressor_lr),
        suppressor_weight_decay=float(args.retained_suppressor_weight_decay),
        keep_threshold=float(args.retained_suppressor_keep_threshold),
        seed=20260531,
    )


def _retained_suppressor_examples(
    *,
    baseline_prediction: dict[str, Any],
    gt_stop_lines: list[dict[str, Any]],
    outputs: dict[str, Any],
    sample_index: int,
    global_sample_index: int,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    lines = [dict(line) for line in baseline_prediction.get("stop_lines", [])]
    labels = _assign_retained_stopline_labels(lines, gt_stop_lines)
    examples: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        label = int(labels[line_index]) if line_index < len(labels) else 0
        feature = _retained_line_feature_vector(
            line,
            predictions=outputs,
            sample_index=int(sample_index),
            meta=meta,
            rank=int(line_index),
            candidate_count=len(lines),
        )
        examples.append(
            {
                "features": feature,
                "label": int(label),
                "sample_index": int(global_sample_index),
                "sample_batch_index": int(sample_index),
                "line_index": int(line_index),
            }
        )
    return examples


def _union_candidate_row(
    candidate: dict[str, Any],
    *,
    batch_index: int,
    sample_index: int,
    meta: dict[str, Any],
    rank: int,
    candidate_count: int,
) -> dict[str, Any]:
    row = _candidate_row(candidate, batch_index=batch_index, sample_index=sample_index, meta=meta)
    source = str(candidate.get("source", ""))
    row.update(
        {
            "source": source,
            "is_union_positive": int(bool(candidate.get("is_union_positive", False))),
            "union_source_retained": float(1.0 if source == "retained_projection_comp" else 0.0),
            "union_source_temporal": float(1.0 if _is_temporal_source(source) else 0.0),
            "union_candidate_rank_norm": float(1.0 / max(int(rank), 1)),
            "union_candidate_count_norm": float(min(float(candidate_count) / 16.0, 4.0)),
        }
    )
    if UNION_SCORE_KEY in candidate:
        row[UNION_SCORE_KEY] = float(candidate.get(UNION_SCORE_KEY, 0.0))
    return row


def _build_temporal_candidates(
    *,
    meta: dict[str, Any],
    mask_probs: np.ndarray | None,
    center_probs: np.ndarray | None,
    selector_probs: np.ndarray | None,
    current_stop_lines: list[dict[str, Any]],
    gt_stop_lines: list[dict[str, Any]],
    neighbor_predictions: list[
        tuple[int, int, dict[str, Any]]
        | tuple[int, int, dict[str, Any], dict[str, float]]
        | tuple[int, int, dict[str, Any], dict[str, float], dict[str, Any]]
    ],
    max_candidates: int,
    temporal_envelope_enabled: bool = False,
    temporal_dense_component_enabled: bool = False,
    max_temporal_dense_components: int = 4,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for neighbor_payload in neighbor_predictions:
        offset, neighbor_dataset_index, neighbor_prediction = neighbor_payload[:3]
        alignment = dict(neighbor_payload[3]) if len(neighbor_payload) > 3 else {}
        neighbor_lines = list(neighbor_prediction.get("stop_lines", []))
        neighbor_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
        for neighbor_rank, line in enumerate(neighbor_lines, start=1):
            candidate = _copy_stop_line(line, score=_line_score(line), source="temporal_neighbor")
            if float(alignment.get("applied", 0.0)) > 0.0:
                candidate["points_xy"] = _warp_stop_line_points(
                    candidate,
                    alignment=alignment,
                    meta=meta,
                )
            candidate["neighbor_offset"] = int(offset)
            candidate["neighbor_dataset_index"] = int(neighbor_dataset_index)
            candidate["neighbor_rank"] = int(neighbor_rank)
            candidate["temporal_alignment_dx"] = float(alignment.get("dx", 0.0))
            candidate["temporal_alignment_dy"] = float(alignment.get("dy", 0.0))
            candidate["temporal_alignment_response_raw"] = float(alignment.get("response", 0.0))
            candidate["temporal_alignment_applied"] = float(alignment.get("applied", 0.0))
            candidate["length"] = _line_length(candidate)
            candidate.update(
                _stopline_temporal_features(
                    candidate,
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    neighbor_offset=int(offset),
                    neighbor_rank=int(neighbor_rank),
                    alignment=alignment,
                    current_stop_lines=current_stop_lines,
                )
            )
            nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
            candidate["nearest_gt_distance"] = float(nearest_distance)
            candidate["nearest_gt_angle_error"] = float(nearest_angle)
            candidate["nearest_gt_index"] = int(nearest_index)
            candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
            candidate["temporal_rank_score"] = (
                float(candidate.get("temporal_proposal_max", 0.0))
                + 0.5 * float(candidate.get("temporal_mask_mean", 0.0))
                + 0.25 * float(candidate.get("temporal_neighbor_score", 0.0))
                - 0.05 * float(abs(int(offset)))
            )
            candidates.append(candidate)
        if bool(temporal_dense_component_enabled) and len(neighbor_payload) > 4:
            dense_payload = dict(neighbor_payload[4])
            neighbor_meta = dict(dense_payload.get("meta", {}))
            dense_lines = _build_dense_component_stop_lines(
                meta=neighbor_meta,
                mask_probs=dense_payload.get("mask_probs"),
                center_probs=dense_payload.get("center_probs"),
                selector_probs=dense_payload.get("selector_probs"),
                max_candidates=int(max_temporal_dense_components),
                source="temporal_dense_component",
            )
            for dense_rank, dense_line in enumerate(dense_lines, start=1):
                candidate = _copy_stop_line(dense_line, score=_line_score(dense_line), source="temporal_dense_component")
                if float(alignment.get("applied", 0.0)) > 0.0:
                    candidate["points_xy"] = _warp_stop_line_points(
                        candidate,
                        alignment=alignment,
                        meta=meta,
                    )
                candidate["neighbor_offset"] = int(offset)
                candidate["neighbor_dataset_index"] = int(neighbor_dataset_index)
                candidate["neighbor_rank"] = int(dense_rank)
                candidate["temporal_alignment_dx"] = float(alignment.get("dx", 0.0))
                candidate["temporal_alignment_dy"] = float(alignment.get("dy", 0.0))
                candidate["temporal_alignment_response_raw"] = float(alignment.get("response", 0.0))
                candidate["temporal_alignment_applied"] = float(alignment.get("applied", 0.0))
                candidate["length"] = _line_length(candidate)
                candidate.update(
                    _stopline_temporal_features(
                        candidate,
                        meta=meta,
                        mask_probs=mask_probs,
                        center_probs=center_probs,
                        selector_probs=selector_probs,
                        neighbor_offset=int(offset),
                        neighbor_rank=int(dense_rank),
                        alignment=alignment,
                        current_stop_lines=current_stop_lines,
                    )
                )
                nearest_distance, nearest_angle, nearest_index = _nearest_gt(candidate, gt_stop_lines)
                candidate["nearest_gt_distance"] = float(nearest_distance)
                candidate["nearest_gt_angle_error"] = float(nearest_angle)
                candidate["nearest_gt_index"] = int(nearest_index)
                candidate["is_oracle_positive"] = bool(nearest_distance <= 40.0)
                candidate["temporal_rank_score"] = (
                    float(candidate.get("temporal_proposal_max", 0.0))
                    + 0.5 * float(candidate.get("temporal_mask_mean", 0.0))
                    + 0.25 * float(candidate.get("temporal_neighbor_score", 0.0))
                    + 0.10 * float(candidate.get("temporal_source_dense_component", 0.0))
                    - 0.05 * float(abs(int(offset)))
                )
                candidates.append(candidate)
    if bool(temporal_envelope_enabled) and current_stop_lines:
        envelope_candidates = _build_temporal_endpoint_envelopes(
            meta=meta,
            temporal_candidates=candidates,
            current_stop_lines=current_stop_lines,
            gt_stop_lines=gt_stop_lines,
            mask_probs=mask_probs,
            center_probs=center_probs,
            selector_probs=selector_probs,
        )
        candidates.extend(envelope_candidates)
    candidates.sort(
        key=lambda item: (
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("temporal_neighbor_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(max_candidates))]


def _collect_records(
    *,
    loader: Any,
    evaluator: Any,
    predictor: _StopLineNeighborPredictor,
    postprocess_config: Any,
    record_index_by_key: dict[tuple[str, str, str], int],
    max_batches: int,
    neighbor_offsets: tuple[int, ...],
    temporal_alignment_mode: str,
    temporal_alignment_size: tuple[int, int],
    temporal_alignment_max_shift_frac: float,
    max_temporal_candidates: int,
    temporal_envelope_enabled: bool,
    temporal_dense_component_enabled: bool,
    max_temporal_dense_components: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    union_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(max_batches):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[stopline_temporal] collect {split_name} batch {batch_index}/{max_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("temporal stop-line probe requires raw batches for metrics")
            gt_samples = _extract_gt_samples(raw_batch)
            encoded = evaluator.prepare_batch(batch)
            outputs = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta_rows = _detach_to_cpu(encoded["meta"])
            baseline_predictions = postprocess_pv26_batch(outputs, meta_rows, config=postprocess_config)
            for sample_index, (meta, baseline_prediction, gt_sample) in enumerate(
                zip(meta_rows, baseline_predictions, gt_samples)
            ):
                sample_id = str(meta.get("sample_id", ""))
                dataset_key = str(meta.get("dataset_key", ""))
                split = str(meta.get("split", ""))
                neighbor_payloads: list[tuple[int, int, dict[str, Any], dict[str, float], dict[str, Any]]] = []
                missing_neighbors = 0
                for offset in neighbor_offsets:
                    neighbor_id = _neighbor_sample_id(sample_id, int(offset))
                    if neighbor_id is None:
                        missing_neighbors += 1
                        continue
                    neighbor_index = record_index_by_key.get((dataset_key, split, neighbor_id))
                    if neighbor_index is None:
                        missing_neighbors += 1
                        continue
                    neighbor_payload = predictor.predict_index(neighbor_index)
                    alignment = _temporal_alignment(
                        current_meta=meta,
                        neighbor_meta=dict(neighbor_payload.get("meta", {})),
                        mode=str(temporal_alignment_mode),
                        size=temporal_alignment_size,
                        max_shift_frac=float(temporal_alignment_max_shift_frac),
                    )
                    dense_payload = {"meta": dict(neighbor_payload.get("meta", {}))}
                    dense_payload.update(dict(neighbor_payload.get("dense", {})))
                    neighbor_payloads.append(
                        (int(offset), int(neighbor_index), dict(neighbor_payload["prediction"]), alignment, dense_payload)
                    )
                mask_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_mask_logits", sample_index), sigmoid=True)
                center_probs = _as_2d_array(_sample_tensor(outputs, "stop_line_center_logits", sample_index), sigmoid=True)
                selector_probs = _as_2d_array(
                    _sample_tensor(outputs, "stop_line_selector_map_logits", sample_index),
                    sigmoid=True,
                )
                candidates = _build_temporal_candidates(
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    current_stop_lines=list(baseline_prediction.get("stop_lines", [])),
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    neighbor_predictions=neighbor_payloads,
                    max_candidates=int(max_temporal_candidates),
                    temporal_envelope_enabled=bool(temporal_envelope_enabled),
                    temporal_dense_component_enabled=bool(temporal_dense_component_enabled),
                    max_temporal_dense_components=int(max_temporal_dense_components),
                )
                baseline_candidates = _build_baseline_union_candidates(
                    meta=meta,
                    mask_probs=mask_probs,
                    center_probs=center_probs,
                    selector_probs=selector_probs,
                    baseline_prediction=baseline_prediction,
                )
                union_candidates = _build_union_candidates(
                    baseline_candidates=baseline_candidates,
                    temporal_candidates=candidates,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                )
                global_sample_index = int(len(records))
                retained_suppressor_examples = _retained_suppressor_examples(
                    baseline_prediction=baseline_prediction,
                    gt_stop_lines=list(gt_sample.get("stop_lines", [])),
                    outputs=outputs,
                    sample_index=int(sample_index),
                    global_sample_index=global_sample_index,
                    meta=meta,
                )
                candidate_rows = [
                    _candidate_row(candidate, batch_index=batch_index, sample_index=sample_index, meta=meta)
                    for candidate in candidates
                ]
                for row in candidate_rows:
                    row["split"] = str(split_name)
                rows.extend(candidate_rows)
                sample_union_rows = [
                    _union_candidate_row(
                        candidate,
                        batch_index=batch_index,
                        sample_index=sample_index,
                        meta=meta,
                        rank=rank,
                        candidate_count=len(union_candidates),
                    )
                    for rank, candidate in enumerate(union_candidates, start=1)
                ]
                for row in sample_union_rows:
                    row["split"] = str(split_name)
                union_rows.extend(sample_union_rows)
                records.append(
                    {
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": sample_id,
                        "raw_batch": _slice_raw_batch_sample(raw_batch, sample_index),
                        "baseline_prediction": dict(baseline_prediction),
                        "gt_stop_lines": list(gt_sample.get("stop_lines", [])),
                        "candidates": candidates,
                        "union_candidates": union_candidates,
                        "candidate_feature_rows": candidate_rows,
                        "union_candidate_rows": sample_union_rows,
                        "retained_suppressor_examples": retained_suppressor_examples,
                    }
                )
                sample_rows.append(
                    {
                        "split": str(split_name),
                        "batch_index": int(batch_index),
                        "sample_index": int(sample_index),
                        "sample_id": sample_id,
                        "dataset_key": dataset_key,
                        "temporal_neighbor_count": int(len(neighbor_payloads)),
                        "missing_neighbor_count": int(missing_neighbors),
                        "temporal_alignment_applied_count": int(
                            sum(1 for payload in neighbor_payloads if float(payload[3].get("applied", 0.0)) > 0.0)
                        ),
                        "baseline_stopline_count": int(len(list(baseline_prediction.get("stop_lines", [])))),
                        "gt_stopline_count": int(len(list(gt_sample.get("stop_lines", [])))),
                        "temporal_candidate_count": int(len(candidates)),
                        "temporal_envelope_candidate_count": int(
                            sum(
                                1
                                for candidate in candidates
                                if str(candidate.get("source", "")) == "temporal_endpoint_envelope"
                            )
                        ),
                        "temporal_dense_component_candidate_count": int(
                            sum(
                                1
                                for candidate in candidates
                                if str(candidate.get("source", "")) == "temporal_dense_component"
                            )
                        ),
                        "union_candidate_count": int(len(union_candidates)),
                        "union_oracle_positive_count": int(
                            sum(1 for candidate in union_candidates if bool(candidate.get("is_union_positive", False)))
                        ),
                        "temporal_oracle_positive_count": int(
                            sum(1 for candidate in candidates if bool(candidate.get("is_oracle_positive", False)))
                        ),
                        "temporal_dense_component_oracle_positive_count": int(
                            sum(
                                1
                                for candidate in candidates
                                if str(candidate.get("source", "")) == "temporal_dense_component"
                                and bool(candidate.get("is_oracle_positive", False))
                            )
                        ),
                    }
                )
    return records, rows, union_rows, sample_rows


def _feature_matrix(records: list[dict[str, Any]], *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for record in records:
        for rank, candidate in enumerate(record.get("candidates", []), start=1):
            if rank > int(top_k):
                continue
            features.append([float(candidate.get(name, 0.0)) for name in TEMPORAL_FEATURES])
            labels.append(float(bool(candidate.get("is_oracle_positive", False))))
    if not features:
        return np.zeros((0, len(TEMPORAL_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _union_feature_vector(candidate: dict[str, Any], *, rank: int, candidate_count: int) -> list[float]:
    source = str(candidate.get("source", ""))
    values = [float(candidate.get(name, 0.0)) for name in TEMPORAL_FEATURES]
    values.extend(
        [
            float(1.0 if source == "retained_projection_comp" else 0.0),
            float(1.0 if _is_temporal_source(source) else 0.0),
            float(1.0 / max(int(rank), 1)),
            float(min(float(candidate_count) / 16.0, 4.0)),
        ]
    )
    return [0.0 if not math.isfinite(float(value)) else float(value) for value in values]


def _union_feature_matrix(records: list[dict[str, Any]], *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for record in records:
        candidates = list(record.get("union_candidates", []))
        for rank, candidate in enumerate(candidates, start=1):
            if rank > int(top_k):
                continue
            features.append(_union_feature_vector(candidate, rank=rank, candidate_count=len(candidates)))
            labels.append(float(bool(candidate.get("is_union_positive", False))))
    if not features:
        return np.zeros((0, len(TEMPORAL_UNION_FEATURES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _attach_scores(records: list[dict[str, Any]], scores: np.ndarray, *, top_k: int) -> None:
    index = 0
    for record in records:
        for rank, (candidate, row) in enumerate(
            zip(record.get("candidates", []), record.get("candidate_feature_rows", [])),
            start=1,
        ):
            if rank > int(top_k):
                continue
            score = float(scores[index])
            candidate[SCORE_KEY] = score
            row[SCORE_KEY] = score
            index += 1
    if index != int(scores.shape[0]):
        raise ValueError(f"score length mismatch: attached {index}, got {int(scores.shape[0])}")


def _attach_union_scores(records: list[dict[str, Any]], scores: np.ndarray, *, top_k: int) -> None:
    index = 0
    for record in records:
        rows = list(record.get("union_candidate_rows", []))
        for rank, candidate in enumerate(record.get("union_candidates", []), start=1):
            if rank > int(top_k):
                continue
            score = float(scores[index])
            candidate[UNION_SCORE_KEY] = score
            if rank - 1 < len(rows):
                rows[rank - 1][UNION_SCORE_KEY] = score
            index += 1
    if index != int(scores.shape[0]):
        raise ValueError(f"union score length mismatch: attached {index}, got {int(scores.shape[0])}")


def _score_records(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    top_k: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    train_x, train_y = _feature_matrix(train_records, top_k=int(top_k))
    val_x, val_y = _feature_matrix(val_records, top_k=int(top_k))
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("temporal verifier requires non-empty train and validation candidates")
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
        "train_candidate_count": int(train_x.shape[0]),
        "train_positive_count": int(train_y.sum()),
        "val_candidate_count": int(val_x.shape[0]),
        "val_positive_count": int(val_y.sum()),
        "feature_dim": int(train_x.shape[1]),
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def _score_union_records(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    top_k: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    train_x, train_y = _union_feature_matrix(train_records, top_k=int(top_k))
    val_x, val_y = _union_feature_matrix(val_records, top_k=int(top_k))
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("temporal union selector requires non-empty train and validation candidates")
    combined_x = np.concatenate([train_x, val_x], axis=0)
    combined_std, mean, std = _standardize_from_train(train_x.astype(np.float64), combined_x.astype(np.float64))
    train_std = combined_std[: train_x.shape[0]].astype(np.float32)
    val_std = combined_std[train_x.shape[0] :].astype(np.float32)
    model = _fit_raw_patch_mlp(train_std, train_y.astype(np.float32), epochs=int(epochs), lr=float(lr))
    train_scores = _predict_raw_patch_mlp(model, train_std)
    val_scores = _predict_raw_patch_mlp(model, val_std)
    _attach_union_scores(train_records, train_scores, top_k=int(top_k))
    _attach_union_scores(val_records, val_scores, top_k=int(top_k))
    return {
        "train_candidate_count": int(train_x.shape[0]),
        "train_positive_count": int(train_y.sum()),
        "val_candidate_count": int(val_x.shape[0]),
        "val_positive_count": int(val_y.sum()),
        "feature_dim": int(train_x.shape[1]),
        "mean_dim": int(mean.shape[0]),
        "std_min": float(std.min()) if std.size else 0.0,
    }


def _select_temporal_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        if rank > int(top_k):
            continue
        if float(candidate.get(score_key, 0.0)) < float(threshold):
            continue
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            float(item.get(score_key, 0.0)),
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        _copy_stop_line(candidate, score=float(candidate.get(score_key, 0.0)), source=str(candidate.get("source", "temporal_neighbor")))
        for candidate in selected
    ]
    predictions.sort(key=_stopline_prediction_sort_key, reverse=True)
    predictions = _dedupe_stop_line_predictions(predictions)
    return predictions[: max(1, int(max_components))]


def _select_union_stop_lines(
    candidates: list[dict[str, Any]],
    *,
    threshold: float,
    top_k: int,
    max_components: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        if rank > int(top_k):
            continue
        if float(candidate.get(UNION_SCORE_KEY, 0.0)) < float(threshold):
            continue
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            float(item.get(UNION_SCORE_KEY, 0.0)),
            float(item.get("union_source_retained", 0.0)),
            float(item.get("temporal_rank_score", 0.0)),
            float(item.get("length", 0.0)),
        ),
        reverse=True,
    )
    predictions = [
        _copy_stop_line(candidate, score=float(candidate.get(UNION_SCORE_KEY, 0.0)), source=str(candidate.get("source", "union")))
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
            stop_lines = _select_temporal_stop_lines(
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


def _union_metrics_row(
    records: list[dict[str, Any]],
    *,
    name: str,
    split: str,
    threshold: float,
    top_k: int,
    max_components: int,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    for record in records:
        raw_batches.append(record["raw_batch"])
        stop_lines = _select_union_stop_lines(
            list(record.get("union_candidates", [])),
            threshold=float(threshold),
            top_k=int(top_k),
            max_components=int(max_components),
        )
        predictions.append({**record["baseline_prediction"], "stop_lines": stop_lines})
    merged_raw = _merge_raw_batches(raw_batches)
    metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
    prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
    row = _row_from_metrics(name, metrics, prediction_count=prediction_count, stats={})
    row.update(
        {
            "split": str(split),
            "score_key": UNION_SCORE_KEY,
            "threshold": float(threshold),
            "top_k": int(top_k),
            "union_baseline": 0,
            "sample_count": int(len(records)),
        }
    )
    return row


def _merge_stop_lines_with_extra(
    base_stop_lines: list[dict[str, Any]],
    extra_stop_lines: list[dict[str, Any]],
    *,
    max_components: int,
) -> list[dict[str, Any]]:
    stop_lines = [dict(line) for line in base_stop_lines] + [dict(line) for line in extra_stop_lines]
    stop_lines.sort(key=_stopline_prediction_sort_key, reverse=True)
    stop_lines = _dedupe_stop_line_predictions(stop_lines)
    return stop_lines[: max(1, int(max_components))]


def _metrics_row_with_base_predictions(
    records: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    *,
    name: str,
    split: str,
    score_key: str = "",
    threshold: float = 0.0,
    top_k: int = 0,
    max_components: int = 2,
) -> dict[str, Any]:
    if len(records) != len(base_predictions):
        raise ValueError("base prediction count must match record count")
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    for record, base_prediction in zip(records, base_predictions):
        raw_batches.append(record["raw_batch"])
        stop_lines = list(base_prediction.get("stop_lines", []))
        if score_key:
            extra_stop_lines = _select_temporal_stop_lines(
                list(record.get("candidates", [])),
                score_key=score_key,
                threshold=float(threshold),
                top_k=int(top_k),
                max_components=int(max_components),
            )
            stop_lines = _merge_stop_lines_with_extra(
                stop_lines,
                extra_stop_lines,
                max_components=int(max_components),
            )
        predictions.append({**record["baseline_prediction"], "stop_lines": stop_lines})
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
            "union_baseline": 1,
            "sample_count": int(len(records)),
        }
    )
    return row


def _flatten_retained_suppressor_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        examples.extend(dict(example) for example in record.get("retained_suppressor_examples", []))
    return examples


def _baseline_prediction_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record["baseline_prediction"]) for record in records]


def _best_threshold(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    max_components: int,
    grid_size: int,
) -> float:
    del max_components
    candidate_rows: list[tuple[float, int]] = []
    for record in records:
        for rank, candidate in enumerate(record.get("candidates", []), start=1):
            if rank > int(top_k):
                continue
            candidate_rows.append((float(candidate.get(score_key, 0.0)), int(bool(candidate.get("is_oracle_positive", False)))))
    if not candidate_rows:
        return 0.5
    total_positive = int(sum(label for _score, label in candidate_rows))
    if total_positive <= 0:
        return 1.0
    best_threshold = 0.5
    best_key: tuple[float, int, int] = (-1.0, -1, 0)
    for threshold in np.linspace(0.0, 1.0, max(2, int(grid_size))).tolist():
        tp = 0
        fp = 0
        for score, label in candidate_rows:
            if float(score) < float(threshold):
                continue
            if label:
                tp += 1
            else:
                fp += 1
        fn = max(0, total_positive - tp)
        f1 = 0.0 if tp <= 0 else (2.0 * float(tp)) / (2.0 * float(tp) + float(fp) + float(fn))
        key = (float(f1), int(tp), -int(fp))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return float(best_threshold)


def _best_union_threshold(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    grid_size: int,
) -> float:
    candidate_rows: list[tuple[float, int]] = []
    for record in records:
        for rank, candidate in enumerate(record.get("union_candidates", []), start=1):
            if rank > int(top_k):
                continue
            candidate_rows.append((float(candidate.get(UNION_SCORE_KEY, 0.0)), int(bool(candidate.get("is_union_positive", False)))))
    if not candidate_rows:
        return 0.5
    total_positive = int(sum(label for _score, label in candidate_rows))
    if total_positive <= 0:
        return 1.0
    best_threshold = 0.5
    best_key: tuple[float, int, int] = (-1.0, -1, 0)
    for threshold in np.linspace(0.0, 1.0, max(2, int(grid_size))).tolist():
        tp = 0
        fp = 0
        for score, label in candidate_rows:
            if float(score) < float(threshold):
                continue
            if label:
                tp += 1
            else:
                fp += 1
        fn = max(0, total_positive - tp)
        f1 = 0.0 if tp <= 0 else (2.0 * float(tp)) / (2.0 * float(tp) + float(fp) + float(fn))
        key = (float(f1), int(tp), -int(fp))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return float(best_threshold)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.checkpoint).strip():
        from tools.probe_pv26_lane_flip_tta import DEFAULT_CHECKPOINT

        args.checkpoint = str(DEFAULT_CHECKPOINT)
    if not str(args.source_run).strip():
        from tools.probe_pv26_lane_flip_tta import SOURCE_RUN

        args.source_run = str(SOURCE_RUN)
    neighbor_offsets = _parse_neighbor_offsets(str(args.neighbor_offsets))
    alignment_size = _parse_alignment_size(str(args.temporal_alignment_size))
    scenario, scenario_path, options, phase, train_config = _build_scenario(args)
    train_config = replace(
        train_config,
        encode_train_batches_in_loader=False,
        encode_val_batches_in_loader=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_temporal] {message}", flush=True),
    )
    train_loader, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if train_loader is None or val_loader is None:
        raise ValueError("temporal stop-line probe requires train and validation loaders")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))
    trainer = train_cli._build_phase_trainer(phase, train_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trainer.load_model_weights(checkpoint, map_location=train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()
    postprocess_config = _postprocess_override_config(args, trainer) or train_cli._build_postprocess_config(train_config)
    predictor = _StopLineNeighborPredictor(
        dataset=dataset,
        evaluator=evaluator,
        postprocess_config=postprocess_config,
    )
    record_index_by_key = {
        (str(record.dataset_key), str(record.split), str(record.sample_id)): int(index)
        for index, record in enumerate(dataset.records)
    }
    train_records, train_candidate_rows, train_union_candidate_rows, train_sample_rows = _collect_records(
        loader=train_loader,
        evaluator=evaluator,
        predictor=predictor,
        postprocess_config=postprocess_config,
        record_index_by_key=record_index_by_key,
        max_batches=int(args.train_record_batches),
        neighbor_offsets=neighbor_offsets,
        temporal_alignment_mode=str(args.temporal_alignment_mode),
        temporal_alignment_size=alignment_size,
        temporal_alignment_max_shift_frac=float(args.temporal_alignment_max_shift_frac),
        max_temporal_candidates=int(args.max_temporal_candidates),
        temporal_envelope_enabled=bool(int(args.temporal_envelope_enabled)),
        temporal_dense_component_enabled=bool(int(args.temporal_dense_component_enabled)),
        max_temporal_dense_components=int(args.max_temporal_dense_components),
        split_name="train",
    )
    val_records, val_candidate_rows, val_union_candidate_rows, val_sample_rows = _collect_records(
        loader=val_loader,
        evaluator=evaluator,
        predictor=predictor,
        postprocess_config=postprocess_config,
        record_index_by_key=record_index_by_key,
        max_batches=int(args.max_val_batches),
        neighbor_offsets=neighbor_offsets,
        temporal_alignment_mode=str(args.temporal_alignment_mode),
        temporal_alignment_size=alignment_size,
        temporal_alignment_max_shift_frac=float(args.temporal_alignment_max_shift_frac),
        max_temporal_candidates=int(args.max_temporal_candidates),
        temporal_envelope_enabled=bool(int(args.temporal_envelope_enabled)),
        temporal_dense_component_enabled=bool(int(args.temporal_dense_component_enabled)),
        max_temporal_dense_components=int(args.max_temporal_dense_components),
        split_name="val",
    )
    score_summary = _score_records(
        train_records=train_records,
        val_records=val_records,
        top_k=int(args.temporal_top_k),
        epochs=int(args.verifier_epochs),
        lr=float(args.verifier_lr),
    )
    threshold = _best_threshold(
        train_records,
        score_key=SCORE_KEY,
        top_k=int(args.temporal_top_k),
        max_components=int(args.max_components),
        grid_size=int(args.threshold_grid),
    )
    union_score_summary: dict[str, Any] = {}
    union_threshold: float | None = None
    if int(args.union_selector_enabled) == 1:
        union_score_summary = _score_union_records(
            train_records=train_records,
            val_records=val_records,
            top_k=int(args.union_top_k),
            epochs=int(args.verifier_epochs),
            lr=float(args.verifier_lr),
        )
        union_threshold = _best_union_threshold(
            train_records,
            top_k=int(args.union_top_k),
            grid_size=int(args.threshold_grid),
        )
    retained_suppressor_summary: dict[str, Any] = {}
    retained_suppressor_decision_audit: dict[str, int] = {}
    retained_suppressed_val_predictions: list[dict[str, Any]] = []
    retained_suppressor_decision_rows: list[dict[str, Any]] = []
    if int(args.retained_suppressor_enabled) == 1:
        suppressor_args = _retained_suppressor_args(args)
        train_retained_examples = _flatten_retained_suppressor_examples(train_records)
        val_retained_examples = _flatten_retained_suppressor_examples(val_records)
        retained_suppressor_model, retained_suppressor_summary = _train_retained_suppressor(
            train_retained_examples,
            args=suppressor_args,
            device=str(train_config.device),
        )
        retained_suppressed_val_predictions, retained_suppressor_decision_rows = _apply_retained_suppressor(
            examples=val_retained_examples,
            baseline_predictions=_baseline_prediction_list(val_records),
            model=retained_suppressor_model,
            args=suppressor_args,
            device=str(train_config.device),
        )
        for row in retained_suppressor_decision_rows:
            row["split"] = "val"
        retained_suppressor_decision_audit = _retained_suppressor_decision_audit(retained_suppressor_decision_rows)
    rows = [
        _metrics_row(train_records, name="baseline", split="train"),
        _metrics_row(val_records, name="baseline", split="val"),
        _metrics_row(
            train_records,
            name="temporal_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="temporal_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            train_records,
            name="baseline_plus_temporal_mlp",
            split="train",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_temporal_mlp",
            split="val",
            score_key=SCORE_KEY,
            threshold=threshold,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
        _metrics_row(
            val_records,
            name="oracle_temporal",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
        ),
        _metrics_row(
            val_records,
            name="baseline_plus_oracle_temporal",
            split="val",
            score_key="is_oracle_positive",
            threshold=0.5,
            top_k=int(args.temporal_top_k),
            max_components=int(args.max_components),
            union_baseline=True,
        ),
    ]
    if int(args.retained_suppressor_enabled) == 1:
        rows.extend(
            [
                _metrics_row_with_base_predictions(
                    val_records,
                    retained_suppressed_val_predictions,
                    name="retained_suppressed_baseline",
                    split="val",
                    max_components=int(args.max_components),
                ),
                _metrics_row_with_base_predictions(
                    val_records,
                    retained_suppressed_val_predictions,
                    name="retained_suppressed_plus_temporal_mlp",
                    split="val",
                    score_key=SCORE_KEY,
                    threshold=float(threshold),
                    top_k=int(args.temporal_top_k),
                    max_components=int(args.max_components),
                ),
                _metrics_row_with_base_predictions(
                    val_records,
                    retained_suppressed_val_predictions,
                    name="retained_suppressed_plus_oracle_temporal",
                    split="val",
                    score_key="is_oracle_positive",
                    threshold=0.5,
                    top_k=int(args.temporal_top_k),
                    max_components=int(args.max_components),
                ),
            ]
        )
    if int(args.union_selector_enabled) == 1 and union_threshold is not None:
        rows.extend(
            [
                _union_metrics_row(
                    train_records,
                    name="temporal_union_selector",
                    split="train",
                    threshold=float(union_threshold),
                    top_k=int(args.union_top_k),
                    max_components=int(args.max_components),
                ),
                _union_metrics_row(
                    val_records,
                    name="temporal_union_selector",
                    split="val",
                    threshold=float(union_threshold),
                    top_k=int(args.union_top_k),
                    max_components=int(args.max_components),
                ),
            ]
        )
    summary = {
        "checkpoint": str(checkpoint),
        "source_run": str(Path(args.source_run).expanduser().resolve()),
        "scenario_path": str(scenario_path),
        "lane60_experiment": str(args.lane60_experiment),
        "phase_index": int(tuple(options["selected_phase_indices"])[0]),
        "train_record_batches": int(args.train_record_batches),
        "max_val_batches": int(args.max_val_batches),
        "validation_epoch": int(args.validation_epoch),
        "neighbor_offsets": list(neighbor_offsets),
        "temporal_alignment_mode": str(args.temporal_alignment_mode),
        "temporal_alignment_size": list(alignment_size),
        "temporal_alignment_max_shift_frac": float(args.temporal_alignment_max_shift_frac),
        "temporal_top_k": int(args.temporal_top_k),
        "max_temporal_candidates": int(args.max_temporal_candidates),
        "temporal_envelope_enabled": int(args.temporal_envelope_enabled),
        "temporal_dense_component_enabled": int(args.temporal_dense_component_enabled),
        "max_temporal_dense_components": int(args.max_temporal_dense_components),
        "max_components": int(args.max_components),
        "union_selector_enabled": int(args.union_selector_enabled),
        "union_top_k": int(args.union_top_k),
        "retained_suppressor_enabled": int(args.retained_suppressor_enabled),
        "retained_suppressor_keep_threshold": float(args.retained_suppressor_keep_threshold),
        "retained_suppressor_summary": retained_suppressor_summary,
        "retained_suppressor_decision_audit": retained_suppressor_decision_audit,
        "threshold": float(threshold),
        "threshold_selection": "train_candidate_oracle_label_f1",
        "union_threshold": None if union_threshold is None else float(union_threshold),
        "union_threshold_selection": None if union_threshold is None else "train_union_candidate_label_f1",
        "score_summary": score_summary,
        "union_score_summary": union_score_summary,
        "train_sample_count": int(len(train_records)),
        "val_sample_count": int(len(val_records)),
        "train_temporal_candidate_count": int(len(train_candidate_rows)),
        "val_temporal_candidate_count": int(len(val_candidate_rows)),
        "train_union_candidate_count": int(len(train_union_candidate_rows)),
        "val_union_candidate_count": int(len(val_union_candidate_rows)),
        "train_temporal_oracle_positive_count": int(
            sum(int(row.get("is_oracle_positive", 0)) for row in train_candidate_rows)
        ),
        "val_temporal_oracle_positive_count": int(sum(int(row.get("is_oracle_positive", 0)) for row in val_candidate_rows)),
        "train_temporal_envelope_candidate_count": int(
            sum(1 for row in train_candidate_rows if str(row.get("source", "")) == "temporal_endpoint_envelope")
        ),
        "val_temporal_envelope_candidate_count": int(
            sum(1 for row in val_candidate_rows if str(row.get("source", "")) == "temporal_endpoint_envelope")
        ),
        "train_temporal_envelope_oracle_positive_count": int(
            sum(
                int(row.get("is_oracle_positive", 0))
                for row in train_candidate_rows
                if str(row.get("source", "")) == "temporal_endpoint_envelope"
            )
        ),
        "val_temporal_envelope_oracle_positive_count": int(
            sum(
                int(row.get("is_oracle_positive", 0))
                for row in val_candidate_rows
                if str(row.get("source", "")) == "temporal_endpoint_envelope"
            )
        ),
        "train_temporal_dense_component_candidate_count": int(
            sum(1 for row in train_candidate_rows if str(row.get("source", "")) == "temporal_dense_component")
        ),
        "val_temporal_dense_component_candidate_count": int(
            sum(1 for row in val_candidate_rows if str(row.get("source", "")) == "temporal_dense_component")
        ),
        "train_temporal_dense_component_oracle_positive_count": int(
            sum(
                int(row.get("is_oracle_positive", 0))
                for row in train_candidate_rows
                if str(row.get("source", "")) == "temporal_dense_component"
            )
        ),
        "val_temporal_dense_component_oracle_positive_count": int(
            sum(
                int(row.get("is_oracle_positive", 0))
                for row in val_candidate_rows
                if str(row.get("source", "")) == "temporal_dense_component"
            )
        ),
        "train_union_oracle_positive_count": int(
            sum(int(row.get("is_union_positive", 0)) for row in train_union_candidate_rows)
        ),
        "val_union_oracle_positive_count": int(
            sum(int(row.get("is_union_positive", 0)) for row in val_union_candidate_rows)
        ),
        "train_temporal_alignment_applied_count": int(
            sum(1 for row in train_candidate_rows if float(row.get("temporal_alignment_applied", 0.0)) > 0.0)
        ),
        "val_temporal_alignment_applied_count": int(
            sum(1 for row in val_candidate_rows if float(row.get("temporal_alignment_applied", 0.0)) > 0.0)
        ),
        "rows": rows,
        "interpretation": (
            "Temporal stop-line candidate probe. Runtime candidates come from neighboring frame predictions "
            "and current-frame dense stop-line map features; the optional endpoint-envelope source creates "
            "a new along-axis extent candidate when retained and aligned neighbor lines support one axis, "
            "and the optional dense-component source extracts stop-line segments directly from neighboring "
            "dense stop-line maps before temporal alignment. "
            "GT is used for train labels, oracle diagnostics, "
            "and final metrics only. The opt-in union selector trains a candidate-level FP-control verifier "
            "over retained projection-comp plus temporal candidates, then emits one fixed-size selected set."
        ),
    }
    return {
        "summary": summary,
        "rows": rows,
        "train_candidate_rows": train_candidate_rows,
        "val_candidate_rows": val_candidate_rows,
        "train_union_candidate_rows": train_union_candidate_rows,
        "val_union_candidate_rows": val_union_candidate_rows,
        "retained_suppressor_decision_rows": retained_suppressor_decision_rows,
        "sample_rows": [*train_sample_rows, *val_sample_rows],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(args)
    _write_csv(output_dir / "temporal_variants.csv", payload["rows"])
    _write_candidate_features_csv(output_dir / "train_candidate_features.csv", payload["train_candidate_rows"])
    _write_candidate_features_csv(output_dir / "val_candidate_features.csv", payload["val_candidate_rows"])
    if payload.get("train_union_candidate_rows") or payload.get("val_union_candidate_rows"):
        _write_candidate_features_csv(output_dir / "train_union_candidate_features.csv", payload["train_union_candidate_rows"])
        _write_candidate_features_csv(output_dir / "val_union_candidate_features.csv", payload["val_union_candidate_rows"])
    if payload.get("retained_suppressor_decision_rows"):
        _write_csv(output_dir / "retained_suppressor_decisions.csv", payload["retained_suppressor_decision_rows"])
    _write_csv(output_dir / "temporal_samples.csv", payload["sample_rows"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
