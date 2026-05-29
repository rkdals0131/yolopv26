from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_NETWORK_HW = (608, 800)
NETWORK_HW_ENV = "PV26_NETWORK_HW"


def _parse_network_hw_env(value: str | None) -> tuple[int, int]:
    if value is None or value.strip() == "":
        return DEFAULT_NETWORK_HW
    normalized = value.lower().replace("x", ",").replace(":", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{NETWORK_HW_ENV} must be formatted as '<height>,<width>' or '<height>x<width>'")
    height, width = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"{NETWORK_HW_ENV} dimensions must be positive")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"{NETWORK_HW_ENV} dimensions must be divisible by 32 for the PV26 feature pyramid")
    return (height, width)


NETWORK_HW = _parse_network_hw_env(os.environ.get(NETWORK_HW_ENV))
PADDING_FILL_UINT8 = 114


@dataclass(frozen=True)
class LetterboxTransform:
    raw_hw: tuple[int, int]
    network_hw: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    resized_hw: tuple[int, int]

    def as_meta(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("raw_hw", None)
        payload.pop("network_hw", None)
        return payload


@dataclass(frozen=True)
class TrainAugmentationConfig:
    horizontal_flip_prob: float = 0.50
    brightness_delta: float = 0.10
    contrast_range: tuple[float, float] = (0.90, 1.10)
    gamma_range: tuple[float, float] = (0.95, 1.05)
    affine_prob: float = 0.0
    affine_degrees: float = 0.0
    affine_translate_frac: float = 0.0
    affine_scale_range: tuple[float, float] = (1.0, 1.0)
    affine_shear_degrees: float = 0.0
    synthetic_stopline_prob: float = 0.0
    synthetic_stopline_thickness_px: float = 5.0
    stopline_focus_crop_prob: float = 0.0
    stopline_focus_crop_scale_range: tuple[float, float] = (1.25, 1.75)
    stopline_focus_crop_jitter: float = 0.10


def compute_letterbox_transform(
    raw_hw: tuple[int, int],
    network_hw: tuple[int, int] = NETWORK_HW,
) -> LetterboxTransform:
    raw_h, raw_w = raw_hw
    net_h, net_w = network_hw
    scale = min(net_w / raw_w, net_h / raw_h)
    resized_w = int(round(raw_w * scale))
    resized_h = int(round(raw_h * scale))
    pad_w = net_w - resized_w
    pad_h = net_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return LetterboxTransform(
        raw_hw=raw_hw,
        network_hw=network_hw,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        resized_hw=(resized_h, resized_w),
    )


def load_letterboxed_image(path: Path, transform: LetterboxTransform) -> torch.FloatTensor:
    with Image.open(path) as raw_image:
        image = raw_image.convert("RGB")
    resized = image.resize((transform.resized_hw[1], transform.resized_hw[0]), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (transform.network_hw[1], transform.network_hw[0]), (PADDING_FILL_UINT8,) * 3)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def transform_from_meta(meta: dict[str, object]) -> LetterboxTransform:
    raw_hw = tuple(int(value) for value in meta["raw_hw"])
    network_hw = tuple(int(value) for value in meta["network_hw"])
    payload = dict(meta["transform"])
    resized_hw = tuple(int(value) for value in payload["resized_hw"])
    return LetterboxTransform(
        raw_hw=raw_hw,
        network_hw=network_hw,
        scale=float(payload["scale"]),
        pad_left=int(payload["pad_left"]),
        pad_top=int(payload["pad_top"]),
        pad_right=int(payload["pad_right"]),
        pad_bottom=int(payload["pad_bottom"]),
        resized_hw=resized_hw,
    )


def transform_box_xyxy(box: Iterable[float], transform: LetterboxTransform) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        x1 * transform.scale + transform.pad_left,
        y1 * transform.scale + transform.pad_top,
        x2 * transform.scale + transform.pad_left,
        y2 * transform.scale + transform.pad_top,
    ]


def clip_box_xyxy(box: Iterable[float], network_hw: tuple[int, int] = NETWORK_HW) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    net_h, net_w = network_hw
    x1 = max(0.0, min(x1, net_w - 1.0))
    y1 = max(0.0, min(y1, net_h - 1.0))
    x2 = max(0.0, min(x2, net_w - 1.0))
    y2 = max(0.0, min(y2, net_h - 1.0))
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return [x1, y1, x2, y2]


def inverse_transform_box_xyxy(box: Iterable[float], transform: LetterboxTransform) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    raw_box = [
        (x1 - transform.pad_left) / transform.scale,
        (y1 - transform.pad_top) / transform.scale,
        (x2 - transform.pad_left) / transform.scale,
        (y2 - transform.pad_top) / transform.scale,
    ]
    return clip_box_xyxy(raw_box, transform.raw_hw)


def transform_points(points: Iterable[Iterable[float]], transform: LetterboxTransform) -> list[list[float]]:
    transformed: list[list[float]] = []
    for point in points:
        x, y = [float(value) for value in point]
        transformed.append(
            [
                x * transform.scale + transform.pad_left,
                y * transform.scale + transform.pad_top,
            ]
        )
    return transformed


def clip_points(points: Iterable[Iterable[float]], network_hw: tuple[int, int] = NETWORK_HW) -> list[list[float]]:
    net_h, net_w = network_hw
    clipped: list[list[float]] = []
    for point in points:
        x, y = [float(value) for value in point]
        clipped.append(
            [
                max(0.0, min(x, net_w - 1.0)),
                max(0.0, min(y, net_h - 1.0)),
            ]
        )
    return clipped


def inverse_transform_points(points: Iterable[Iterable[float]], transform: LetterboxTransform) -> list[list[float]]:
    restored: list[list[float]] = []
    for point in points:
        x, y = [float(value) for value in point]
        restored.append(
            [
                (x - transform.pad_left) / transform.scale,
                (y - transform.pad_top) / transform.scale,
            ]
        )
    return clip_points(restored, transform.raw_hw)


def unique_point_count(points: Iterable[Iterable[float]]) -> int:
    return len({(float(point[0]), float(point[1])) for point in points})


def _clone_geometry_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    cloned: list[dict[str, object]] = []
    for row in rows:
        copied: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                copied[key] = value.clone()
            else:
                copied[key] = value
        cloned.append(copied)
    return cloned


def _flip_box_xyxy(box: Iterable[float], network_hw: tuple[int, int]) -> list[float]:
    _, net_w = network_hw
    x1, y1, x2, y2 = [float(value) for value in box]
    max_x = float(net_w - 1)
    return [
        max(0.0, max_x - x2),
        y1,
        max(0.0, max_x - x1),
        y2,
    ]


def _flip_points_tensor(points: torch.Tensor, network_hw: tuple[int, int]) -> torch.Tensor:
    _, net_w = network_hw
    flipped = points.clone()
    flipped[..., 0] = float(net_w - 1) - flipped[..., 0]
    return flipped


def _crop_zoom_points_tensor(
    points: torch.Tensor,
    *,
    crop_left: float,
    crop_top: float,
    scale_x: float,
    scale_y: float,
    network_hw: tuple[int, int],
) -> torch.Tensor:
    transformed = points.clone()
    transformed[..., 0] = (transformed[..., 0] - float(crop_left)) * float(scale_x)
    transformed[..., 1] = (transformed[..., 1] - float(crop_top)) * float(scale_y)
    net_h, net_w = network_hw
    transformed[..., 0] = transformed[..., 0].clamp(0.0, float(net_w - 1))
    transformed[..., 1] = transformed[..., 1].clamp(0.0, float(net_h - 1))
    return transformed


def _crop_zoom_box_xyxy(
    box: Iterable[float],
    *,
    crop_left: float,
    crop_top: float,
    scale_x: float,
    scale_y: float,
    network_hw: tuple[int, int],
) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    transformed = [
        (x1 - float(crop_left)) * float(scale_x),
        (y1 - float(crop_top)) * float(scale_y),
        (x2 - float(crop_left)) * float(scale_x),
        (y2 - float(crop_top)) * float(scale_y),
    ]
    return clip_box_xyxy(transformed, network_hw)


def _affine_matrix(
    *,
    network_hw: tuple[int, int],
    degrees: float,
    translate_xy: tuple[float, float],
    scale: float,
    shear_degrees: float,
) -> torch.Tensor:
    net_h, net_w = network_hw
    center_x = (float(net_w) - 1.0) * 0.5
    center_y = (float(net_h) - 1.0) * 0.5
    angle = math.radians(float(degrees))
    shear = math.radians(float(shear_degrees))
    cos_a = math.cos(angle) * float(scale)
    sin_a = math.sin(angle) * float(scale)
    shear_tan = math.tan(shear)
    translate_x, translate_y = translate_xy

    to_origin = torch.tensor(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotate_scale = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    shear_x = torch.tensor(
        [[1.0, shear_tan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    from_origin = torch.tensor(
        [[1.0, 0.0, center_x + float(translate_x)], [0.0, 1.0, center_y + float(translate_y)], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return from_origin @ shear_x @ rotate_scale @ to_origin


def _pixel_affine_to_grid_theta(matrix: torch.Tensor, *, network_hw: tuple[int, int]) -> torch.Tensor:
    net_h, net_w = network_hw
    inv_matrix = torch.linalg.inv(matrix.to(dtype=torch.float32))
    pixel_to_norm = torch.tensor(
        [
            [2.0 / max(float(net_w - 1), 1.0), 0.0, -1.0],
            [0.0, 2.0 / max(float(net_h - 1), 1.0), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    norm_to_pixel = torch.tensor(
        [
            [max(float(net_w - 1), 1.0) * 0.5, 0.0, max(float(net_w - 1), 1.0) * 0.5],
            [0.0, max(float(net_h - 1), 1.0) * 0.5, max(float(net_h - 1), 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return (pixel_to_norm @ inv_matrix @ norm_to_pixel)[:2]


def _affine_points_tensor(
    points: torch.Tensor,
    *,
    matrix: torch.Tensor,
    network_hw: tuple[int, int],
) -> torch.Tensor:
    transformed = points.clone().to(dtype=torch.float32)
    flat = transformed.reshape(-1, 2)
    ones = torch.ones((flat.shape[0], 1), dtype=flat.dtype, device=flat.device)
    matrix = matrix.to(device=flat.device, dtype=flat.dtype)
    warped = torch.cat([flat, ones], dim=1) @ matrix.T
    net_h, net_w = network_hw
    warped_xy = warped[:, :2]
    warped_xy[:, 0] = warped_xy[:, 0].clamp(0.0, float(net_w - 1))
    warped_xy[:, 1] = warped_xy[:, 1].clamp(0.0, float(net_h - 1))
    return warped_xy.reshape_as(transformed).to(dtype=points.dtype)


def _affine_box_xyxy(
    box: Iterable[float],
    *,
    matrix: torch.Tensor,
    network_hw: tuple[int, int],
) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    corners = torch.tensor(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=torch.float32,
    )
    transformed = _affine_points_tensor(corners, matrix=matrix, network_hw=network_hw)
    min_xy = transformed.min(dim=0).values
    max_xy = transformed.max(dim=0).values
    return clip_box_xyxy(
        [float(min_xy[0].item()), float(min_xy[1].item()), float(max_xy[0].item()), float(max_xy[1].item())],
        network_hw,
    )


def _lane_x_at_y(points: torch.Tensor, y: float) -> float | None:
    reshaped = points.to(dtype=torch.float32).reshape(-1, 2)
    if reshaped.shape[0] < 2 or not torch.isfinite(reshaped).all():
        return None
    best_x: float | None = None
    best_distance = float("inf")
    target_y = float(y)
    for start, end in zip(reshaped[:-1], reshaped[1:]):
        x0, y0 = float(start[0].item()), float(start[1].item())
        x1, y1 = float(end[0].item()), float(end[1].item())
        dy = y1 - y0
        if abs(dy) > 1.0e-4:
            t = (target_y - y0) / dy
            if 0.0 <= t <= 1.0:
                return x0 + t * (x1 - x0)
        distance = min(abs(target_y - y0), abs(target_y - y1))
        if distance < best_distance:
            best_distance = distance
            best_x = x0 if abs(target_y - y0) <= abs(target_y - y1) else x1
    if best_distance <= 16.0:
        return best_x
    return None


def _synthetic_stopline_points(
    lanes: list[dict[str, object]],
    *,
    network_hw: tuple[int, int],
    rng: random.Random,
) -> torch.Tensor | None:
    net_h, net_w = network_hw
    if net_h <= 2 or net_w <= 2:
        return None
    min_span = min(max(4.0, 0.07 * float(net_w)), 0.35 * float(net_w))
    max_span = max(min_span, 0.45 * float(net_w))
    for _ in range(8):
        y = rng.uniform(0.45 * float(net_h), 0.88 * float(net_h))
        xs: list[float] = []
        for row in lanes:
            points = _row_points(row)
            if points is None:
                continue
            x = _lane_x_at_y(points, y)
            if x is None or x < 2.0 or x > float(net_w - 3):
                continue
            if all(abs(x - existing) > 4.0 for existing in xs):
                xs.append(float(x))
        if len(xs) < 2:
            continue
        xs.sort()
        pairs = [(left, right, right - left) for left, right in zip(xs[:-1], xs[1:]) if right - left >= min_span]
        if not pairs:
            continue
        left, right, span = min(pairs, key=lambda pair: abs(pair[2] - 0.22 * float(net_w)))
        if span > max_span:
            mid_x = 0.5 * (left + right)
            half = 0.5 * max_span
            left = mid_x - half
            right = mid_x + half
            span = max_span
        pad = min(0.12 * span, 0.035 * float(net_w))
        left = max(0.0, left - pad)
        right = min(float(net_w - 1), right + pad)
        if right - left < min_span:
            continue
        slope = rng.uniform(-0.018, 0.018) * (right - left)
        y_left = max(0.0, min(float(net_h - 1), y - 0.5 * slope))
        y_right = max(0.0, min(float(net_h - 1), y + 0.5 * slope))
        return torch.tensor([[left, y_left], [right, y_right]], dtype=torch.float32)
    return None


def _draw_synthetic_stopline(
    image: torch.FloatTensor,
    points: torch.Tensor,
    *,
    thickness_px: float,
) -> torch.FloatTensor:
    if points.shape != (2, 2):
        return image
    _, height, width = image.shape
    start = points[0].to(device=image.device, dtype=image.dtype)
    end = points[1].to(device=image.device, dtype=image.dtype)
    segment = end - start
    length_sq = float(torch.dot(segment, segment).item())
    if length_sq <= 1.0:
        return image
    yy = torch.arange(height, device=image.device, dtype=image.dtype).view(height, 1)
    xx = torch.arange(width, device=image.device, dtype=image.dtype).view(1, width)
    rel_x = xx - start[0]
    rel_y = yy - start[1]
    t = ((rel_x * segment[0] + rel_y * segment[1]) / length_sq).clamp(0.0, 1.0)
    nearest_x = start[0] + t * segment[0]
    nearest_y = start[1] + t * segment[1]
    radius = max(1.0, float(thickness_px) * 0.5)
    mask = (xx - nearest_x).square() + (yy - nearest_y).square() <= radius * radius
    if not bool(mask.any().item()):
        return image
    painted = image.clone()
    opacity = 0.78
    intensity = 0.92
    painted[:, mask] = painted[:, mask] * (1.0 - opacity) + intensity * opacity
    return painted.clamp(0.0, 1.0).contiguous()


def _apply_synthetic_stopline(
    image: torch.FloatTensor,
    *,
    lanes: list[dict[str, object]],
    stop_lines: list[dict[str, object]],
    network_hw: tuple[int, int],
    config: TrainAugmentationConfig,
    rng: random.Random,
) -> tuple[torch.FloatTensor, list[dict[str, object]], dict[str, object] | None]:
    if stop_lines or rng.random() >= float(config.synthetic_stopline_prob):
        return image, stop_lines, None
    points = _synthetic_stopline_points(lanes, network_hw=network_hw, rng=rng)
    if points is None:
        return image, stop_lines, None
    thickness = max(1.0, float(config.synthetic_stopline_thickness_px))
    rendered = _draw_synthetic_stopline(image, points, thickness_px=thickness)
    augmented_stop_lines = [*stop_lines, {"points_xy": points, "synthetic": True}]
    meta_points = [
        [float(points[0, 0].item()), float(points[0, 1].item())],
        [float(points[1, 0].item()), float(points[1, 1].item())],
    ]
    return (
        rendered,
        augmented_stop_lines,
        {
            "applied": True,
            "points_xy": meta_points,
            "thickness_px": float(thickness),
        },
    )


def _apply_shared_affine(
    image: torch.FloatTensor,
    *,
    det_boxes: list[list[float]],
    lanes: list[dict[str, object]],
    stop_lines: list[dict[str, object]],
    crosswalks: list[dict[str, object]],
    network_hw: tuple[int, int],
    config: TrainAugmentationConfig,
    rng: random.Random,
) -> tuple[
    torch.FloatTensor,
    list[list[float]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
]:
    if rng.random() >= float(config.affine_prob):
        return image, det_boxes, lanes, stop_lines, crosswalks, None
    degrees = float(config.affine_degrees)
    translate_frac = max(0.0, float(config.affine_translate_frac))
    scale_min, scale_max = sorted(float(value) for value in config.affine_scale_range)
    scale_min = max(scale_min, 1.0e-3)
    scale_max = max(scale_min, scale_max)
    shear_degrees = float(config.affine_shear_degrees)

    net_h, net_w = network_hw
    sampled_degrees = rng.uniform(-degrees, degrees) if degrees > 0.0 else 0.0
    sampled_scale = rng.uniform(scale_min, scale_max)
    sampled_shear = rng.uniform(-shear_degrees, shear_degrees) if shear_degrees > 0.0 else 0.0
    sampled_tx = rng.uniform(-translate_frac, translate_frac) * float(net_w)
    sampled_ty = rng.uniform(-translate_frac, translate_frac) * float(net_h)
    matrix = _affine_matrix(
        network_hw=network_hw,
        degrees=sampled_degrees,
        translate_xy=(sampled_tx, sampled_ty),
        scale=sampled_scale,
        shear_degrees=sampled_shear,
    )
    theta = _pixel_affine_to_grid_theta(matrix, network_hw=network_hw).to(device=image.device, dtype=image.dtype)
    grid = F.affine_grid(
        theta.unsqueeze(0),
        size=(1, int(image.shape[0]), int(net_h), int(net_w)),
        align_corners=True,
    )
    fill = float(PADDING_FILL_UINT8) / 255.0
    warped = F.grid_sample(
        (image.unsqueeze(0) - fill),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0) + fill
    warped = warped.clamp(0.0, 1.0).contiguous()

    transformed_det: list[list[float]] = []
    for box in det_boxes:
        transformed_box = _affine_box_xyxy(box, matrix=matrix, network_hw=network_hw)
        if transformed_box is not None:
            transformed_det.append(transformed_box)
    for rows in (lanes, stop_lines, crosswalks):
        for row in rows:
            points = _row_points(row)
            if points is None:
                continue
            row["points_xy"] = _affine_points_tensor(points, matrix=matrix, network_hw=network_hw)

    return (
        warped,
        transformed_det,
        lanes,
        stop_lines,
        crosswalks,
        {
            "applied": True,
            "degrees": float(sampled_degrees),
            "translate": [float(sampled_tx), float(sampled_ty)],
            "scale": float(sampled_scale),
            "shear_degrees": float(sampled_shear),
        },
    )


def _row_points(row: dict[str, object]) -> torch.Tensor | None:
    points_xy = row.get("points_xy")
    if not isinstance(points_xy, torch.Tensor) or points_xy.numel() < 2:
        return None
    return points_xy


def _stopline_focus_center(stop_lines: list[dict[str, object]], *, rng: random.Random) -> tuple[float, float] | None:
    candidates: list[torch.Tensor] = []
    for row in stop_lines:
        points = _row_points(row)
        if points is None:
            continue
        reshaped = points.to(dtype=torch.float32).reshape(-1, 2)
        if int(reshaped.shape[0]) >= 2 and torch.isfinite(reshaped).all():
            candidates.append(reshaped)
    if not candidates:
        return None
    points = candidates[rng.randrange(len(candidates))]
    center = points.mean(dim=0)
    return float(center[0].item()), float(center[1].item())


def _apply_stopline_focus_crop(
    image: torch.FloatTensor,
    *,
    det_boxes: list[list[float]],
    lanes: list[dict[str, object]],
    stop_lines: list[dict[str, object]],
    crosswalks: list[dict[str, object]],
    network_hw: tuple[int, int],
    config: TrainAugmentationConfig,
    rng: random.Random,
) -> tuple[
    torch.FloatTensor,
    list[list[float]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
]:
    if rng.random() >= float(config.stopline_focus_crop_prob):
        return image, det_boxes, lanes, stop_lines, crosswalks, None
    if det_boxes:
        # This augmentation can drop boxes, while class rows are owned by the
        # dataset loader. Keep detector-supervised samples on the stable path.
        return image, det_boxes, lanes, stop_lines, crosswalks, None
    focus_center = _stopline_focus_center(stop_lines, rng=rng)
    if focus_center is None:
        return image, det_boxes, lanes, stop_lines, crosswalks, None

    net_h, net_w = network_hw
    scale_min, scale_max = sorted(float(value) for value in config.stopline_focus_crop_scale_range)
    scale_min = max(1.0, scale_min)
    scale_max = max(scale_min, scale_max)
    zoom = rng.uniform(scale_min, scale_max)
    crop_w = max(2, min(int(net_w), int(round(float(net_w) / zoom))))
    crop_h = max(2, min(int(net_h), int(round(float(net_h) / zoom))))
    jitter = max(0.0, float(config.stopline_focus_crop_jitter))
    center_x = float(focus_center[0]) + rng.uniform(-jitter, jitter) * float(crop_w)
    center_y = float(focus_center[1]) + rng.uniform(-jitter, jitter) * float(crop_h)
    crop_left = int(round(center_x - 0.5 * float(crop_w)))
    crop_top = int(round(center_y - 0.5 * float(crop_h)))
    crop_left = max(0, min(crop_left, int(net_w) - crop_w))
    crop_top = max(0, min(crop_top, int(net_h) - crop_h))
    crop_right = crop_left + crop_w
    crop_bottom = crop_top + crop_h
    if crop_right <= crop_left + 1 or crop_bottom <= crop_top + 1:
        return image, det_boxes, lanes, stop_lines, crosswalks, None

    cropped = image[:, crop_top:crop_bottom, crop_left:crop_right].unsqueeze(0)
    zoomed = F.interpolate(cropped, size=(int(net_h), int(net_w)), mode="bilinear", align_corners=False).squeeze(0)
    scale_x = float(net_w) / float(crop_w)
    scale_y = float(net_h) / float(crop_h)

    transformed_det: list[list[float]] = []
    for box in det_boxes:
        transformed_box = _crop_zoom_box_xyxy(
            box,
            crop_left=float(crop_left),
            crop_top=float(crop_top),
            scale_x=scale_x,
            scale_y=scale_y,
            network_hw=network_hw,
        )
        if transformed_box is not None:
            transformed_det.append(transformed_box)

    for rows in (lanes, stop_lines, crosswalks):
        for row in rows:
            points = _row_points(row)
            if points is None:
                continue
            row["points_xy"] = _crop_zoom_points_tensor(
                points,
                crop_left=float(crop_left),
                crop_top=float(crop_top),
                scale_x=scale_x,
                scale_y=scale_y,
                network_hw=network_hw,
            )

    return (
        zoomed.contiguous(),
        transformed_det,
        lanes,
        stop_lines,
        crosswalks,
        {
            "applied": True,
            "zoom": float(zoom),
            "crop_left": int(crop_left),
            "crop_top": int(crop_top),
            "crop_right": int(crop_right),
            "crop_bottom": int(crop_bottom),
            "focus_center": [float(focus_center[0]), float(focus_center[1])],
        },
    )


def _apply_photometric_jitter(
    image: torch.FloatTensor,
    *,
    config: TrainAugmentationConfig,
    rng: random.Random,
) -> tuple[torch.FloatTensor, dict[str, float]]:
    brightness = 1.0 + rng.uniform(-float(config.brightness_delta), float(config.brightness_delta))
    contrast = rng.uniform(float(config.contrast_range[0]), float(config.contrast_range[1]))
    gamma = rng.uniform(float(config.gamma_range[0]), float(config.gamma_range[1]))

    jittered = image.clone()
    jittered = torch.clamp(jittered * brightness, 0.0, 1.0)
    mean = jittered.mean(dim=(1, 2), keepdim=True)
    jittered = torch.clamp((jittered - mean) * contrast + mean, 0.0, 1.0)
    jittered = torch.clamp(jittered, 0.0, 1.0).pow(gamma)
    return jittered, {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "gamma": float(gamma),
    }


def apply_train_augmentations(
    image: torch.FloatTensor,
    *,
    det_boxes: Iterable[Iterable[float]],
    lanes: Iterable[dict[str, object]],
    stop_lines: Iterable[dict[str, object]],
    crosswalks: Iterable[dict[str, object]],
    network_hw: tuple[int, int] = NETWORK_HW,
    config: TrainAugmentationConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[
    torch.FloatTensor,
    list[list[float]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    config = config or TrainAugmentationConfig()
    rng = rng or random.Random()

    augmented_image, photo_meta = _apply_photometric_jitter(image, config=config, rng=rng)
    augmented_det = [[float(value) for value in box] for box in det_boxes]
    augmented_lanes = _clone_geometry_rows(lanes)
    augmented_stop_lines = _clone_geometry_rows(stop_lines)
    augmented_crosswalks = _clone_geometry_rows(crosswalks)
    (
        augmented_image,
        augmented_det,
        augmented_lanes,
        augmented_stop_lines,
        augmented_crosswalks,
        focus_crop_meta,
    ) = _apply_stopline_focus_crop(
        augmented_image,
        det_boxes=augmented_det,
        lanes=augmented_lanes,
        stop_lines=augmented_stop_lines,
        crosswalks=augmented_crosswalks,
        network_hw=network_hw,
        config=config,
        rng=rng,
    )
    (
        augmented_image,
        augmented_det,
        augmented_lanes,
        augmented_stop_lines,
        augmented_crosswalks,
        affine_meta,
    ) = _apply_shared_affine(
        augmented_image,
        det_boxes=augmented_det,
        lanes=augmented_lanes,
        stop_lines=augmented_stop_lines,
        crosswalks=augmented_crosswalks,
        network_hw=network_hw,
        config=config,
        rng=rng,
    )
    (
        augmented_image,
        augmented_stop_lines,
        synthetic_stopline_meta,
    ) = _apply_synthetic_stopline(
        augmented_image,
        lanes=augmented_lanes,
        stop_lines=augmented_stop_lines,
        network_hw=network_hw,
        config=config,
        rng=rng,
    )
    applied_flip = rng.random() < float(config.horizontal_flip_prob)

    if applied_flip:
        augmented_image = torch.flip(augmented_image, dims=(2,))
        augmented_det = [_flip_box_xyxy(box, network_hw) for box in augmented_det]
        for rows in (augmented_lanes, augmented_stop_lines, augmented_crosswalks):
            for row in rows:
                points_xy = row.get("points_xy")
                if isinstance(points_xy, torch.Tensor):
                    row["points_xy"] = _flip_points_tensor(points_xy, network_hw)

    return (
        augmented_image,
        augmented_det,
        augmented_lanes,
        augmented_stop_lines,
        augmented_crosswalks,
        {
            "horizontal_flip": bool(applied_flip),
            "shared_affine": affine_meta,
            "synthetic_stopline": synthetic_stopline_meta,
            "stopline_focus_crop": focus_crop_meta,
            **photo_meta,
        },
    )
