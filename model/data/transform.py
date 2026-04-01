from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


NETWORK_HW = (608, 800)
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
