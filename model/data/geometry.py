"""Focused-model letterbox metadata and raw-image coordinate conversion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from common.schema import DEFAULT_IMAGE_HW


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
    network_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> LetterboxTransform:
    raw_h, raw_w = (int(raw_hw[0]), int(raw_hw[1]))
    net_h, net_w = (int(network_hw[0]), int(network_hw[1]))
    if raw_h <= 0 or raw_w <= 0 or net_h <= 0 or net_w <= 0:
        raise ValueError(
            f"letterbox dimensions must be positive: raw_hw={(raw_h, raw_w)} network_hw={(net_h, net_w)}"
        )
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
        raw_hw=(raw_h, raw_w),
        network_hw=(net_h, net_w),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        resized_hw=(resized_h, resized_w),
    )


def transform_from_meta(meta: dict[str, object]) -> LetterboxTransform:
    raw_hw = tuple(int(value) for value in meta["raw_hw"])
    network_hw = tuple(int(value) for value in meta["network_hw"])
    payload = dict(meta["transform"])
    resized_hw = tuple(int(value) for value in payload["resized_hw"])
    scale = float(payload["scale"])
    pad_left = int(payload["pad_left"])
    pad_top = int(payload["pad_top"])
    pad_right = int(payload["pad_right"])
    pad_bottom = int(payload["pad_bottom"])
    raw_h, raw_w = raw_hw
    net_h, net_w = network_hw
    resized_h, resized_w = resized_hw
    if raw_h <= 0 or raw_w <= 0 or net_h <= 0 or net_w <= 0 or resized_h <= 0 or resized_w <= 0:
        raise ValueError("letterbox transform dimensions must be positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("letterbox transform scale must be positive and finite")
    if pad_left < 0 or pad_top < 0 or pad_right < 0 or pad_bottom < 0:
        raise ValueError("letterbox transform padding must be non-negative")
    if resized_h + pad_top + pad_bottom != net_h or resized_w + pad_left + pad_right != net_w:
        raise ValueError("letterbox resized/padding must match network_hw")
    if resized_h != int(round(raw_h * scale)) or resized_w != int(round(raw_w * scale)):
        raise ValueError("letterbox resized_hw must match raw_hw and scale")
    expected = compute_letterbox_transform((raw_h, raw_w), network_hw=(net_h, net_w))
    if (
        not math.isclose(scale, expected.scale, rel_tol=1.0e-9, abs_tol=1.0e-9)
        or pad_left != expected.pad_left
        or pad_top != expected.pad_top
        or pad_right != expected.pad_right
        or pad_bottom != expected.pad_bottom
        or (resized_h, resized_w) != expected.resized_hw
    ):
        raise ValueError("letterbox transform must match raw_hw/network_hw")
    return LetterboxTransform(
        raw_hw=(raw_h, raw_w),
        network_hw=(net_h, net_w),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        resized_hw=(resized_h, resized_w),
    )


def inverse_transform_box_xyxy(box: Iterable[float], transform: LetterboxTransform) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    raw_box = [
        (x1 - transform.pad_left) / transform.scale,
        (y1 - transform.pad_top) / transform.scale,
        (x2 - transform.pad_left) / transform.scale,
        (y2 - transform.pad_top) / transform.scale,
    ]
    raw_h, raw_w = transform.raw_hw
    x1, y1, x2, y2 = raw_box
    x1 = max(0.0, min(x1, raw_w - 1.0))
    y1 = max(0.0, min(y1, raw_h - 1.0))
    x2 = max(0.0, min(x2, raw_w - 1.0))
    y2 = max(0.0, min(y2, raw_h - 1.0))
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return [x1, y1, x2, y2]
