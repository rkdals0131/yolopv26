from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

SIGNAL_ATTR_CROP_REASON_VALID = "valid"
SIGNAL_ATTR_CROP_REASON_INVALID_ROI = "signal_attr_teacher_invalid_roi"
SIGNAL_ATTR_CROP_REASONS = (
    SIGNAL_ATTR_CROP_REASON_VALID,
    SIGNAL_ATTR_CROP_REASON_INVALID_ROI,
)


@dataclass(frozen=True)
class SignalAttrCropConfig:
    input_size: int = 128
    padding_ratio: float = 0.15
    min_crop_side_px: int = 4
    clip_to_image: bool = True
    color_space: str = "rgb"
    normalization: str = "imagenet"
    interpolation: str = "bilinear"


@dataclass(frozen=True)
class SignalAttrCropResult:
    crop_image: Any | None
    crop_box: tuple[int, int, int, int] | None
    clipped_box: tuple[float, float, float, float] | None
    reason: str

    @property
    def valid(self) -> bool:
        return self.reason == SIGNAL_ATTR_CROP_REASON_VALID


DEFAULT_SIGNAL_ATTR_CROP_CONFIG = SignalAttrCropConfig()


def signal_attr_crop_config_to_dict(config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG) -> dict[str, Any]:
    return asdict(config)


def _invalid_crop() -> SignalAttrCropResult:
    return SignalAttrCropResult(
        crop_image=None,
        crop_box=None,
        clipped_box=None,
        reason=SIGNAL_ATTR_CROP_REASON_INVALID_ROI,
    )


def _finite_float_tuple(values: Sequence[Any]) -> tuple[float, float, float, float] | None:
    if len(values) != 4:
        return None
    try:
        coords = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    return coords[0], coords[1], coords[2], coords[3]


def _image_size_tuple(image_size: Sequence[Any]) -> tuple[int, int] | None:
    if len(image_size) != 2:
        return None
    try:
        width = int(image_size[0])
        height = int(image_size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def signal_attr_crop_window(
    bbox: Sequence[Any],
    image_size: Sequence[Any],
    *,
    config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
) -> SignalAttrCropResult:
    size = _image_size_tuple(image_size)
    coords = _finite_float_tuple(bbox)
    if size is None or coords is None:
        return _invalid_crop()

    image_width, image_height = size
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return _invalid_crop()

    if config.clip_to_image:
        clipped_x1 = max(0.0, min(x1, float(image_width)))
        clipped_y1 = max(0.0, min(y1, float(image_height)))
        clipped_x2 = max(0.0, min(x2, float(image_width)))
        clipped_y2 = max(0.0, min(y2, float(image_height)))
    elif x1 < 0.0 or y1 < 0.0 or x2 > image_width or y2 > image_height:
        return _invalid_crop()
    else:
        clipped_x1, clipped_y1, clipped_x2, clipped_y2 = x1, y1, x2, y2

    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return _invalid_crop()
    if min(clipped_width, clipped_height) < int(config.min_crop_side_px):
        return _invalid_crop()

    pad_x = clipped_width * float(config.padding_ratio)
    pad_y = clipped_height * float(config.padding_ratio)
    padded_x1 = clipped_x1 - pad_x
    padded_y1 = clipped_y1 - pad_y
    padded_x2 = clipped_x2 + pad_x
    padded_y2 = clipped_y2 + pad_y
    if config.clip_to_image:
        padded_x1 = max(0.0, min(padded_x1, float(image_width)))
        padded_y1 = max(0.0, min(padded_y1, float(image_height)))
        padded_x2 = max(0.0, min(padded_x2, float(image_width)))
        padded_y2 = max(0.0, min(padded_y2, float(image_height)))

    crop_box = (
        max(0, min(int(math.floor(padded_x1)), image_width)),
        max(0, min(int(math.floor(padded_y1)), image_height)),
        max(0, min(int(math.ceil(padded_x2)), image_width)),
        max(0, min(int(math.ceil(padded_y2)), image_height)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return _invalid_crop()

    return SignalAttrCropResult(
        crop_image=None,
        crop_box=crop_box,
        clipped_box=(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
        reason=SIGNAL_ATTR_CROP_REASON_VALID,
    )


def _pil_resampling_filter(interpolation: str) -> int:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    interpolation_key = str(interpolation).strip().lower()
    if interpolation_key == "nearest":
        return int(resampling.NEAREST)
    if interpolation_key == "bicubic":
        return int(resampling.BICUBIC)
    return int(resampling.BILINEAR)


def crop_signal_attr_roi(
    image: Any,
    bbox: Sequence[Any],
    *,
    config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
) -> SignalAttrCropResult:
    window = signal_attr_crop_window(bbox, (image.width, image.height), config=config)
    if not window.valid or window.crop_box is None:
        return window

    crop = image.crop(window.crop_box)
    color_space = str(config.color_space).strip().lower()
    if color_space == "rgb":
        crop = crop.convert("RGB")
    elif color_space:
        crop = crop.convert(color_space.upper())
    crop = crop.resize(
        (int(config.input_size), int(config.input_size)),
        resample=_pil_resampling_filter(config.interpolation),
    )
    return SignalAttrCropResult(
        crop_image=crop,
        crop_box=window.crop_box,
        clipped_box=window.clipped_box,
        reason=window.reason,
    )


def count_crop_reasons(results: Iterable[SignalAttrCropResult]) -> dict[str, int]:
    counts = Counter(result.reason for result in results)
    return {reason: int(counts.get(reason, 0)) for reason in SIGNAL_ATTR_CROP_REASONS}
