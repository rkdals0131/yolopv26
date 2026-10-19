"""Train-only SignalAttr crop augmentation for distant and blurred traffic lights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class SignalAttrAugmentConfig:
    """Probabilities and ranges. Horizontal flip and hue shifts are excluded on purpose:
    a flip turns the left arrow into a right arrow, and hue changes the label."""

    jitter_prob: float = 0.5
    jitter_scale: tuple[float, float] = (0.9, 1.1)
    jitter_shift: float = 0.08
    photometric_prob: float = 0.5
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.2
    lowres_prob: float = 0.5
    lowres_min_short_side_px: float = 8.0
    blur_prob: float = 0.3
    blur_radius_px: tuple[float, float] = (0.3, 0.8)
    jpeg_prob: float = 0.3
    jpeg_quality: tuple[int, int] = (50, 90)


def signal_attr_augment_config_from_mapping(payload: Mapping[str, Any] | None) -> SignalAttrAugmentConfig | None:
    """``None`` or ``enabled: false`` disables augmentation; other keys override defaults."""
    if not payload or not bool(payload.get("enabled", True)):
        return None
    fields = {key: value for key, value in payload.items() if key != "enabled"}
    unknown = set(fields) - set(SignalAttrAugmentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown SignalAttr augment keys: {sorted(unknown)}")
    for key in ("jitter_scale", "blur_radius_px", "jpeg_quality"):
        if key in fields:
            fields[key] = tuple(fields[key])
    return SignalAttrAugmentConfig(**fields)


def signal_attr_augment_config_to_dict(config: SignalAttrAugmentConfig | None) -> dict[str, Any]:
    if config is None:
        return {"enabled": False}
    payload = asdict(config)
    return {"enabled": True, **{key: list(value) if isinstance(value, tuple) else value
                                for key, value in payload.items()}}


def _native_size(crop_box: Sequence[Any] | None, fallback: int) -> tuple[float, float]:
    """Pixel size of the crop window in the source image, before the resize to the input size."""
    if crop_box is not None and len(crop_box) == 4:
        width = float(crop_box[2]) - float(crop_box[0])
        height = float(crop_box[3]) - float(crop_box[1])
        if width > 0 and height > 0:
            return width, height
    return float(fallback), float(fallback)


def _jitter(image: Image.Image, rng: random.Random, config: SignalAttrAugmentConfig) -> Image.Image:
    """Shift and rescale the crop window like a predicted box that misses the label box."""
    size = image.width
    scale = rng.uniform(*config.jitter_scale)
    window = size * scale
    center_x = size / 2 + rng.uniform(-config.jitter_shift, config.jitter_shift) * size
    center_y = size / 2 + rng.uniform(-config.jitter_shift, config.jitter_shift) * size
    # Edge padding keeps zoom-out from adding black borders the runtime crop never has.
    pad = int(math.ceil(window / 2 + config.jitter_shift * size))
    padded = Image.fromarray(np.pad(np.asarray(image), ((pad, pad), (pad, pad), (0, 0)), mode="edge"))
    box = (center_x - window / 2 + pad, center_y - window / 2 + pad,
           center_x + window / 2 + pad, center_y + window / 2 + pad)
    return padded.resize((size, size), resample=Image.Resampling.BILINEAR, box=box)


def _photometric(image: Image.Image, rng: random.Random, config: SignalAttrAugmentConfig) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - config.brightness, 1 + config.brightness))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - config.contrast, 1 + config.contrast))
    return ImageEnhance.Color(image).enhance(rng.uniform(1 - config.saturation, 1 + config.saturation))


def augment_signal_attr_crop(
    image: Image.Image, crop_box: Sequence[Any] | None, rng: random.Random, config: SignalAttrAugmentConfig,
) -> Image.Image:
    """Return an RGB crop of the same size.

    Low resolution, blur and JPEG run at the source-image pixel size of the crop window, so
    a shrunken crop goes through the same upscale the runtime applies to a distant light.
    """
    image = image.convert("RGB")
    size = image.width
    if rng.random() < config.jitter_prob:
        image = _jitter(image, rng, config)
    if rng.random() < config.photometric_prob:
        image = _photometric(image, rng, config)

    native_w, native_h = _native_size(crop_box, size)
    factor = 1.0
    if rng.random() < config.lowres_prob:
        short = min(native_w, native_h)
        if short > config.lowres_min_short_side_px:
            # Log-uniform so 2x and 4x farther are drawn as often as each other.
            factor = math.exp(rng.uniform(math.log(config.lowres_min_short_side_px / short), 0.0))
    use_blur = rng.random() < config.blur_prob
    use_jpeg = rng.random() < config.jpeg_prob
    if factor == 1.0 and not use_blur and not use_jpeg:
        return image

    small = image.resize(
        (max(1, round(native_w * factor)), max(1, round(native_h * factor))),
        resample=Image.Resampling.BOX,
    )
    if use_blur:
        small = small.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*config.blur_radius_px)))
    if use_jpeg:
        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=rng.randint(*config.jpeg_quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            small = decoded.convert("RGB")
    return small.resize((size, size), resample=Image.Resampling.BILINEAR)


__all__ = [
    "SignalAttrAugmentConfig",
    "augment_signal_attr_crop",
    "signal_attr_augment_config_from_mapping",
    "signal_attr_augment_config_to_dict",
]
