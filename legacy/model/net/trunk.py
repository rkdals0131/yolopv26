from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    from ultralytics import __version__ as ULTRALYTICS_VERSION
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    YOLO = None
    ULTRALYTICS_VERSION = "0.0.0"


MIN_YOLO26_VERSION = "8.4.0"
DEFAULT_YOLO26_VARIANT = "s"
YOLO26_DEFAULT_WEIGHTS = {
    "n": "yolo26n.pt",
    "s": "yolo26s.pt",
}
YOLO26_DETECT_SOURCE_STRIDES = (8, 16, 32)
YOLO26_ROADMARK_SOURCE_INDICES = (2, 16, 19, 22)
YOLO26_ROADMARK_SOURCE_STRIDES = (4, 8, 16, 32)
YOLO26_PYRAMID_CHANNELS = {
    "n": (64, 128, 256),
    "s": (128, 256, 512),
}


@dataclass
class UltralyticsYOLO26TrunkAdapter:
    weights: str
    ultralytics_version: str
    raw_model: nn.Module
    trunk: nn.Sequential
    detect_head: nn.Module
    detect_head_index: int
    feature_source_indices: tuple[int, ...] = field(default_factory=tuple)
    feature_source_strides: tuple[int, ...] = field(default_factory=tuple)
    resolved_feature_channels: tuple[int, ...] = field(default_factory=tuple)

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True


def summarize_trunk_adapter(adapter: UltralyticsYOLO26TrunkAdapter) -> dict[str, Any]:
    trunk_layers = list(adapter.trunk.children())
    trunk_parameter_count = sum(parameter.numel() for parameter in adapter.trunk.parameters())
    detect_parameter_count = sum(parameter.numel() for parameter in adapter.detect_head.parameters())
    raw_layer_container = getattr(adapter.raw_model, "model", None)
    raw_layer_count = len(raw_layer_container) if raw_layer_container is not None else None
    return {
        "weights": adapter.weights,
        "ultralytics_version": adapter.ultralytics_version,
        "raw_model_class": adapter.raw_model.__class__.__name__,
        "raw_layer_count": raw_layer_count,
        "detect_head_index": adapter.detect_head_index,
        "detect_head_class": adapter.detect_head.__class__.__name__,
        "feature_source_indices": list(adapter.feature_source_indices),
        "feature_source_strides": list(adapter.feature_source_strides),
        "resolved_feature_channels": list(adapter.resolved_feature_channels),
        "trunk_layer_count": len(trunk_layers),
        "trunk_layer_classes": [layer.__class__.__name__ for layer in trunk_layers],
        "trunk_parameter_count": trunk_parameter_count,
        "detect_parameter_count": detect_parameter_count,
        "yaml_file": getattr(adapter.raw_model, "yaml", {}).get("yaml_file")
        if hasattr(adapter.raw_model, "yaml")
        else None,
    }


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = "".join(character for character in raw_part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def ensure_yolo26_support(version: str | None = None) -> None:
    version = version or ULTRALYTICS_VERSION
    if _parse_version(version) < _parse_version(MIN_YOLO26_VERSION):
        raise RuntimeError(
            "YOLO26 support requires ultralytics>="
            f"{MIN_YOLO26_VERSION}, but {version} is installed."
        )


def infer_yolo26_variant(weights: str | Path) -> str | None:
    filename = Path(weights).name.lower()
    for variant in YOLO26_DEFAULT_WEIGHTS:
        if f"26{variant}" in filename:
            return variant
    return None


def resolve_yolo26_weights(*, variant: str | None = None, weights: str | None = None) -> str:
    if weights:
        return str(weights)
    normalized_variant = str(variant or DEFAULT_YOLO26_VARIANT).strip().lower()
    try:
        return YOLO26_DEFAULT_WEIGHTS[normalized_variant]
    except KeyError as exc:
        raise KeyError(
            f"unsupported YOLO26 backbone variant: {normalized_variant!r}; "
            f"expected one of {sorted(YOLO26_DEFAULT_WEIGHTS)}"
        ) from exc


def expected_pyramid_channels(*, variant: str | None = None, weights: str | None = None) -> tuple[int, ...] | None:
    resolved_variant = variant or infer_yolo26_variant(weights or "")
    if resolved_variant is None:
        return None
    return YOLO26_PYRAMID_CHANNELS.get(resolved_variant)


def expected_roadmark_pyramid_channels(
    *,
    variant: str | None = None,
    weights: str | None = None,
) -> tuple[int, ...] | None:
    detect_channels = expected_pyramid_channels(variant=variant, weights=weights)
    if detect_channels is None:
        return None
    return (detect_channels[0], *detect_channels)


def _extract_model_layers(torch_model: nn.Module) -> list[nn.Module]:
    model_layers = getattr(torch_model, "model", None)
    if model_layers is None:
        raise TypeError("Ultralytics model does not expose a .model layer container.")
    if isinstance(model_layers, (nn.ModuleList, nn.Sequential)):
        layers = list(model_layers.children())
    elif isinstance(model_layers, (list, tuple)):
        layers = list(model_layers)
    else:
        raise TypeError("Unsupported Ultralytics layer container type for trunk extraction.")
    if len(layers) < 2:
        raise ValueError("Ultralytics model must expose at least trunk + detect head layers.")
    return layers


def build_yolo26_trunk(
    *,
    variant: str | None = None,
    weights: str | None = None,
    feature_source_indices: tuple[int, ...] | None = None,
    feature_source_strides: tuple[int, ...] | None = None,
) -> UltralyticsYOLO26TrunkAdapter:
    ensure_yolo26_support()
    if YOLO is None:  # pragma: no cover - dependency absence handled above.
        raise RuntimeError("ultralytics is not installed.")

    resolved_weights = resolve_yolo26_weights(variant=variant, weights=weights)
    yolo = YOLO(resolved_weights)
    torch_model = yolo.model
    layers = _extract_model_layers(torch_model)
    detect_head_index = len(layers) - 1
    trunk = nn.Sequential(*layers[:detect_head_index])
    detect_head = layers[detect_head_index]
    trunk.requires_grad_(True)
    default_source_indices = tuple(int(index) for index in getattr(detect_head, "f", ()))
    resolved_source_indices = (
        tuple(int(index) for index in default_source_indices)
        if feature_source_indices is None
        else tuple(int(index) for index in feature_source_indices)
    )
    if feature_source_strides is None:
        if resolved_source_indices == default_source_indices:
            resolved_source_strides = YOLO26_DETECT_SOURCE_STRIDES
        elif resolved_source_indices == YOLO26_ROADMARK_SOURCE_INDICES:
            resolved_source_strides = YOLO26_ROADMARK_SOURCE_STRIDES
        else:
            resolved_source_strides = tuple()
    else:
        resolved_source_strides = tuple(int(value) for value in feature_source_strides)
    return UltralyticsYOLO26TrunkAdapter(
        weights=resolved_weights,
        ultralytics_version=ULTRALYTICS_VERSION,
        raw_model=torch_model,
        trunk=trunk,
        detect_head=detect_head,
        detect_head_index=detect_head_index,
        feature_source_indices=resolved_source_indices,
        feature_source_strides=resolved_source_strides,
        resolved_feature_channels=expected_pyramid_channels(variant=variant, weights=resolved_weights) or (),
    )


def build_yolo26n_trunk(weights: str = "yolo26n.pt") -> UltralyticsYOLO26TrunkAdapter:
    return build_yolo26_trunk(weights=weights)


def build_yolo26s_trunk(weights: str = "yolo26s.pt") -> UltralyticsYOLO26TrunkAdapter:
    return build_yolo26_trunk(weights=weights)


def build_yolo26_roadmark_trunk(
    *,
    variant: str | None = None,
    weights: str | None = None,
) -> UltralyticsYOLO26TrunkAdapter:
    resolved_weights = resolve_yolo26_weights(variant=variant, weights=weights)
    adapter = build_yolo26_trunk(
        variant=variant,
        weights=resolved_weights,
        feature_source_indices=YOLO26_ROADMARK_SOURCE_INDICES,
        feature_source_strides=YOLO26_ROADMARK_SOURCE_STRIDES,
    )
    expected_channels = expected_roadmark_pyramid_channels(variant=variant, weights=resolved_weights)
    if expected_channels is not None:
        adapter.resolved_feature_channels = tuple(int(channel) for channel in expected_channels)
    return adapter


def forward_selected_features(
    adapter: UltralyticsYOLO26TrunkAdapter,
    image: Any,
    source_indices: tuple[int, ...] | list[int],
) -> list[Any]:
    resolved_source_indices = tuple(int(index) for index in source_indices)
    if not resolved_source_indices:
        raise ValueError("detect head does not expose source feature indices.")
    layers = list(getattr(adapter.raw_model, "model"))
    max_index = max(resolved_source_indices)
    outputs: list[Any] = []
    current = image

    for layer_index, layer in enumerate(layers[: max_index + 1]):
        from_index = getattr(layer, "f", -1)
        if from_index == -1:
            layer_input = current
        elif isinstance(from_index, int):
            layer_input = outputs[from_index]
        else:
            layer_input = [current if int(index) == -1 else outputs[int(index)] for index in from_index]
        current = layer(layer_input)
        outputs.append(current)

    return [outputs[index] for index in resolved_source_indices]


def forward_pyramid_features(adapter: UltralyticsYOLO26TrunkAdapter, image: Any) -> list[Any]:
    return forward_selected_features(adapter, image, adapter.feature_source_indices)


def infer_pyramid_channels(
    adapter: UltralyticsYOLO26TrunkAdapter,
    *,
    network_hw: tuple[int, int] = (608, 800),
) -> tuple[int, ...]:
    if adapter.resolved_feature_channels:
        return adapter.resolved_feature_channels
    with torch.no_grad():
        features = forward_pyramid_features(
            adapter,
            torch.zeros(1, 3, int(network_hw[0]), int(network_hw[1])),
        )
    channels = tuple(int(feature.shape[1]) for feature in features)
    adapter.resolved_feature_channels = channels
    return channels


def resolve_pyramid_channels(
    adapter: UltralyticsYOLO26TrunkAdapter,
    *,
    network_hw: tuple[int, int] = (608, 800),
) -> tuple[int, ...]:
    return infer_pyramid_channels(adapter, network_hw=network_hw)


def load_matching_state_dict(target: nn.Module, source_state_dict: dict[str, Any]) -> dict[str, Any]:
    target_state = target.state_dict()
    matched: dict[str, Any] = {}
    skipped_shape_keys: list[str] = []
    missing_target_keys: list[str] = []

    for key, value in source_state_dict.items():
        if key not in target_state:
            missing_target_keys.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape_keys.append(key)
            continue
        matched[key] = value

    target_state.update(matched)
    target.load_state_dict(target_state)
    return {
        "loaded_count": len(matched),
        "loaded_keys": sorted(matched.keys()),
        "skipped_shape_keys": sorted(skipped_shape_keys),
        "missing_target_keys": sorted(missing_target_keys),
    }


__all__ = [
    "MIN_YOLO26_VERSION",
    "ULTRALYTICS_VERSION",
    "UltralyticsYOLO26TrunkAdapter",
    "DEFAULT_YOLO26_VARIANT",
    "YOLO26_DEFAULT_WEIGHTS",
    "YOLO26_DETECT_SOURCE_STRIDES",
    "YOLO26_PYRAMID_CHANNELS",
    "YOLO26_ROADMARK_SOURCE_INDICES",
    "YOLO26_ROADMARK_SOURCE_STRIDES",
    "build_yolo26_trunk",
    "build_yolo26_roadmark_trunk",
    "build_yolo26n_trunk",
    "build_yolo26s_trunk",
    "ensure_yolo26_support",
    "expected_pyramid_channels",
    "expected_roadmark_pyramid_channels",
    "forward_selected_features",
    "forward_pyramid_features",
    "infer_pyramid_channels",
    "infer_yolo26_variant",
    "load_matching_state_dict",
    "resolve_pyramid_channels",
    "resolve_yolo26_weights",
    "summarize_trunk_adapter",
]
