from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

try:
    from ultralytics import YOLO
    from ultralytics import __version__ as ULTRALYTICS_VERSION
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    YOLO = None
    ULTRALYTICS_VERSION = "0.0.0"


MIN_YOLO26_VERSION = "8.4.0"


@dataclass
class UltralyticsYOLO26TrunkAdapter:
    weights: str
    ultralytics_version: str
    raw_model: nn.Module
    trunk: nn.Sequential
    detect_head: nn.Module
    detect_head_index: int

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True


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


def build_yolo26n_trunk(weights: str = "yolo26n.pt") -> UltralyticsYOLO26TrunkAdapter:
    ensure_yolo26_support()
    if YOLO is None:  # pragma: no cover - dependency absence handled above.
        raise RuntimeError("ultralytics is not installed.")

    yolo = YOLO(weights)
    torch_model = yolo.model
    layers = _extract_model_layers(torch_model)
    detect_head_index = len(layers) - 1
    trunk = nn.Sequential(*layers[:detect_head_index])
    detect_head = layers[detect_head_index]
    return UltralyticsYOLO26TrunkAdapter(
        weights=weights,
        ultralytics_version=ULTRALYTICS_VERSION,
        raw_model=torch_model,
        trunk=trunk,
        detect_head=detect_head,
        detect_head_index=detect_head_index,
    )


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
    "build_yolo26n_trunk",
    "ensure_yolo26_support",
    "load_matching_state_dict",
]
