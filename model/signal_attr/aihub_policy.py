"""AIHub traffic-light attributes used by the product crop classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TL_BITS = ("red", "yellow", "green", "arrow")
PRODUCT_SIGNAL_TYPES = ("car", "pedestrian")


@dataclass(frozen=True)
class ProductSignalAttrTarget:
    light_type: str
    base_color: str
    left_arrow: int
    state_valid: bool
    reason: str

    @property
    def tl_bits(self) -> dict[str, int]:
        bits = {bit: 0 for bit in TL_BITS}
        if self.base_color in ("red", "yellow", "green"):
            bits[self.base_color] = 1
        bits["arrow"] = self.left_arrow
        return bits


def _attribute_map(raw_attribute: Any) -> dict[str, str] | None:
    entries = raw_attribute if isinstance(raw_attribute, list) else [raw_attribute]
    for entry in entries:
        if isinstance(entry, Mapping):
            return {str(key).strip().lower(): str(value).strip().lower() for key, value in entry.items()}
    return None


def extract_product_signal_attr_target(
    annotation: Mapping[str, Any], *, all_off_is_valid: bool
) -> ProductSignalAttrTarget:
    light_type = str(annotation.get("type") or "").strip().lower()
    if light_type not in PRODUCT_SIGNAL_TYPES:
        return ProductSignalAttrTarget(light_type, "off", 0, False, "unsupported_light_type")
    attributes = _attribute_map(annotation.get("attribute"))
    if attributes is None:
        return ProductSignalAttrTarget(light_type, "off", 0, False, "missing_attribute_map")

    required = ("red", "green") if light_type == "pedestrian" else ("red", "yellow", "green", "left_arrow")
    if any(attributes.get(key) not in ("on", "off") for key in required):
        return ProductSignalAttrTarget(light_type, "off", 0, False, "missing_state_field")
    active_colors = [key for key in ("red", "yellow", "green") if attributes.get(key) == "on"]
    left_arrow = int(light_type == "car" and attributes.get("left_arrow") == "on")
    other_arrow_on = attributes.get("others_arrow") == "on"
    base_color = active_colors[0] if len(active_colors) == 1 else "off"
    target = ProductSignalAttrTarget(light_type, base_color, left_arrow, True, "valid")
    if attributes.get("x_light") == "on":
        return ProductSignalAttrTarget(light_type, base_color, left_arrow, False, "x_light_active")
    if light_type == "pedestrian" and (
        attributes.get("yellow") == "on" or attributes.get("left_arrow") == "on" or other_arrow_on
    ):
        return ProductSignalAttrTarget(light_type, base_color, 0, False, "pedestrian_unsupported_state")
    if len(active_colors) > 1:
        return ProductSignalAttrTarget(light_type, "off", left_arrow, False, "multi_color_active")
    if not active_colors and not left_arrow and not other_arrow_on and not all_off_is_valid:
        return ProductSignalAttrTarget(light_type, "off", 0, False, "all_off_unverified")
    return target


__all__ = ["TL_BITS", "ProductSignalAttrTarget", "extract_product_signal_attr_target"]
