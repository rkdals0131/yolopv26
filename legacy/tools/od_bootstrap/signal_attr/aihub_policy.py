from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from common.pv26_schema import TL_BITS

from ..source.shared.raw import normalize_text

AIHUB_TL_VALID_REASON = "valid"
AIHUB_TL_INVALID_REASONS = (
    "non_car_traffic_light",
    "missing_attribute_map",
    "x_light_active",
    "multi_color_active",
)

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
        bits = _empty_tl_bits()
        if self.base_color in ("red", "yellow", "green"):
            bits[self.base_color] = 1
        bits["arrow"] = self.left_arrow
        return bits


@dataclass(frozen=True)
class AIHubTrafficLightAttrLabel:
    tl_bits: dict[str, int]
    tl_attr_valid: int
    collapse_reason: str

    @property
    def arrow(self) -> int:
        return int(self.tl_bits.get("arrow", 0))

    @property
    def base_color(self) -> str:
        active = [bit for bit in ("red", "yellow", "green") if self.tl_bits.get(bit)]
        if len(active) == 1:
            return active[0]
        if active:
            return "multi"
        return "off"

    def as_traffic_worker_tuple(self) -> tuple[dict[str, int], int, str]:
        return dict(self.tl_bits), int(self.tl_attr_valid), str(self.collapse_reason)


def _empty_tl_bits() -> dict[str, int]:
    return {bit: 0 for bit in TL_BITS}


def _first_attribute_map(raw_attribute: Any) -> dict[str, str] | None:
    candidate_items = raw_attribute if isinstance(raw_attribute, list) else [raw_attribute]
    for item in candidate_items:
        if isinstance(item, Mapping):
            return {str(key): str(value).strip().lower() for key, value in item.items()}
    return None


def collapse_aihub_traffic_light_attr(annotation: Mapping[str, Any]) -> AIHubTrafficLightAttrLabel:
    bits = _empty_tl_bits()
    light_type = normalize_text(annotation.get("type"))
    if light_type != "car":
        return AIHubTrafficLightAttrLabel(bits, 0, "non_car_traffic_light")

    attribute_map = _first_attribute_map(annotation.get("attribute"))
    if attribute_map is None:
        return AIHubTrafficLightAttrLabel(bits, 0, "missing_attribute_map")

    red_on = attribute_map.get("red") == "on"
    yellow_on = attribute_map.get("yellow") == "on"
    green_on = attribute_map.get("green") == "on"
    arrow_on = attribute_map.get("left_arrow") == "on" or attribute_map.get("others_arrow") == "on"
    x_light_on = attribute_map.get("x_light") == "on"

    bits["red"] = int(red_on)
    bits["yellow"] = int(yellow_on)
    bits["green"] = int(green_on)
    bits["arrow"] = int(arrow_on)

    base_on_count = sum(int(flag) for flag in (red_on, yellow_on, green_on))
    if x_light_on:
        return AIHubTrafficLightAttrLabel(bits, 0, "x_light_active")
    if base_on_count > 1:
        return AIHubTrafficLightAttrLabel(bits, 0, "multi_color_active")
    return AIHubTrafficLightAttrLabel(bits, 1, AIHUB_TL_VALID_REASON)


def extract_product_signal_attr_target(
    annotation: Mapping[str, Any],
    *,
    all_off_is_valid: bool,
) -> ProductSignalAttrTarget:
    """Decode raw AIHub state for the car/pedestrian product contract.

    The old ``collapse_aihub_traffic_light_attr`` remains the Plan B teacher
    contract. This decoder uses the original attribute map because the
    canonical ``tl_bits.arrow`` field has merged left and other arrows.
    """
    light_type = normalize_text(annotation.get("type"))
    if light_type not in PRODUCT_SIGNAL_TYPES:
        return ProductSignalAttrTarget(light_type, "off", 0, False, "unsupported_light_type")
    attributes = _first_attribute_map(annotation.get("attribute"))
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


def combo_name(bits: Mapping[str, int]) -> str:
    active = [key for key in TL_BITS if bits.get(key)]
    return "+".join(active) if active else "off"
