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


def combo_name(bits: Mapping[str, int]) -> str:
    active = [key for key in TL_BITS if bits.get(key)]
    return "+".join(active) if active else "off"
