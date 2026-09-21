from __future__ import annotations

from common.pv26_schema import OD_CLASSES, TL_BITS


def det_class_map_yaml() -> str:
    lines = [
        "version: pv26-det-v1",
        "classes:",
    ]
    for index, class_name in enumerate(OD_CLASSES):
        lines.append(f"  {index}: {class_name}")
    lines.extend(
        [
            "tl_bits:",
        ]
    )
    for bit in TL_BITS:
        lines.append(f"  - {bit}")
    lines.extend(
        [
            "tl_attribute_policy:",
            "  base_type: car",
            "  arrow_sources:",
            "    - left_arrow",
            "    - others_arrow",
            "  masked_cases:",
            "    - x_light_active",
            "    - multi_color_active",
            "    - non_car_traffic_light",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "det_class_map_yaml",
]
