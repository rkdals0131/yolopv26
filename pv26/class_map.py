from __future__ import annotations

from typing import List

from .constants import (
    CLASSMAP_VERSION_V2,
    DET_CLASSES_CANONICAL,
    SEG_ID_TO_NAME_V2,
)


def render_class_map_yaml(*, classmap_version: str = CLASSMAP_VERSION_V2) -> str:
    """
    Render meta/class_map.yaml without external YAML deps.
    """
    if classmap_version != CLASSMAP_VERSION_V2:
        raise ValueError(f"unsupported classmap_version: {classmap_version}")

    lines: List[str] = []
    lines.append(f"classmap_version: {classmap_version}")
    lines.append("segmentation:")
    lines.append("  id_to_name:")
    for seg_id in sorted(SEG_ID_TO_NAME_V2.keys()):
        lines.append(f"    {seg_id}: {SEG_ID_TO_NAME_V2[seg_id]}")
    lines.append("detection:")
    lines.append("  id_to_name:")
    for c in DET_CLASSES_CANONICAL:
        lines.append(f"    {c.det_id}: {c.name}")
    lines.append("")
    return "\n".join(lines)

