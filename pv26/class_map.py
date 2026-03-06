from __future__ import annotations

from typing import List

from .constants import (
    CLASSMAP_VERSION_V2,
    CLASSMAP_VERSION_V3,
    DEFAULT_CLASSMAP_VERSION,
    DET_CLASSES_CANONICAL,
    SEG_ID_TO_NAME_V2,
    SEG_ID_TO_NAME_V3,
)


def render_class_map_yaml(*, classmap_version: str = DEFAULT_CLASSMAP_VERSION) -> str:
    """
    Render meta/class_map.yaml without external YAML deps.
    """
    version = str(classmap_version).strip()
    if version == CLASSMAP_VERSION_V2:
        seg_map = SEG_ID_TO_NAME_V2
    elif version == CLASSMAP_VERSION_V3:
        seg_map = SEG_ID_TO_NAME_V3
    else:
        raise ValueError(f"unsupported classmap_version: {classmap_version}")

    lines: List[str] = []
    lines.append(f"classmap_version: {version}")
    lines.append("segmentation:")
    lines.append("  id_to_name:")
    for seg_id in sorted(seg_map.keys()):
        lines.append(f"    {seg_id}: {seg_map[seg_id]}")
    lines.append("detection:")
    lines.append("  id_to_name:")
    for c in DET_CLASSES_CANONICAL:
        lines.append(f"    {c.det_id}: {c.name}")
    lines.append("")
    return "\n".join(lines)
