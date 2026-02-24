from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DetClass:
    det_id: int
    name: str


# Canonical OD classes (11) from docs/PRD.md and docs/DATASET_CONVERSION_SPEC.md.
DET_CLASSES_CANONICAL: List[DetClass] = [
    DetClass(0, "car"),
    DetClass(1, "bus"),
    DetClass(2, "truck"),
    DetClass(3, "motorcycle"),
    DetClass(4, "bicycle"),
    DetClass(5, "pedestrian"),
    DetClass(6, "traffic_cone"),
    DetClass(7, "barrier"),
    DetClass(8, "bollard"),
    DetClass(9, "road_obstacle"),
    DetClass(10, "sign_pole"),
]

DET_NAME_TO_ID: Dict[str, int] = {c.name: c.det_id for c in DET_CLASSES_CANONICAL}


# Semantic classmap constants (v2) from docs/PRD.md.
CLASSMAP_VERSION_V2 = "classmap-v2"

SEG_ID_BACKGROUND = 0
SEG_ID_DRIVABLE = 1
SEG_ID_LANE_MARKING = 2
SEG_ID_STOP_LINE = 3

SEG_ID_TO_NAME_V2: Dict[int, str] = {
    SEG_ID_BACKGROUND: "background",
    SEG_ID_DRIVABLE: "drivable_area",
    SEG_ID_LANE_MARKING: "lane_marking",
    SEG_ID_STOP_LINE: "stop_line",
}

