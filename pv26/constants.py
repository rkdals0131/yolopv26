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


# Semantic classmap constants.
CLASSMAP_VERSION_V2 = "classmap-v2"
CLASSMAP_VERSION_V3 = "classmap-v3"
DEFAULT_CLASSMAP_VERSION = CLASSMAP_VERSION_V3

# v2 ids
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

# v3 ids
SEG3_ID_BACKGROUND = 0
SEG3_ID_DRIVABLE = 1
SEG3_ID_LANE_WHITE_SOLID = 2
SEG3_ID_LANE_WHITE_DASHED = 3
SEG3_ID_LANE_YELLOW_SOLID = 4
SEG3_ID_LANE_YELLOW_DASHED = 5
SEG3_ID_ROAD_MARKER_NON_LANE = 6
SEG3_ID_STOP_LINE = 7

SEG_ID_TO_NAME_V3: Dict[int, str] = {
    SEG3_ID_BACKGROUND: "background",
    SEG3_ID_DRIVABLE: "drivable_area",
    SEG3_ID_LANE_WHITE_SOLID: "lane_white_solid",
    SEG3_ID_LANE_WHITE_DASHED: "lane_white_dashed",
    SEG3_ID_LANE_YELLOW_SOLID: "lane_yellow_solid",
    SEG3_ID_LANE_YELLOW_DASHED: "lane_yellow_dashed",
    SEG3_ID_ROAD_MARKER_NON_LANE: "road_marker_non_lane",
    SEG3_ID_STOP_LINE: "stop_line",
}
