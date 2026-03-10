from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DetClass:
    det_id: int
    name: str


# Canonical OD classes (7) for the coarse unified PV26 taxonomy.
DET_CLASSES_CANONICAL: List[DetClass] = [
    DetClass(0, "vehicle"),
    DetClass(1, "bike"),
    DetClass(2, "pedestrian"),
    DetClass(3, "traffic_cone"),
    DetClass(4, "obstacle"),
    DetClass(5, "traffic_light"),
    DetClass(6, "sign_pole"),
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
