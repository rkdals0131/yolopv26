from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..labels import DET_NAME_TO_ID
from ..yolo import BoxXYXY, format_yolo_line, xyxy_to_yolo_normalized


WAYMO_BOX_VEHICLE = 1
WAYMO_BOX_PEDESTRIAN = 2
WAYMO_BOX_SIGN = 3
WAYMO_BOX_CYCLIST = 4

WAYMO_CAMERA_BOX_TO_PV26_DET_ID: Dict[int, int] = {
    WAYMO_BOX_VEHICLE: DET_NAME_TO_ID["vehicle"],
    WAYMO_BOX_PEDESTRIAN: DET_NAME_TO_ID["pedestrian"],
    WAYMO_BOX_SIGN: DET_NAME_TO_ID["sign_pole"],
    WAYMO_BOX_CYCLIST: DET_NAME_TO_ID["bike"],
}

WAYMO_CAMERA_BOX_DET_ANNOTATED_IDS: Tuple[int, ...] = tuple(sorted(set(WAYMO_CAMERA_BOX_TO_PV26_DET_ID.values())))
WAYMO_CAMERA_BOX_DET_ANNOTATED_CLASS_IDS_CSV = ",".join(str(v) for v in WAYMO_CAMERA_BOX_DET_ANNOTATED_IDS)

WAYMO_SEM_CAR = 2
WAYMO_SEM_TRUCK = 3
WAYMO_SEM_BUS = 4
WAYMO_SEM_OTHER_LARGE_VEHICLE = 5
WAYMO_SEM_BICYCLE = 6
WAYMO_SEM_MOTORCYCLE = 7
WAYMO_SEM_TRAILER = 8
WAYMO_SEM_PEDESTRIAN = 9
WAYMO_SEM_CYCLIST = 10
WAYMO_SEM_MOTORCYCLIST = 11
WAYMO_SEM_CONSTRUCTION_CONE_POLE = 14
WAYMO_SEM_POLE = 15
WAYMO_SEM_PEDESTRIAN_OBJECT = 16
WAYMO_SEM_SIGN = 17
WAYMO_SEM_TRAFFIC_LIGHT = 18


WAYMO_SEM_ROAD = 20
WAYMO_SEM_LANE_MARKER = 21
WAYMO_SEM_ROAD_MARKER = 22
WAYMO_SEM_SIDEWALK = 23


WAYMO_PANOPTIC_TO_PV26_DET_ID: Dict[int, int] = {
    WAYMO_SEM_CAR: DET_NAME_TO_ID["vehicle"],
    WAYMO_SEM_TRUCK: DET_NAME_TO_ID["vehicle"],
    WAYMO_SEM_BUS: DET_NAME_TO_ID["vehicle"],
    WAYMO_SEM_OTHER_LARGE_VEHICLE: DET_NAME_TO_ID["vehicle"],
    WAYMO_SEM_BICYCLE: DET_NAME_TO_ID["bike"],
    WAYMO_SEM_MOTORCYCLE: DET_NAME_TO_ID["bike"],
    WAYMO_SEM_TRAILER: DET_NAME_TO_ID["vehicle"],
    WAYMO_SEM_PEDESTRIAN: DET_NAME_TO_ID["pedestrian"],
    WAYMO_SEM_CYCLIST: DET_NAME_TO_ID["bike"],
    WAYMO_SEM_MOTORCYCLIST: DET_NAME_TO_ID["bike"],
    WAYMO_SEM_CONSTRUCTION_CONE_POLE: DET_NAME_TO_ID["traffic_cone"],
    WAYMO_SEM_POLE: DET_NAME_TO_ID["sign_pole"],
    WAYMO_SEM_PEDESTRIAN_OBJECT: DET_NAME_TO_ID["obstacle"],
    WAYMO_SEM_SIGN: DET_NAME_TO_ID["sign_pole"],
    WAYMO_SEM_TRAFFIC_LIGHT: DET_NAME_TO_ID["traffic_light"],
}

WAYMO_PANOPTIC_DET_ANNOTATED_IDS: Tuple[int, ...] = tuple(sorted(set(WAYMO_PANOPTIC_TO_PV26_DET_ID.values())))
WAYMO_PANOPTIC_DET_ANNOTATED_CLASS_IDS_CSV = ",".join(str(v) for v in WAYMO_PANOPTIC_DET_ANNOTATED_IDS)


@dataclass(frozen=True)
class WodPanopticInstanceDetBox:
    semantic_id: int
    instance_id: int
    det_id: int
    box_xyxy: BoxXYXY


def decode_panoptic_to_semantic_instance(
    panoptic_png: bytes,
    *,
    divisor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode Waymo panoptic PNG into semantic id and instance id maps.

    Waymo panoptic values follow:
      panoptic_value = semantic_id * divisor + instance_id
    """
    if divisor <= 0:
        raise ValueError(f"invalid panoptic divisor: {divisor}")
    panoptic = np.array(Image.open(io.BytesIO(panoptic_png)), dtype=np.uint16)
    semantic = (panoptic // np.uint16(divisor)).astype(np.int32)
    instance = (panoptic % np.uint16(divisor)).astype(np.int32)
    return semantic, instance


def _bbox_from_mask(mask: np.ndarray) -> Optional[BoxXYXY]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def panoptic_to_pv26_det_boxes(
    *,
    semantic_id: np.ndarray,
    instance_id: np.ndarray,
    width: int,
    height: int,
) -> List[WodPanopticInstanceDetBox]:
    """
    Convert Waymo panoptic thing instances into PV26 detection boxes.

    Only semantic classes explicitly mapped in `WAYMO_PANOPTIC_TO_PV26_DET_ID`
    are exported. Stuff classes such as road / lane marker / road marker remain
    segmentation-only.
    """
    if semantic_id.ndim != 2 or instance_id.ndim != 2:
        raise ValueError("semantic_id and instance_id must be 2D")
    if semantic_id.shape != instance_id.shape:
        raise ValueError(f"semantic/instance shape mismatch: {semantic_id.shape} vs {instance_id.shape}")
    if semantic_id.shape != (height, width):
        raise ValueError(f"semantic/instance shape must match image size: {(height, width)} vs {semantic_id.shape}")

    out: List[WodPanopticInstanceDetBox] = []
    sem_values = sorted(int(v) for v in np.unique(semantic_id).tolist())
    for sem_val in sem_values:
        det_id = WAYMO_PANOPTIC_TO_PV26_DET_ID.get(int(sem_val))
        if det_id is None:
            continue
        sem_mask = semantic_id == int(sem_val)
        inst_values = sorted(int(v) for v in np.unique(instance_id[sem_mask]).tolist())
        for inst_val in inst_values:
            if inst_val <= 0:
                continue
            mask = sem_mask & (instance_id == int(inst_val))
            box = _bbox_from_mask(mask)
            if box is None:
                continue
            box = box.clip(width=width, height=height)
            if box.area_px() <= 0.0:
                continue
            out.append(
                WodPanopticInstanceDetBox(
                    semantic_id=int(sem_val),
                    instance_id=int(inst_val),
                    det_id=int(det_id),
                    box_xyxy=box,
                )
            )
    return out


def panoptic_to_pv26_det_yolo_lines(
    *,
    semantic_id: np.ndarray,
    instance_id: np.ndarray,
    width: int,
    height: int,
) -> List[str]:
    """
    Export panoptic-derived PV26 detection labels as YOLO txt lines.
    """
    out: List[str] = []
    for det_box in panoptic_to_pv26_det_boxes(
        semantic_id=semantic_id,
        instance_id=instance_id,
        width=width,
        height=height,
    ):
        cx, cy, bw, bh = xyxy_to_yolo_normalized(det_box.box_xyxy, width=width, height=height)
        if bw <= 0.0 or bh <= 0.0:
            continue
        out.append(format_yolo_line(det_box.det_id, cx, cy, bw, bh))
    return out


def semantic_to_pv26_da_rm_masks(semantic_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Waymo Perception v2 camera semantic ids into PV26 masks.

    Returns:
      da_mask, rm_lane_marker, rm_road_marker_non_lane
    """
    if semantic_id.ndim != 2:
        raise ValueError(f"expected 2D semantic id mask, got shape={semantic_id.shape}")
    da = (semantic_id == WAYMO_SEM_ROAD).astype(np.uint8)
    lane = (semantic_id == WAYMO_SEM_LANE_MARKER).astype(np.uint8)
    road = (semantic_id == WAYMO_SEM_ROAD_MARKER).astype(np.uint8)
    return da, lane, road
