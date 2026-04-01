from __future__ import annotations

OD_CLASSES = (
    "vehicle",
    "bike",
    "pedestrian",
    "traffic_cone",
    "obstacle",
    "traffic_light",
    "sign",
)
OD_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(OD_CLASSES)}
LANE_CLASSES = ("white_lane", "yellow_lane", "blue_lane")
LANE_TYPES = ("solid", "dotted")
TL_BITS = ("red", "yellow", "green", "arrow")

AIHUB_LANE_DATASET_KEY = "aihub_lane_seoul"
AIHUB_OBSTACLE_DATASET_KEY = "aihub_obstacle_seoul"
AIHUB_TRAFFIC_DATASET_KEY = "aihub_traffic_seoul"
BDD100K_DATASET_KEY = "bdd100k_det_100k"

EXHAUSTIVE_DATASET_KEY_BY_SOURCE = {
    BDD100K_DATASET_KEY: "pv26_exhaustive_bdd100k_det_100k",
    AIHUB_TRAFFIC_DATASET_KEY: "pv26_exhaustive_aihub_traffic_seoul",
    AIHUB_OBSTACLE_DATASET_KEY: "pv26_exhaustive_aihub_obstacle_seoul",
}

SOURCE_MASK_BY_DATASET = {
    "pv26_exhaustive_bdd100k_det_100k": {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    "pv26_exhaustive_aihub_traffic_seoul": {
        "det": True,
        "tl_attr": True,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    "pv26_exhaustive_aihub_obstacle_seoul": {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    AIHUB_TRAFFIC_DATASET_KEY: {
        "det": True,
        "tl_attr": True,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    AIHUB_OBSTACLE_DATASET_KEY: {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    AIHUB_LANE_DATASET_KEY: {
        "det": False,
        "tl_attr": False,
        "lane": True,
        "stop_line": True,
        "crosswalk": True,
    },
    BDD100K_DATASET_KEY: {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
}

DET_SUPERVISION_BY_DATASET = {
    "pv26_exhaustive_bdd100k_det_100k": {
        "class_names": OD_CLASSES,
        "allow_objectness_negatives": True,
        "allow_unmatched_class_negatives": True,
    },
    "pv26_exhaustive_aihub_traffic_seoul": {
        "class_names": OD_CLASSES,
        "allow_objectness_negatives": True,
        "allow_unmatched_class_negatives": True,
    },
    "pv26_exhaustive_aihub_obstacle_seoul": {
        "class_names": OD_CLASSES,
        "allow_objectness_negatives": True,
        "allow_unmatched_class_negatives": True,
    },
    AIHUB_TRAFFIC_DATASET_KEY: {
        "class_names": ("traffic_light", "sign"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
    AIHUB_OBSTACLE_DATASET_KEY: {
        "class_names": ("traffic_cone", "obstacle"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
    AIHUB_LANE_DATASET_KEY: {
        "class_names": (),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": False,
    },
    BDD100K_DATASET_KEY: {
        "class_names": ("vehicle", "bike", "pedestrian"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
}
