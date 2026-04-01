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

