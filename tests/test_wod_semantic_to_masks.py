import unittest
import io

import numpy as np
from PIL import Image

from pv26.dataset.sources.wod import (
    WAYMO_CAMERA_BOX_DET_ANNOTATED_CLASS_IDS_CSV,
    WAYMO_PANOPTIC_DET_ANNOTATED_CLASS_IDS_CSV,
    WAYMO_SEM_CAR,
    WAYMO_SEM_LANE_MARKER,
    WAYMO_SEM_ROAD,
    WAYMO_SEM_ROAD_MARKER,
    WAYMO_SEM_TRAFFIC_LIGHT,
    decode_panoptic_to_semantic_instance,
    panoptic_to_pv26_det_yolo_lines,
    semantic_to_pv26_da_rm_masks,
)


class TestWodSemanticMapping(unittest.TestCase):
    def test_semantic_to_masks(self):
        sem = np.array(
            [
                [0, WAYMO_SEM_ROAD, WAYMO_SEM_LANE_MARKER],
                [WAYMO_SEM_ROAD_MARKER, WAYMO_SEM_LANE_MARKER, WAYMO_SEM_ROAD],
            ],
            dtype=np.int32,
        )
        da, lane, road = semantic_to_pv26_da_rm_masks(sem)
        self.assertEqual(da.tolist(), [[0, 1, 0], [0, 0, 1]])
        self.assertEqual(lane.tolist(), [[0, 0, 1], [0, 1, 0]])
        self.assertEqual(road.tolist(), [[0, 0, 0], [1, 0, 0]])

    def test_decode_panoptic_to_semantic_instance(self):
        divisor = 1000
        panoptic = np.array(
            [
                [WAYMO_SEM_CAR * divisor + 7, WAYMO_SEM_CAR * divisor + 7],
                [WAYMO_SEM_TRAFFIC_LIGHT * divisor + 5, 0],
            ],
            dtype=np.uint16,
        )
        buf = io.BytesIO()
        Image.fromarray(panoptic).save(buf, format="PNG")

        sem, inst = decode_panoptic_to_semantic_instance(buf.getvalue(), divisor=divisor)
        self.assertEqual(sem.tolist(), [[WAYMO_SEM_CAR, WAYMO_SEM_CAR], [WAYMO_SEM_TRAFFIC_LIGHT, 0]])
        self.assertEqual(inst.tolist(), [[7, 7], [5, 0]])

    def test_panoptic_to_pv26_det_yolo_lines_extracts_instance_boxes(self):
        sem = np.array(
            [
                [0, 0, 0, 0, WAYMO_SEM_TRAFFIC_LIGHT, WAYMO_SEM_TRAFFIC_LIGHT],
                [0, WAYMO_SEM_CAR, WAYMO_SEM_CAR, WAYMO_SEM_CAR, WAYMO_SEM_TRAFFIC_LIGHT, WAYMO_SEM_TRAFFIC_LIGHT],
                [0, WAYMO_SEM_CAR, WAYMO_SEM_CAR, WAYMO_SEM_CAR, 0, 0],
                [WAYMO_SEM_ROAD, WAYMO_SEM_ROAD, WAYMO_SEM_LANE_MARKER, WAYMO_SEM_LANE_MARKER, WAYMO_SEM_ROAD_MARKER, WAYMO_SEM_ROAD_MARKER],
                [WAYMO_SEM_ROAD, WAYMO_SEM_ROAD, WAYMO_SEM_LANE_MARKER, WAYMO_SEM_LANE_MARKER, WAYMO_SEM_ROAD_MARKER, WAYMO_SEM_ROAD_MARKER],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        inst = np.array(
            [
                [0, 0, 0, 0, 2, 2],
                [0, 7, 7, 7, 2, 2],
                [0, 7, 7, 7, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        lines = panoptic_to_pv26_det_yolo_lines(
            semantic_id=sem,
            instance_id=inst,
            width=6,
            height=6,
        )

        self.assertEqual(
            lines,
            [
                "0 0.416667 0.333333 0.500000 0.333333",
                "10 0.833333 0.166667 0.333333 0.333333",
            ],
        )

    def test_wod_annotated_class_ids_contracts(self):
        self.assertEqual(WAYMO_CAMERA_BOX_DET_ANNOTATED_CLASS_IDS_CSV, "0,4,5,10")
        self.assertEqual(WAYMO_PANOPTIC_DET_ANNOTATED_CLASS_IDS_CSV, "0,1,2,3,4,5,6,9,10")


if __name__ == "__main__":
    unittest.main()
