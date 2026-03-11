import unittest

import numpy as np

from pv26.dataset.sources.bdd import (
    bdd_record_to_image_name,
    bdd_record_to_rm_masks,
    bdd_record_to_rm_masks_with_lane_subclass,
    bdd_record_to_yolo_lines,
)
from pv26.dataset.masks import IGNORE_VALUE


class TestBddAdapter(unittest.TestCase):
    def test_record_name_adds_jpg_extension(self):
        rec = {"name": "abc123"}
        self.assertEqual(bdd_record_to_image_name(rec), "abc123.jpg")

    def test_frames_objects_are_converted_to_yolo(self):
        rec = {
            "name": "sample",
            "frames": [
                {
                    "timestamp": 1000,
                    "objects": [
                        {
                            "category": "car",
                            "box2d": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0},
                        },
                        {
                            "category": "lane/single white",
                            "poly2d": [],
                        },
                    ],
                }
            ],
        }
        lines = bdd_record_to_yolo_lines(rec, width=100, height=100)
        self.assertEqual(len(lines), 1)
        # class 0 vehicle, cx=0.2, cy=0.4, w=0.2, h=0.4
        self.assertEqual(lines[0], "0 0.200000 0.400000 0.200000 0.400000")

    def test_bdd_coarse_7class_mapping_distinguishes_light_and_obstacle(self):
        rec = {
            "name": "sample",
            "frames": [
                {
                    "objects": [
                        {
                            "category": "traffic light",
                            "box2d": {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 50.0},
                        },
                        {
                            "category": "barrier",
                            "box2d": {"x1": 40.0, "y1": 20.0, "x2": 90.0, "y2": 60.0},
                        },
                    ],
                }
            ],
        }
        lines = bdd_record_to_yolo_lines(rec, width=100, height=100)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "5 0.200000 0.300000 0.200000 0.400000")
        self.assertEqual(lines[1], "4 0.650000 0.400000 0.500000 0.400000")

    def test_lane_poly2d_rasterization(self):
        rec = {
            "name": "sample",
            "frames": [
                {
                    "objects": [
                        {
                            "category": "lane/single white",
                            "poly2d": [[10.0, 10.0, "L"], [90.0, 10.0, "L"]],
                        },
                        {
                            "category": "lane/crosswalk",
                            "poly2d": [[20.0, 20.0, "L"], [80.0, 20.0, "L"], [80.0, 40.0, "L"], [20.0, 40.0, "L"]],
                        },
                    ]
                }
            ],
        }
        rm_lane, rm_road, rm_stop, has_lane, has_road, has_stop = bdd_record_to_rm_masks(rec, width=100, height=60)
        self.assertEqual(has_lane, 1)
        self.assertEqual(has_road, 1)
        self.assertEqual(has_stop, 0)
        self.assertTrue(np.any(rm_lane == 1))
        self.assertTrue(np.any(rm_road == 1))
        self.assertTrue(np.all(rm_stop == IGNORE_VALUE))

    def test_lane_subclass_rasterization(self):
        rec = {
            "name": "sample",
            "frames": [
                {
                    "objects": [
                        {
                            "category": "lane/single white",
                            "attributes": {"style": "solid"},
                            "poly2d": [[10.0, 10.0, "L"], [90.0, 10.0, "L"]],
                        },
                        {
                            "category": "lane/single yellow",
                            "attributes": {"style": "dashed"},
                            "poly2d": [[10.0, 20.0, "L"], [90.0, 20.0, "L"]],
                        },
                        {
                            "category": "lane/single other",
                            "attributes": {"style": "solid"},
                            "poly2d": [[10.0, 30.0, "L"], [90.0, 30.0, "L"]],
                        },
                    ]
                }
            ],
        }
        rm_lane, _rm_road, _rm_stop, rm_lane_sub, has_lane, _has_road, _has_stop, has_lane_sub = (
            bdd_record_to_rm_masks_with_lane_subclass(rec, width=100, height=60)
        )
        self.assertEqual(has_lane, 1)
        self.assertEqual(has_lane_sub, 1)
        self.assertTrue(np.any(rm_lane == 1))
        self.assertTrue(np.any(rm_lane_sub == 1))  # white solid
        self.assertTrue(np.any(rm_lane_sub == 4))  # yellow dashed
        self.assertTrue(np.any(rm_lane_sub == IGNORE_VALUE))  # lane other -> ignore


if __name__ == "__main__":
    unittest.main()
