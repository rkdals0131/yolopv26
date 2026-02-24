import unittest

import numpy as np

from pv26.bdd import bdd_record_to_image_name, bdd_record_to_rm_masks, bdd_record_to_yolo_lines
from pv26.masks import IGNORE_VALUE


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
        # class 0 car, cx=0.2, cy=0.4, w=0.2, h=0.4
        self.assertEqual(lines[0], "0 0.200000 0.400000 0.200000 0.400000")

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


if __name__ == "__main__":
    unittest.main()
