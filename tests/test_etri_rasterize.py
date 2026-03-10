import unittest

import numpy as np

from pv26.dataset.masks import IGNORE_VALUE
from pv26.dataset.sources.etri import rasterize_etri_type_a_masks


class TestEtriRasterize(unittest.TestCase):
    def test_rasterize_simple_polygons(self):
        data = {
            "imgHeight": 10,
            "imgWidth": 10,
            "objects": [
                {
                    "label": "out of roi",
                    "deleted": 1,
                    "polygon": [[0, 0], [9, 0], [9, 2], [0, 2]],
                },
                {
                    "label": "road",
                    "deleted": 0,
                    "polygon": [[0, 5], [9, 5], [9, 9], [0, 9]],
                },
                {
                    "label": "whsol",
                    "deleted": 0,
                    "polygon": [[1, 6], [3, 6], [3, 7], [1, 7]],
                },
                {
                    "label": "crosswalk",
                    "deleted": 0,
                    "polygon": [[5, 6], [7, 6], [7, 7], [5, 7]],
                },
                {
                    "label": "stop line",
                    "deleted": 0,
                    "polygon": [[5, 8], [7, 8], [7, 9], [5, 9]],
                },
            ],
        }

        da, rm_lane, rm_road, rm_stop, rm_lane_sub = rasterize_etri_type_a_masks(data, width=10, height=10)

        # out-of-roi must be ignored across all masks
        self.assertEqual(int(da[0, 0]), IGNORE_VALUE)
        self.assertEqual(int(rm_lane[0, 0]), IGNORE_VALUE)
        self.assertEqual(int(rm_road[0, 0]), IGNORE_VALUE)
        self.assertEqual(int(rm_stop[0, 0]), IGNORE_VALUE)
        self.assertEqual(int(rm_lane_sub[0, 0]), IGNORE_VALUE)

        # drivable road region
        self.assertEqual(int(da[6, 6]), 1)
        self.assertEqual(int(da[4, 4]), 0)

        # lane marker + subclass
        self.assertEqual(int(rm_lane[6, 2]), 1)
        self.assertEqual(int(rm_lane_sub[6, 2]), 1)  # whsol -> white solid

        # non-lane road marker
        self.assertEqual(int(rm_road[6, 6]), 1)  # crosswalk

        # stop line maps to stop_line and also road_marker_non_lane
        self.assertEqual(int(rm_stop[8, 6]), 1)
        self.assertEqual(int(rm_road[8, 6]), 1)

        # Masks must be uint8
        for m in [da, rm_lane, rm_road, rm_stop, rm_lane_sub]:
            self.assertEqual(m.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()

