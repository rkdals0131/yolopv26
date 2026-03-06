import unittest

import numpy as np

from pv26.dataset.sources.wod import WAYMO_SEM_LANE_MARKER, WAYMO_SEM_ROAD, WAYMO_SEM_ROAD_MARKER, semantic_to_pv26_da_rm_masks


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


if __name__ == "__main__":
    unittest.main()

