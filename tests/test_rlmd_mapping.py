import unittest

import numpy as np

from pv26.dataset.masks import IGNORE_VALUE
from pv26.dataset.sources.rlmd import RlmdRgbClass, rlmd_code_mask_to_pv26_rm_masks


class TestRlmdMapping(unittest.TestCase):
    def test_rlmd_code_to_pv26_masks_basic(self):
        bg = RlmdRgbClass(0, "background", (0, 0, 0))
        white_solid = RlmdRgbClass(4, "solid single white", (237, 28, 36))
        yellow_dashed = RlmdRgbClass(10, "dashed single yellow", (153, 217, 234))
        red_solid = RlmdRgbClass(6, "solid single red", (185, 122, 87))
        stop_line = RlmdRgbClass(3, "stop line", (61, 72, 204))
        left_arrow = RlmdRgbClass(11, "left arrow", (158, 159, 76))

        palette = {
            bg.code: bg,
            white_solid.code: white_solid,
            yellow_dashed.code: yellow_dashed,
            red_solid.code: red_solid,
            stop_line.code: stop_line,
            left_arrow.code: left_arrow,
        }

        code = np.array(
            [
                [bg.code, white_solid.code, left_arrow.code],
                [stop_line.code, red_solid.code, yellow_dashed.code],
            ],
            dtype=np.uint32,
        )

        rm_lane, rm_road, rm_stop, rm_lane_sub, unknown = rlmd_code_mask_to_pv26_rm_masks(code, palette=palette)
        self.assertEqual(unknown, 0)

        self.assertEqual(rm_lane.tolist(), [[0, 1, 0], [0, 1, 1]])
        self.assertEqual(rm_road.tolist(), [[0, 0, 1], [1, 0, 0]])
        self.assertEqual(rm_stop.tolist(), [[0, 0, 0], [1, 0, 0]])

        # white solid -> 1, yellow dashed -> 4, red lane marker -> ignore
        self.assertEqual(rm_lane_sub[0, 1], 1)
        self.assertEqual(rm_lane_sub[1, 2], 4)
        self.assertEqual(rm_lane_sub[1, 1], IGNORE_VALUE)

    def test_unknown_codes_become_ignore(self):
        palette = {RlmdRgbClass(0, "background", (0, 0, 0)).code: RlmdRgbClass(0, "background", (0, 0, 0))}
        code = np.array([[123456]], dtype=np.uint32)  # not in palette
        rm_lane, rm_road, rm_stop, rm_lane_sub, unknown = rlmd_code_mask_to_pv26_rm_masks(code, palette=palette)
        self.assertEqual(unknown, 1)
        self.assertEqual(rm_lane.tolist(), [[IGNORE_VALUE]])
        self.assertEqual(rm_road.tolist(), [[IGNORE_VALUE]])
        self.assertEqual(rm_stop.tolist(), [[IGNORE_VALUE]])
        self.assertEqual(rm_lane_sub.tolist(), [[IGNORE_VALUE]])


if __name__ == "__main__":
    unittest.main()
