import unittest

import numpy as np

from pv26.dataset.masks import (
    IGNORE_VALUE,
    compose_semantic_id_v2,
    compose_semantic_id_v3,
    validate_binary_mask_u8,
    validate_lane_subclass_mask_u8,
    validate_semantic_id_u8,
)


class TestMasks(unittest.TestCase):
    def test_validate_binary_mask_u8_ok(self):
        m = np.array([[0, 1, IGNORE_VALUE]], dtype=np.uint8)
        validate_binary_mask_u8(m, allow_ignore=True)

    def test_validate_binary_mask_u8_rejects_values(self):
        m = np.array([[2]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_binary_mask_u8(m, allow_ignore=True)

    def test_compose_semantic_id_v2_priority(self):
        da = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        rm_lane = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        rm_road = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rm_stop = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        res = compose_semantic_id_v2(da, rm_lane, rm_road, rm_stop)
        self.assertTrue(res.ok)
        # (0,0): rm_road -> lane_marking(2)
        # (0,1): rm_lane -> lane_marking(2)
        # (1,0): background(0)
        # (1,1): stop_line(3)
        self.assertEqual(res.semantic_id.tolist(), [[2, 2], [0, 3]])

    def test_compose_semantic_id_refuses_ignore(self):
        da = np.array([[IGNORE_VALUE]], dtype=np.uint8)
        rm = np.array([[0]], dtype=np.uint8)
        res = compose_semantic_id_v2(da, rm, rm, rm)
        self.assertFalse(res.ok)

    def test_validate_semantic_id_u8(self):
        sem = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        validate_semantic_id_u8(sem, allowed_ids={0, 1, 2, 3})

    def test_validate_lane_subclass_mask_u8(self):
        m = np.array([[0, 1, 2, 3, 4, IGNORE_VALUE]], dtype=np.uint8)
        validate_lane_subclass_mask_u8(m, allow_ignore=True)

    def test_compose_semantic_id_v3_priority(self):
        da = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        lane_sub = np.array([[1, 2], [0, 0]], dtype=np.uint8)
        rm_road = np.array([[0, 0], [1, 0]], dtype=np.uint8)
        rm_stop = np.array([[0, 0], [1, 0]], dtype=np.uint8)
        res = compose_semantic_id_v3(da, lane_sub, rm_road, rm_stop)
        self.assertTrue(res.ok)
        # (0,0): lane white solid(2)
        # (0,1): lane white dashed(3)
        # (1,0): stop line(7) overrides road marker/drivable
        # (1,1): background(0)
        self.assertEqual(res.semantic_id.tolist(), [[2, 3], [7, 0]])


if __name__ == "__main__":
    unittest.main()
