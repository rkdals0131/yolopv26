from __future__ import annotations

import unittest

import torch

from tools.merge_pv26_lane_family_task_heads import _replace_task_weights


class MergePV26LaneFamilyTaskHeadsTest(unittest.TestCase):
    def test_missing_source_task_key_raises_by_default(self) -> None:
        merged_state = {
            "roadmark_heads.roadmark_heads.stop_line_head.mask.weight": torch.zeros(1),
        }
        source_state = {
            "roadmark_heads.roadmark_heads.stop_line_head.haf_endpoint.weight": torch.ones(1),
        }

        with self.assertRaisesRegex(KeyError, "missing in base checkpoint"):
            _replace_task_weights(merged_state, source_state, task_name="stop_line")

    def test_allow_source_extra_keys_adds_new_task_head_weights(self) -> None:
        merged_state = {
            "roadmark_heads.roadmark_heads.stop_line_head.mask.weight": torch.zeros(1),
        }
        source_state = {
            "roadmark_heads.roadmark_heads.stop_line_head.mask.weight": torch.ones(1),
            "roadmark_heads.roadmark_heads.stop_line_head.haf_endpoint.weight": torch.full((1,), 2.0),
        }

        replaced = _replace_task_weights(
            merged_state,
            source_state,
            task_name="stop_line",
            allow_source_extra_keys=True,
        )

        self.assertEqual(replaced, 2)
        self.assertTrue(
            torch.equal(
                merged_state["roadmark_heads.roadmark_heads.stop_line_head.mask.weight"],
                torch.ones(1),
            )
        )
        self.assertTrue(
            torch.equal(
                merged_state["roadmark_heads.roadmark_heads.stop_line_head.haf_endpoint.weight"],
                torch.full((1,), 2.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
