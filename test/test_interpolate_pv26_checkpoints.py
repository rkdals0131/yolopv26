from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from tools.interpolate_pv26_checkpoints import interpolate_checkpoints


class InterpolatePV26CheckpointsTests(unittest.TestCase):
    def test_stop_line_scope_interpolates_only_stop_line_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "stage": "lane_family",
                "adapter_state_dict": {"trunk.weight": torch.tensor([1.0])},
                "heads_state_dict": {
                    "lane_head.weight": torch.tensor([1.0]),
                    "stop_line_head.weight": torch.tensor([2.0]),
                    "crosswalk_head.weight": torch.tensor([3.0]),
                    "stop_line_head.counter": torch.tensor([7], dtype=torch.int64),
                },
                "optimizer_state_dict": {"state": {"stale": True}},
                "scheduler_state_dict": {"stale": True},
            }
            trained = {
                "adapter_state_dict": {"trunk.weight": torch.tensor([9.0])},
                "heads_state_dict": {
                    "lane_head.weight": torch.tensor([9.0]),
                    "stop_line_head.weight": torch.tensor([6.0]),
                    "crosswalk_head.weight": torch.tensor([9.0]),
                    "stop_line_head.counter": torch.tensor([99], dtype=torch.int64),
                },
            }
            base_path = root / "base.pt"
            trained_path = root / "trained.pt"
            output_path = root / "out.pt"
            torch.save(base, base_path)
            torch.save(trained, trained_path)

            summary = interpolate_checkpoints(
                base_checkpoint=base_path,
                trained_checkpoint=trained_path,
                output_path=output_path,
                alpha=0.25,
                scope="stop_line_head",
            )

            self.assertEqual(summary["states"]["adapter_state_dict"]["kept_base"], 1)
            checkpoint = torch.load(output_path, map_location="cpu")
            self.assertAlmostEqual(float(checkpoint["heads_state_dict"]["lane_head.weight"][0]), 1.0)
            self.assertAlmostEqual(float(checkpoint["heads_state_dict"]["crosswalk_head.weight"][0]), 3.0)
            self.assertAlmostEqual(float(checkpoint["heads_state_dict"]["stop_line_head.weight"][0]), 3.0)
            self.assertEqual(int(checkpoint["heads_state_dict"]["stop_line_head.counter"][0]), 7)
            self.assertEqual(checkpoint["optimizer_state_dict"], {})
            self.assertNotIn("scheduler_state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main()
