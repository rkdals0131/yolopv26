from __future__ import annotations

import math
import unittest

from common.scalars import flatten_scalar_tree


class ScalarTreeTests(unittest.TestCase):
    def test_flatten_scalar_tree_handles_nested_mappings_sequences_and_bools(self) -> None:
        payload = {
            "loss": {"total": 1.5, "enabled": True},
            "lr": [0.1, 0.01],
        }

        self.assertEqual(
            flatten_scalar_tree("train", payload),
            [
                ("train/loss/total", 1.5),
                ("train/loss/enabled", 1.0),
                ("train/lr/0", 0.1),
                ("train/lr/1", 0.01),
            ],
        )

    def test_flatten_scalar_tree_skips_non_finite_and_non_scalar_values(self) -> None:
        payload = {"ok": 1, "bad": math.inf, "text": "ignore"}

        self.assertEqual(flatten_scalar_tree("", payload), [("ok", 1.0)])
