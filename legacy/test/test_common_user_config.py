from __future__ import annotations

import unittest

from common.user_config import deep_merge_mappings


class CommonUserConfigTests(unittest.TestCase):
    def test_deep_merge_mappings_recursively_merges_without_mutating_inputs(self) -> None:
        base = {
            "dataset": {
                "root": "/tmp/default",
                "options": {"shuffle": True, "batch_size": 4},
            },
            "run": {"name": "baseline"},
        }
        overrides = {
            "dataset": {
                "options": {"batch_size": 8},
                "additional_roots": ["/tmp/extra"],
            },
            "preview": {"enabled": False},
        }

        merged = deep_merge_mappings(base, overrides)

        self.assertEqual(
            merged,
            {
                "dataset": {
                    "root": "/tmp/default",
                    "options": {"shuffle": True, "batch_size": 8},
                    "additional_roots": ["/tmp/extra"],
                },
                "run": {"name": "baseline"},
                "preview": {"enabled": False},
            },
        )
        self.assertEqual(base["dataset"]["options"]["batch_size"], 4)
        self.assertNotIn("additional_roots", base["dataset"])
