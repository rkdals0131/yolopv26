from __future__ import annotations

import unittest

import torch

from model.engine.multitask_conflict import (
    accumulate_pcgrad_trunk_update,
    compute_pcgrad_trunk_update,
    current_pcgrad_trunk_update,
    init_multitask_conflict_state,
    normalize_multitask_conflict,
    reset_multitask_conflict_state,
)


class MultiTaskConflictTests(unittest.TestCase):
    def test_normalize_pcgrad_style_uses_pv26_task_set(self) -> None:
        config = normalize_multitask_conflict(
            {
                "enabled": True,
                "mode": "pcgrad_style",
                "tasks": ["det", "tl_attr", "lane", "stop_line", "crosswalk"],
            }
        )
        self.assertTrue(config["enabled"])
        self.assertEqual(config["mode"], "pcgrad_style")
        self.assertEqual(config["tasks"], ("det", "tl_attr", "lane", "stop_line", "crosswalk"))

    def test_compute_pcgrad_trunk_update_returns_snapshot(self) -> None:
        p1 = torch.nn.Parameter(torch.tensor([1.0, -1.0], requires_grad=True))
        p2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        losses = {
            "det": (p1[0] - p1[1] + p2[0]) ** 2,
            "tl_attr": (p1[0] + p1[1]) ** 2,
            "lane": (p1[0] + p1[1] - p2[0]) ** 2,
            "stop_line": (p1[0] + 2 * p2[0]) ** 2,
            "crosswalk": (p1[1] - p2[0]) ** 2,
        }
        grads, snapshot = compute_pcgrad_trunk_update(
            losses,
            params=[p1, p2],
            config={
                "enabled": True,
                "mode": "pcgrad_style",
                "tasks": ("det", "tl_attr", "lane", "stop_line", "crosswalk"),
            },
        )
        self.assertIsNotNone(grads)
        assert grads is not None
        self.assertEqual(len(grads), 2)
        self.assertTrue(snapshot["enabled"])
        self.assertEqual(snapshot["mode"], "pcgrad_style")
        self.assertEqual(snapshot["tasks"], ["det", "tl_attr", "lane", "stop_line", "crosswalk"])
        self.assertIn("pairwise_dots", snapshot)
        self.assertGreaterEqual(snapshot["combined_grad_norm"], 0.0)

    def test_pcgrad_state_accumulates_microbatches(self) -> None:
        p = torch.nn.Parameter(torch.tensor([1.0, -1.0], requires_grad=True))
        config = {"enabled": True, "mode": "pcgrad_style", "tasks": ("det", "lane", "crosswalk")}
        state = init_multitask_conflict_state(config)
        grads1, _ = compute_pcgrad_trunk_update(
            {
                "det": (p[0] + p[1]) ** 2,
                "lane": (p[0] - p[1]) ** 2,
                "crosswalk": (p[0]) ** 2,
            },
            params=[p],
            config=config,
        )
        grads2, _ = compute_pcgrad_trunk_update(
            {
                "det": (2 * p[0]) ** 2,
                "lane": (3 * p[1]) ** 2,
                "crosswalk": (p[0] + 2 * p[1]) ** 2,
            },
            params=[p],
            config=config,
        )
        state = accumulate_pcgrad_trunk_update(state, grads1)
        state = accumulate_pcgrad_trunk_update(state, grads2)
        combined = current_pcgrad_trunk_update(state)
        self.assertIsNotNone(combined)
        assert combined is not None
        self.assertEqual(state["accumulated_micro_steps"], 2)
        self.assertEqual(len(combined), 1)
        self.assertIsNotNone(combined[0])
        state = reset_multitask_conflict_state(state)
        self.assertEqual(state["accumulated_micro_steps"], 0)
        self.assertEqual(state["accumulated_trunk_grads"], [])

    def test_pcgrad_ignores_non_differentiable_losses(self) -> None:
        p = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        grads, snapshot = compute_pcgrad_trunk_update(
            {
                "det": torch.tensor(1.0),
                "lane": (p[0] * 2.0) ** 2,
            },
            params=[p],
            config={"enabled": True, "mode": "pcgrad_style", "tasks": ("det", "lane")},
        )
        self.assertIsNotNone(grads)
        self.assertEqual(snapshot["tasks"], ["lane"])


if __name__ == "__main__":
    unittest.main()
