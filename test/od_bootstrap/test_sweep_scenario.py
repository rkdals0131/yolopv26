from __future__ import annotations

import unittest

from tools.od_bootstrap.data.sweep_types import REQUIRED_TEACHER_ORDER
from tools.od_bootstrap.presets import build_sweep_preset


class ODBootstrapScenarioTests(unittest.TestCase):
    def test_build_sweep_preset_uses_model_centric_run_and_required_teacher_order(self) -> None:
        scenario = build_sweep_preset()

        self.assertEqual(scenario.run.execution_mode, "model-centric")
        self.assertEqual(tuple(teacher.name for teacher in scenario.teachers), REQUIRED_TEACHER_ORDER)

    def test_build_sweep_preset_class_policy_covers_required_classes(self) -> None:
        scenario = build_sweep_preset()

        self.assertEqual(
            sorted(scenario.class_policy),
            ["bike", "obstacle", "pedestrian", "sign", "traffic_cone", "traffic_light", "vehicle"],
        )
        self.assertEqual(scenario.class_policy["traffic_light"].score_threshold, 0.30)
        self.assertEqual(scenario.class_policy["obstacle"].min_box_size, 4)
