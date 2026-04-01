from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.od_bootstrap.data.sweep_types import REQUIRED_TEACHER_ORDER
from tools.od_bootstrap.presets import build_sweep_preset


def _build_isolated_sweep_preset(
    *,
    calibration_root: Path | None = None,
    hyperparameters_config: dict[str, object] | None = None,
):
    if calibration_root is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            return _build_isolated_sweep_preset(
                calibration_root=Path(temp_dir),
                hyperparameters_config=hyperparameters_config,
            )
    with patch(
        "tools.od_bootstrap.presets.load_user_paths_config",
        return_value={"od_bootstrap": {"runs": {"calibration_root": str(calibration_root)}}},
    ), patch(
        "tools.od_bootstrap.presets.load_user_hyperparameters_config",
        return_value=hyperparameters_config or {},
    ):
        return build_sweep_preset()


class ODBootstrapScenarioTests(unittest.TestCase):
    def test_build_sweep_preset_uses_model_centric_run_and_required_teacher_order(self) -> None:
        scenario = _build_isolated_sweep_preset()

        self.assertEqual(scenario.run.execution_mode, "model-centric")
        self.assertEqual(tuple(teacher.name for teacher in scenario.teachers), REQUIRED_TEACHER_ORDER)

    def test_build_sweep_preset_class_policy_covers_required_classes(self) -> None:
        scenario = _build_isolated_sweep_preset()

        self.assertEqual(
            sorted(scenario.class_policy),
            ["bike", "obstacle", "pedestrian", "sign", "traffic_cone", "traffic_light", "vehicle"],
        )
        self.assertEqual(scenario.class_policy["traffic_light"].score_threshold, 0.30)
        self.assertEqual(scenario.class_policy["obstacle"].min_box_size, 4)

    def test_build_sweep_preset_prefers_generated_class_policy_yaml_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "class_policy.yaml"
            policy_path.write_text(
                "\n".join(
                    [
                        "vehicle:",
                        "  score_threshold: 0.91",
                        "  nms_iou_threshold: 0.42",
                        "  min_box_size: 9",
                        "  center_y_range: [0.1, 0.8]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scenario = _build_isolated_sweep_preset(
                calibration_root=root,
                hyperparameters_config={
                    "od_bootstrap": {
                        "exhaustive_od": {
                            "class_policy_defaults": {
                                "vehicle": {"score_threshold": 0.25, "nms_iou_threshold": 0.55, "min_box_size": 4},
                                "bike": {"score_threshold": 0.25, "nms_iou_threshold": 0.55, "min_box_size": 4},
                                "pedestrian": {"score_threshold": 0.25, "nms_iou_threshold": 0.55, "min_box_size": 4},
                                "traffic_light": {"score_threshold": 0.30, "nms_iou_threshold": 0.50, "min_box_size": 4},
                                "sign": {"score_threshold": 0.25, "nms_iou_threshold": 0.50, "min_box_size": 4},
                                "traffic_cone": {"score_threshold": 0.25, "nms_iou_threshold": 0.55, "min_box_size": 4},
                                "obstacle": {"score_threshold": 0.25, "nms_iou_threshold": 0.55, "min_box_size": 4},
                            }
                        }
                    }
                },
            )

        self.assertEqual(scenario.class_policy_path, policy_path.resolve())
        self.assertEqual(scenario.class_policy["vehicle"].score_threshold, 0.91)
        self.assertEqual(scenario.class_policy["vehicle"].nms_iou_threshold, 0.42)
        self.assertEqual(scenario.class_policy["vehicle"].min_box_size, 9)
        self.assertEqual(scenario.class_policy["vehicle"].center_y_range, (0.1, 0.8))
        self.assertEqual(scenario.class_policy["traffic_light"].score_threshold, 0.30)
