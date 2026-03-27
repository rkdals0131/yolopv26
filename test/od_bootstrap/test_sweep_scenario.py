from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from tools.od_bootstrap.sweep.scenario import REQUIRED_TEACHER_ORDER, load_sweep_scenario


class ODBootstrapScenarioTests(unittest.TestCase):
    def test_shipped_sweep_default_and_dryrun_configs_have_expected_dry_run_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        default_payload = yaml.safe_load(
            (repo_root / "tools" / "od_bootstrap" / "config" / "sweep" / "model_centric.default.yaml").read_text(
                encoding="utf-8"
            )
        )
        dryrun_payload = yaml.safe_load(
            (repo_root / "tools" / "od_bootstrap" / "config" / "sweep" / "model_centric.dryrun.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(bool(default_payload["run"]["dry_run"]))
        self.assertTrue(bool(dryrun_payload["run"]["dry_run"]))

    def test_load_sweep_scenario_resolves_paths_and_keeps_teacher_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            data_dir = root / "data"
            config_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            scenario_path = config_dir / "bootstrap.yaml"
            image_list_manifest = data_dir / "image_list.jsonl"
            image_list_manifest.write_text("", encoding="utf-8")
            class_policy_path = data_dir / "class_policy.yaml"
            class_policy_path.write_text(
                textwrap.dedent(
                    """
                    vehicle: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                    bike: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                    pedestrian: {score_threshold: 0.35, nms_iou_threshold: 0.50, min_box_size: 8}
                    traffic_light: {score_threshold: 0.40, nms_iou_threshold: 0.45, min_box_size: 6, allowed_source_datasets: [aihub_traffic_seoul], center_y_range: [0.0, 0.7]}
                    sign: {score_threshold: 0.40, nms_iou_threshold: 0.50, min_box_size: 8, suppress_with_classes: [traffic_light], cross_class_iou_threshold: 0.35}
                    traffic_cone: {score_threshold: 0.45, nms_iou_threshold: 0.45, min_box_size: 8}
                    obstacle: {score_threshold: 0.55, nms_iou_threshold: 0.40, min_box_size: 12}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario_path.write_text(
                textwrap.dedent(
                    """
                    run:
                      output_root: ../runs/od_bootstrap
                      execution_mode: model-centric
                      dry_run: false
                      predict_conf: 0.001
                      predict_iou: 0.99
                    image_list:
                      manifest_path: ../data/image_list.jsonl
                    materialization:
                      output_root: ../seg_dataset/pv26_od_bootstrap/exhaustive_od
                    class_policy_path: ../data/class_policy.yaml
                    teachers:
                      - name: mobility
                        base_model: yolov26n
                        checkpoint_path: ../weights/mobility.pt
                        model_version: mobility_v1
                        classes: [vehicle, bike, pedestrian]
                      - name: signal
                        base_model: yolov26n
                        checkpoint_path: ../weights/signal.pt
                        model_version: signal_v1
                        classes: [traffic_light, sign]
                      - name: obstacle
                        base_model: yolov26n
                        checkpoint_path: ../weights/obstacle.pt
                        model_version: obstacle_v1
                        classes: [traffic_cone, obstacle]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_sweep_scenario(scenario_path)

            self.assertEqual(scenario.run.output_root, (root / "runs" / "od_bootstrap").resolve())
            self.assertEqual(scenario.image_list.manifest_path, image_list_manifest.resolve())
            self.assertEqual(scenario.class_policy_path, class_policy_path.resolve())
            self.assertEqual(tuple(teacher.name for teacher in scenario.teachers), REQUIRED_TEACHER_ORDER)
            self.assertEqual(scenario.teachers[0].checkpoint_path, (root / "weights" / "mobility.pt").resolve())
            self.assertEqual(scenario.class_policy["obstacle"].min_box_size, 12)
            self.assertEqual(scenario.class_policy["traffic_light"].allowed_source_datasets, ("aihub_traffic_seoul",))
            self.assertEqual(scenario.class_policy["sign"].suppress_with_classes, ("traffic_light",))
            self.assertEqual(scenario.class_policy["traffic_light"].center_y_range, (0.0, 0.7))
            self.assertAlmostEqual(scenario.run.predict_conf, 0.001, places=6)
            self.assertEqual(
                scenario.materialization.output_root,
                (root / "seg_dataset" / "pv26_od_bootstrap" / "exhaustive_od").resolve(),
            )

    def test_load_sweep_scenario_rejects_wrong_teacher_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = root / "bootstrap.yaml"
            class_policy_path = root / "class_policy.yaml"
            class_policy_path.write_text(
                textwrap.dedent(
                    """
                    vehicle: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                    bike: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                    pedestrian: {score_threshold: 0.35, nms_iou_threshold: 0.50, min_box_size: 8}
                    traffic_light: {score_threshold: 0.40, nms_iou_threshold: 0.45, min_box_size: 6}
                    sign: {score_threshold: 0.40, nms_iou_threshold: 0.50, min_box_size: 8}
                    traffic_cone: {score_threshold: 0.45, nms_iou_threshold: 0.45, min_box_size: 8}
                    obstacle: {score_threshold: 0.55, nms_iou_threshold: 0.40, min_box_size: 12}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    image_list:
                      manifest_path: image_list.jsonl
                    materialization:
                      output_root: exhaustive_od
                    class_policy_path: class_policy.yaml
                    teachers:
                      - name: signal
                        base_model: yolov26n
                        checkpoint_path: signal.pt
                        model_version: signal_v1
                        classes: [traffic_light, sign]
                      - name: mobility
                        base_model: yolov26n
                        checkpoint_path: mobility.pt
                        model_version: mobility_v1
                        classes: [vehicle, bike, pedestrian]
                      - name: obstacle
                        base_model: yolov26n
                        checkpoint_path: obstacle.pt
                        model_version: obstacle_v1
                        classes: [traffic_cone, obstacle]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (root / "image_list.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "teachers must be ordered"):
                load_sweep_scenario(scenario_path)
