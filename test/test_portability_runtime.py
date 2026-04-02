from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path


class PV26PortabilityRuntimeTests(unittest.TestCase):
    def test_standardization_defaults_follow_repo_root(self) -> None:
        from tools.od_bootstrap.source import aihub, bdd100k

        repo_root = Path(aihub.__file__).resolve().parents[3]
        self.assertEqual(aihub.DEFAULT_REPO_ROOT, repo_root)
        self.assertEqual(bdd100k.DEFAULT_REPO_ROOT, repo_root)
        self.assertTrue(aihub.DEFAULT_AIHUB_ROOT.is_absolute())
        self.assertTrue(aihub.DEFAULT_OBSTACLE_ROOT.is_absolute())
        self.assertTrue(bdd100k.DEFAULT_BDD_ROOT.is_absolute())
        self.assertEqual(aihub.DEFAULT_AIHUB_ROOT.name, "AIHUB")
        self.assertEqual(aihub.DEFAULT_OBSTACLE_ROOT.name, "도로장애물·표면 인지 영상(수도권)")
        self.assertEqual(bdd100k.DEFAULT_BDD_ROOT.name, "BDD100K")

    def test_preflight_report_is_available_without_runtime_side_effects(self) -> None:
        from tools.check_env import check_env

        report = check_env(check_yolo_runtime=False)

        self.assertIn("versions", report)
        self.assertIn("checks", report)
        self.assertIn("torchvision_nms", report["checks"])
        self.assertIn("yolo26", report["checks"])

    def test_check_env_interactive_mode_requires_tty_without_strict_or_json(self) -> None:
        from tools.check_env import _should_run_interactive

        base_args = argparse.Namespace(strict=False, json=False, check_yolo_runtime=False)
        self.assertTrue(_should_run_interactive(base_args, stdin_isatty=True, stdout_isatty=True))
        self.assertFalse(_should_run_interactive(base_args, stdin_isatty=False, stdout_isatty=True))
        self.assertFalse(_should_run_interactive(base_args, stdin_isatty=True, stdout_isatty=False))
        self.assertFalse(
            _should_run_interactive(
                argparse.Namespace(strict=True, json=False, check_yolo_runtime=False),
                stdin_isatty=True,
                stdout_isatty=True,
            )
        )
        self.assertFalse(
            _should_run_interactive(
                argparse.Namespace(strict=False, json=True, check_yolo_runtime=False),
                stdin_isatty=True,
                stdout_isatty=True,
            )
        )

    def test_manifest_header_loader_reads_prefix_without_full_samples_array(self) -> None:
        from tools.check_env import _manifest_header

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "teacher_dataset_manifest.json"
            payload = {
                "version": "od-bootstrap-teacher-dataset-v1",
                "teacher_name": "mobility",
                "sample_count": 3,
                "detection_count": 7,
                "samples": [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}],
            }
            manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            header = _manifest_header(manifest_path)

        self.assertIsNotNone(header)
        assert header is not None
        self.assertEqual(header["sample_count"], 3)
        self.assertEqual(header["detection_count"], 7)
        self.assertNotIn("samples", header)

    def test_workspace_scan_prefers_compact_summaries_for_large_artifacts(self) -> None:
        from tools.check_env import PipelinePaths, scan_workspace_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_root = root / "bootstrap"
            teacher_root = root / "teacher_datasets"
            train_root = root / "train"
            eval_root = root / "eval"
            calibration_root = root / "calibration"
            exhaustive_dataset_root = root / "exhaustive"
            final_dataset_root = root / "final_dataset"
            pv26_run_root = root / "pv26_runs"

            (bootstrap_root / "meta").mkdir(parents=True, exist_ok=True)
            (bootstrap_root / "meta" / "source_prep_manifest.json").write_text("{}", encoding="utf-8")
            (bootstrap_root / "meta" / "bootstrap_image_list.jsonl").write_text("{}", encoding="utf-8")
            (bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta").mkdir(parents=True, exist_ok=True)
            (bootstrap_root / "canonical" / "aihub_standardized" / "meta").mkdir(parents=True, exist_ok=True)
            (bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta" / "conversion_report.json").write_text(
                json.dumps(
                    {
                        "dataset": {"processed_samples": 100},
                        "source_inventory_snapshot": {"dataset": {"local_inventory": {"splits": {"train": {"json_files": 100}}}}},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta" / "source_inventory.json").write_text(
                json.dumps({"dataset": {"local_inventory": {"splits": {"train": {"json_files": 100}}}}}, indent=2) + "\n",
                encoding="utf-8",
            )
            (bootstrap_root / "canonical" / "aihub_standardized" / "meta" / "conversion_report.json").write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"dataset_key": "aihub_lane_seoul", "processed_samples": 50},
                            {"dataset_key": "aihub_traffic_seoul", "processed_samples": 60},
                            {"dataset_key": "aihub_obstacle_seoul", "processed_samples": 40},
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (bootstrap_root / "canonical" / "aihub_standardized" / "meta" / "source_inventory.json").write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"dataset_key": "aihub_lane_seoul", "local_inventory": {"splits": {"train": {"json_files": 50}}}},
                            {"dataset_key": "aihub_traffic_seoul", "local_inventory": {"splits": {"train": {"json_files": 60}}}},
                            {"dataset_key": "aihub_obstacle_seoul", "local_inventory": {"splits": {"train": {"json_files": 40}}}},
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            for teacher_name, sample_count in (("mobility", 100), ("signal", 60), ("obstacle", 40)):
                meta_root = teacher_root / teacher_name / "meta"
                meta_root.mkdir(parents=True, exist_ok=True)
                (meta_root / "teacher_dataset_summary.json").write_text(
                    json.dumps({"teacher_name": teacher_name, "sample_count": sample_count}, indent=2) + "\n",
                    encoding="utf-8",
                )
                (train_root / teacher_name / "weights").mkdir(parents=True, exist_ok=True)
                (train_root / teacher_name / "weights" / "best.pt").write_text("checkpoint", encoding="utf-8")
                (train_root / teacher_name / "run_summary.json").write_text("{}", encoding="utf-8")

            (calibration_root).mkdir(parents=True, exist_ok=True)
            (calibration_root / "class_policy.yaml").write_text("vehicle: {}\n", encoding="utf-8")
            (calibration_root / "hard_negative_manifest.json").write_text("{}", encoding="utf-8")
            (calibration_root / "calibration_report.json").write_text(
                json.dumps(
                    {
                        "teachers": [{"teacher_name": "mobility", "sample_count": 10}],
                        "classes": {name: {"meets_precision_floor": True} for name in ("vehicle", "bike", "pedestrian", "traffic_light", "sign", "traffic_cone", "obstacle")},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            latest_exhaustive = exhaustive_dataset_root / "run_001" / "meta"
            latest_exhaustive.mkdir(parents=True, exist_ok=True)
            (latest_exhaustive / "materialization_summary.json").write_text(
                json.dumps({"run_id": "run_001", "sample_count": 200}, indent=2) + "\n",
                encoding="utf-8",
            )

            (final_dataset_root / "meta").mkdir(parents=True, exist_ok=True)
            (final_dataset_root / "meta" / "final_dataset_summary.json").write_text(
                json.dumps({"sample_count": 250, "dataset_counts": {"aihub_lane_seoul": 50}, "rerun_mode": "atomic_overwrite"}, indent=2) + "\n",
                encoding="utf-8",
            )
            (final_dataset_root / "meta" / "final_dataset_publish_state.json").write_text(
                json.dumps({"status": "completed", "rerun_mode": "atomic_overwrite"}, indent=2) + "\n",
                encoding="utf-8",
            )

            latest_pv26 = pv26_run_root / "run_001"
            latest_pv26.mkdir(parents=True, exist_ok=True)
            (latest_pv26 / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed_phases": 4,
                        "total_phases": 4,
                        "train_defaults": {"backbone_variant": "s"},
                        "latest_phase_stage": "stage_4_lane_family_finetune",
                        "latest_backbone_variant": "s",
                        "latest_selection_metric_path": "val.metrics.lane_family.mean_f1",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            paths = PipelinePaths(
                repo_root=root,
                raw_bdd_root=root / "bdd_raw",
                raw_aihub_root=root / "aihub_raw",
                bootstrap_root=bootstrap_root,
                teacher_dataset_root=teacher_root,
                teacher_train_root=train_root,
                teacher_eval_root=eval_root,
                calibration_root=calibration_root,
                exhaustive_run_root=root / "exhaustive_runs",
                exhaustive_dataset_root=exhaustive_dataset_root,
                final_dataset_root=final_dataset_root,
                pv26_run_root=pv26_run_root,
                user_paths_config_path=root / "config" / "user_paths.yaml",
                od_hyperparameters_config_path=root / "config" / "od.yaml",
                pv26_hyperparameters_config_path=root / "config" / "pv26.yaml",
            )
            paths.raw_bdd_root.mkdir(parents=True, exist_ok=True)
            paths.raw_aihub_root.mkdir(parents=True, exist_ok=True)

            report = {
                "versions": {"torch": "2.0.0", "torchvision": "0.1.0", "ultralytics": "8.0.0"},
                "checks": {
                    "torchvision_nms": {"callable": True},
                    "yolo26": {"importable": True, "runtime_load_ok": True},
                },
            }
            snapshot = scan_workspace_status(report, paths=paths)

        row_map = {row.stage: row for row in snapshot.rows}
        self.assertEqual(row_map["Teacher dataset"].verdict, "OK")
        self.assertIn("mobility 100", row_map["Teacher dataset"].current_state)
        self.assertEqual(row_map["Calibration"].verdict, "OK")
        self.assertEqual(row_map["PV26 학습 run"].verdict, "OK")
        self.assertIn("rerun=atomic_overwrite", row_map["최종 병합 데이터셋"].current_state)
        self.assertIn("publish=completed", row_map["최종 병합 데이터셋"].current_state)
        self.assertIn("phases=4/4", row_map["PV26 학습 run"].current_state)
        self.assertIn("stage=stage_4_lane_family_finetune", row_map["PV26 학습 run"].current_state)
        self.assertIn("backbone=s", row_map["PV26 학습 run"].current_state)
        self.assertIn("selection=val.metrics.lane_family.mean_f1", row_map["PV26 학습 run"].current_state)

    def test_empty_final_dataset_directory_stays_todo(self) -> None:
        from tools.check_env import PipelinePaths, scan_workspace_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dataset_root = root / "final_dataset"
            final_dataset_root.mkdir(parents=True, exist_ok=True)
            paths = PipelinePaths(
                repo_root=root,
                raw_bdd_root=root / "bdd_raw",
                raw_aihub_root=root / "aihub_raw",
                bootstrap_root=root / "bootstrap",
                teacher_dataset_root=root / "teacher_datasets",
                teacher_train_root=root / "teacher_runs",
                teacher_eval_root=root / "teacher_eval",
                calibration_root=root / "calibration",
                exhaustive_run_root=root / "exhaustive_runs",
                exhaustive_dataset_root=root / "exhaustive",
                final_dataset_root=final_dataset_root,
                pv26_run_root=root / "pv26_runs",
                user_paths_config_path=root / "config" / "user_paths.yaml",
                od_hyperparameters_config_path=root / "config" / "od.yaml",
                pv26_hyperparameters_config_path=root / "config" / "pv26.yaml",
            )
            paths.raw_bdd_root.mkdir(parents=True, exist_ok=True)
            paths.raw_aihub_root.mkdir(parents=True, exist_ok=True)
            report = {
                "versions": {"torch": "2.0.0", "torchvision": "0.1.0", "ultralytics": "8.0.0"},
                "checks": {
                    "torchvision_nms": {"callable": True},
                    "yolo26": {"importable": True, "runtime_load_ok": True},
                },
            }

            snapshot = scan_workspace_status(report, paths=paths)

        row_map = {row.stage: row for row in snapshot.rows}
        self.assertEqual(row_map["최종 병합 데이터셋"].verdict, "TODO")
        self.assertEqual(row_map["최종 병합 데이터셋"].current_state, "없음")


if __name__ == "__main__":
    unittest.main()
