from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.od_bootstrap.build.exhaustive_od import EXHAUSTIVE_MATERIALIZATION_SUMMARY_NAME
from tools.od_bootstrap.build.final_dataset import FINAL_DATASET_SUMMARY_NAME
from tools.od_bootstrap.build.final_dataset_stats import FINAL_DATASET_STATS_NAME


class PV26PortabilityRuntimeTests(unittest.TestCase):
    def test_ascii_input_ignores_invalid_utf8_bytes(self) -> None:
        from rich.console import Console
        from tools.check_env.launch import _ascii_input

        class _FakeStdin:
            def __init__(self, payload: bytes) -> None:
                self.buffer = io.BytesIO(payload)

        console = Console(file=io.StringIO())
        stdout = io.StringIO()

        with unittest.mock.patch.object(sys, "stdin", _FakeStdin(b"\xff\xfe yes\n")):
            with unittest.mock.patch.object(sys, "stdout", stdout):
                raw = _ascii_input(console, "yes/no > ")

        self.assertEqual(raw, "yes")
        self.assertEqual(stdout.getvalue(), "yes/no > ")

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

    def test_check_env_main_json_mode_prints_report_without_interactive_loop(self) -> None:
        from tools import check_env as check_env_module

        report = {
            "repo_root": "/tmp/repo",
            "python": "3.11.0",
            "versions": {
                "torch": "2.0.0",
                "torchvision": "0.1.0",
                "ultralytics": "8.0.0",
                "numpy": "1.26.0",
                "scipy": "1.11.0",
                "PIL": "10.0.0",
            },
            "checks": {
                "torchvision_nms": {"callable": True},
                "yolo26": {"importable": True, "runtime_load_ok": True},
            },
        }
        stdout = io.StringIO()

        with unittest.mock.patch.object(check_env_module, "check_env", return_value=report):
            with unittest.mock.patch.object(check_env_module, "_interactive_loop") as interactive_loop:
                with contextlib.redirect_stdout(stdout):
                    exit_code = check_env_module.main(["--json"])

        self.assertEqual(exit_code, 0)
        interactive_loop.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_check_env_main_interactive_mode_delegates_to_interactive_loop(self) -> None:
        from tools import check_env as check_env_module

        with unittest.mock.patch.object(check_env_module, "_should_run_interactive", return_value=True):
            with unittest.mock.patch.object(check_env_module, "_interactive_loop", return_value=7) as interactive_loop:
                exit_code = check_env_module.main([])

        self.assertEqual(exit_code, 7)
        interactive_loop.assert_called_once()

    def test_check_env_action_catalog_includes_export_and_resume_actions(self) -> None:
        from tools.check_env import PipelinePaths, _action_catalog

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = PipelinePaths(
                repo_root=root,
                raw_bdd_root=root / "bdd_raw",
                raw_aihub_root=root / "aihub_raw",
                bootstrap_root=root / "bootstrap",
                teacher_dataset_root=root / "teacher_datasets",
                teacher_train_root=root / "teacher_train",
                teacher_eval_root=root / "teacher_eval",
                calibration_root=root / "calibration",
                exhaustive_run_root=root / "exhaustive_runs",
                exhaustive_dataset_root=root / "exhaustive",
                final_dataset_root=root / "final_dataset",
                pv26_run_root=root / "pv26_runs",
                user_paths_config_path=root / "config" / "user_paths.yaml",
                od_hyperparameters_config_path=root / "config" / "od.yaml",
                pv26_hyperparameters_config_path=root / "config" / "pv26.yaml",
            )

            actions = _action_catalog(paths)

        stress_action = next(item for item in actions if item.key == "D")
        self.assertIn("interactive", stress_action.command_display)
        self.assertEqual(stress_action.argv, ())
        self.assertIn("phase", stress_action.label.lower())
        resume_action = next(item for item in actions if item.key == "E")
        self.assertIn("resume", resume_action.command_display)
        self.assertEqual(resume_action.argv, ())
        self.assertIn("resume", resume_action.label.lower())
        retrain_action = next(item for item in actions if item.key == "K")
        self.assertIn("retrain", retrain_action.label.lower())
        self.assertEqual(retrain_action.argv, ())
        stats_action = next(item for item in actions if item.key == "L")
        self.assertIn("stats", stats_action.label.lower())
        self.assertEqual(stats_action.argv, ())
        export_action = next(item for item in actions if item.key == "F")
        self.assertIn("TorchScript", export_action.label)
        self.assertEqual(export_action.argv, ())
        teacher_export_action = next(item for item in actions if item.key == "G")
        self.assertIn("Mobility", teacher_export_action.label)
        self.assertEqual(teacher_export_action.argv, ())

    def test_scan_pv26_resume_candidates_filters_completed_runs(self) -> None:
        from tools.check_env import _scan_pv26_resume_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "pv26_runs"
            resumable_run = run_root / "run_active"
            completed_run = run_root / "run_done"
            (resumable_run / "phase_2" / "checkpoints").mkdir(parents=True, exist_ok=True)
            (completed_run).mkdir(parents=True, exist_ok=True)
            (resumable_run / "phase_2" / "checkpoints" / "last.pt").write_text("checkpoint", encoding="utf-8")
            (resumable_run / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "updated_at": "2026-04-02T12:34:56",
                        "phases": [
                            {
                                "name": "head_warmup",
                                "stage": "stage_1_frozen_trunk_warmup",
                                "status": "completed",
                                "run_dir": str(resumable_run / "phase_1"),
                                "best_checkpoint_path": str(resumable_run / "phase_1" / "checkpoints" / "best.pt"),
                            },
                            {
                                "name": "partial_unfreeze",
                                "stage": "stage_2_partial_unfreeze",
                                "status": "running",
                                "run_dir": str(resumable_run / "phase_2"),
                                "best_checkpoint_path": None,
                            },
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (completed_run / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "updated_at": "2026-04-02T12:35:00",
                        "phases": [
                            {
                                "name": "head_warmup",
                                "stage": "stage_1_frozen_trunk_warmup",
                                "status": "completed",
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = _scan_pv26_resume_candidates(run_root)

        self.assertEqual([item.run_name for item in candidates], ["run_active"])
        self.assertEqual(candidates[0].resume_source, "phase_2 last.pt")
        self.assertEqual(candidates[0].next_phase_stage, "stage_2_partial_unfreeze")

    def test_scan_pv26_resume_candidates_supports_skipped_phases_and_lineage_seed(self) -> None:
        from tools.check_env import _scan_pv26_resume_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "pv26_runs"
            derived_run = run_root / "run_derived"
            seed_checkpoint = root / "seed" / "best.pt"
            seed_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            seed_checkpoint.write_text("checkpoint", encoding="utf-8")
            derived_run.mkdir(parents=True, exist_ok=True)
            (derived_run / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "updated_at": "2026-04-05T08:00:00",
                        "lineage": {
                            "mode": "derived_run",
                            "seed_checkpoint_path": str(seed_checkpoint),
                        },
                        "phases": [
                            {"name": "head_warmup", "stage": "stage_1_frozen_trunk_warmup", "status": "skipped"},
                            {"name": "partial_unfreeze", "stage": "stage_2_partial_unfreeze", "status": "skipped"},
                            {"name": "end_to_end_finetune", "stage": "stage_3_end_to_end_finetune", "status": "pending"},
                            {"name": "lane_family_finetune", "stage": "stage_4_lane_family_finetune", "status": "pending"},
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = _scan_pv26_resume_candidates(run_root)

        self.assertEqual([item.run_name for item in candidates], ["run_derived"])
        self.assertEqual(candidates[0].completed_phases, 0)
        self.assertEqual(candidates[0].total_phases, 2)
        self.assertEqual(candidates[0].next_phase_stage, "stage_3_end_to_end_finetune")
        self.assertEqual(candidates[0].resume_source, "lineage seed checkpoint")

    def test_scan_pv26_retrain_candidates_finds_runs_with_seed_checkpoints(self) -> None:
        from tools.check_env import _scan_pv26_retrain_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "pv26_runs"
            candidate_run = run_root / "run_done"
            phase3_checkpoint = candidate_run / "phase_3" / "checkpoints" / "best.pt"
            phase3_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            phase3_checkpoint.write_text("checkpoint", encoding="utf-8")
            (candidate_run / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "updated_at": "2026-04-05T08:10:00",
                        "phases": [
                            {"name": "head_warmup", "stage": "stage_1_frozen_trunk_warmup", "status": "completed"},
                            {"name": "partial_unfreeze", "stage": "stage_2_partial_unfreeze", "status": "completed"},
                            {
                                "name": "end_to_end_finetune",
                                "stage": "stage_3_end_to_end_finetune",
                                "status": "completed",
                                "best_checkpoint_path": str(phase3_checkpoint),
                            },
                            {"name": "lane_family_finetune", "stage": "stage_4_lane_family_finetune", "status": "completed"},
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = _scan_pv26_retrain_candidates(run_root)

        self.assertEqual([item.run_name for item in candidates], ["run_done"])
        self.assertIn("stage_3_end_to_end_finetune", candidates[0].available_seed_stages)

    def test_action_config_lines_include_teacher_and_pv26_hyperparameters(self) -> None:
        from tools.check_env import ActionSpec, _action_config_lines

        teacher_lines = _action_config_lines(
            ActionSpec(
                key="3",
                label="Mobility teacher 학습",
                command_display="python -m tools.od_bootstrap train --teacher mobility",
                argv=(),
                output_hint="unused",
            )
        )
        pv26_lines = _action_config_lines(
            ActionSpec(
                key="C",
                label="PV26 기본 학습",
                command_display="python3 tools/run_pv26_train.py --preset default",
                argv=(),
                output_hint="unused",
            )
        )

        self.assertTrue(any("teacher=mobility" in line for line in teacher_lines))
        self.assertTrue(any("epochs=" in line for line in teacher_lines))
        self.assertTrue(any("dataset roots:" in line for line in pv26_lines))
        self.assertTrue(any("phase_1" in line for line in pv26_lines))
        self.assertTrue(any("preview:" in line for line in pv26_lines))

        export_lines = _action_config_lines(
            ActionSpec(
                key="G",
                label="Mobility teacher TorchScript export",
                command_display="interactive",
                argv=(),
                output_hint="unused",
            )
        )
        self.assertTrue(any("checkpoint:" in line for line in export_lines))
        self.assertTrue(any("format=torchscript" in line for line in export_lines))

    def test_scan_pv26_export_candidates_filters_completed_runs_with_checkpoints(self) -> None:
        from tools.check_env import _scan_pv26_export_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_root = root / "pv26_runs"
            completed = run_root / "run_done"
            incomplete = run_root / "run_active"
            checkpoint = completed / "phase_4" / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")
            incomplete.mkdir(parents=True, exist_ok=True)
            (completed / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "final_checkpoint_path": str(checkpoint),
                        "latest_phase_stage": "stage_4_lane_family_finetune",
                        "latest_selection_metric_path": "val.metrics.lane_family.mean_f1",
                        "latest_backbone_variant": "s",
                        "updated_at": "2026-04-04T01:02:03",
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (incomplete / "summary.json").write_text(
                json.dumps({"status": "running"}, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            candidates = _scan_pv26_export_candidates(run_root)

        self.assertEqual([item.run_name for item in candidates], ["run_done"])
        self.assertEqual(candidates[0].checkpoint_path, checkpoint.resolve())
        self.assertEqual(candidates[0].artifact_path, checkpoint.with_suffix(".torchscript.pt").resolve())

    def test_stage3_stress_action_resolution_appends_runtime_parameters(self) -> None:
        from rich.console import Console
        from tools.check_env import ActionSpec, _resolve_stage3_stress_action

        action = ActionSpec(
            key="D",
            label="PV26 phase VRAM stress",
            command_display="python3 tools/run_pv26_train.py --preset default --stage3-vram-stress --stress-batch-size <BATCH> --stress-iters <ITERS>",
            argv=("python3", "tools/run_pv26_train.py", "--preset", "default", "--stage3-vram-stress"),
            output_hint="stdout only",
        )
        console = Console(file=io.StringIO())

        with unittest.mock.patch("tools.check_env.launch._default_phase_stress_batch_size", return_value=40):
            with unittest.mock.patch("tools.check_env.launch._ascii_input", side_effect=["4", "48", "18"]):
                resolved = _resolve_stage3_stress_action(console, action)

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(
            resolved.argv[-6:],
            ("--stress-stage", "stage_4_lane_family_finetune", "--stress-batch-size", "48", "--stress-iters", "18"),
        )
        self.assertIn("stage_4_lane_family_finetune", resolved.command_display)
        self.assertIn("batch_size=48", resolved.command_display)
        self.assertIn("stress_iters=18", resolved.command_display)

    def test_check_env_launch_runs_pv26_from_repo_root(self) -> None:
        from rich.console import Console
        from tools.check_env import ActionSpec, PipelinePaths, WorkspaceSnapshot
        from tools.check_env.launch import _run_subprocess_action

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = WorkspaceSnapshot(
                paths=PipelinePaths(
                    repo_root=root,
                    raw_bdd_root=root / "bdd_raw",
                    raw_aihub_root=root / "aihub_raw",
                    bootstrap_root=root / "bootstrap",
                    teacher_dataset_root=root / "teacher_datasets",
                    teacher_train_root=root / "teacher_train",
                    teacher_eval_root=root / "teacher_eval",
                    calibration_root=root / "calibration",
                    exhaustive_run_root=root / "exhaustive_runs",
                    exhaustive_dataset_root=root / "exhaustive",
                    final_dataset_root=root / "final_dataset",
                    pv26_run_root=root / "pv26_runs",
                    user_paths_config_path=root / "config" / "user_paths.yaml",
                    od_hyperparameters_config_path=root / "config" / "od.yaml",
                    pv26_hyperparameters_config_path=root / "config" / "pv26.yaml",
                ),
                rows=(),
                flags={},
                recommendation="",
                notes=(),
            )
            action = ActionSpec(
                key="C",
                label="PV26 기본 학습",
                command_display="python3 tools/run_pv26_train.py --preset default",
                argv=(sys.executable, "tools/run_pv26_train.py", "--preset", "default"),
                output_hint=str(root / "runs"),
            )
            console = Console(file=io.StringIO())

            with unittest.mock.patch("tools.check_env.launch._confirm", return_value=True), unittest.mock.patch(
                "tools.check_env.launch._pause_to_continue",
                return_value=True,
            ), unittest.mock.patch(
                "tools.check_env.launch.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as mocked_run:
                _run_subprocess_action(console, snapshot, action, config_line_factory=lambda _: [])

        mocked_run.assert_called_once_with(action.argv, cwd=str(root), check=False)

    def test_check_env_launch_renders_final_dataset_stats_from_existing_file(self) -> None:
        from rich.console import Console
        from tools.check_env import ActionSpec, PipelinePaths, WorkspaceSnapshot
        from tools.check_env.launch import _run_final_dataset_stats

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dataset_root = root / "final_dataset"
            (final_dataset_root / "meta").mkdir(parents=True, exist_ok=True)
            (final_dataset_root / "meta" / FINAL_DATASET_STATS_NAME).write_text(
                json.dumps(
                    {
                        "dataset_root": str(final_dataset_root),
                        "sample_count": 3,
                        "dataset_counts": {"aihub_lane_seoul": 3},
                        "split_counts": {"train": 2, "val": 1},
                        "source_kind_counts": {"lane": 3},
                        "presence_counts": {"lane": 3},
                        "detector": {
                            "classes": {
                                "vehicle": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "bike": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "pedestrian": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "traffic_cone": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "obstacle": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "traffic_light": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                                "sign": {"image_count": 0, "instance_count": 0, "split_image_counts": {}},
                            },
                            "traffic_light_bbox_buckets": {},
                        },
                        "traffic_light_attr": {
                            "valid_count": 0,
                            "invalid_count": 0,
                            "valid_image_count": 0,
                            "combo_counts": {},
                            "bit_positive_counts": {},
                            "invalid_reason_counts": {},
                        },
                        "lane": {
                            "classes": {"white_lane": {"image_count": 0, "instance_count": 0}, "yellow_lane": {"image_count": 1, "instance_count": 1}, "blue_lane": {"image_count": 0, "instance_count": 0}},
                            "types": {"solid": {"image_count": 1, "instance_count": 1}, "dotted": {"image_count": 0, "instance_count": 0}, "missing": {"image_count": 0, "instance_count": 0}},
                        },
                        "stop_line": {"image_count": 0, "instance_count": 0},
                        "crosswalk": {"image_count": 0, "instance_count": 0},
                        "audit": {
                            "manifest_found": True,
                            "manifest_sample_count": 3,
                            "scanned_scene_count": 3,
                            "manifest_scene_path_valid_count": 3,
                            "manifest_scene_path_invalid_count": 0,
                            "manifest_image_path_valid_count": 3,
                            "manifest_image_path_invalid_count": 0,
                            "manifest_det_path_present_count": 0,
                            "manifest_det_path_valid_count": 0,
                            "manifest_det_path_invalid_count": 0,
                            "manifest_source_dataset_counts": {"aihub_lane_seoul": 3},
                            "scanned_source_dataset_counts": {"aihub_lane_seoul": 3},
                            "rebuild_needed": False,
                        },
                        "warnings": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            snapshot = WorkspaceSnapshot(
                paths=PipelinePaths(
                    repo_root=root,
                    raw_bdd_root=root / "bdd_raw",
                    raw_aihub_root=root / "aihub_raw",
                    bootstrap_root=root / "bootstrap",
                    teacher_dataset_root=root / "teacher_datasets",
                    teacher_train_root=root / "teacher_train",
                    teacher_eval_root=root / "teacher_eval",
                    calibration_root=root / "calibration",
                    exhaustive_run_root=root / "exhaustive_runs",
                    exhaustive_dataset_root=root / "exhaustive",
                    final_dataset_root=final_dataset_root,
                    pv26_run_root=root / "pv26_runs",
                    user_paths_config_path=root / "config" / "user_paths.yaml",
                    od_hyperparameters_config_path=root / "config" / "od.yaml",
                    pv26_hyperparameters_config_path=root / "config" / "pv26.yaml",
                ),
                rows=(),
                flags={"final_dataset": True},
                recommendation="",
                notes=(),
            )
            action = ActionSpec(
                key="L",
                label="최종 데이터셋 full stats",
                command_display="interactive",
                argv=(),
                output_hint=str(final_dataset_root / "meta"),
            )
            console = Console(file=io.StringIO())

            with unittest.mock.patch("tools.check_env.launch._pause_to_continue", return_value=True):
                result = _run_final_dataset_stats(console, snapshot, action)

        self.assertTrue(result)

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
            (latest_exhaustive / EXHAUSTIVE_MATERIALIZATION_SUMMARY_NAME).write_text(
                json.dumps({"run_id": "run_001", "sample_count": 200}, indent=2) + "\n",
                encoding="utf-8",
            )

            (final_dataset_root / "meta").mkdir(parents=True, exist_ok=True)
            (final_dataset_root / "meta" / FINAL_DATASET_SUMMARY_NAME).write_text(
                json.dumps({"sample_count": 250, "dataset_counts": {"aihub_lane_seoul": 50}, "rerun_mode": "atomic_overwrite"}, indent=2) + "\n",
                encoding="utf-8",
            )
            (final_dataset_root / "meta" / FINAL_DATASET_STATS_NAME).write_text(
                json.dumps(
                    {
                        "sample_count": 250,
                        "presence_counts": {"det": 22},
                        "detector": {
                            "classes": {
                                "vehicle": {"image_count": 20, "instance_count": 30},
                                "bike": {"image_count": 0, "instance_count": 0},
                                "pedestrian": {"image_count": 0, "instance_count": 0},
                                "traffic_cone": {"image_count": 0, "instance_count": 0},
                                "obstacle": {"image_count": 0, "instance_count": 0},
                                "traffic_light": {
                                    "image_count": 15,
                                    "instance_count": 18,
                                    "split_image_counts": {"val": 6},
                                },
                                "sign": {"image_count": 2, "instance_count": 2},
                            }
                        },
                        "audit": {
                            "manifest_det_path_present_count": 80,
                            "manifest_scene_path_invalid_count": 3,
                        },
                        "warnings": ["manifest_paths_stale"],
                    },
                    indent=2,
                )
                + "\n",
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
        self.assertEqual(row_map["최종 병합 데이터셋"].verdict, "WARN")
        self.assertIn("det_images=22", row_map["최종 병합 데이터셋"].current_state)
        self.assertIn("tl_val=6", row_map["최종 병합 데이터셋"].current_state)
        self.assertIn("stale_scene_paths=3", row_map["최종 병합 데이터셋"].current_state)
        self.assertIn("warnings=manifest_paths_stale", row_map["최종 병합 데이터셋"].current_state)
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
