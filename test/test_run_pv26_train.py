from __future__ import annotations

import io
import json
import os
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.pv26_train.artifacts import load_or_init_meta_manifest
from tools.run_pv26_train import (
    PRESET_PATH_ROOT,
    PhaseConfig,
    PhaseTransitionController,
    PreviewConfig,
    SelectionConfig,
    _build_phase_train_loaders,
    _build_arg_parser,
    _build_postprocess_config,
    _phase_manifest_extra,
    _configure_torch_multiprocessing,
    _phase_entry_is_completed,
    _phase_entry_is_terminal,
    _recover_phase_entry_from_run_dir,
    _sample_preview_selection,
    _scenario_phase_defaults,
    load_meta_train_derived_scenario,
    load_meta_train_resume_context,
    load_meta_train_resume_scenario,
    load_meta_train_scenario,
    main,
    run_phase_vram_sweep,
    run_phase_vram_stress,
    run_stage3_vram_stress,
)
from tools.pv26_train.config import (
    DatasetConfig,
    MetaTrainScenario,
    PreviewConfig as ScenarioPreviewConfig,
    RunConfig,
    TrainDefaultsConfig,
    scenario_to_mapping,
)


def _epoch_summary(epoch: int, metric_value: float) -> dict:
    return {
        "epoch": int(epoch),
        "val": {
            "losses": {
                "total": {
                    "mean": float(metric_value),
                }
            }
        },
    }


class RunPV26TrainScenarioTests(unittest.TestCase):
    def test_public_facade_exports_remain_available(self) -> None:
        from tools import run_pv26_train as module

        expected_names = {
            "PRESET_PATH_ROOT",
            "PhaseConfig",
            "PhaseTransitionController",
            "PreviewConfig",
            "SelectionConfig",
            "_build_arg_parser",
            "_build_postprocess_config",
            "_configure_torch_multiprocessing",
            "_phase_entry_is_completed",
            "_recover_phase_entry_from_run_dir",
            "_sample_preview_selection",
            "_scenario_phase_defaults",
            "load_meta_train_resume_scenario",
            "load_meta_train_scenario",
            "main",
            "run_meta_train_scenario",
            "run_phase_vram_sweep",
            "run_phase_vram_stress",
            "run_stage3_vram_stress",
        }

        for name in expected_names:
            self.assertTrue(hasattr(module, name), msg=name)

    def test_meta_manifest_persists_scenario_snapshot(self) -> None:
        scenario = load_meta_train_scenario("default")
        snapshot = scenario_to_mapping(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "resume_target"
            manifest, _ = load_or_init_meta_manifest(
                scenario=scenario,
                scenario_path=PRESET_PATH_ROOT / "default",
                run_dir=run_dir,
                meta_manifest_version="pv26-meta-train-v1",
                scenario_snapshot=snapshot,
            )

        self.assertIn("scenario_snapshot", manifest)
        self.assertEqual(manifest["scenario_snapshot"]["run"]["run_dir"], str(run_dir))
        self.assertEqual(
            manifest["scenario_snapshot"]["train_defaults"]["batch_size"],
            scenario.train_defaults.batch_size,
        )

    def test_load_meta_train_scenario_applies_user_yaml_overrides(self) -> None:
        user_paths_config = {
            "pv26_train": {
                "dataset_root": "custom/pv26_dataset",
                "additional_roots": ["custom/extra_a", "custom/extra_b"],
                "run_root": "custom/runs/default",
            }
        }
        user_hyperparameters_config = {
            "pv26_train": {
                "presets": {
                    "default": {
                        "train_defaults": {
                            "batch_size": 12,
                            "num_workers": 3,
                            "amp_init_scale": 1024.0,
                            "skip_non_finite_loss": True,
                            "oom_guard": True,
                            "det_conf_threshold": 0.33,
                            "lane_obj_threshold": 0.61,
                        },
                        "preview": {
                            "dataset_keys": ["custom_preview_dataset"],
                            "max_samples_per_dataset": 2,
                        },
                        "phases": [
                            {
                                "name": "head_warmup",
                                "stage": "stage_1_frozen_trunk_warmup",
                                "min_epochs": 2,
                                "max_epochs": 4,
                                "patience": 2,
                                "min_improvement_pct": 2.0,
                                "overrides": {
                                    "head_lr": 0.0015,
                                },
                            },
                            {
                                "name": "partial_unfreeze",
                                "stage": "stage_2_partial_unfreeze",
                                "min_epochs": 3,
                                "max_epochs": 6,
                                "patience": 2,
                                "min_improvement_pct": 1.0,
                                "overrides": {
                                    "head_lr": 0.0008,
                                },
                            },
                            {
                                "name": "end_to_end_finetune",
                                "stage": "stage_3_end_to_end_finetune",
                                "min_epochs": 4,
                                "max_epochs": 10,
                                "patience": 3,
                                "min_improvement_pct": 0.25,
                                "overrides": {
                                    "head_lr": 0.0004,
                                },
                            },
                            {
                                "name": "lane_family_finetune",
                                "stage": "stage_4_lane_family_finetune",
                                "min_epochs": 4,
                                "max_epochs": 8,
                                "patience": 3,
                                "min_improvement_pct": 0.25,
                                "selection": {
                                    "metric_path": "val.metrics.lane_family.mean_f1",
                                    "mode": "max",
                                    "eps": 1.0e-8,
                                },
                                "loss_weights": {
                                    "det": 0.0,
                                    "tl_attr": 0.0,
                                    "lane": 1.5,
                                    "stop_line": 1.25,
                                    "crosswalk": 1.0,
                                },
                                "overrides": {
                                    "trunk_lr": 0.0,
                                    "head_lr": 0.0002,
                                },
                            },
                        ],
                    }
                }
            }
        }

        with patch("tools.run_pv26_train.load_user_paths_config", return_value=user_paths_config):
            with patch("tools.run_pv26_train.load_user_hyperparameters_config", return_value=user_hyperparameters_config):
                scenario = load_meta_train_scenario("default")

        self.assertEqual(scenario.dataset.root.parts[-2:], ("custom", "pv26_dataset"))
        self.assertEqual(
            tuple(path.parts[-2:] for path in scenario.dataset.additional_roots),
            (("custom", "extra_a"), ("custom", "extra_b")),
        )
        self.assertEqual(scenario.run.run_root.parts[-3:], ("custom", "runs", "default"))
        self.assertEqual(scenario.preview.dataset_keys, ("custom_preview_dataset",))
        self.assertEqual(scenario.preview.max_samples_per_dataset, 2)
        self.assertEqual(scenario.train_defaults.batch_size, 12)
        self.assertEqual(scenario.train_defaults.num_workers, 3)
        self.assertEqual(scenario.train_defaults.amp_init_scale, 1024.0)
        self.assertTrue(scenario.train_defaults.skip_non_finite_loss)
        self.assertTrue(scenario.train_defaults.oom_guard)
        self.assertAlmostEqual(scenario.train_defaults.det_conf_threshold, 0.33)
        self.assertAlmostEqual(scenario.train_defaults.lane_obj_threshold, 0.61)
        self.assertEqual(scenario.phases[3].selection.metric_path, "val.metrics.lane_family.mean_f1")
        self.assertEqual(scenario.phases[3].selection.mode, "max")
        self.assertEqual(scenario.phases[3].loss_weights["det"], 0.0)
        phase_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[0].overrides)
        self.assertAlmostEqual(phase_train.head_lr, 0.0015)

    def test_load_meta_train_scenario_preserves_defaults_without_user_yaml(self) -> None:
        with patch("tools.run_pv26_train.load_user_paths_config", return_value={}):
            with patch("tools.run_pv26_train.load_user_hyperparameters_config", return_value={}):
                scenario = load_meta_train_scenario("default")

        self.assertEqual(scenario.dataset.root.parts[-2:], ("seg_dataset", "pv26_exhaustive_od_lane_dataset"))
        self.assertEqual(scenario.run.run_root.parts[-2:], ("runs", "pv26_exhaustive_od_lane_train"))
        self.assertEqual(scenario.train_defaults.batch_size, 40)
        self.assertEqual(scenario.train_defaults.backbone_variant, "s")
        self.assertAlmostEqual(scenario.train_defaults.det_conf_threshold, 0.25)
        self.assertAlmostEqual(scenario.train_defaults.det_iou_threshold, 0.70)
        self.assertAlmostEqual(scenario.train_defaults.lane_obj_threshold, 0.50)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_obj_threshold, 0.50)
        self.assertAlmostEqual(scenario.train_defaults.crosswalk_obj_threshold, 0.50)
        self.assertEqual(tuple(phase.stage for phase in scenario.phases), (
            "stage_1_frozen_trunk_warmup",
            "stage_2_partial_unfreeze",
            "stage_3_end_to_end_finetune",
            "stage_4_lane_family_finetune",
        ))

    def test_default_preset_uses_exhaustive_dataset_and_stage_order(self) -> None:
        with patch("tools.run_pv26_train.load_user_paths_config", return_value={}):
            with patch("tools.run_pv26_train.load_user_hyperparameters_config", return_value={}):
                scenario = load_meta_train_scenario("default")

        self.assertEqual(scenario.dataset.root.parts[-2:], ("seg_dataset", "pv26_exhaustive_od_lane_dataset"))
        self.assertEqual(scenario.dataset.additional_roots, ())
        self.assertEqual(scenario.dataset.roots[-1].parts[-2:], ("seg_dataset", "pv26_exhaustive_od_lane_dataset"))
        self.assertEqual(scenario.run.run_root.parts[-2:], ("runs", "pv26_exhaustive_od_lane_train"))
        self.assertEqual(
            scenario.preview.dataset_keys,
            (
                "pv26_exhaustive_bdd100k_det_100k",
                "pv26_exhaustive_aihub_traffic_seoul",
                "pv26_exhaustive_aihub_obstacle_seoul",
                "aihub_lane_seoul",
            ),
        )
        self.assertEqual(tuple(phase.stage for phase in scenario.phases), (
            "stage_1_frozen_trunk_warmup",
            "stage_2_partial_unfreeze",
            "stage_3_end_to_end_finetune",
            "stage_4_lane_family_finetune",
        ))
        phase_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[0].overrides)
        self.assertEqual(phase_train.batch_size, 40)
        self.assertAlmostEqual(phase_train.head_lr, 0.003)
        self.assertTrue(phase_train.encode_val_batches_in_loader)
        self.assertEqual(scenario.train_defaults.backbone_variant, "s")
        self.assertAlmostEqual(scenario.train_defaults.det_conf_threshold, 0.25)
        self.assertEqual(scenario.selection.metric_path, "selection_metrics.phase_objective")
        self.assertEqual(scenario.selection.mode, "max")
        self.assertIsNone(scenario.phases[3].selection)
        self.assertAlmostEqual(scenario.phases[0].min_delta_abs, 0.005)
        self.assertAlmostEqual(scenario.phases[3].min_delta_abs, 0.003)
        self.assertEqual(scenario.phases[3].freeze_policy, "lane_family_heads_only")
        phase4_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[3].overrides)
        self.assertEqual(phase4_train.sampler_ratios["aihub_lane"], 1.0)
        self.assertEqual(phase4_train.sampler_ratios["bdd100k"], 0.0)

    def test_removed_stage3_stress_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unsupported PV26 meta-train preset: stage3_vram_stress"):
            load_meta_train_scenario("stage3_vram_stress")

    def test_removed_stress_run_root_override_fails_fast(self) -> None:
        with patch("tools.run_pv26_train.load_user_paths_config", return_value={"pv26_train": {"stress_run_root": "legacy/stress"}}):
            with patch("tools.run_pv26_train.load_user_hyperparameters_config", return_value={}):
                with self.assertRaisesRegex(ValueError, "pv26_train.stress_run_root is no longer supported"):
                    load_meta_train_scenario("default")

    def test_legacy_dataset_mapping_keys_fail_fast(self) -> None:
        legacy_hyperparameters_config = {
            "pv26_train": {
                "presets": {
                    "default": {
                        "dataset": {
                            "aihub_root": "custom/aihub_only",
                            "bdd_root": "custom/bdd_legacy",
                            "include_bdd": True,
                        }
                    }
                }
            }
        }

        with patch("tools.run_pv26_train.load_user_paths_config", return_value={}):
            with patch(
                "tools.run_pv26_train.load_user_hyperparameters_config",
                return_value=legacy_hyperparameters_config,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "unsupported dataset config keys: .*dataset.root and dataset.additional_roots",
                ):
                    load_meta_train_scenario("default")

    def test_removed_preset_override_section_fails_fast(self) -> None:
        with patch("tools.run_pv26_train.load_user_paths_config", return_value={}):
            with patch(
                "tools.run_pv26_train.load_user_hyperparameters_config",
                return_value={"pv26_train": {"presets": {"stage3_vram_stress": {}}}},
            ):
                with self.assertRaisesRegex(KeyError, "unsupported PV26 meta-train preset overrides"):
                    load_meta_train_scenario("default")

    def test_load_meta_train_scenario_rejects_unknown_preset(self) -> None:
        with self.assertRaisesRegex(KeyError, "unsupported PV26 meta-train preset"):
            load_meta_train_scenario("does-not-exist")

    def test_build_postprocess_config_uses_train_defaults_thresholds(self) -> None:
        train_defaults = TrainDefaultsConfig(
            det_conf_threshold=0.31,
            det_iou_threshold=0.66,
            lane_obj_threshold=0.41,
            stop_line_obj_threshold=0.42,
            crosswalk_obj_threshold=0.43,
        )

        config = _build_postprocess_config(train_defaults)

        self.assertAlmostEqual(config.det_conf_threshold, 0.31)
        self.assertAlmostEqual(config.det_iou_threshold, 0.66)
        self.assertAlmostEqual(config.lane_obj_threshold, 0.41)
        self.assertAlmostEqual(config.stop_line_obj_threshold, 0.42)
        self.assertAlmostEqual(config.crosswalk_obj_threshold, 0.43)
        self.assertFalse(config.allow_python_nms_fallback)

    def test_phase_manifest_extra_includes_resolved_postprocess_thresholds(self) -> None:
        scenario = load_meta_train_scenario("default")
        phase_train = _scenario_phase_defaults(
            scenario.train_defaults,
            {
                "det_conf_threshold": 0.29,
                "lane_obj_threshold": 0.44,
            },
        )

        manifest_extra = _phase_manifest_extra(
            scenario_path=PRESET_PATH_ROOT / "default",
            phase_index=1,
            phase=scenario.phases[0],
            train_config=phase_train,
            scenario=scenario,
        )

        self.assertEqual(manifest_extra["postprocess"]["det_conf_threshold"], 0.29)
        self.assertEqual(manifest_extra["postprocess"]["det_iou_threshold"], 0.7)
        self.assertEqual(manifest_extra["postprocess"]["lane_obj_threshold"], 0.44)
        self.assertEqual(manifest_extra["postprocess"]["stop_line_obj_threshold"], 0.5)
        self.assertEqual(manifest_extra["postprocess"]["crosswalk_obj_threshold"], 0.5)
        self.assertFalse(manifest_extra["postprocess"]["allow_python_nms_fallback"])

    def test_load_meta_train_scenario_rejects_invalid_stage_order(self) -> None:
        scenario = load_meta_train_scenario("default")
        bad_scenario = SimpleNamespace(**scenario.__dict__)
        bad_scenario.phases = (
            PhaseConfig(
                name="wrong_first",
                stage="stage_2_partial_unfreeze",
                min_epochs=1,
                max_epochs=2,
                patience=1,
                min_improvement_pct=1.0,
            ),
            scenario.phases[1],
            scenario.phases[2],
            scenario.phases[3],
        )

        with self.assertRaisesRegex(ValueError, "phase 1 must use stage"):
            from tools.run_pv26_train import _validate_meta_train_scenario

            _validate_meta_train_scenario(bad_scenario)

    def test_validate_meta_train_scenario_rejects_phase_objective_without_validation(self) -> None:
        scenario = load_meta_train_scenario("default")
        bad_scenario = SimpleNamespace(**scenario.__dict__)
        bad_scenario.train_defaults = TrainDefaultsConfig(**{
            **scenario.train_defaults.__dict__,
            "val_batches": 0,
        })

        with self.assertRaisesRegex(ValueError, "requires val"):
            from tools.run_pv26_train import _validate_meta_train_scenario

            _validate_meta_train_scenario(bad_scenario)

    def test_phase_recovery_roundtrip_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            phase_run_dir = Path(temp_dir) / "phase_2"
            summary_path = phase_run_dir / "summary.json"
            best_checkpoint = phase_run_dir / "checkpoints" / "best.pt"
            last_checkpoint = phase_run_dir / "checkpoints" / "last.pt"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                textwrap.dedent(
                    f"""
                    {{
                      "completed_epochs": 4,
                      "best_metric_value": 1.23,
                      "best_epoch": 2,
                      "checkpoint_paths": {{
                        "best": "{best_checkpoint}",
                        "last": "{last_checkpoint}"
                      }},
                      "early_exit": {{
                        "reason": "plateau",
                        "phase_state": {{
                          "epoch": 3
                        }}
                      }}
                    }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            entry = {
                "status": "running",
                "run_dir": str(phase_run_dir),
                "best_checkpoint_path": None,
            }
            phase = PhaseConfig(
                name="partial_unfreeze",
                stage="stage_2_partial_unfreeze",
                min_epochs=2,
                max_epochs=8,
                patience=2,
                min_improvement_pct=1.0,
            )

            recovered = _recover_phase_entry_from_run_dir(entry, phase)

            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["status"], "completed")
            self.assertEqual(recovered["best_checkpoint_path"], str(best_checkpoint))
            self.assertEqual(recovered["last_checkpoint_path"], str(last_checkpoint))
            self.assertEqual(recovered["promotion_reason"], "plateau")
            self.assertEqual(recovered["phase_state"]["epoch"], 3)

    def test_load_meta_train_resume_scenario_uses_manifest_snapshot(self) -> None:
        scenario = load_meta_train_scenario("default")
        snapshot = scenario_to_mapping(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "resume_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "status": "running",
                "scenario_path": str(PRESET_PATH_ROOT / "default"),
                "scenario_snapshot": {
                    **snapshot,
                    "run": {
                        **snapshot["run"],
                        "run_dir": str(run_dir),
                    },
                },
                "phases": [
                    {"name": phase.name, "stage": phase.stage, "status": "pending"}
                    for phase in scenario.phases
                ],
            }
            (run_dir / "meta_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            resumed_scenario, scenario_path = load_meta_train_resume_scenario(
                run_dir,
                preset_name="default",
            )

        self.assertEqual(scenario_path, PRESET_PATH_ROOT / "default")
        self.assertEqual(resumed_scenario.run.run_dir, run_dir.resolve())
        self.assertEqual(resumed_scenario.train_defaults.batch_size, scenario.train_defaults.batch_size)
        self.assertEqual(
            tuple(phase.stage for phase in resumed_scenario.phases),
            tuple(phase.stage for phase in scenario.phases),
        )

    def test_load_meta_train_resume_scenario_rejects_missing_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_run_dir = Path(temp_dir) / "missing_run"

            with self.assertRaisesRegex(SystemExit, "resume run directory does not exist"):
                load_meta_train_resume_scenario(missing_run_dir, preset_name="default")

    def test_load_meta_train_resume_scenario_rejects_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "completed_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta_manifest.json").write_text(
                json.dumps({"status": "completed", "phases": []}, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "exact resume only supports incomplete runs"):
                load_meta_train_resume_scenario(run_dir, preset_name="default")

    def test_load_meta_train_resume_context_reads_selected_phase_window_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "derived_resume"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "selected_phase_window": {
                            "selected_phase_indices": [3, 4],
                            "start_phase_stage": "stage_3_end_to_end_finetune",
                            "end_phase_stage": "stage_4_lane_family_finetune",
                        },
                        "lineage": {
                            "mode": "derived_run",
                            "source_run_dir": "/tmp/source_run",
                            "seed_checkpoint_path": "/tmp/source_run/phase_3/checkpoints/best.pt",
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

            context = load_meta_train_resume_context(run_dir)

        self.assertEqual(context["selected_phase_window"]["selected_phase_indices"], [3, 4])
        self.assertEqual(context["lineage"]["mode"], "derived_run")

    def test_load_meta_train_derived_scenario_uses_current_config_and_source_seed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_run_dir = Path(temp_dir) / "source_run"
            phase3_checkpoint = source_run_dir / "phase_3" / "checkpoints" / "best.pt"
            phase3_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            phase3_checkpoint.write_text("checkpoint", encoding="utf-8")
            (source_run_dir / "meta_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
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

            derived_scenario, scenario_path, derived_options = load_meta_train_derived_scenario(
                source_run_dir,
                preset_name="default",
                start_stage="stage_3_end_to_end_finetune",
                end_stage="stage_4_lane_family_finetune",
            )

        self.assertEqual(scenario_path, PRESET_PATH_ROOT / "default")
        self.assertIsNone(derived_scenario.run.run_dir)
        self.assertIn("source_run", derived_scenario.run.run_name_prefix)
        self.assertEqual(derived_options["selected_phase_indices"], (3, 4))
        self.assertEqual(derived_options["initial_best_checkpoint"], phase3_checkpoint.resolve())
        self.assertEqual(derived_options["lineage"]["source_run_dir"], str(source_run_dir.resolve()))
        self.assertEqual(derived_options["lineage"]["seed_checkpoint_source"], "phase_3 best.pt")

    def test_phase_entry_is_terminal_treats_skipped_as_terminal(self) -> None:
        phase = PhaseConfig(
            name="lane_family_finetune",
            stage="stage_4_lane_family_finetune",
            min_epochs=1,
            max_epochs=1,
            patience=1,
            min_improvement_pct=0.25,
        )
        entry = {"status": "skipped", "run_dir": "/tmp/unused"}

        self.assertTrue(_phase_entry_is_terminal(entry, phase))

    def test_load_meta_train_resume_scenario_loads_compatible_legacy_manifest(self) -> None:
        scenario = load_meta_train_scenario("default")
        scenario_mapping = scenario_to_mapping(scenario)
        legacy_manifest = {
            "status": "running",
            "dataset": scenario_mapping["dataset"],
            "train_defaults": scenario_mapping["train_defaults"],
            "selection": scenario_mapping["selection"],
            "preview": scenario_mapping["preview"],
            "phases": [
                {"name": phase.name, "stage": phase.stage, "status": "pending"}
                for phase in scenario.phases
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "legacy_resume"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta_manifest.json").write_text(
                json.dumps(legacy_manifest, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            resumed_scenario, scenario_path = load_meta_train_resume_scenario(
                run_dir,
                preset_name="default",
            )

        self.assertEqual(scenario_path, PRESET_PATH_ROOT / "default")
        self.assertEqual(resumed_scenario.run.run_dir, run_dir.resolve())
        self.assertEqual(
            tuple(phase.stage for phase in resumed_scenario.phases),
            tuple(phase.stage for phase in scenario.phases),
        )

    def test_load_meta_train_resume_scenario_rejects_legacy_mismatch(self) -> None:
        scenario = load_meta_train_scenario("default")
        scenario_mapping = scenario_to_mapping(scenario)
        legacy_manifest = {
            "status": "running",
            "scenario_path": str(PRESET_PATH_ROOT / "default"),
            "dataset": scenario_mapping["dataset"],
            "train_defaults": {
                **scenario_mapping["train_defaults"],
                "batch_size": scenario.train_defaults.batch_size + 7,
            },
            "selection": scenario_mapping["selection"],
            "preview": scenario_mapping["preview"],
            "phases": [
                {"name": phase.name, "stage": phase.stage, "status": "pending"}
                for phase in scenario.phases
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "legacy_resume"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta_manifest.json").write_text(
                json.dumps(legacy_manifest, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "legacy resume run is incompatible"):
                load_meta_train_resume_scenario(run_dir, preset_name="default")

    def test_phase_transition_controller_supports_phase_specific_max_metric(self) -> None:
        phase = PhaseConfig(
            name="lane_family_finetune",
            stage="stage_4_lane_family_finetune",
            min_epochs=2,
            max_epochs=6,
            patience=2,
            min_improvement_pct=0.25,
            selection=SelectionConfig(
                metric_path="val.metrics.lane_family.mean_f1",
                mode="max",
                eps=1.0e-8,
            ),
        )
        controller = PhaseTransitionController(
            phase=phase,
            selection=phase.selection,
        )

        def _summary(epoch: int, metric: float) -> dict:
            return {
                "epoch": int(epoch),
                "val": {
                    "metrics": {
                        "lane_family": {
                            "mean_f1": float(metric),
                        }
                    }
                },
            }

        self.assertIsNone(controller.observe_epoch(_summary(1, 0.40)))
        self.assertIsNone(controller.observe_epoch(_summary(2, 0.45)))
        stop_state = controller.observe_epoch(_summary(3, 0.451))
        self.assertIsNone(stop_state)
        stop_state = controller.observe_epoch(_summary(4, 0.4515))
        self.assertIsNotNone(stop_state)
        self.assertEqual(stop_state["reason"], "plateau")
        self.assertEqual(stop_state["phase_state"]["improvement_policy"], "relative_pct")

    def test_phase_transition_controller_uses_phase_objective_absolute_delta(self) -> None:
        phase = PhaseConfig(
            name="lane_family_finetune",
            stage="stage_4_lane_family_finetune",
            min_epochs=2,
            max_epochs=6,
            patience=2,
            min_improvement_pct=0.25,
            min_delta_abs=0.003,
        )
        selection = SelectionConfig(metric_path="selection_metrics.phase_objective", mode="max", eps=1.0e-8)
        controller = PhaseTransitionController(
            phase=phase,
            selection=selection,
        )

        def _summary(epoch: int, lane_f1: float) -> dict:
            return {
                "epoch": int(epoch),
                "val": {
                    "metrics": {
                        "lane": {
                            "tp": 120,
                            "fn": 0,
                            "f1": float(lane_f1),
                            "mean_point_distance": 10.0,
                            "color_accuracy": 0.9,
                            "type_accuracy": 0.8,
                        },
                        "stop_line": {
                            "tp": 40,
                            "fn": 0,
                            "f1": 0.45,
                            "mean_point_distance": 10.0,
                            "mean_angle_error": 5.0,
                        },
                        "crosswalk": {
                            "tp": 40,
                            "fn": 0,
                            "f1": 0.35,
                            "mean_polygon_iou": 0.8,
                            "mean_vertex_distance": 8.0,
                        },
                    }
                },
            }

        first = _summary(1, 0.40)
        self.assertIsNone(controller.observe_epoch(first))
        self.assertIn("selection_metrics", first)
        self.assertIn("phase_objective", first["selection_metrics"])
        self.assertIsNone(controller.observe_epoch(_summary(2, 0.50)))
        stop_state = controller.observe_epoch(_summary(3, 0.504))
        self.assertIsNone(stop_state)
        stop_state = controller.observe_epoch(_summary(4, 0.508))
        self.assertIsNotNone(stop_state)
        self.assertEqual(stop_state["reason"], "plateau")
        self.assertEqual(stop_state["phase_state"]["improvement_policy"], "absolute_delta")
        self.assertGreater(stop_state["phase_state"]["best_phase_objective"], 0.0)
        self.assertGreater(stop_state["phase_state"]["last_improvement_abs"], 0.0)

    def test_stage4_build_phase_train_loaders_filter_to_lane_family_dataset(self) -> None:
        class _FakeDataset:
            def __init__(self) -> None:
                self.records = []
                self.samples = []
                for split in ("train", "val"):
                    for dataset_key in ("bdd100k_det_100k", "aihub_lane_seoul"):
                        for index in range(2):
                            record = SimpleNamespace(
                                dataset_key=dataset_key,
                                split=split,
                                sample_id=f"{dataset_key}_{split}_{index}",
                            )
                            self.records.append(record)
                            self.samples.append(
                                {
                                    "image": torch.zeros((3, 608, 800), dtype=torch.float32),
                                    "det_targets": {
                                        "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
                                        "classes": torch.zeros((0,), dtype=torch.long),
                                    },
                                    "tl_attr_targets": {
                                        "bits": torch.zeros((0, 4), dtype=torch.float32),
                                        "is_traffic_light": torch.zeros((0,), dtype=torch.bool),
                                        "collapse_reason": [],
                                    },
                                    "lane_targets": {
                                        "lanes": [],
                                        "stop_lines": [],
                                        "crosswalks": [],
                                    },
                                    "source_mask": {
                                        "det": dataset_key != "aihub_lane_seoul",
                                        "tl_attr": False,
                                        "lane": dataset_key == "aihub_lane_seoul",
                                        "stop_line": dataset_key == "aihub_lane_seoul",
                                        "crosswalk": dataset_key == "aihub_lane_seoul",
                                    },
                                    "valid_mask": {
                                        "det": torch.zeros((0,), dtype=torch.bool),
                                        "tl_attr": torch.zeros((0,), dtype=torch.bool),
                                        "lane": torch.zeros((0,), dtype=torch.bool),
                                        "stop_line": torch.zeros((0,), dtype=torch.bool),
                                        "crosswalk": torch.zeros((0,), dtype=torch.bool),
                                    },
                                    "meta": {
                                        "sample_id": record.sample_id,
                                        "dataset_key": dataset_key,
                                        "split": split,
                                        "image_path": f"/tmp/{record.sample_id}.jpg",
                                        "raw_hw": (720, 1280),
                                        "network_hw": (608, 800),
                                        "transform": {
                                            "scale": 0.625,
                                            "pad_left": 0,
                                            "pad_top": 79,
                                            "pad_right": 0,
                                            "pad_bottom": 79,
                                            "resized_hw": (450, 800),
                                        },
                                        "det_supervised_classes": [],
                                        "det_supervised_class_ids": [],
                                        "det_allow_objectness_negatives": False,
                                        "det_allow_unmatched_class_negatives": False,
                                    },
                                }
                            )

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, index: int) -> dict:
                return self.samples[index]

        phase = PhaseConfig(
            name="lane_family_finetune",
            stage="stage_4_lane_family_finetune",
            min_epochs=4,
            max_epochs=12,
            patience=3,
            min_improvement_pct=0.25,
            selection=SelectionConfig(metric_path="val.metrics.lane_family.mean_f1", mode="max", eps=1.0e-8),
            loss_weights={"det": 0.0, "tl_attr": 0.0, "lane": 1.5, "stop_line": 1.25, "crosswalk": 1.0},
            freeze_policy="lane_family_heads_only",
        )
        train_config = TrainDefaultsConfig(
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
            sampler_ratios={
                "bdd100k": 0.0,
                "aihub_traffic": 0.0,
                "aihub_lane": 1.0,
                "aihub_obstacle": 0.0,
            },
        )

        train_loader, val_loader = _build_phase_train_loaders(_FakeDataset(), train_config=train_config, phase=phase)
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        self.assertTrue(all(item["dataset_key"] == "aihub_lane_seoul" for item in train_batch["meta"]))
        self.assertTrue(all(item["dataset_key"] == "aihub_lane_seoul" for item in val_batch["meta"]))

    def test_sample_preview_selection_uses_record_metadata_before_loading_samples(self) -> None:
        class _FakeDataset:
            def __init__(self) -> None:
                self.records = [
                    SimpleNamespace(dataset_key="aihub_traffic_seoul", split="train", sample_id="train_a"),
                    SimpleNamespace(dataset_key="bdd100k_det_100k", split="train", sample_id="train_b"),
                    SimpleNamespace(dataset_key="aihub_traffic_seoul", split="val", sample_id="val_a"),
                    SimpleNamespace(dataset_key="bdd100k_det_100k", split="val", sample_id="val_b"),
                ]
                self.loaded_indices: list[int] = []

            def __getitem__(self, index: int) -> dict:
                self.loaded_indices.append(index)
                record = self.records[index]
                return {
                    "meta": {
                        "sample_id": record.sample_id,
                        "dataset_key": record.dataset_key,
                        "split": record.split,
                    }
                }

        dataset = _FakeDataset()

        selected = _sample_preview_selection(
            dataset,
            PreviewConfig(
                enabled=True,
                split="val",
                dataset_keys=("aihub_traffic_seoul", "bdd100k_det_100k"),
                max_samples_per_dataset=1,
                write_overlay=False,
            ),
        )

        self.assertEqual(dataset.loaded_indices, [2, 3])
        self.assertEqual([item["meta"]["sample_id"] for item in selected], ["val_a", "val_b"])

    def test_sample_preview_selection_tolerates_missing_dataset_keys(self) -> None:
        class _FakeDataset:
            def __init__(self) -> None:
                self.records = [
                    SimpleNamespace(dataset_key="aihub_lane_seoul", split="val", sample_id="lane_val"),
                ]
                self.loaded_indices: list[int] = []

            def __getitem__(self, index: int) -> dict:
                self.loaded_indices.append(index)
                record = self.records[index]
                return {
                    "meta": {
                        "sample_id": record.sample_id,
                        "dataset_key": record.dataset_key,
                        "split": record.split,
                    }
                }

        dataset = _FakeDataset()
        messages: list[str] = []

        from tools.run_pv26_train import _sample_preview_selection_with_logging

        selected = _sample_preview_selection_with_logging(
            dataset,
            PreviewConfig(
                enabled=True,
                split="val",
                dataset_keys=(
                    "pv26_exhaustive_bdd100k_det_100k",
                    "pv26_exhaustive_aihub_traffic_seoul",
                    "aihub_lane_seoul",
                ),
                max_samples_per_dataset=1,
                write_overlay=False,
            ),
            progress_callback=messages.append,
        )

        self.assertEqual(dataset.loaded_indices, [0])
        self.assertEqual([item["meta"]["sample_id"] for item in selected], ["lane_val"])
        self.assertEqual(len(messages), 1)
        self.assertIn("preview selection fallback", messages[0])
        self.assertIn("aihub_lane_seoul", messages[0])

    def test_main_accepts_preset_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loaded_scenario = SimpleNamespace()

            with patch("tools.run_pv26_train.load_meta_train_scenario", return_value=loaded_scenario) as mocked_load:
                with patch(
                    "tools.run_pv26_train.run_meta_train_scenario",
                    return_value={"status": "ok", "scenario_path": str(PRESET_PATH_ROOT / "default")},
                ) as mocked_run:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        main(["--preset", "default"])

            mocked_load.assert_called_once_with("default")
            mocked_run.assert_called_once_with(
                loaded_scenario,
                scenario_path=PRESET_PATH_ROOT / "default",
                selected_phase_indices=None,
                initial_best_checkpoint=None,
                lineage=None,
            )
            self.assertIn('"status": "ok"', buffer.getvalue())

    def test_main_accepts_resume_run_argument(self) -> None:
        loaded_scenario = SimpleNamespace(run=SimpleNamespace(run_dir="/tmp/existing_run"))

        with patch(
            "tools.run_pv26_train.load_meta_train_resume_scenario",
            return_value=(loaded_scenario, PRESET_PATH_ROOT / "default"),
        ) as mocked_resume:
            with patch(
                "tools.run_pv26_train.load_meta_train_resume_context",
                return_value={"selected_phase_window": None, "lineage": None},
            ) as mocked_resume_context:
                with patch(
                    "tools.run_pv26_train.run_meta_train_scenario",
                    return_value={"status": "ok", "scenario_path": str(PRESET_PATH_ROOT / "default")},
                ) as mocked_run:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        main(["--resume-run", "/tmp/existing_run"])

        mocked_resume.assert_called_once_with("/tmp/existing_run", preset_name="default")
        mocked_resume_context.assert_called_once_with("/tmp/existing_run")
        mocked_run.assert_called_once_with(
            loaded_scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            selected_phase_indices=None,
            initial_best_checkpoint=None,
            lineage=None,
        )
        self.assertIn('"status": "ok"', buffer.getvalue())

    def test_main_accepts_derive_run_argument(self) -> None:
        loaded_scenario = SimpleNamespace()
        derived_options = {
            "selected_phase_indices": (3, 3),
            "initial_best_checkpoint": Path("/tmp/source_run/phase_3/checkpoints/best.pt"),
            "lineage": {"mode": "derived_run"},
        }

        with patch(
            "tools.run_pv26_train.load_meta_train_derived_scenario",
            return_value=(loaded_scenario, PRESET_PATH_ROOT / "default", derived_options),
        ) as mocked_derive:
            with patch(
                "tools.run_pv26_train.run_meta_train_scenario",
                return_value={"status": "ok", "scenario_path": str(PRESET_PATH_ROOT / "default")},
            ) as mocked_run:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    main(
                        [
                            "--derive-run",
                            "/tmp/source_run",
                            "--start-stage",
                            "stage_3_end_to_end_finetune",
                            "--end-stage",
                            "stage_3_end_to_end_finetune",
                        ]
                    )

        mocked_derive.assert_called_once_with(
            "/tmp/source_run",
            preset_name="default",
            start_stage="stage_3_end_to_end_finetune",
            end_stage="stage_3_end_to_end_finetune",
        )
        mocked_run.assert_called_once_with(
            loaded_scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            selected_phase_indices=(3, 3),
            initial_best_checkpoint=Path("/tmp/source_run/phase_3/checkpoints/best.pt"),
            lineage={"mode": "derived_run"},
        )
        self.assertIn('"status": "ok"', buffer.getvalue())

    def test_main_dispatches_stage3_vram_stress_mode(self) -> None:
        loaded_scenario = SimpleNamespace()

        with patch("tools.run_pv26_train.load_meta_train_scenario", return_value=loaded_scenario) as mocked_load:
            with patch("tools.run_pv26_train.run_phase_vram_stress", return_value={"status": "ok", "mode": "phase_vram_stress"}) as mocked_stress:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    main(
                        [
                            "--preset",
                            "default",
                            "--stage3-vram-stress",
                            "--stress-stage",
                            "stage_2_partial_unfreeze",
                            "--stress-batch-size",
                            "24",
                            "--stress-iters",
                            "16",
                        ]
                    )

        mocked_load.assert_called_once_with("default")
        mocked_stress.assert_called_once_with(
            loaded_scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            stage="stage_2_partial_unfreeze",
            batch_size=24,
            stress_iters=16,
        )
        self.assertIn('"mode": "phase_vram_stress"', buffer.getvalue())

    def test_main_dispatches_phase_vram_sweep_mode(self) -> None:
        loaded_scenario = SimpleNamespace()

        with patch("tools.run_pv26_train.load_meta_train_scenario", return_value=loaded_scenario) as mocked_load:
            with patch("tools.run_pv26_train.run_phase_vram_sweep", return_value={"status": "ok", "mode": "phase_vram_sweep"}) as mocked_sweep:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    main(
                        [
                            "--preset",
                            "default",
                            "--phase-vram-sweep",
                            "--stress-stages",
                            "stage_1_frozen_trunk_warmup,stage_4_lane_family_finetune",
                            "--stress-batch-sizes",
                            "1,2,4",
                            "--stress-iters",
                            "6",
                        ]
                    )

        mocked_load.assert_called_once_with("default")
        mocked_sweep.assert_called_once_with(
            loaded_scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            stages="stage_1_frozen_trunk_warmup,stage_4_lane_family_finetune",
            batch_sizes="1,2,4",
            stress_iters=6,
        )
        self.assertIn('"mode": "phase_vram_sweep"', buffer.getvalue())

    def test_arg_parser_keeps_resume_and_stage3_runtime_flags(self) -> None:
        parser = _build_arg_parser()

        args = parser.parse_args(
            [
                "--preset",
                "default",
                "--resume-run",
                "/tmp/existing_run",
                "--stage3-vram-stress",
                "--stress-stage",
                "stage_4_lane_family_finetune",
                "--stress-batch-size",
                "24",
                "--stress-iters",
                "16",
            ]
        )

        self.assertEqual(args.preset, "default")
        self.assertEqual(args.resume_run, "/tmp/existing_run")
        self.assertTrue(args.stage3_vram_stress)
        self.assertEqual(args.stress_stage, "stage_4_lane_family_finetune")
        self.assertEqual(args.stress_batch_size, 24)
        self.assertEqual(args.stress_iters, 16)

    def test_arg_parser_accepts_phase_vram_sweep_runtime_flags(self) -> None:
        parser = _build_arg_parser()

        args = parser.parse_args(
            [
                "--preset",
                "default",
                "--phase-vram-sweep",
                "--stress-stages",
                "stage_1_frozen_trunk_warmup,stage_3_end_to_end_finetune",
                "--stress-batch-sizes",
                "1,2,4,8",
                "--stress-iters",
                "6",
            ]
        )

        self.assertTrue(args.phase_vram_sweep)
        self.assertEqual(args.stress_stages, "stage_1_frozen_trunk_warmup,stage_3_end_to_end_finetune")
        self.assertEqual(args.stress_batch_sizes, "1,2,4,8")
        self.assertEqual(args.stress_iters, 6)

    def test_arg_parser_accepts_derive_run_and_stage_window(self) -> None:
        parser = _build_arg_parser()

        args = parser.parse_args(
            [
                "--preset",
                "default",
                "--derive-run",
                "/tmp/source_run",
                "--start-stage",
                "stage_3_end_to_end_finetune",
                "--end-stage",
                "stage_4_lane_family_finetune",
            ]
        )

        self.assertEqual(args.derive_run, "/tmp/source_run")
        self.assertEqual(args.start_stage, "stage_3_end_to_end_finetune")
        self.assertEqual(args.end_stage, "stage_4_lane_family_finetune")

    def test_main_rejects_resume_run_with_stage3_vram_stress(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--resume-run cannot be combined with VRAM probe modes"):
            main(["--resume-run", "/tmp/existing_run", "--stage3-vram-stress"])

    def test_main_rejects_derive_run_with_stage3_vram_stress(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--derive-run cannot be combined with VRAM probe modes"):
            main(["--derive-run", "/tmp/source_run", "--stage3-vram-stress"])

    def test_main_rejects_combined_vram_probe_modes(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--stage3-vram-stress cannot be combined with --phase-vram-sweep"):
            main(["--stage3-vram-stress", "--phase-vram-sweep"])

    def test_main_rejects_resume_run_with_derive_run(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--resume-run cannot be combined with --derive-run"):
            main(["--resume-run", "/tmp/existing_run", "--derive-run", "/tmp/source_run"])

    def test_main_exits_with_code_2_when_stage3_vram_stress_reports_non_ok_status(self) -> None:
        loaded_scenario = SimpleNamespace()

        with patch("tools.run_pv26_train.load_meta_train_scenario", return_value=loaded_scenario):
            with patch(
                "tools.run_pv26_train.run_phase_vram_stress",
                return_value={"status": "oom", "mode": "phase_vram_stress"},
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    with redirect_stdout(io.StringIO()):
                        main(["--preset", "default", "--stage3-vram-stress"])

        self.assertEqual(exc_info.exception.code, 2)

    def test_configure_torch_multiprocessing_uses_file_system_sharing(self) -> None:
        torch_root = Path("/tmp/fake_torch")
        mock_torch = SimpleNamespace(
            __file__=str(torch_root / "__init__.py"),
            multiprocessing=SimpleNamespace(
                get_sharing_strategy=lambda: "file_descriptor",
                set_sharing_strategy=lambda strategy: setattr(self, "_sharing_strategy", strategy),
            )
        )

        with patch("tools.run_pv26_train.Path.is_dir", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch.dict(os.environ, {}, clear=True):
                    self._sharing_strategy = None
                    _configure_torch_multiprocessing()
                    self.assertEqual(os.environ["LD_LIBRARY_PATH"], str(torch_root / "lib"))

        self.assertEqual(self._sharing_strategy, "file_system")

    def test_run_stage3_vram_stress_validates_runtime_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=Path(tmpdir)),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(),
                selection=SelectionConfig(),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="end_to_end_finetune",
                        stage="stage_3_end_to_end_finetune",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                    ),
                ),
            )
            with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
                    with self.assertRaisesRegex(ValueError, "stress batch size must be > 0"):
                        run_stage3_vram_stress(
                            scenario,
                            scenario_path=Path(tmpdir) / "default",
                            batch_size=0,
                            stress_iters=12,
                        )
                    with self.assertRaisesRegex(ValueError, "stress iterations must be > 0"):
                        run_stage3_vram_stress(
                            scenario,
                            scenario_path=Path(tmpdir) / "default",
                            batch_size=4,
                            stress_iters=0,
                        )

    def test_run_stage3_vram_stress_rejects_missing_dataset_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_root = Path(tmpdir) / "missing_dataset"
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=missing_root),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(),
                selection=SelectionConfig(),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="end_to_end_finetune",
                        stage="stage_3_end_to_end_finetune",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                    ),
                ),
            )
            with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
                    with self.assertRaisesRegex(SystemExit, "canonical dataset roots not found"):
                        run_stage3_vram_stress(
                            scenario,
                            scenario_path=Path(tmpdir) / "default",
                            batch_size=4,
                            stress_iters=2,
                        )

    def test_run_stage3_vram_stress_uses_single_process_loader(self) -> None:
        class FakeCudaDevice:
            type = "cuda"

            def __str__(self) -> str:
                return "cuda:0"

        class FakeTrainer:
            def __init__(self) -> None:
                self.device = FakeCudaDevice()
                self.oom_guard = True

            def train_epoch(self, *args, **kwargs) -> dict[str, object]:
                return {"loss": 1.0}

        captured_train_config: dict[str, object] = {}

        def fake_build_phase_train_loaders(dataset, *, train_config, phase=None):
            captured_train_config["train_config"] = train_config
            captured_train_config["phase"] = phase
            return object(), None

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=Path(tmpdir)),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(num_workers=6, persistent_workers=True, prefetch_factor=2),
                selection=SelectionConfig(),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="end_to_end_finetune",
                        stage="stage_3_end_to_end_finetune",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                    ),
                ),
            )
            mock_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    empty_cache=lambda: None,
                    reset_peak_memory_stats=lambda device: None,
                )
            )

            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                    with patch("tools.run_pv26_train.PV26CanonicalDataset", return_value=SimpleNamespace(records=[])):
                        with patch(
                            "tools.run_pv26_train._build_phase_train_loaders",
                            side_effect=fake_build_phase_train_loaders,
                        ):
                            with patch("tools.run_pv26_train._build_phase_trainer", return_value=FakeTrainer()):
                                with patch(
                                    "tools.run_pv26_train._cuda_memory_stats",
                                    return_value={"device": "cuda:0"},
                                ):
                                    result = run_stage3_vram_stress(
                                        scenario,
                                        scenario_path=Path(tmpdir) / "default",
                                        batch_size=32,
                                        stress_iters=12,
                                    )

        train_config = captured_train_config["train_config"]
        self.assertEqual(train_config.num_workers, 0)
        self.assertFalse(train_config.persistent_workers)
        self.assertIsNone(train_config.prefetch_factor)
        self.assertEqual(train_config.batch_size, 32)
        self.assertEqual(train_config.train_batches, 12)
        self.assertEqual(train_config.val_batches, 0)
        self.assertEqual(captured_train_config["phase"].stage, "stage_3_end_to_end_finetune")
        self.assertEqual(result["status"], "ok")

    def test_run_phase_vram_stress_uses_phase_override_batch_size_by_default(self) -> None:
        class FakeCudaDevice:
            type = "cuda"

            def __str__(self) -> str:
                return "cuda:0"

        class FakeTrainer:
            def __init__(self) -> None:
                self.device = FakeCudaDevice()
                self.oom_guard = True

            def train_epoch(self, *args, **kwargs) -> dict[str, object]:
                return {"loss": 1.0}

        captured_train_config: dict[str, object] = {}

        def fake_build_phase_train_loaders(dataset, *, train_config, phase=None):
            captured_train_config["train_config"] = train_config
            captured_train_config["phase"] = phase
            return object(), None

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=Path(tmpdir)),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(batch_size=16),
                selection=SelectionConfig(),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="head_warmup",
                        stage="stage_1_frozen_trunk_warmup",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                        overrides={"batch_size": 10},
                    ),
                ),
            )
            mock_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    empty_cache=lambda: None,
                    reset_peak_memory_stats=lambda device: None,
                )
            )

            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                    with patch("tools.run_pv26_train.PV26CanonicalDataset", return_value=SimpleNamespace(records=[])):
                        with patch(
                            "tools.run_pv26_train._build_phase_train_loaders",
                            side_effect=fake_build_phase_train_loaders,
                        ):
                            with patch("tools.run_pv26_train._build_phase_trainer", return_value=FakeTrainer()):
                                with patch(
                                    "tools.run_pv26_train._cuda_memory_stats",
                                    return_value={"device": "cuda:0"},
                                ):
                                    result = run_phase_vram_stress(
                                        scenario,
                                        scenario_path=Path(tmpdir) / "default",
                                        stage="stage_1_frozen_trunk_warmup",
                                        batch_size=None,
                                        stress_iters=6,
                                    )

        train_config = captured_train_config["train_config"]
        self.assertEqual(train_config.batch_size, 10)
        self.assertEqual(train_config.train_batches, 6)
        self.assertEqual(captured_train_config["phase"].stage, "stage_1_frozen_trunk_warmup")
        self.assertEqual(result["status"], "ok")

    def test_run_phase_vram_sweep_reuses_dataset_and_reports_phase_bounds(self) -> None:
        class FakeCudaDevice:
            type = "cuda"

            def __str__(self) -> str:
                return "cuda:0"

        class FakeTrainer:
            def __init__(self) -> None:
                self.device = FakeCudaDevice()
                self.oom_guard = True

            def train_epoch(self, *args, **kwargs) -> dict[str, object]:
                return {"loss": 1.0}

        captured_train_configs: list[object] = []

        def fake_build_phase_train_loaders(dataset, *, train_config, phase=None):
            captured_train_configs.append(train_config)
            return object(), None

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=Path(tmpdir)),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(batch_size=4, num_workers=6, persistent_workers=True, prefetch_factor=2),
                selection=SelectionConfig(),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="head_warmup",
                        stage="stage_1_frozen_trunk_warmup",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                    ),
                    PhaseConfig(
                        name="lane_family_finetune",
                        stage="stage_4_lane_family_finetune",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                        min_improvement_pct=0.25,
                    ),
                ),
            )

            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache"):
                    with patch("torch.cuda.reset_peak_memory_stats"):
                        with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                            with patch("tools.run_pv26_train.PV26CanonicalDataset", return_value=SimpleNamespace(records=[])) as mocked_dataset:
                                with patch(
                                    "tools.run_pv26_train._build_phase_train_loaders",
                                    side_effect=fake_build_phase_train_loaders,
                                ):
                                    with patch("tools.run_pv26_train._build_phase_trainer", return_value=FakeTrainer()):
                                        with patch(
                                            "tools.run_pv26_train._cuda_memory_stats",
                                            return_value={"device": "cuda:0"},
                                        ):
                                            result = run_phase_vram_sweep(
                                                scenario,
                                                scenario_path=Path(tmpdir) / "default",
                                                stages="stage_1_frozen_trunk_warmup,stage_4_lane_family_finetune",
                                                batch_sizes="1,2",
                                                stress_iters=3,
                                            )

        mocked_dataset.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "phase_vram_sweep")
        self.assertEqual(result["batch_sizes"], [1, 2])
        self.assertEqual(len(result["phase_results"]), 2)
        self.assertEqual(result["phase_results"][0]["max_ok_batch_size"], 2)
        self.assertFalse(result["phase_results"][0]["ceiling_observed"])
        self.assertEqual(len(captured_train_configs), 4)
        self.assertTrue(all(config.num_workers == 0 for config in captured_train_configs))
        self.assertTrue(all(config.val_batches == 0 for config in captured_train_configs))


if __name__ == "__main__":
    unittest.main()
