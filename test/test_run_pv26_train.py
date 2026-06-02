from __future__ import annotations

import copy
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

from tools.pv26_train import runtime as train_runtime
from tools.pv26_train.artifacts import load_or_init_meta_manifest
from tools.run_pv26_train import (
    PRESET_PATH_ROOT,
    PhaseConfig,
    PhaseTransitionController,
    PreviewConfig,
    SelectionConfig,
    _build_backbone_adapter,
    _execute_phase,
    _build_phase_trainer,
    _build_phase_train_loaders,
    _build_arg_parser,
    _build_postprocess_config,
    _dataset_for_phase,
    _phase_manifest_extra,
    _configure_torch_multiprocessing,
    _phase_entry_is_completed,
    _phase_entry_is_terminal,
    _recover_phase_entry_from_run_dir,
    _resolve_head_channels,
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
from tools.pv26_train.epoch_visualization import _gt_scene_from_sample


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

    def test_build_backbone_adapter_prefers_roadmark_trunk_contract(self) -> None:
        train_config = TrainDefaultsConfig(backbone_variant="s", backbone_weights="custom.pt")
        adapter = object()

        with patch("tools.run_pv26_train.resolve_yolo26_weights", return_value="resolved.pt") as resolve_mock:
            with patch("tools.run_pv26_train.build_yolo26_roadmark_trunk", return_value=adapter) as roadmark_mock:
                with patch("tools.run_pv26_train.build_yolo26_trunk") as generic_mock:
                    with patch("tools.run_pv26_train.build_yolo26n_trunk") as compat_mock:
                        result = _build_backbone_adapter(train_config)

        self.assertIs(result, adapter)
        resolve_mock.assert_called_once_with(variant="s", weights="custom.pt")
        roadmark_mock.assert_called_once_with(variant="s", weights="resolved.pt")
        generic_mock.assert_not_called()
        compat_mock.assert_not_called()

    def test_resolve_head_channels_reconstructs_four_level_contract_from_detect_adapter(self) -> None:
        train_config = TrainDefaultsConfig(backbone_variant="s")
        adapter = object()

        with patch("tools.run_pv26_train.infer_pyramid_channels", return_value=(128, 256, 512)):
            channels = _resolve_head_channels(adapter, train_config)

        self.assertEqual(channels, (128, 128, 256, 512))

    def test_runtime_meta_train_scenario_tracks_manifest_lifecycle_for_selected_window(self) -> None:
        class _FakeDataset:
            records = (
                SimpleNamespace(split="train", dataset_key="aihub_lane_seoul"),
                SimpleNamespace(split="val", dataset_key="pv26_exhaustive_bdd100k_det_100k"),
            )

            def __init__(self, roots, **kwargs) -> None:
                self.roots = roots
                self.kwargs = kwargs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            run_dir = root / "runs" / "selected"
            scenario_path = root / "default.yaml"
            seed_checkpoint = root / "seed" / "best.pt"
            seed_checkpoint.parent.mkdir(parents=True)
            seed_checkpoint.write_text("seed", encoding="utf-8")
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=dataset_root),
                run=RunConfig(run_root=root / "runs", run_dir=run_dir),
                train_defaults=TrainDefaultsConfig(train_augmentation=True, train_augmentation_seed=123),
                selection=SelectionConfig(metric_path="val.losses.total.mean", mode="min"),
                preview=ScenarioPreviewConfig(enabled=False),
                phases=(
                    PhaseConfig(
                        name="warmup",
                        stage="stage_1_frozen_trunk_warmup",
                        min_epochs=1,
                        max_epochs=1,
                        patience=1,
                    ),
                    PhaseConfig(
                        name="unfreeze",
                        stage="stage_2_partial_unfreeze",
                        min_epochs=1,
                        max_epochs=2,
                        patience=1,
                    ),
                    PhaseConfig(
                        name="lane",
                        stage="stage_4_lane_family_finetune",
                        min_epochs=1,
                        max_epochs=3,
                        patience=1,
                    ),
                ),
            )
            manifest = {
                "version": "pv26-meta-train-v1",
                "status": "running",
                "scenario_path": str(scenario_path),
                "run_dir": str(run_dir),
                "active_phase_index": None,
                "active_phase_name": None,
                "phases": [
                    {"index": 1, "name": "warmup", "stage": "stage_1_frozen_trunk_warmup", "status": "pending"},
                    {"index": 2, "name": "unfreeze", "stage": "stage_2_partial_unfreeze", "status": "pending"},
                    {"index": 3, "name": "lane", "stage": "stage_4_lane_family_finetune", "status": "pending"},
                ],
            }
            manifest_path = run_dir / "meta_manifest.json"
            logs: list[str] = []
            manifest_writes: list[dict] = []
            summary_writes: list[dict] = []
            execute_calls: list[dict] = []

            def fake_load_or_init_meta_manifest(**kwargs):
                self.assertEqual(kwargs["selected_phase_window"]["selected_phase_indices"], [2, 3])
                self.assertEqual(kwargs["lineage"]["mode"], "derived_run")
                return manifest, manifest_path

            def fake_write_meta_manifest(path, payload):
                self.assertEqual(path, manifest_path)
                manifest_writes.append(copy.deepcopy(payload))

            def fake_write_meta_summary(path, payload):
                self.assertEqual(path, run_dir)
                summary_writes.append(copy.deepcopy(payload))

            def fake_execute_phase(**kwargs):
                phase_index = int(kwargs["phase_index"])
                execute_calls.append(
                    {
                        "phase_index": phase_index,
                        "phase_name": kwargs["phase"].name,
                        "previous_best_checkpoint": kwargs["previous_best_checkpoint"],
                        "preview_samples": kwargs["preview_samples"],
                    }
                )
                best_checkpoint = root / f"phase_{phase_index}_best.pt"
                return {
                    "index": phase_index,
                    "name": kwargs["phase"].name,
                    "stage": kwargs["phase"].stage,
                    "status": "completed",
                    "run_dir": str(run_dir / f"phase_{phase_index}"),
                    "summary_path": str(run_dir / f"phase_{phase_index}" / "summary.json"),
                    "run_manifest_path": str(run_dir / f"phase_{phase_index}" / "run_manifest.json"),
                    "best_checkpoint_path": str(best_checkpoint),
                    "last_checkpoint_path": str(root / f"phase_{phase_index}_last.pt"),
                    "completed_epochs": phase_index,
                    "best_metric_value": 1.0 / phase_index,
                    "best_epoch": phase_index,
                    "promotion_reason": "completed",
                    "phase_state": {"phase_index": phase_index},
                    "selection": {"metric_path": "val.losses.total.mean"},
                    "backbone": {"variant": "s", "weights": None},
                    "postprocess": {},
                    "head_channels": [128, 128, 256, 512],
                    "preview": {"best": None, "last": None},
                    "phase_train_config": {},
                    "run_summary": {"completed_epochs": phase_index},
                }

            result = train_runtime.run_meta_train_scenario(
                scenario,
                scenario_path=scenario_path,
                configure_torch_multiprocessing=lambda: logs.append("configured"),
                log_meta_train=logs.append,
                canonical_dataset_cls=_FakeDataset,
                resolve_meta_run_dir=lambda scenario, *, scenario_path: run_dir,
                sample_preview_selection_with_logging=lambda dataset, preview, *, progress_callback: [
                    {"meta": {"sample_id": "preview-1", "dataset_key": "aihub_lane_seoul"}}
                ],
                load_or_init_meta_manifest=fake_load_or_init_meta_manifest,
                phase_entry_is_completed=lambda entry, phase: entry.get("status") == "completed",
                recover_phase_entry_from_run_dir=lambda entry, phase: None,
                scenario_snapshot_for_run=lambda scenario, *, run_dir: {"snapshot_run_dir": str(run_dir)},
                write_meta_manifest=fake_write_meta_manifest,
                write_meta_summary=fake_write_meta_summary,
                resolve_phase_selection=lambda selection, phase: phase.selection or selection,
                execute_phase=fake_execute_phase,
                phase_entry_is_terminal=lambda entry, phase: entry.get("status") in {"completed", "skipped"},
                selected_phase_indices=(2, 3),
                initial_best_checkpoint=seed_checkpoint,
                lineage={"mode": "derived_run", "source_run_dir": str(root / "source")},
            )

        self.assertIn("configured", logs)
        self.assertEqual(len(execute_calls), 2)
        self.assertEqual(execute_calls[0]["phase_index"], 2)
        self.assertEqual(execute_calls[0]["previous_best_checkpoint"], seed_checkpoint)
        self.assertEqual(execute_calls[1]["phase_index"], 3)
        self.assertEqual(execute_calls[1]["previous_best_checkpoint"], root / "phase_2_best.pt")
        self.assertEqual(execute_calls[0]["preview_samples"][0]["meta"]["sample_id"], "preview-1")
        self.assertGreaterEqual(len(manifest_writes), 6)
        self.assertEqual(manifest_writes[0]["phases"][0]["status"], "skipped")
        self.assertEqual(manifest_writes[0]["phases"][0]["promotion_reason"], "window_excluded")
        phase2_running = next(write for write in manifest_writes if write.get("active_phase_index") == 2)
        self.assertEqual(phase2_running["active_phase_name"], "unfreeze")
        self.assertEqual(phase2_running["phases"][1]["status"], "running")
        phase3_running = next(write for write in manifest_writes if write.get("active_phase_index") == 3)
        self.assertEqual(phase3_running["active_phase_name"], "lane")
        final_manifest = manifest_writes[-1]
        self.assertEqual(final_manifest["status"], "completed")
        self.assertIsNone(final_manifest["active_phase_index"])
        self.assertIsNone(final_manifest["active_phase_name"])
        self.assertEqual([phase["status"] for phase in final_manifest["phases"]], ["skipped", "completed", "completed"])
        self.assertEqual(final_manifest["selected_phase_window"]["selected_phase_indices"], [2, 3])
        self.assertEqual(final_manifest["lineage"]["mode"], "derived_run")
        self.assertEqual(summary_writes[-1]["status"], "completed")
        self.assertEqual(result["completed_phases"], 2)
        self.assertEqual(result["skipped_phases"], 1)
        self.assertEqual(result["selected_phase_window"]["start_phase_index"], 2)
        self.assertEqual(result["lineage"]["mode"], "derived_run")
        self.assertEqual(result["final_checkpoint_path"], root / "phase_3_best.pt")

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
                            "train_augmentation": True,
                            "train_augmentation_seed": 42,
                            "train_aug_stopline_focus_crop_prob": 0.7,
                            "train_aug_stopline_focus_crop_scale_min": 1.2,
                            "train_aug_stopline_focus_crop_scale_max": 1.6,
                            "train_aug_stopline_focus_crop_jitter": 0.05,
                            "train_aug_affine_prob": 0.6,
                            "train_aug_affine_degrees": 2.5,
                            "train_aug_affine_translate_frac": 0.03,
                            "train_aug_affine_scale_min": 0.95,
                            "train_aug_affine_scale_max": 1.08,
                            "train_aug_affine_shear_degrees": 1.5,
                            "train_aug_synthetic_stopline_prob": 0.4,
                            "train_aug_synthetic_stopline_thickness_px": 6.0,
                            "train_aug_stopline_copy_paste_prob": 0.3,
                            "train_aug_stopline_copy_paste_margin_px": 12.0,
                            "train_aug_stopline_copy_paste_alpha": 0.75,
                            "roadmark_architecture": "current_family",
                            "lane_head_mode": "row_native",
                            "lane_family_query_objectness_target_mode": "metric_quality",
                            "det_conf_threshold": 0.33,
                            "lane_obj_threshold": 0.61,
                            "lane_segfirst_centerline_target_mode": "core",
                            "lane_segfirst_centerline_max_positive_weight": 8.0,
                            "lane_segfirst_residual_risk_core_weight": 0.5,
                            "lane_segfirst_residual_risk_ring_weight": 0.35,
                            "lane_segfirst_residual_risk_ring_margin": 0.2,
                            "lane_segfirst_center_offset_aux_weight": 0.45,
                            "lane_segfirst_anchor_offset_aux_weight": 0.25,
                            "lane_segfirst_task_conflict_negative_mode": "crosswalk",
                            "lane_segfirst_task_conflict_negative_weight": 0.4,
                            "lane_segfirst_task_conflict_negative_margin": 0.1,
                            "lane_conditional_row_aux_weight": 0.7,
                            "lane_segfirst_row_link_aux_weight": 0.55,
                            "lane_conditional_seed_aux_weight": 0.3,
                            "lane_conditional_seed_target_mode": "bottom_anchor",
                            "lane_conditional_objectness_target_mode": "metric_quality",
                            "lane_conditional_row_x_weight": 0.4,
                            "lane_conditional_denoise_aux_weight": 0.8,
                            "lane_conditional_denoise_hard_negative_count": 2,
                            "lane_conditional_denoise_hard_negative_offset_px": 72.0,
                            "lane_segfirst_instance_embedding_aux_weight": 0.6,
                            "lane_conditional_row_enabled": True,
                            "lane_conditional_row_dense_gate_enabled": True,
                            "lane_conditional_row_dense_min_mean_centerline": 0.42,
                            "lane_conditional_row_dense_min_mean_support": 0.37,
                            "lane_conditional_row_dense_min_points": 6,
                            "lane_segfirst_track_mode": "row_scan",
                            "lane_segfirst_max_row_gap": 24,
                            "lane_segfirst_max_link_dx": 12.0,
                            "lane_segfirst_seed_threshold": 0.62,
                            "lane_segfirst_seed_trace_max_seeds": 12,
                            "lane_segfirst_center_offset_enabled": True,
                            "lane_segfirst_center_offset_max_shift_px": 3.5,
                            "lane_segfirst_center_offset_min_support_score": 0.4,
                            "lane_segfirst_loss_weights": {
                                "centerline_bce": 1.25,
                                "centerline_dice": 1.5,
                            },
                            "lane_segfirst_color_class_weights": {
                                "yellow_lane": 1.75,
                            },
                            "stopline_selector_target_mode": "rowx_band",
                            "stopline_endpoint_pair_aux_weight": 0.9,
                            "stopline_endpoint_pair_segment_aux_weight": 0.7,
                            "stopline_endpoint_pair_verifier_aux_weight": 0.6,
                            "stopline_segment_denoise_aux_weight": 0.8,
                            "stopline_context_segment_set_aux_weight": 0.55,
                            "stopline_context_segment_verifier_aux_weight": 0.25,
                            "stopline_midpoint_aux_weight": 0.65,
                            "stopline_axis_distance_aux_weight": 0.55,
                            "stopline_axis_segment_set_aux_weight": 0.6,
                            "stopline_axis_segment_verifier_aux_weight": 0.4,
                            "stopline_patch_segment_set_aux_weight": 0.45,
                            "stopline_patch_segment_verifier_aux_weight": 0.35,
                            "stopline_segment_verifier_target_mode": "metric_quality",
                            "stopline_segment_objectness_target_mode": "metric_quality",
                            "stopline_segment_verifier_quality_tau_px": 18.0,
                            "stopline_empty_sample_mode": "positive_only",
                            "stopline_task_conflict_negative_mode": "lane_crosswalk",
                            "stopline_task_conflict_negative_weight": 0.45,
                            "stopline_task_conflict_negative_margin": 0.2,
                            "stop_line_endpoint_pair_enabled": True,
                            "stop_line_endpoint_pair_score_threshold": 0.58,
                            "stop_line_endpoint_pair_topk": 9,
                            "stop_line_endpoint_pair_max_segments": 4,
                            "stop_line_endpoint_pair_segment_enabled": True,
                            "stop_line_endpoint_pair_segment_score_threshold": 0.51,
                            "stop_line_endpoint_pair_segment_max_segments": 2,
                            "stop_line_endpoint_pair_verifier_score_weight": 0.75,
                            "stop_line_axis_distance_enabled": True,
                            "stop_line_axis_distance_valid_threshold": 0.82,
                            "stop_line_axis_distance_min_votes": 4,
                            "stop_line_axis_distance_cluster_endpoint_tolerance": 5.0,
                            "stop_line_axis_distance_max_endpoint_covariance": 12.0,
                            "stop_line_axis_distance_min_support_score": 0.42,
                            "stop_line_axis_distance_max_segments": 4,
                            "stop_line_axis_segment_set_enabled": True,
                            "stop_line_axis_segment_set_score_threshold": 0.57,
                            "stop_line_axis_segment_set_max_segments": 5,
                            "stop_line_axis_segment_verifier_score_weight": 0.75,
                            "stop_line_patch_segment_set_enabled": True,
                            "stop_line_patch_segment_set_score_threshold": 0.56,
                            "stop_line_patch_segment_set_max_segments": 4,
                            "stop_line_patch_segment_verifier_score_weight": 0.65,
                            "stop_line_context_segment_set_enabled": True,
                            "stop_line_context_segment_set_score_threshold": 0.54,
                            "stop_line_context_segment_set_max_segments": 3,
                            "stop_line_context_segment_verifier_score_weight": 0.85,
                            "stop_line_projection_comp_enabled": True,
                            "stop_line_projection_comp_proposal_source": "midpoint",
                            "stop_line_projection_comp_min_gap": 5.0,
                            "stop_line_projection_comp_topk": 40,
                            "stop_line_projection_comp_union_min_score": 0.81,
                            "stop_line_projection_comp_single_min_score": 0.91,
                            "stop_line_projection_comp_angle_threshold_deg": 15.0,
                            "stop_line_projection_comp_offset_threshold_px": 44.0,
                            "stop_line_projection_comp_min_cluster_count": 3,
                            "stop_line_projection_comp_projection_gap_px": 280.0,
                            "stop_line_projection_comp_max_predictions": 2,
                            "stop_line_projection_comp_second_min_score": 0.2,
                            "stop_line_projection_comp_second_min_fragment_count": 4,
                            "stop_line_projection_comp_second_min_length_ratio": 0.1,
                            "stop_line_component_gate_source": "selector",
                            "stopline_lane_context_fusion_enabled": True,
                            "stopline_lane_context_detach": False,
                            "stopline_crosswalk_context_fusion_enabled": True,
                            "stopline_crosswalk_context_detach": False,
                            "distill_enabled": True,
                            "distill_teacher_checkpoint": "runs/teacher.pt",
                            "distill_task_teacher_checkpoints": {
                                "stop_line": "runs/stopline_teacher.pt",
                            },
                            "distill_loss_weights": {
                                "lane": 0.0,
                                "stop_line": 0.5,
                                "crosswalk": 0.0,
                            },
                            "lane_family_include_det_source_distill_only": True,
                            "distill_sample_mode": "det_source_only",
                            "distill_confidence_mode": "teacher_positive",
                            "distill_confidence_threshold": 0.72,
                            "distill_normalize_mode": "ema",
                            "distill_ema_decay": 0.9,
                            "distill_ema_warmup_steps": 2,
                            "distill_ema_eps": 1.0e-5,
                            "task_loss_normalize_mode": "ema",
                            "task_loss_normalize_tasks": ["lane", "stop_line"],
                            "task_loss_ema_decay": 0.85,
                            "task_loss_ema_warmup_steps": 3,
                            "task_loss_ema_eps": 1.0e-4,
                            "task_loss_scale_min": 0.5,
                            "task_loss_scale_max": 2.5,
                            "task_uncertainty_weighting_enabled": True,
                            "task_uncertainty_tasks": ["lane", "crosswalk"],
                            "task_uncertainty_init_log_vars": {
                                "lane": 0.1,
                                "crosswalk": -0.2,
                            },
                            "task_uncertainty_log_var_min": -1.0,
                            "task_uncertainty_log_var_max": 1.0,
                            "lane_family_cross_stitch_enabled": True,
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
        self.assertTrue(scenario.train_defaults.train_augmentation)
        self.assertEqual(scenario.train_defaults.train_augmentation_seed, 42)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_focus_crop_prob, 0.7)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_focus_crop_scale_min, 1.2)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_focus_crop_scale_max, 1.6)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_focus_crop_jitter, 0.05)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_prob, 0.6)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_degrees, 2.5)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_translate_frac, 0.03)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_scale_min, 0.95)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_scale_max, 1.08)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_affine_shear_degrees, 1.5)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_synthetic_stopline_prob, 0.4)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_synthetic_stopline_thickness_px, 6.0)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_copy_paste_prob, 0.3)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_copy_paste_margin_px, 12.0)
        self.assertAlmostEqual(scenario.train_defaults.train_aug_stopline_copy_paste_alpha, 0.75)
        self.assertEqual(scenario.train_defaults.roadmark_architecture, "current_family")
        self.assertEqual(scenario.train_defaults.lane_head_mode, "row_native")
        self.assertEqual(scenario.train_defaults.lane_family_query_objectness_target_mode, "metric_quality")
        self.assertAlmostEqual(scenario.train_defaults.det_conf_threshold, 0.33)
        self.assertAlmostEqual(scenario.train_defaults.lane_obj_threshold, 0.61)
        self.assertEqual(scenario.train_defaults.lane_segfirst_centerline_target_mode, "core")
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_centerline_max_positive_weight, 8.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_core_weight, 0.5)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_ring_weight, 0.35)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_ring_margin, 0.2)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_center_offset_aux_weight, 0.45)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_anchor_offset_aux_weight, 0.25)
        self.assertEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_mode, "crosswalk")
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_weight, 0.4)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_margin, 0.1)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_row_aux_weight, 0.7)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_row_link_aux_weight, 0.55)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_seed_aux_weight, 0.3)
        self.assertEqual(scenario.train_defaults.lane_conditional_seed_target_mode, "bottom_anchor")
        self.assertEqual(scenario.train_defaults.lane_conditional_objectness_target_mode, "metric_quality")
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_row_x_weight, 0.4)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_denoise_aux_weight, 0.8)
        self.assertEqual(scenario.train_defaults.lane_conditional_denoise_hard_negative_count, 2)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_denoise_hard_negative_offset_px, 72.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_instance_embedding_aux_weight, 0.6)
        self.assertTrue(scenario.train_defaults.lane_conditional_row_enabled)
        self.assertTrue(scenario.train_defaults.lane_conditional_row_dense_gate_enabled)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_row_dense_min_mean_centerline, 0.42)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_row_dense_min_mean_support, 0.37)
        self.assertEqual(scenario.train_defaults.lane_conditional_row_dense_min_points, 6)
        self.assertEqual(scenario.train_defaults.lane_segfirst_track_mode, "row_scan")
        self.assertEqual(scenario.train_defaults.lane_segfirst_max_row_gap, 24)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_max_link_dx, 12.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_seed_threshold, 0.62)
        self.assertEqual(scenario.train_defaults.lane_segfirst_seed_trace_max_seeds, 12)
        self.assertTrue(scenario.train_defaults.lane_segfirst_center_offset_enabled)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_center_offset_max_shift_px, 3.5)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_center_offset_min_support_score, 0.4)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_loss_weights["centerline_bce"], 1.25)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_loss_weights["centerline_dice"], 1.5)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_color_class_weights["yellow_lane"], 1.75)
        self.assertEqual(scenario.train_defaults.stopline_selector_target_mode, "rowx_band")
        self.assertAlmostEqual(scenario.train_defaults.stopline_endpoint_pair_aux_weight, 0.9)
        self.assertAlmostEqual(scenario.train_defaults.stopline_endpoint_pair_segment_aux_weight, 0.7)
        self.assertAlmostEqual(scenario.train_defaults.stopline_endpoint_pair_verifier_aux_weight, 0.6)
        self.assertAlmostEqual(scenario.train_defaults.stopline_segment_denoise_aux_weight, 0.8)
        self.assertAlmostEqual(scenario.train_defaults.stopline_context_segment_set_aux_weight, 0.55)
        self.assertAlmostEqual(scenario.train_defaults.stopline_context_segment_verifier_aux_weight, 0.25)
        self.assertAlmostEqual(scenario.train_defaults.stopline_midpoint_aux_weight, 0.65)
        self.assertAlmostEqual(scenario.train_defaults.stopline_axis_distance_aux_weight, 0.55)
        self.assertAlmostEqual(scenario.train_defaults.stopline_axis_segment_set_aux_weight, 0.6)
        self.assertAlmostEqual(scenario.train_defaults.stopline_axis_segment_verifier_aux_weight, 0.4)
        self.assertAlmostEqual(scenario.train_defaults.stopline_patch_segment_set_aux_weight, 0.45)
        self.assertAlmostEqual(scenario.train_defaults.stopline_patch_segment_verifier_aux_weight, 0.35)
        self.assertEqual(scenario.train_defaults.stopline_segment_verifier_target_mode, "metric_quality")
        self.assertEqual(scenario.train_defaults.stopline_segment_objectness_target_mode, "metric_quality")
        self.assertAlmostEqual(scenario.train_defaults.stopline_segment_verifier_quality_tau_px, 18.0)
        self.assertEqual(scenario.train_defaults.stopline_empty_sample_mode, "positive_only")
        self.assertEqual(scenario.train_defaults.stopline_task_conflict_negative_mode, "lane_crosswalk")
        self.assertAlmostEqual(scenario.train_defaults.stopline_task_conflict_negative_weight, 0.45)
        self.assertAlmostEqual(scenario.train_defaults.stopline_task_conflict_negative_margin, 0.2)
        self.assertTrue(scenario.train_defaults.stop_line_endpoint_pair_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_endpoint_pair_score_threshold, 0.58)
        self.assertEqual(scenario.train_defaults.stop_line_endpoint_pair_topk, 9)
        self.assertEqual(scenario.train_defaults.stop_line_endpoint_pair_max_segments, 4)
        self.assertTrue(scenario.train_defaults.stop_line_endpoint_pair_segment_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_endpoint_pair_segment_score_threshold, 0.51)
        self.assertEqual(scenario.train_defaults.stop_line_endpoint_pair_segment_max_segments, 2)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_endpoint_pair_verifier_score_weight, 0.75)
        self.assertTrue(scenario.train_defaults.stop_line_axis_distance_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_distance_valid_threshold, 0.82)
        self.assertEqual(scenario.train_defaults.stop_line_axis_distance_min_votes, 4)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_distance_cluster_endpoint_tolerance, 5.0)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_distance_max_endpoint_covariance, 12.0)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_distance_min_support_score, 0.42)
        self.assertEqual(scenario.train_defaults.stop_line_axis_distance_max_segments, 4)
        self.assertTrue(scenario.train_defaults.stop_line_axis_segment_set_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_segment_set_score_threshold, 0.57)
        self.assertEqual(scenario.train_defaults.stop_line_axis_segment_set_max_segments, 5)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_axis_segment_verifier_score_weight, 0.75)
        self.assertTrue(scenario.train_defaults.stop_line_patch_segment_set_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_patch_segment_set_score_threshold, 0.56)
        self.assertEqual(scenario.train_defaults.stop_line_patch_segment_set_max_segments, 4)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_patch_segment_verifier_score_weight, 0.65)
        self.assertTrue(scenario.train_defaults.stop_line_context_segment_set_enabled)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_context_segment_set_score_threshold, 0.54)
        self.assertEqual(scenario.train_defaults.stop_line_context_segment_set_max_segments, 3)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_context_segment_verifier_score_weight, 0.85)
        self.assertTrue(scenario.train_defaults.stop_line_projection_comp_enabled)
        self.assertEqual(scenario.train_defaults.stop_line_projection_comp_proposal_source, "midpoint")
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_min_gap, 5.0)
        self.assertEqual(scenario.train_defaults.stop_line_projection_comp_topk, 40)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_union_min_score, 0.81)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_single_min_score, 0.91)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_angle_threshold_deg, 15.0)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_offset_threshold_px, 44.0)
        self.assertEqual(scenario.train_defaults.stop_line_projection_comp_min_cluster_count, 3)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_projection_gap_px, 280.0)
        self.assertEqual(scenario.train_defaults.stop_line_projection_comp_max_predictions, 2)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_second_min_score, 0.2)
        self.assertEqual(scenario.train_defaults.stop_line_projection_comp_second_min_fragment_count, 4)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_projection_comp_second_min_length_ratio, 0.1)
        self.assertEqual(scenario.train_defaults.stop_line_component_gate_source, "selector")
        self.assertTrue(scenario.train_defaults.distill_enabled)
        self.assertEqual(scenario.train_defaults.distill_teacher_checkpoint, "runs/teacher.pt")
        self.assertEqual(
            scenario.train_defaults.distill_task_teacher_checkpoints,
            {"stop_line": "runs/stopline_teacher.pt"},
        )
        self.assertAlmostEqual(scenario.train_defaults.distill_loss_weights["lane"], 0.0)
        self.assertAlmostEqual(scenario.train_defaults.distill_loss_weights["stop_line"], 0.5)
        self.assertAlmostEqual(scenario.train_defaults.distill_loss_weights["crosswalk"], 0.0)
        self.assertTrue(scenario.train_defaults.lane_family_include_det_source_distill_only)
        self.assertEqual(scenario.train_defaults.distill_sample_mode, "det_source_only")
        self.assertEqual(scenario.train_defaults.distill_confidence_mode, "teacher_positive")
        self.assertAlmostEqual(scenario.train_defaults.distill_confidence_threshold, 0.72)
        self.assertEqual(scenario.train_defaults.distill_normalize_mode, "ema")
        self.assertAlmostEqual(scenario.train_defaults.distill_ema_decay, 0.9)
        self.assertEqual(scenario.train_defaults.distill_ema_warmup_steps, 2)
        self.assertAlmostEqual(scenario.train_defaults.distill_ema_eps, 1.0e-5)
        self.assertEqual(scenario.train_defaults.task_loss_normalize_mode, "ema")
        self.assertEqual(scenario.train_defaults.task_loss_normalize_tasks, ("lane", "stop_line"))
        self.assertAlmostEqual(scenario.train_defaults.task_loss_ema_decay, 0.85)
        self.assertEqual(scenario.train_defaults.task_loss_ema_warmup_steps, 3)
        self.assertAlmostEqual(scenario.train_defaults.task_loss_ema_eps, 1.0e-4)
        self.assertAlmostEqual(scenario.train_defaults.task_loss_scale_min, 0.5)
        self.assertAlmostEqual(scenario.train_defaults.task_loss_scale_max, 2.5)
        self.assertTrue(scenario.train_defaults.task_uncertainty_weighting_enabled)
        self.assertEqual(scenario.train_defaults.task_uncertainty_tasks, ("lane", "crosswalk"))
        self.assertAlmostEqual(scenario.train_defaults.task_uncertainty_init_log_vars["lane"], 0.1)
        self.assertAlmostEqual(scenario.train_defaults.task_uncertainty_init_log_vars["crosswalk"], -0.2)
        self.assertAlmostEqual(scenario.train_defaults.task_uncertainty_log_var_min, -1.0)
        self.assertAlmostEqual(scenario.train_defaults.task_uncertainty_log_var_max, 1.0)
        self.assertTrue(scenario.train_defaults.lane_family_cross_stitch_enabled)
        self.assertTrue(scenario.train_defaults.stopline_lane_context_fusion_enabled)
        self.assertFalse(scenario.train_defaults.stopline_lane_context_detach)
        self.assertTrue(scenario.train_defaults.stopline_crosswalk_context_fusion_enabled)
        self.assertFalse(scenario.train_defaults.stopline_crosswalk_context_detach)
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
        self.assertFalse(scenario.train_defaults.amp)
        self.assertEqual(scenario.train_defaults.task_positive_task, "multi:lane,stopline,crosswalk")
        self.assertAlmostEqual(scenario.train_defaults.task_positive_fraction, 0.75)
        self.assertEqual(scenario.train_defaults.lane_family_unlabeled_negative_mode, "none")
        self.assertEqual(scenario.train_defaults.backbone_variant, "s")
        self.assertEqual(scenario.train_defaults.lane_head_mode, "seg_first")
        self.assertEqual(scenario.train_defaults.lane_objectness_target_mode, "binary")
        self.assertEqual(scenario.train_defaults.lane_family_query_objectness_target_mode, "quality_floor")
        self.assertAlmostEqual(scenario.train_defaults.lane_objectness_quality_min, 0.25)
        self.assertAlmostEqual(scenario.train_defaults.lane_objectness_quality_tau, 10.0)
        self.assertAlmostEqual(scenario.train_defaults.det_conf_threshold, 0.25)
        self.assertAlmostEqual(scenario.train_defaults.det_iou_threshold, 0.70)
        self.assertAlmostEqual(scenario.train_defaults.lane_obj_threshold, 0.45)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_core_weight, 0.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_ring_weight, 0.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_residual_risk_ring_margin, 0.2)
        self.assertEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_mode, "none")
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_weight, 0.0)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_task_conflict_negative_margin, 0.15)
        self.assertEqual(scenario.train_defaults.stopline_task_conflict_negative_mode, "none")
        self.assertAlmostEqual(scenario.train_defaults.stopline_task_conflict_negative_weight, 0.0)
        self.assertAlmostEqual(scenario.train_defaults.stopline_task_conflict_negative_margin, 0.15)
        self.assertAlmostEqual(scenario.train_defaults.lane_conditional_row_aux_weight, 0.0)
        self.assertFalse(scenario.train_defaults.lane_conditional_row_enabled)
        self.assertEqual(scenario.train_defaults.lane_segfirst_track_mode, "component")
        self.assertEqual(scenario.train_defaults.lane_segfirst_max_row_gap, 12)
        self.assertAlmostEqual(scenario.train_defaults.lane_segfirst_max_link_dx, 8.0)
        self.assertFalse(scenario.train_defaults.distill_enabled)
        self.assertIsNone(scenario.train_defaults.distill_teacher_checkpoint)
        self.assertAlmostEqual(scenario.train_defaults.stop_line_obj_threshold, 0.50)
        self.assertAlmostEqual(scenario.train_defaults.crosswalk_obj_threshold, 0.50)
        self.assertEqual(tuple(phase.stage for phase in scenario.phases), (
            "stage_1_frozen_trunk_warmup",
            "stage_2_partial_unfreeze",
            "stage_3_end_to_end_finetune",
            "stage_4_lane_family_finetune",
        ))

    def test_lane_family_phase_keeps_det_records_only_for_unlabeled_negative_train_view(self) -> None:
        class FakeDataset:
            def __init__(self) -> None:
                self.records = [
                    SimpleNamespace(dataset_key="aihub_lane_seoul"),
                    SimpleNamespace(dataset_key="pv26_exhaustive_aihub_traffic_seoul"),
                    SimpleNamespace(dataset_key="pv26_exhaustive_aihub_obstacle_seoul"),
                ]

            def __getitem__(self, index: int) -> object:
                return self.records[index]

        phase = PhaseConfig(
            name="lane_family",
            stage="stage_4_lane_family_finetune",
            min_epochs=1,
            max_epochs=1,
            patience=1,
            freeze_policy="lane_family_heads_only",
        )
        dataset = FakeDataset()
        train_config = TrainDefaultsConfig(lane_family_unlabeled_negative_mode="det_source_stop_line")

        train_view = _dataset_for_phase(
            dataset,
            phase=phase,
            train_config=train_config,
            include_unlabeled_negatives=True,
        )
        val_view = _dataset_for_phase(
            dataset,
            phase=phase,
            train_config=train_config,
            include_unlabeled_negatives=False,
        )

        self.assertEqual([record.dataset_key for record in train_view.records], [
            "aihub_lane_seoul",
            "pv26_exhaustive_aihub_traffic_seoul",
            "pv26_exhaustive_aihub_obstacle_seoul",
        ])
        self.assertEqual([record.dataset_key for record in val_view.records], ["aihub_lane_seoul"])

    def test_lane_family_phase_can_include_det_records_for_distill_only_train_view(self) -> None:
        class FakeDataset:
            def __init__(self) -> None:
                self.records = [
                    SimpleNamespace(dataset_key="aihub_lane_seoul"),
                    SimpleNamespace(dataset_key="pv26_exhaustive_bdd100k_det_100k"),
                    SimpleNamespace(dataset_key="pv26_exhaustive_aihub_traffic_seoul"),
                ]

            def __getitem__(self, index: int) -> object:
                return self.records[index]

        phase = PhaseConfig(
            name="lane_family",
            stage="stage_4_lane_family_finetune",
            min_epochs=1,
            max_epochs=1,
            patience=1,
            freeze_policy="lane_family_heads_static_trunk",
        )
        dataset = FakeDataset()
        train_config = TrainDefaultsConfig(
            distill_enabled=True,
            lane_family_include_det_source_distill_only=True,
        )

        train_view = _dataset_for_phase(
            dataset,
            phase=phase,
            train_config=train_config,
            include_unlabeled_negatives=True,
        )
        val_view = _dataset_for_phase(
            dataset,
            phase=phase,
            train_config=train_config,
            include_unlabeled_negatives=False,
        )

        self.assertEqual([record.dataset_key for record in train_view.records], [
            "aihub_lane_seoul",
            "pv26_exhaustive_bdd100k_det_100k",
            "pv26_exhaustive_aihub_traffic_seoul",
        ])
        self.assertEqual([record.dataset_key for record in val_view.records], ["aihub_lane_seoul"])

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
        self.assertEqual(phase4_train.task_positive_task, "multi:lane,stopline,crosswalk")
        self.assertAlmostEqual(phase4_train.task_positive_fraction, 1.0)

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
            lane_conditional_row_dense_gate_enabled=True,
            lane_conditional_row_dense_min_mean_centerline=0.44,
            lane_conditional_row_dense_min_mean_support=0.39,
            lane_conditional_row_dense_min_points=5,
            stop_line_obj_threshold=0.42,
            crosswalk_obj_threshold=0.43,
        )

        config = _build_postprocess_config(train_defaults)

        self.assertAlmostEqual(config.det_conf_threshold, 0.31)
        self.assertAlmostEqual(config.det_iou_threshold, 0.66)
        self.assertAlmostEqual(config.lane_obj_threshold, 0.41)
        self.assertTrue(config.lane_conditional_row_dense_gate_enabled)
        self.assertAlmostEqual(config.lane_conditional_row_dense_min_mean_centerline, 0.44)
        self.assertAlmostEqual(config.lane_conditional_row_dense_min_mean_support, 0.39)
        self.assertEqual(config.lane_conditional_row_dense_min_points, 5)
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
            task_positive_task=None,
            task_positive_fraction=None,
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

    def test_build_phase_train_loaders_propagates_encoded_loader_flags(self) -> None:
        train_config = TrainDefaultsConfig(
            batch_size=3,
            train_batches=7,
            val_batches=5,
            num_workers=2,
            pin_memory=True,
            encode_train_batches_in_loader=False,
            encode_val_batches_in_loader=True,
            persistent_workers=True,
            prefetch_factor=4,
            task_positive_task="multi:lane,stopline",
            task_positive_fraction=0.5,
        )
        dataset = object()

        with patch("tools.run_pv26_train.build_pv26_train_dataloader", return_value="train-loader") as train_mock:
            with patch("tools.run_pv26_train.build_pv26_eval_dataloader", return_value="val-loader") as val_mock:
                train_loader, val_loader = _build_phase_train_loaders(
                    dataset,  # type: ignore[arg-type]
                    train_config=train_config,
                    phase=None,
                )

        self.assertEqual(train_loader, "train-loader")
        self.assertEqual(val_loader, "val-loader")
        train_kwargs = train_mock.call_args.kwargs
        val_kwargs = val_mock.call_args.kwargs
        self.assertIs(train_mock.call_args.args[0], dataset)
        self.assertIs(val_mock.call_args.args[0], dataset)
        self.assertEqual(train_kwargs["batch_size"], 3)
        self.assertEqual(train_kwargs["num_batches"], 7)
        self.assertEqual(train_kwargs["split"], "train")
        self.assertEqual(train_kwargs["encode_batches"], False)
        self.assertEqual(train_kwargs["num_workers"], 2)
        self.assertEqual(train_kwargs["pin_memory"], True)
        self.assertEqual(train_kwargs["persistent_workers"], True)
        self.assertEqual(train_kwargs["prefetch_factor"], 4)
        self.assertEqual(train_kwargs["task_positive_task"], "multi:lane,stopline")
        self.assertEqual(train_kwargs["task_positive_fraction"], 0.5)
        self.assertEqual(val_kwargs["batch_size"], 3)
        self.assertEqual(val_kwargs["num_batches"], 5)
        self.assertEqual(val_kwargs["split"], "val")
        self.assertEqual(val_kwargs["encode_batches"], True)
        self.assertEqual(val_kwargs["num_workers"], 2)
        self.assertEqual(val_kwargs["pin_memory"], True)
        self.assertEqual(val_kwargs["persistent_workers"], True)
        self.assertEqual(val_kwargs["prefetch_factor"], 4)

    def test_build_phase_trainer_propagates_model_loss_and_runtime_config(self) -> None:
        class _FakeEvaluator:
            def __init__(self) -> None:
                self.evaluate_config = None
                self.predict_config = None

            def evaluate_batch(
                self,
                batch,
                *,
                include_predictions=False,
                compute_loss=True,
                config=None,
            ):
                self.evaluate_config = config
                return {
                    "batch": batch,
                    "include_predictions": include_predictions,
                    "compute_loss": compute_loss,
                }

            def predict_batch(self, batch, *, config=None):
                self.predict_config = config
                return [{"batch": batch}]

        class _FakeTrainer:
            def __init__(self) -> None:
                self.optimizer = object()
                self.scheduler = None

            def build_evaluator(self) -> _FakeEvaluator:
                return _FakeEvaluator()

        phase = PhaseConfig(
            name="stage4",
            stage="stage_4_lane_family_finetune",
            min_epochs=1,
            max_epochs=9,
            patience=2,
            loss_weights={"lane": 1.5, "stop_line": 2.5},
            freeze_policy="heads_only",
        )
        train_config = TrainDefaultsConfig(
            device="cpu",
            trunk_lr=1.0e-5,
            head_lr=2.0e-4,
            criterion_lr=3.0e-4,
            weight_decay=4.0e-5,
            schedule="linear",
            amp=True,
            amp_init_scale=512.0,
            accumulate_steps=3,
            grad_clip_norm=4.5,
            skip_non_finite_loss=True,
            oom_guard=True,
            task_mode="roadmark_only",
            roadmark_architecture="current_family_dense_seed_sigmoid",
            lane_head_mode="row_native",
            lane_conditional_row_coordinate_mode="delta",
            lane_conditional_row_max_delta_px=12.5,
            lane_conditional_denoise_hard_negative_count=6,
            lane_conditional_denoise_hard_negative_offset_px=8.0,
            lane_family_shared_adapter_enabled=True,
            lane_family_task_adapter_enabled=True,
            lane_family_cross_stitch_enabled=True,
            stopline_lane_context_fusion_enabled=True,
            stopline_lane_context_detach=True,
            stopline_crosswalk_context_fusion_enabled=True,
            stopline_crosswalk_context_detach=True,
            lane_assignment_mode="hungarian",
            lane_objectness_target_mode="quality",
            lane_family_query_objectness_target_mode="quality",
            lane_family_query_target_source="dense",
            lane_objectness_quality_min=0.2,
            lane_objectness_quality_tau=0.7,
            lane_dynamic_coverage_weight=0.11,
            lane_centerline_focal_weight=0.12,
            lane_centerline_dice_weight=0.13,
            lane_segfirst_loss_weights={"centerline": 1.2},
            lane_segfirst_centerline_target_mode="thick",
            lane_segfirst_centerline_max_positive_weight=3.5,
            lane_segfirst_residual_risk_core_weight=0.21,
            lane_segfirst_residual_risk_ring_weight=0.22,
            lane_segfirst_residual_risk_ring_margin=5.0,
            lane_segfirst_center_offset_aux_weight=0.23,
            lane_segfirst_anchor_offset_aux_weight=0.24,
            lane_segfirst_task_conflict_negative_mode="margin",
            lane_segfirst_task_conflict_negative_weight=0.25,
            lane_segfirst_task_conflict_negative_margin=0.26,
            lane_conditional_row_aux_weight=0.27,
            lane_segfirst_row_link_aux_weight=0.28,
            lane_conditional_seed_aux_weight=0.29,
            lane_conditional_seed_target_mode="dense",
            lane_conditional_objectness_target_mode="quality",
            lane_conditional_row_x_weight=0.31,
            lane_conditional_denoise_aux_weight=0.32,
            lane_segfirst_instance_embedding_aux_weight=0.33,
            lane_segfirst_color_class_weights={"white": 1.1},
            stopline_local_x_aux_weight=0.41,
            stopline_selector_aux_weight=0.42,
            stopline_selector_target_mode="quality",
            stopline_geometry_aux_weight=0.43,
            stopline_center_target_mode="thick",
            stopline_centerline_target_weight=0.44,
            stopline_midpoint_aux_weight=0.45,
            stopline_haf_aux_weight=0.46,
            stopline_axis_distance_aux_weight=0.47,
            stopline_endpoint_pair_aux_weight=0.48,
            stopline_endpoint_pair_segment_aux_weight=0.49,
            stopline_endpoint_pair_verifier_aux_weight=0.50,
            stopline_segment_set_aux_weight=0.51,
            stopline_segment_verifier_aux_weight=0.52,
            stopline_segment_denoise_aux_weight=0.53,
            stopline_context_segment_set_aux_weight=0.54,
            stopline_context_segment_verifier_aux_weight=0.55,
            stopline_axis_segment_set_aux_weight=0.56,
            stopline_axis_segment_verifier_aux_weight=0.57,
            stopline_patch_segment_set_aux_weight=0.58,
            stopline_patch_segment_verifier_aux_weight=0.59,
            stopline_segment_verifier_target_mode="quality",
            stopline_segment_objectness_target_mode="quality",
            stopline_segment_verifier_quality_tau_px=18.0,
            stopline_empty_sample_mode="ignore",
            lane_family_unlabeled_negative_mode="ignore",
            stopline_task_conflict_negative_mode="margin",
            stopline_task_conflict_negative_weight=0.61,
            stopline_task_conflict_negative_margin=0.62,
            distill_enabled=True,
            distill_teacher_mode="cache",
            distill_sample_mode="positive",
            distill_confidence_mode="threshold",
            distill_confidence_threshold=0.77,
            distill_loss_weights={"lane": 0.8},
            distill_normalize_mode="teacher",
            distill_ema_decay=0.91,
            distill_ema_warmup_steps=7,
            distill_ema_eps=1.0e-5,
            task_loss_normalize_mode="ema",
            task_loss_normalize_tasks=("lane", "stop_line"),
            task_loss_ema_decay=0.92,
            task_loss_ema_warmup_steps=11,
            task_loss_ema_eps=2.0e-5,
            task_loss_scale_min=0.4,
            task_loss_scale_max=2.2,
            task_uncertainty_weighting_enabled=True,
            task_uncertainty_tasks=("lane", "crosswalk"),
            task_uncertainty_init_log_vars={"lane": -0.2},
            task_uncertainty_log_var_min=-2.0,
            task_uncertainty_log_var_max=2.0,
            multitask_conflict={"enabled": True, "mode": "pcgrad", "tasks": ["lane"]},
        )
        adapter = object()
        heads = object()
        criterion = object()
        distill_teacher = object()
        scheduler = object()
        postprocess_config = object()
        fake_trainer = _FakeTrainer()

        with patch("tools.run_pv26_train._build_backbone_adapter", return_value=adapter) as adapter_mock:
            with patch("tools.run_pv26_train._resolve_head_channels", return_value=(11, 22, 33, 44)) as channels_mock:
                with patch("tools.run_pv26_train.PV26Heads", return_value=heads) as heads_mock:
                    with patch("tools.run_pv26_train.PV26MultiTaskLoss", return_value=criterion) as loss_mock:
                        with patch("tools.run_pv26_train._build_distill_teacher", return_value=distill_teacher) as distill_mock:
                            with patch("tools.run_pv26_train.PV26Trainer", return_value=fake_trainer) as trainer_mock:
                                with patch("tools.run_pv26_train.build_pv26_scheduler", return_value=scheduler) as scheduler_mock:
                                    with patch(
                                        "tools.run_pv26_train._build_postprocess_config",
                                        return_value=postprocess_config,
                                    ) as postprocess_mock:
                                        trainer = _build_phase_trainer(phase, train_config)

        self.assertIs(trainer, fake_trainer)
        adapter_mock.assert_called_once_with(train_config)
        channels_mock.assert_called_once_with(adapter, train_config)
        heads_kwargs = heads_mock.call_args.kwargs
        for key, expected in {
            "in_channels": (11, 22, 33, 44),
            "roadmark_architecture": "current_family_dense_seed_sigmoid",
            "lane_head_mode": "row_native",
            "lane_conditional_row_coordinate_mode": "delta",
            "lane_conditional_row_max_delta_px": 12.5,
            "lane_conditional_denoise_hard_negative_count": 6,
            "lane_conditional_denoise_hard_negative_offset_px": 8.0,
            "lane_family_shared_adapter_enabled": True,
            "lane_family_task_adapter_enabled": True,
            "lane_family_cross_stitch_enabled": True,
            "stopline_lane_context_fusion_enabled": True,
            "stopline_lane_context_detach": True,
            "stopline_crosswalk_context_fusion_enabled": True,
            "stopline_crosswalk_context_detach": True,
        }.items():
            self.assertEqual(heads_kwargs[key], expected)

        loss_kwargs = loss_mock.call_args.kwargs
        for key, expected in {
            "stage": "stage_4_lane_family_finetune",
            "loss_weights": {"lane": 1.5, "stop_line": 2.5},
            "task_mode": "roadmark_only",
            "lane_segfirst_loss_weights": {"centerline": 1.2},
            "lane_segfirst_color_class_weights": {"white": 1.1},
            "stopline_segment_verifier_quality_tau_px": 18.0,
            "distill_enabled": True,
            "distill_loss_weights": {"lane": 0.8},
            "task_loss_normalize_mode": "ema",
            "task_uncertainty_weighting_enabled": True,
            "task_uncertainty_init_log_vars": {"lane": -0.2},
        }.items():
            self.assertEqual(loss_kwargs[key], expected)

        distill_mock.assert_called_once_with(train_config)
        trainer_kwargs = trainer_mock.call_args.kwargs
        self.assertIs(trainer_mock.call_args.args[0], adapter)
        self.assertIs(trainer_mock.call_args.args[1], heads)
        for key, expected in {
            "stage": "stage_4_lane_family_finetune",
            "device": "cpu",
            "loss_weights": {"lane": 1.5, "stop_line": 2.5},
            "freeze_policy": "heads_only",
            "trunk_lr": 1.0e-5,
            "head_lr": 2.0e-4,
            "criterion_lr": 3.0e-4,
            "weight_decay": 4.0e-5,
            "amp": True,
            "amp_init_scale": 512.0,
            "accumulate_steps": 3,
            "grad_clip_norm": 4.5,
            "skip_non_finite_loss": True,
            "oom_guard": True,
            "multitask_conflict": {"enabled": True, "mode": "pcgrad", "tasks": ["lane"]},
        }.items():
            self.assertEqual(trainer_kwargs[key], expected)
        self.assertIs(trainer_kwargs["criterion"], criterion)
        self.assertIs(trainer_kwargs["distill_teacher"], distill_teacher)
        scheduler_mock.assert_called_once_with(fake_trainer.optimizer, epochs=9, schedule="linear")
        self.assertIs(fake_trainer.scheduler, scheduler)
        postprocess_mock.assert_called_once_with(train_config)
        self.assertIs(getattr(fake_trainer, "postprocess_config"), postprocess_config)

        evaluator = fake_trainer.build_evaluator()
        eval_result = evaluator.evaluate_batch({"image": "batch"}, include_predictions=True, compute_loss=False)
        predictions = evaluator.predict_batch({"image": "batch"})
        self.assertEqual(eval_result["include_predictions"], True)
        self.assertEqual(eval_result["compute_loss"], False)
        self.assertEqual(predictions, [{"batch": {"image": "batch"}}])
        self.assertIs(evaluator.evaluate_config, postprocess_config)
        self.assertIs(evaluator.predict_config, postprocess_config)
        self.assertIs(getattr(evaluator, "postprocess_config"), postprocess_config)

    def test_execute_phase_propagates_fit_io_and_returns_manifest_ready_result(self) -> None:
        class _FakeTrainer:
            def __init__(self) -> None:
                self.loaded_weights: list[tuple[Path, str]] = []
                self.fit_train_loader = None
                self.fit_kwargs = None
                self.heads = SimpleNamespace(in_channels=(11, 22, 33, 44))

            def load_model_weights(self, checkpoint_path: Path, *, map_location: str) -> None:
                self.loaded_weights.append((checkpoint_path, map_location))

            def fit(self, train_loader, **kwargs):
                self.fit_train_loader = train_loader
                self.fit_kwargs = kwargs
                checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_checkpoint = checkpoint_dir / "best.pt"
                last_checkpoint = checkpoint_dir / "last.pt"
                best_checkpoint.write_text("best", encoding="utf-8")
                last_checkpoint.write_text("last", encoding="utf-8")
                return {
                    "completed_epochs": 3,
                    "best_metric_value": 0.812,
                    "best_epoch": 2,
                    "checkpoint_paths": {
                        "best": str(best_checkpoint),
                        "last": str(last_checkpoint),
                    },
                    "early_exit": {
                        "reason": "plateau",
                        "phase_state": {"best_phase_objective": 0.812},
                    },
                    "history_paths": {
                        "train_steps": str(Path(kwargs["run_dir"]) / "history" / "train_steps.jsonl"),
                        "epochs": str(Path(kwargs["run_dir"]) / "history" / "epochs.jsonl"),
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "audit"
            previous_best_checkpoint = root / "previous" / "best.pt"
            previous_best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            previous_best_checkpoint.write_text("previous", encoding="utf-8")
            phase = PhaseConfig(
                name="lane_finetune",
                stage="stage_4_lane_family_finetune",
                min_epochs=1,
                max_epochs=5,
                patience=2,
                selection=SelectionConfig(metric_path="val.custom_metric", mode="max"),
                loss_weights={"lane": 1.5},
                freeze_policy="heads_only",
                overrides={
                    "batch_size": 6,
                    "checkpoint_every": 3,
                    "train_batches": 4,
                    "val_batches": 2,
                    "log_every_n_steps": 7,
                    "profile_window": 8,
                    "profile_device_sync": False,
                    "step_history_enabled": True,
                    "step_history_every_n_steps": 9,
                    "step_history_include_grad_details": True,
                    "pcgrad_diagnostics_enabled": True,
                    "pcgrad_aggregate_every_n_steps": 10,
                    "pcgrad_keep_raw_every_n_steps": 11,
                },
            )
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=root / "dataset"),
                run=RunConfig(run_root=root / "runs"),
                train_defaults=TrainDefaultsConfig(device="cpu", batch_size=2, checkpoint_every=1),
                selection=SelectionConfig(metric_path="val.losses.total.mean", mode="min"),
                preview=ScenarioPreviewConfig(enabled=True),
                phases=(phase,),
            )
            fake_trainer = _FakeTrainer()

            with patch("tools.run_pv26_train._build_phase_train_loaders", return_value=("train-loader", "val-loader")) as loaders_mock:
                with patch("tools.run_pv26_train._build_phase_trainer", return_value=fake_trainer) as trainer_mock:
                    with patch("tools.run_pv26_train.build_epoch_comparison_grid_callback", return_value="epoch-callback") as callback_mock:
                        with patch(
                            "tools.run_pv26_train._generate_phase_preview_bundle",
                            side_effect=lambda **kwargs: {
                                "enabled": True,
                                "kind": kwargs["preview_kind"],
                                "checkpoint": str(kwargs["checkpoint_path"]),
                            },
                        ) as preview_mock:
                            result = _execute_phase(
                                scenario=scenario,
                                scenario_path=root / "default.yaml",
                                dataset=object(),  # type: ignore[arg-type]
                                preview_samples=[{"meta": {"sample_id": "sample-1", "dataset_key": "aihub_lane_seoul"}}],
                                phase_index=1,
                                phase=phase,
                                run_dir=run_dir,
                                previous_best_checkpoint=previous_best_checkpoint,
                            )

        phase_train_config = loaders_mock.call_args.kwargs["train_config"]
        self.assertEqual(phase_train_config.batch_size, 6)
        self.assertEqual(phase_train_config.checkpoint_every, 3)
        self.assertEqual(phase_train_config.train_batches, 4)
        self.assertEqual(phase_train_config.val_batches, 2)
        loaders_mock.assert_called_once()
        trainer_mock.assert_called_once_with(phase, phase_train_config)
        self.assertEqual(fake_trainer.loaded_weights, [(previous_best_checkpoint, "cpu")])
        self.assertEqual(fake_trainer.fit_train_loader, "train-loader")

        fit_kwargs = fake_trainer.fit_kwargs
        self.assertEqual(fit_kwargs["epochs"], 5)
        self.assertEqual(fit_kwargs["phase_index"], 1)
        self.assertEqual(fit_kwargs["phase_count"], 1)
        self.assertEqual(fit_kwargs["phase_name"], "lane_finetune")
        self.assertEqual(fit_kwargs["val_loader"], "val-loader")
        self.assertEqual(fit_kwargs["run_dir"], run_dir / "phase_1")
        self.assertEqual(fit_kwargs["checkpoint_every"], 3)
        self.assertEqual(fit_kwargs["max_train_batches"], 4)
        self.assertEqual(fit_kwargs["max_val_batches"], 2)
        self.assertEqual(fit_kwargs["best_metric"], "val.custom_metric")
        self.assertEqual(fit_kwargs["best_mode"], "max")
        self.assertEqual(fit_kwargs["auto_resume"], True)
        self.assertEqual(fit_kwargs["enable_tensorboard"], True)
        self.assertEqual(fit_kwargs["epoch_end_callback"], "epoch-callback")
        for key, expected in {
            "log_every_n_steps": 7,
            "profile_window": 8,
            "profile_device_sync": False,
            "step_history_enabled": True,
            "step_history_every_n_steps": 9,
            "step_history_include_grad_details": True,
            "pcgrad_diagnostics_enabled": True,
            "pcgrad_aggregate_every_n_steps": 10,
            "pcgrad_keep_raw_every_n_steps": 11,
        }.items():
            self.assertEqual(fit_kwargs[key], expected)
        self.assertEqual(fit_kwargs["run_manifest_extra"]["phase_train_config"]["batch_size"], 6)
        self.assertEqual(fit_kwargs["run_manifest_extra"]["phase"]["selection"]["metric_path"], "val.custom_metric")
        self.assertEqual(fit_kwargs["run_manifest_extra"]["head_channels"], [11, 22, 33, 44])
        callback_mock.assert_called_once()
        self.assertEqual(preview_mock.call_count, 2)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["run_dir"], str(run_dir / "phase_1"))
        self.assertEqual(result["summary_path"], str(run_dir / "phase_1" / "summary.json"))
        self.assertEqual(result["run_manifest_path"], str(run_dir / "phase_1" / "run_manifest.json"))
        self.assertTrue(str(result["best_checkpoint_path"]).endswith("/phase_1/checkpoints/best.pt"))
        self.assertTrue(str(result["last_checkpoint_path"]).endswith("/phase_1/checkpoints/last.pt"))
        self.assertEqual(result["completed_epochs"], 3)
        self.assertEqual(result["best_metric_value"], 0.812)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["promotion_reason"], "plateau")
        self.assertEqual(result["phase_state"], {"best_phase_objective": 0.812})
        self.assertEqual(result["selection"]["metric_path"], "val.custom_metric")
        self.assertEqual(result["head_channels"], [11, 22, 33, 44])
        self.assertEqual(result["phase_train_config"]["checkpoint_every"], 3)
        self.assertEqual(result["run_summary"]["completed_epochs"], 3)
        self.assertEqual(result["preview"]["best"]["kind"], "best")
        self.assertEqual(result["preview"]["last"]["kind"], "last")

    def test_epoch_comparison_ground_truth_overlay_uses_raw_coordinates(self) -> None:
        sample = {
            "det_targets": {
                "boxes_xyxy": torch.tensor([[62.5, 141.5, 125.0, 204.0]], dtype=torch.float32),
                "classes": torch.tensor([0], dtype=torch.long),
            },
            "lane_targets": {
                "lanes": [
                    {
                        "points_xy": torch.tensor([[62.5, 141.5], [125.0, 204.0]], dtype=torch.float32),
                        "color": 0,
                        "lane_type": 0,
                    }
                ],
                "stop_lines": [
                    {"points_xy": torch.tensor([[187.5, 266.5], [250.0, 266.5]], dtype=torch.float32)}
                ],
                "crosswalks": [
                    {
                        "points_xy": torch.tensor(
                            [[312.5, 329.0], [375.0, 329.0], [375.0, 391.5], [312.5, 391.5]],
                            dtype=torch.float32,
                        )
                    }
                ],
            },
            "valid_mask": {
                "lane": torch.tensor([True], dtype=torch.bool),
                "stop_line": torch.tensor([True], dtype=torch.bool),
                "crosswalk": torch.tensor([True], dtype=torch.bool),
            },
            "meta": {
                "sample_id": "sample",
                "dataset_key": "aihub_lane_seoul",
                "split": "val",
                "image_path": "/tmp/sample.jpg",
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
            },
        }

        scene = _gt_scene_from_sample(sample)

        self.assertEqual(scene["detections"][0]["bbox"], [100.0, 100.0, 200.0, 200.0])
        self.assertEqual(scene["lanes"][0]["points"], [[100.0, 100.0], [200.0, 200.0]])
        self.assertEqual(scene["stop_lines"][0]["points"], [[300.0, 300.0], [400.0, 300.0]])
        self.assertEqual(
            scene["crosswalks"][0]["points"],
            [[500.0, 400.0], [600.0, 400.0], [600.0, 500.0], [500.0, 500.0]],
        )

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

    def test_sample_preview_selection_prioritizes_scene_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weak_scene = root / "weak.json"
            rich_scene = root / "rich.json"
            weak_scene.write_text(json.dumps({"tasks": {"has_lane": 1}, "lanes": [{}]}), encoding="utf-8")
            rich_scene.write_text(
                json.dumps(
                    {
                        "tasks": {"has_lane": 1, "has_stop_line": 1, "has_crosswalk": 1},
                        "lanes": [{}],
                        "stop_lines": [{}],
                        "crosswalks": [{}],
                    }
                ),
                encoding="utf-8",
            )

            class _FakeDataset:
                def __init__(self) -> None:
                    self.records = [
                        SimpleNamespace(
                            dataset_key="aihub_lane_seoul",
                            split="val",
                            sample_id="weak",
                            scene_path=weak_scene,
                        ),
                        SimpleNamespace(
                            dataset_key="aihub_lane_seoul",
                            split="val",
                            sample_id="rich",
                            scene_path=rich_scene,
                        ),
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
                    dataset_keys=("aihub_lane_seoul",),
                    max_samples_per_dataset=1,
                    write_overlay=False,
                ),
            )

        self.assertEqual(dataset.loaded_indices, [1])
        self.assertEqual([item["meta"]["sample_id"] for item in selected], ["rich"])

    def test_sample_preview_selection_tolerates_non_object_scene_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_scene = root / "bad.json"
            rich_scene = root / "rich.json"
            bad_scene.write_text("[]\n", encoding="utf-8")
            rich_scene.write_text(
                json.dumps(
                    {
                        "tasks": {"has_lane": 1, "has_stop_line": 1, "has_crosswalk": 1},
                        "lanes": [{}],
                        "stop_lines": [{}],
                        "crosswalks": [{}],
                    }
                ),
                encoding="utf-8",
            )

            class _FakeDataset:
                def __init__(self) -> None:
                    self.records = [
                        SimpleNamespace(
                            dataset_key="aihub_lane_seoul",
                            split="val",
                            sample_id="bad",
                            scene_path=bad_scene,
                        ),
                        SimpleNamespace(
                            dataset_key="aihub_lane_seoul",
                            split="val",
                            sample_id="rich",
                            scene_path=rich_scene,
                        ),
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
                    dataset_keys=("aihub_lane_seoul",),
                    max_samples_per_dataset=1,
                    write_overlay=False,
                ),
            )

        self.assertEqual(dataset.loaded_indices, [1])
        self.assertEqual([item["meta"]["sample_id"] for item in selected], ["rich"])

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

    def test_main_resume_run_forwards_manifest_window_and_seed_checkpoint(self) -> None:
        loaded_scenario = SimpleNamespace(run=SimpleNamespace(run_dir="/tmp/existing_run"))

        with tempfile.TemporaryDirectory() as temp_dir:
            seed_checkpoint = Path(temp_dir) / "source_run" / "phase_2" / "checkpoints" / "best.pt"
            resume_context = {
                "selected_phase_window": {
                    "start_phase_index": 2,
                    "end_phase_index": 4,
                    "selected_phase_indices": [2, 3, 4],
                    "total_phases": 4,
                },
                "lineage": {
                    "mode": "derived_run",
                    "source_run_dir": str(Path(temp_dir) / "source_run"),
                    "seed_checkpoint_path": str(seed_checkpoint),
                },
            }
            with patch(
                "tools.run_pv26_train.load_meta_train_resume_scenario",
                return_value=(loaded_scenario, PRESET_PATH_ROOT / "default"),
            ):
                with patch(
                    "tools.run_pv26_train.load_meta_train_resume_context",
                    return_value=resume_context,
                ):
                    with patch(
                        "tools.run_pv26_train.run_meta_train_scenario",
                        return_value={"status": "ok", "scenario_path": str(PRESET_PATH_ROOT / "default")},
                    ) as mocked_run:
                        buffer = io.StringIO()
                        with redirect_stdout(buffer):
                            main(["--resume-run", "/tmp/existing_run"])

            mocked_run.assert_called_once_with(
                loaded_scenario,
                scenario_path=PRESET_PATH_ROOT / "default",
                selected_phase_indices=(2, 3, 4),
                initial_best_checkpoint=seed_checkpoint.resolve(),
                lineage=resume_context["lineage"],
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

    def test_run_phase_vram_sweep_stops_phase_on_non_finite_skips(self) -> None:
        class FakeCudaDevice:
            type = "cuda"

            def __str__(self) -> str:
                return "cuda:0"

        class FakeTrainer:
            def __init__(self, batch_size: int) -> None:
                self.device = FakeCudaDevice()
                self.oom_guard = True
                self.batch_size = int(batch_size)

            def train_epoch(self, *args, **kwargs) -> dict[str, object]:
                if self.batch_size >= 2:
                    return {
                        "attempted_batches": 3,
                        "successful_batches": 2,
                        "skipped_batches": 1,
                        "skipped_reasons": {"non_finite_loss": 1},
                    }
                return {
                    "attempted_batches": 3,
                    "successful_batches": 3,
                    "skipped_batches": 0,
                    "skipped_reasons": {},
                }

        captured_train_configs: list[object] = []

        def fake_build_phase_train_loaders(dataset, *, train_config, phase=None):
            captured_train_configs.append(train_config)
            return object(), None

        def fake_build_phase_trainer(phase, train_config):
            return FakeTrainer(train_config.batch_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = MetaTrainScenario(
                dataset=DatasetConfig(root=Path(tmpdir)),
                run=RunConfig(run_root=Path(tmpdir) / "runs"),
                train_defaults=TrainDefaultsConfig(batch_size=4),
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
                ),
            )

            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache"):
                    with patch("torch.cuda.reset_peak_memory_stats"):
                        with patch("tools.run_pv26_train._configure_torch_multiprocessing"):
                            with patch("tools.run_pv26_train.PV26CanonicalDataset", return_value=SimpleNamespace(records=[])):
                                with patch(
                                    "tools.run_pv26_train._build_phase_train_loaders",
                                    side_effect=fake_build_phase_train_loaders,
                                ):
                                    with patch("tools.run_pv26_train._build_phase_trainer", side_effect=fake_build_phase_trainer):
                                        with patch(
                                            "tools.run_pv26_train._cuda_memory_stats",
                                            return_value={"device": "cuda:0"},
                                        ):
                                            result = run_phase_vram_sweep(
                                                scenario,
                                                scenario_path=Path(tmpdir) / "default",
                                                stages="stage_1_frozen_trunk_warmup",
                                                batch_sizes="1,2,4",
                                                stress_iters=3,
                                            )

        phase_result = result["phase_results"][0]
        self.assertEqual(phase_result["max_ok_batch_size"], 1)
        self.assertEqual(phase_result["first_failure_batch_size"], 2)
        self.assertIsNone(phase_result["first_oom_batch_size"])
        self.assertEqual(phase_result["first_non_finite_batch_size"], 2)
        self.assertEqual(phase_result["failure_status"], "non_finite")
        self.assertEqual(len(phase_result["attempts"]), 2)
        self.assertEqual(len(captured_train_configs), 2)


if __name__ == "__main__":
    unittest.main()
