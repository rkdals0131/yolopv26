from __future__ import annotations

import io
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.run_pv26_train import (
    PRESET_PATH_ROOT,
    PhaseConfig,
    PhaseTransitionController,
    PreviewConfig,
    SelectionConfig,
    _configure_torch_multiprocessing,
    _phase_entry_is_completed,
    _recover_phase_entry_from_run_dir,
    _sample_preview_selection,
    _scenario_phase_defaults,
    load_meta_train_scenario,
    main,
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
                                "patience": 2,
                                "min_improvement_pct": 0.5,
                                "overrides": {
                                    "head_lr": 0.0004,
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
        phase_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[0].overrides)
        self.assertAlmostEqual(phase_train.head_lr, 0.0015)

    def test_load_meta_train_scenario_preserves_defaults_without_user_yaml(self) -> None:
        with patch("tools.run_pv26_train.load_user_paths_config", return_value={}):
            with patch("tools.run_pv26_train.load_user_hyperparameters_config", return_value={}):
                scenario = load_meta_train_scenario("default")

        self.assertEqual(scenario.dataset.root.parts[-2:], ("seg_dataset", "pv26_exhaustive_od_lane_dataset"))
        self.assertEqual(scenario.run.run_root.parts[-2:], ("runs", "pv26_exhaustive_od_lane_train"))
        self.assertEqual(scenario.train_defaults.batch_size, 40)
        self.assertEqual(tuple(phase.stage for phase in scenario.phases), (
            "stage_1_frozen_trunk_warmup",
            "stage_2_partial_unfreeze",
            "stage_3_end_to_end_finetune",
        ))

    def test_default_preset_uses_exhaustive_dataset_and_stage_order(self) -> None:
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
        ))
        phase_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[0].overrides)
        self.assertEqual(phase_train.batch_size, 6)
        self.assertAlmostEqual(phase_train.head_lr, 0.003)
        self.assertTrue(phase_train.encode_val_batches_in_loader)

    def test_removed_stage3_stress_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unsupported PV26 meta-train preset: stage3_vram_stress"):
            load_meta_train_scenario("stage3_vram_stress")

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
                    "dataset.root and dataset.additional_roots",
                ):
                    load_meta_train_scenario("default")

    def test_load_meta_train_scenario_rejects_unknown_preset(self) -> None:
        with self.assertRaisesRegex(KeyError, "unsupported PV26 meta-train preset"):
            load_meta_train_scenario("does-not-exist")

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
        )

        with self.assertRaisesRegex(ValueError, "phase 1 must use stage"):
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
            mocked_run.assert_called_once_with(loaded_scenario, scenario_path=PRESET_PATH_ROOT / "default")
            self.assertIn('"status": "ok"', buffer.getvalue())

    def test_configure_torch_multiprocessing_uses_file_system_sharing(self) -> None:
        mock_torch = SimpleNamespace(
            multiprocessing=SimpleNamespace(
                get_sharing_strategy=lambda: "file_descriptor",
                set_sharing_strategy=lambda strategy: setattr(self, "_sharing_strategy", strategy),
            )
        )

        with patch.dict("sys.modules", {"torch": mock_torch}):
            self._sharing_strategy = None
            _configure_torch_multiprocessing()

        self.assertEqual(self._sharing_strategy, "file_system")


if __name__ == "__main__":
    unittest.main()
