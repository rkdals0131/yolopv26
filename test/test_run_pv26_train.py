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
    PhaseConfig,
    PhaseTransitionController,
    PreviewConfig,
    SelectionConfig,
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
    def test_load_meta_train_scenario_resolves_relative_paths_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = config_dir / "meta.yaml"
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    dataset:
                      aihub_root: ../data/aihub
                      include_bdd: true
                      bdd_root: ../data/bdd
                    run:
                      run_root: ../runs
                      run_name_prefix: baseline
                      run_dir: ../runs/meta_baseline
                      tensorboard_mode: curated
                    train_defaults:
                      device: cpu
                      batch_size: 8
                      encode_train_batches_in_loader: true
                      encode_val_batches_in_loader: false
                    selection:
                      metric_path: val.losses.total.mean
                      mode: min
                    preview:
                      enabled: true
                      split: val
                      dataset_keys: [aihub_traffic_seoul, bdd100k_det_100k]
                      max_samples_per_dataset: 2
                      write_overlay: false
                    phases:
                      - name: head_warmup
                        stage: stage_1_frozen_trunk_warmup
                        min_epochs: 2
                        max_epochs: 5
                        patience: 2
                        min_improvement_pct: 2.0
                        overrides:
                          head_lr: 0.01
                      - name: partial_unfreeze
                        stage: stage_2_partial_unfreeze
                        min_epochs: 2
                        max_epochs: 4
                        patience: 2
                        min_improvement_pct: 1.0
                        overrides:
                          trunk_lr: 0.00005
                      - name: full_finetune
                        stage: stage_3_end_to_end_finetune
                        min_epochs: 1
                        max_epochs: 3
                        patience: 1
                        min_improvement_pct: 0.5
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_meta_train_scenario(scenario_path)

            self.assertEqual(scenario.dataset.aihub_root, (root / "data" / "aihub").resolve())
            self.assertEqual(scenario.dataset.bdd_root, (root / "data" / "bdd").resolve())
            self.assertEqual(scenario.run.run_dir, (root / "runs" / "meta_baseline").resolve())
            self.assertEqual(scenario.preview.dataset_keys, ("aihub_traffic_seoul", "bdd100k_det_100k"))
            phase_train = _scenario_phase_defaults(scenario.train_defaults, scenario.phases[0].overrides)
            self.assertEqual(phase_train.batch_size, 8)
            self.assertAlmostEqual(phase_train.head_lr, 0.01)
            self.assertFalse(phase_train.encode_val_batches_in_loader)

    def test_load_exhaustive_od_lane_scenario_uses_final_dataset_keys(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scenario_path = repo_root / "tools" / "od_bootstrap" / "config" / "pv26_train" / "pv26_exhaustive_od_lane.default.yaml"

        scenario = load_meta_train_scenario(scenario_path)

        self.assertFalse(scenario.dataset.include_bdd)
        self.assertEqual(scenario.dataset.aihub_root, (repo_root / "seg_dataset" / "pv26_exhaustive_od_lane_dataset").resolve())
        self.assertEqual(scenario.run.run_root, (repo_root / "runs" / "pv26_exhaustive_od_lane_train").resolve())
        self.assertEqual(
            scenario.preview.dataset_keys,
            (
                "pv26_exhaustive_bdd100k_det_100k",
                "pv26_exhaustive_aihub_traffic_seoul",
                "pv26_exhaustive_aihub_obstacle_seoul",
                "aihub_lane_seoul",
            ),
        )

    def test_load_exhaustive_od_lane_smoke_scenario_uses_smoke_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scenario_path = (
            repo_root / "tools" / "od_bootstrap" / "config" / "pv26_train" / "pv26_exhaustive_od_lane.smoke.yaml"
        )

        scenario = load_meta_train_scenario(scenario_path)

        self.assertFalse(scenario.dataset.include_bdd)
        self.assertEqual(
            scenario.dataset.aihub_root,
            (repo_root / "seg_dataset" / "pv26_exhaustive_od_lane_dataset_smoke").resolve(),
        )
        self.assertEqual(
            scenario.run.run_root,
            (repo_root / "runs" / "pv26_exhaustive_od_lane_train_smoke").resolve(),
        )
        self.assertEqual(scenario.train_defaults.train_batches, 4)
        self.assertEqual(scenario.train_defaults.val_batches, 2)
        self.assertEqual(scenario.preview.max_samples_per_dataset, 1)

    def test_load_meta_train_scenario_rejects_invalid_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "invalid.yaml"
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    phases:
                      - name: wrong_first
                        stage: stage_2_partial_unfreeze
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                      - name: second
                        stage: stage_1_frozen_trunk_warmup
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                      - name: third
                        stage: stage_3_end_to_end_finetune
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "phase 1 must use stage"):
                load_meta_train_scenario(scenario_path)

    def test_load_meta_train_scenario_rejects_val_metric_without_val_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "invalid_val.yaml"
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    train_defaults:
                      val_batches: 0
                    phases:
                      - name: head_warmup
                        stage: stage_1_frozen_trunk_warmup
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                      - name: partial_unfreeze
                        stage: stage_2_partial_unfreeze
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                      - name: full_finetune
                        stage: stage_3_end_to_end_finetune
                        min_epochs: 1
                        max_epochs: 2
                        patience: 1
                        min_improvement_pct: 1.0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires val"):
                load_meta_train_scenario(scenario_path)

    def test_phase_transition_controller_stops_after_plateau(self) -> None:
        controller = PhaseTransitionController(
            phase=PhaseConfig(
                name="head_warmup",
                stage="stage_1_frozen_trunk_warmup",
                min_epochs=2,
                max_epochs=8,
                patience=2,
                min_improvement_pct=1.0,
            ),
            selection=SelectionConfig(metric_path="val.losses.total.mean", mode="min", eps=1e-8),
        )

        self.assertIsNone(controller.observe_epoch(_epoch_summary(epoch=1, metric_value=100.0)))
        self.assertIsNone(controller.observe_epoch(_epoch_summary(epoch=2, metric_value=95.0)))
        self.assertIsNone(controller.observe_epoch(_epoch_summary(epoch=3, metric_value=94.7)))
        stop_state = controller.observe_epoch(_epoch_summary(epoch=4, metric_value=94.5))

        self.assertIsNotNone(stop_state)
        self.assertEqual(stop_state["reason"], "plateau")
        self.assertEqual(stop_state["phase_state"]["plateau_count"], 2)
        self.assertEqual(controller.best_epoch, 4)

    def test_phase_transition_controller_stops_at_max_epochs(self) -> None:
        controller = PhaseTransitionController(
            phase=PhaseConfig(
                name="full_finetune",
                stage="stage_3_end_to_end_finetune",
                min_epochs=1,
                max_epochs=3,
                patience=10,
                min_improvement_pct=10.0,
            ),
            selection=SelectionConfig(metric_path="val.losses.total.mean", mode="min", eps=1e-8),
        )

        self.assertIsNone(controller.observe_epoch(_epoch_summary(epoch=1, metric_value=100.0)))
        self.assertIsNone(controller.observe_epoch(_epoch_summary(epoch=2, metric_value=92.0)))
        stop_state = controller.observe_epoch(_epoch_summary(epoch=3, metric_value=91.5))

        self.assertIsNotNone(stop_state)
        self.assertEqual(stop_state["reason"], "max_epochs_reached")
        self.assertEqual(stop_state["phase_state"]["epoch"], 3)

    def test_phase_entry_is_completed_when_summary_contains_early_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            phase_run_dir = Path(temp_dir) / "phase_1"
            phase_run_dir.mkdir(parents=True, exist_ok=True)
            (phase_run_dir / "summary.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "completed_epochs": 4,
                      "early_exit": {
                        "reason": "plateau"
                      }
                    }
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

            self.assertTrue(_phase_entry_is_completed(entry, phase))

    def test_recover_phase_entry_from_run_dir_uses_summary_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            phase_run_dir = Path(temp_dir) / "phase_2"
            checkpoint_dir = phase_run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = checkpoint_dir / "best.pt"
            last_checkpoint = checkpoint_dir / "last.pt"
            best_checkpoint.write_bytes(b"best")
            last_checkpoint.write_bytes(b"last")
            (phase_run_dir / "summary.json").write_text(
                textwrap.dedent(
                    f"""
                    {{
                      "completed_epochs": 3,
                      "best_metric_value": 12.5,
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
        self.assertEqual(
            [item["meta"]["sample_id"] for item in selected],
            ["val_a", "val_b"],
        )

    def test_main_accepts_config_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "smoke.yaml"
            scenario_path.write_text("phases: []\n", encoding="utf-8")
            loaded_scenario = SimpleNamespace()

            with patch("tools.run_pv26_train.load_meta_train_scenario", return_value=loaded_scenario) as mocked_load:
                with patch(
                    "tools.run_pv26_train.run_meta_train_scenario",
                    return_value={"status": "ok", "scenario_path": str(scenario_path)},
                ) as mocked_run:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        main(["--config", str(scenario_path)])

            mocked_load.assert_called_once_with(scenario_path.resolve())
            mocked_run.assert_called_once_with(loaded_scenario, scenario_path=scenario_path.resolve())
            self.assertIn('"status": "ok"', buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
