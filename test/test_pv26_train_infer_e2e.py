from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from model.data import PV26CanonicalDataset, collate_pv26_samples
from pv26_prepared_dataset_fixture import (
    DEFAULT_PREPARED_DATASET_KEYS,
    build_prepared_dataset_e2e_scenario,
    create_prepared_pv26_dataset,
    select_prepared_samples,
)
from tools.run_pv26_train import _build_phase_trainer, run_meta_train_scenario


def _assert_nested_finite(test_case: unittest.TestCase, value, *, field_name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_nested_finite(test_case, item, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_nested_finite(test_case, item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _assert_nested_finite(test_case, item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        test_case.assertTrue(math.isfinite(float(value)), msg=field_name)


class PV26PreparedDatasetE2ETests(unittest.TestCase):
    def test_prepared_dataset_train_eval_predict_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = create_prepared_pv26_dataset(root / "pv26_prepared_dataset")
            run_root = root / "runs"
            scenario = build_prepared_dataset_e2e_scenario(
                dataset_root=dataset_root,
                run_root=run_root,
            )
            scenario_path = root / "prepared_dataset_e2e.yaml"
            scenario_path.write_text("prepared_dataset_e2e: true\n", encoding="utf-8")

            summary = run_meta_train_scenario(
                scenario,
                scenario_path=scenario_path,
                selected_phase_indices=(1,),
            )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_phases"], 1)
            self.assertTrue(Path(summary["run_dir"]).is_dir())
            self.assertTrue(Path(summary["summary_path"]).is_file())
            self.assertTrue(Path(summary["meta_manifest_path"]).is_file())

            phase_summary = summary["phases"][0]
            self.assertEqual(phase_summary["status"], "completed")
            self.assertEqual(phase_summary["completed_epochs"], 1)
            self.assertTrue(Path(phase_summary["run_dir"]).is_dir())
            self.assertTrue(Path(phase_summary["summary_path"]).is_file())
            self.assertTrue(Path(phase_summary["run_manifest_path"]).is_file())
            self.assertTrue(Path(phase_summary["best_checkpoint_path"]).is_file())
            self.assertTrue(Path(phase_summary["last_checkpoint_path"]).is_file())

            run_summary = phase_summary["run_summary"]
            self.assertEqual(run_summary["completed_epochs"], 1)
            self.assertEqual(run_summary["last_epoch"]["train"]["skipped_batches"], 0)
            self.assertEqual(run_summary["last_epoch"]["val"]["batches"], 1)
            self.assertTrue(math.isfinite(float(run_summary["last_epoch"]["train"]["losses"]["total"]["mean"])))
            self.assertTrue(math.isfinite(float(run_summary["last_epoch"]["val"]["losses"]["total"]["mean"])))
            self.assertGreater(run_summary["last_epoch"]["val"]["counts"]["det_gt"], 0)
            self.assertGreater(run_summary["last_epoch"]["val"]["counts"]["lane_rows"], 0)
            self.assertTrue(Path(run_summary["history_paths"]["train_steps"]).is_file())
            self.assertTrue(Path(run_summary["history_paths"]["epochs"]).is_file())

            summary_payload = json.loads(Path(phase_summary["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["completed_epochs"], 1)
            self.assertEqual(summary_payload["last_epoch"]["train"]["skipped_batches"], 0)
            self.assertEqual(summary_payload["last_epoch"]["val"]["batches"], 1)

            dataset = PV26CanonicalDataset([dataset_root])
            for record in dataset.records:
                self.assertTrue(record.scene_path.is_file())
                self.assertTrue(record.image_path.is_file())
                if record.det_path is not None:
                    self.assertTrue(record.det_path.is_file())
            val_samples = select_prepared_samples(
                dataset,
                split="val",
                dataset_keys=DEFAULT_PREPARED_DATASET_KEYS,
            )
            batch = collate_pv26_samples(val_samples)

            phase = scenario.phases[0]
            train_config = scenario.train_defaults
            trainer = _build_phase_trainer(phase, train_config)
            trainer.load_model_weights(Path(phase_summary["best_checkpoint_path"]), map_location=train_config.device)
            evaluator = trainer.build_evaluator()

            eval_summary = evaluator.evaluate_batch(batch, include_predictions=True)
            predictions = evaluator.predict_batch(batch)

            self.assertEqual(len(eval_summary["predictions"]), len(predictions))
            self.assertEqual(len(predictions), len(DEFAULT_PREPARED_DATASET_KEYS))
            self.assertTrue(math.isfinite(float(eval_summary["losses"]["total"])))
            self.assertIn("metrics", eval_summary)

            detection_count = sum(len(item["detections"]) for item in predictions)
            lane_family_count = sum(
                len(item["lanes"]) + len(item["stop_lines"]) + len(item["crosswalks"])
                for item in predictions
            )
            self.assertGreater(detection_count, 0)
            self.assertGreater(lane_family_count, 0)
            _assert_nested_finite(self, eval_summary["losses"], field_name="eval_summary.losses")
            _assert_nested_finite(self, predictions, field_name="predictions")


if __name__ == "__main__":
    unittest.main()
