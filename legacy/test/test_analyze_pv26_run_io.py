from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_pv26_run import build_phase_infos, build_phase_overview_rows, iter_jsonl, read_json


class AnalyzePV26RunIOTests(unittest.TestCase):
    def test_read_json_rejects_non_object_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_path = Path(temp_dir) / "summary.json"
            payload_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "JSON root must be an object"):
                read_json(payload_path)

    def test_iter_jsonl_skips_blank_lines_and_yields_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_path = Path(temp_dir) / "epochs.jsonl"
            payload_path.write_text('\n{"epoch": 1}\n\n{"epoch": 2}\n', encoding="utf-8")

            self.assertEqual(list(iter_jsonl(payload_path)), [{"epoch": 1}, {"epoch": 2}])

    def test_phase_overview_uses_trainer_best_epoch_for_min_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            phase_dir = run_dir / "phase_1"
            history_dir = phase_dir / "history"
            history_dir.mkdir(parents=True)
            (phase_dir / "summary.json").write_text(
                json.dumps({"stage": "stage_1_frozen_trunk_warmup"}, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            (history_dir / "train_steps.jsonl").write_text(
                json.dumps(
                    {
                        "batch_size": 2,
                        "trainable": {
                            "freeze_policy": "none",
                            "head_training_policy": "all",
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            epoch_rows = [
                {
                    "epoch": 1,
                    "selection_metrics": {"phase_objective": 0.1},
                    "train": {"losses": {"total": {"mean": 5.0}}, "duration_sec": 1.0, "batches": 1},
                    "val": {
                        "losses": {"total": {"mean": 4.0}},
                        "metrics": {},
                        "duration_sec": 1.0,
                        "batches": 1,
                    },
                },
                {
                    "epoch": 2,
                    "selection_metrics": {"phase_objective": 0.9},
                    "train": {"losses": {"total": {"mean": 7.0}}, "duration_sec": 1.0, "batches": 1},
                    "val": {
                        "losses": {"total": {"mean": 9.0}},
                        "metrics": {},
                        "duration_sec": 1.0,
                        "batches": 1,
                    },
                },
            ]
            (history_dir / "epochs.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in epoch_rows) + "\n",
                encoding="utf-8",
            )
            top_summary = {
                "phases": [
                    {
                        "name": "head_warmup",
                        "stage": "stage_1_frozen_trunk_warmup",
                        "run_dir": str(phase_dir),
                        "status": "completed",
                        "best_epoch": 1,
                        "best_metric_value": 4.0,
                        "selection": {"metric_path": "val.losses.total.mean", "mode": "min"},
                    }
                ]
            }

            rows = build_phase_overview_rows(run_dir, build_phase_infos(run_dir, top_summary))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["best_epoch"], 1)
            self.assertEqual(rows[0]["best_metric_value"], 4.0)
            self.assertEqual(rows[0]["phase_objective_best_epoch"], 1)
            self.assertEqual(rows[0]["phase_objective_best"], 0.1)

    def test_phase_overview_preserves_weights_only_handoff_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            phase_dir = run_dir / "phase_2"
            history_dir = phase_dir / "history"
            history_dir.mkdir(parents=True)
            (phase_dir / "summary.json").write_text(
                json.dumps({"stage": "stage_4_lane_family_finetune"}, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            (history_dir / "train_steps.jsonl").write_text(
                json.dumps(
                    {
                        "batch_size": 2,
                        "trainable": {
                            "freeze_policy": "heads_only",
                            "head_training_policy": "lane_family",
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (history_dir / "epochs.jsonl").write_text(
                json.dumps(
                    {
                        "epoch": 1,
                        "selection_metrics": {"phase_objective": 0.42},
                        "train": {"losses": {"total": {"mean": 3.0}}, "duration_sec": 1.0, "batches": 1},
                        "val": {
                            "losses": {"total": {"mean": 2.0}},
                            "metrics": {},
                            "duration_sec": 1.0,
                            "batches": 1,
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            top_summary = {
                "phases": [
                    {
                        "name": "lane_handoff",
                        "stage": "stage_4_lane_family_finetune",
                        "run_dir": str(phase_dir),
                        "status": "completed",
                        "best_epoch": 1,
                        "best_metric_value": 0.42,
                        "selection": {"metric_path": "selection_metrics.phase_objective", "mode": "max"},
                        "weights_only_handoff": {
                            "checkpoint_path": "/runs/source/phase_1/checkpoints/best.pt",
                            "load_policy": "shape_aware_partial",
                            "adapter_load_report": {
                                "loaded_count": 12,
                                "skipped_shape_keys": ["adapter.stale"],
                            },
                            "heads_load_report": {
                                "loaded_count": 7,
                                "skipped_shape_keys": ["heads.lane.location_head.weight"],
                            },
                            "checkpoint_metadata": {
                                "architecture_generation": "pv26-road-marking-v3",
                                "spec_version": "pv26-loss-spec-v1",
                            },
                        },
                    }
                ]
            }

            rows = build_phase_overview_rows(run_dir, build_phase_infos(run_dir, top_summary))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["weights_only_handoff_checkpoint_path"], "/runs/source/phase_1/checkpoints/best.pt")
            self.assertEqual(rows[0]["weights_only_handoff_load_policy"], "shape_aware_partial")
            self.assertEqual(rows[0]["weights_only_handoff_adapter_loaded_count"], 12)
            self.assertEqual(rows[0]["weights_only_handoff_heads_loaded_count"], 7)
            self.assertEqual(rows[0]["weights_only_handoff_adapter_skipped_shape_count"], 1)
            self.assertEqual(rows[0]["weights_only_handoff_heads_skipped_shape_count"], 1)
            self.assertEqual(
                rows[0]["weights_only_handoff_checkpoint_metadata.architecture_generation"],
                "pv26-road-marking-v3",
            )
            self.assertEqual(
                rows[0]["weights_only_handoff_checkpoint_metadata.spec_version"],
                "pv26-loss-spec-v1",
            )


if __name__ == "__main__":
    unittest.main()
