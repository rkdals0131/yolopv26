from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.analyze_pv26_lane_repairability_model_replay import BaselineCounts, build_model_replay, run_audit


class LaneRepairabilityModelReplayTests(unittest.TestCase):
    def test_model_replay_uses_oof_scores_and_improves_f1(self) -> None:
        rows = [
            self._row(batch=0, sample=0, center=0.95, length=420.0, label=True),
            self._row(batch=0, sample=1, center=0.92, length=400.0, label=True),
            self._row(batch=1, sample=0, center=0.10, length=90.0, label=False),
            self._row(batch=1, sample=1, center=0.12, length=100.0, label=False),
            self._row(batch=2, sample=0, center=0.88, length=390.0, label=True),
            self._row(batch=2, sample=1, center=0.15, length=110.0, label=False),
        ]

        summary = build_model_replay(
            rows,
            baseline=BaselineCounts(tp=10, fp=6, fn=8),
            epochs=200,
            learning_rate=0.12,
            l2=0.0,
        )

        tight = summary["labels"]["repairable_le80_center050"]
        best = tight["best_replay_by_f1"]
        self.assertGreater(tight["oof_auc"], 0.80)
        self.assertGreater(best["lane_f1"], summary["baseline"]["lane_f1"])
        self.assertGreaterEqual(best["selected_repairable"], 2)
        self.assertEqual(tight["full_model"]["features"][0], "pred_point_count")
        self.assertEqual(len(tight["full_model"]["weights"]), len(tight["full_model"]["features"]))

    def test_run_audit_writes_summary_and_replay_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "lane_unmatched_prediction_repair_rows.csv"
            with rows_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._row(batch=0, sample=0, center=0.9, length=300.0, label=True)))
                writer.writeheader()
                for index in range(8):
                    writer.writerow(
                        self._row(
                            batch=index,
                            sample=index,
                            center=0.9 if index < 4 else 0.1,
                            length=300.0 if index < 4 else 80.0,
                            label=index < 4,
                        )
                    )

            out = root / "out"
            summary = run_audit(
                unmatched_rows=rows_path,
                output_dir=out,
                baseline=BaselineCounts(tp=20, fp=8, fn=10),
                epochs=80,
            )

            self.assertEqual(summary["row_count"], 8)
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "repairability_model_replay.csv").exists())
            self.assertTrue((out / "repairability_model_weights.csv").exists())
            self.assertTrue((out / "repairability_model_parameters.json").exists())

    @staticmethod
    def _row(*, batch: int, sample: int, center: float, length: float, label: bool) -> dict[str, str]:
        return {
            "batch_index": str(batch),
            "sample_index": str(sample),
            "repairable_le80_center050": str(bool(label)),
            "repairable_le120_any_center": str(bool(label)),
            "pred_point_count": "12",
            "pred_polyline_length": str(length),
            "pred_bbox_width": str(length / 4.0),
            "pred_bbox_height": "60.0",
            "pred_bbox_aspect": str(length / 240.0),
            "pred_track_pixels": str(length / 2.0),
            "pred_center_mask_mean": str(center),
            "pred_center_mask_q10": str(center),
            "pred_center_mask_active05": str(center),
            "pred_center_point_mean": str(center),
            "pred_center_point_q10": str(center),
            "pred_center_point_active05": str(center),
            "pred_center_point_low_run05": str(1.0 - center),
            "pred_support_point_mean": str(center),
            "pred_support_point_q10": str(center),
            "pred_support_mask_mean": str(center),
            "nearest_other_pred_distance": "100.0",
            "sample_pred_lane_count": "3",
            "sample_unmatched_pred_count": "1",
        }


if __name__ == "__main__":
    unittest.main()
