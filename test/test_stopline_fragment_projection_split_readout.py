import tempfile
import unittest
from pathlib import Path

from tools.probe_pv26_stopline_fragment_projection_split_readout import (
    ProjectionSplitVariant,
    _projection_split_predictions,
    run_replay,
)


def _candidate_row(
    *,
    sample_id: str = "sample-a",
    rank: int,
    score: float,
    points: list[list[float]],
    gt_count: int = 2,
    gt_points: list[list[list[float]]] | None = None,
    min_gap: float = 4.0,
) -> dict[str, str]:
    import json

    return {
        "sample_id": sample_id,
        "gt_stop_line_count": str(gt_count),
        "gt_stop_line_points_json": json.dumps(
            gt_points or [[[0.0, 0.0], [30.0, 0.0]], [[500.0, 0.0], [530.0, 0.0]]]
        ),
        "proposal_source": "max",
        "proposal_min_gap": str(min_gap),
        "proposal_rank": str(rank),
        "score": str(score),
        "candidate_points_json": json.dumps(points),
    }


class StopLineFragmentProjectionSplitReadoutTest(unittest.TestCase):
    def test_projection_gap_splits_over_merged_union_group(self) -> None:
        variant = ProjectionSplitVariant(
            "projection_split",
            min_gap=4.0,
            top_k=50,
            min_score=0.5,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
            projection_gap_px=100.0,
            max_predictions=2,
        )

        predictions, stats = _projection_split_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.8, points=[[20.0, 0.0], [30.0, 0.0]]),
                _candidate_row(rank=3, score=0.85, points=[[500.0, 0.0], [510.0, 0.0]]),
                _candidate_row(rank=4, score=0.75, points=[[520.0, 0.0], [530.0, 0.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["cluster_count"], 1)
        self.assertEqual(stats["split_group_count"], 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual([prediction["fragment_count"] for prediction in predictions], [2, 2])

    def test_projection_gap_keeps_nearby_fragments_together(self) -> None:
        variant = ProjectionSplitVariant(
            "projection_split",
            min_gap=4.0,
            top_k=50,
            min_score=0.5,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
            projection_gap_px=100.0,
            max_predictions=2,
        )

        predictions, stats = _projection_split_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.8, points=[[20.0, 0.0], [30.0, 0.0]]),
                _candidate_row(rank=3, score=0.85, points=[[80.0, 0.0], [90.0, 0.0]]),
                _candidate_row(rank=4, score=0.75, points=[[100.0, 0.0], [110.0, 0.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["cluster_count"], 1)
        self.assertEqual(stats["split_group_count"], 1)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["fragment_count"], 4)

    def test_run_replay_writes_projection_split_summary(self) -> None:
        import csv
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate_features.csv"
            summary_path = root / "summary.json"
            output_dir = root / "out"
            with candidate_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "gt_stop_line_count",
                        "gt_stop_line_points_json",
                        "proposal_source",
                        "proposal_min_gap",
                        "proposal_rank",
                        "score",
                        "candidate_points_json",
                    ],
                )
                writer.writeheader()
                writer.writerow(_candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]))
                writer.writerow(_candidate_row(rank=2, score=0.8, points=[[20.0, 0.0], [30.0, 0.0]]))
                writer.writerow(_candidate_row(rank=3, score=0.85, points=[[500.0, 0.0], [510.0, 0.0]]))
                writer.writerow(_candidate_row(rank=4, score=0.75, points=[[520.0, 0.0], [530.0, 0.0]]))
            summary_path.write_text(
                json.dumps(
                    {
                        "variants": [
                            {
                                "variant": "baseline",
                                "lane_f1": 0.5,
                                "stop_line_tp": 2,
                                "stop_line_fn": 0,
                                "stop_line_f1": 1.0,
                                "crosswalk_f1": 0.6,
                            }
                        ]
                    }
                )
            )

            result = run_replay(
                candidate_features_path=candidate_path,
                summary_path=summary_path,
                output_dir=output_dir,
                reference_variant="baseline",
                variants=(
                    ProjectionSplitVariant(
                        "projection_split",
                        min_gap=4.0,
                        top_k=50,
                        min_score=0.5,
                        angle_threshold_deg=8.0,
                        offset_threshold_px=4.0,
                        min_cluster_count=2,
                        projection_gap_px=100.0,
                        max_predictions=2,
                    ),
                ),
            )

            self.assertEqual(result["missing_gt_count"], 0)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(result["variants"][0]["stop_line_tp"], 2)
            self.assertEqual(result["variants"][0]["stop_line_fp"], 0)


if __name__ == "__main__":
    unittest.main()
