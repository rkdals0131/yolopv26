import tempfile
import unittest
from pathlib import Path

from tools.probe_pv26_stopline_fragment_union_readout import (
    FragmentUnionVariant,
    _evaluate_predictions,
    _union_predictions,
    run_replay,
)


def _candidate_row(
    *,
    sample_id: str = "sample-a",
    rank: int,
    score: float,
    points: list[list[float]],
    gt_count: int = 1,
    gt_points: list[list[list[float]]] | None = None,
    min_gap: float = 4.0,
) -> dict[str, str]:
    import json

    return {
        "sample_id": sample_id,
        "gt_stop_line_count": str(gt_count),
        "gt_stop_line_points_json": json.dumps(gt_points or [[[0.0, 0.0], [30.0, 0.0]]]),
        "proposal_source": "max",
        "proposal_min_gap": str(min_gap),
        "proposal_rank": str(rank),
        "score": str(score),
        "candidate_points_json": json.dumps(points),
    }


class StopLineFragmentUnionReadoutTest(unittest.TestCase):
    def test_collinear_fragments_merge_into_longer_segment(self) -> None:
        variant = FragmentUnionVariant(
            "union",
            min_gap=4.0,
            top_k=50,
            min_score=0.5,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
        )

        predictions, stats = _union_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.8, points=[[20.0, 0.0], [30.0, 0.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0]["length"], 30.0, places=5)
        self.assertEqual(predictions[0]["fragment_count"], 2)

    def test_offset_separated_fragments_do_not_merge(self) -> None:
        variant = FragmentUnionVariant(
            "union",
            min_gap=4.0,
            top_k=50,
            min_score=0.5,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
        )

        predictions, stats = _union_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.8, points=[[20.0, 12.0], [30.0, 12.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(len(predictions), 0)

    def test_missing_gt_count_is_added_to_fn(self) -> None:
        summary = _evaluate_predictions(
            {
                "sample-a": {
                    "gt_stop_lines": [
                        {"points_xy": [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]}
                    ]
                }
            },
            {"sample-a": []},
            missing_gt_count=3,
        )

        self.assertEqual(summary["stop_line_tp"], 0)
        self.assertEqual(summary["stop_line_fp"], 0)
        self.assertEqual(summary["stop_line_fn"], 4)
        self.assertEqual(summary["stop_line_support"], 4)

    def test_run_replay_writes_summary_with_missing_gt_support(self) -> None:
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
                writer.writerow(
                    _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]])
                )
                writer.writerow(
                    _candidate_row(rank=2, score=0.8, points=[[20.0, 0.0], [30.0, 0.0]])
                )
            summary_path.write_text(
                json.dumps(
                    {
                        "variants": [
                            {
                                "variant": "baseline",
                                "lane_f1": 0.5,
                                "stop_line_tp": 1,
                                "stop_line_fn": 2,
                                "stop_line_f1": 0.4,
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
                    FragmentUnionVariant(
                        "union",
                        min_gap=4.0,
                        top_k=50,
                        min_score=0.5,
                        angle_threshold_deg=8.0,
                        offset_threshold_px=4.0,
                        min_cluster_count=2,
                    ),
                ),
            )

            self.assertEqual(result["missing_gt_count"], 2)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(result["variants"][0]["stop_line_tp"], 1)
            self.assertEqual(result["variants"][0]["stop_line_fn"], 2)


if __name__ == "__main__":
    unittest.main()
