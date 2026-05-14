import tempfile
import unittest
from pathlib import Path

from tools.probe_pv26_stopline_fragment_projection_competition_readout import (
    ProjectionCompetitionVariant,
    _projection_competition_predictions,
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
    component_svd_length: float | None = None,
) -> dict[str, str]:
    import json

    length = ((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2) ** 0.5
    component_length = float(component_svd_length if component_svd_length is not None else length)
    return {
        "sample_id": sample_id,
        "gt_stop_line_count": str(gt_count),
        "gt_stop_line_points_json": json.dumps(gt_points or [[[0.0, 0.0], [60.0, 0.0]]]),
        "proposal_source": "max",
        "proposal_min_gap": str(min_gap),
        "proposal_rank": str(rank),
        "score": str(score),
        "length": str(length),
        "component_svd_length": str(component_length),
        "candidate_points_json": json.dumps(points),
    }


class StopLineFragmentProjectionCompetitionReadoutTest(unittest.TestCase):
    def test_length_ranked_single_can_compete_with_projection_split_group(self) -> None:
        variant = ProjectionCompetitionVariant(
            "projection_comp",
            min_gap=4.0,
            top_k=50,
            union_min_score=0.8,
            single_min_score=0.9,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
            projection_gap_px=100.0,
            rank_feature="length",
            max_predictions=1,
        )

        predictions, stats = _projection_competition_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.85, points=[[20.0, 0.0], [30.0, 0.0]]),
                _candidate_row(rank=3, score=0.95, points=[[0.0, 20.0], [60.0, 20.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["selected_kind"], "single")
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0]["length"], 60.0, places=5)

    def test_second_prediction_gate_applies_after_competition(self) -> None:
        variant = ProjectionCompetitionVariant(
            "projection_comp",
            min_gap=4.0,
            top_k=50,
            union_min_score=0.8,
            single_min_score=0.8,
            angle_threshold_deg=8.0,
            offset_threshold_px=4.0,
            min_cluster_count=2,
            projection_gap_px=100.0,
            rank_feature="length",
            max_predictions=2,
            second_min_fragment_count=5,
        )

        predictions, stats = _projection_competition_predictions(
            [
                _candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]),
                _candidate_row(rank=2, score=0.85, points=[[20.0, 0.0], [30.0, 0.0]]),
                _candidate_row(rank=3, score=0.95, points=[[0.0, 40.0], [60.0, 40.0]]),
            ],
            variant,
        )

        self.assertEqual(stats["proposal_count"], 2)
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0]["length"], 60.0, places=5)

    def test_run_replay_writes_projection_competition_summary(self) -> None:
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
                        "length",
                        "component_svd_length",
                        "candidate_points_json",
                    ],
                )
                writer.writeheader()
                writer.writerow(_candidate_row(rank=1, score=0.9, points=[[0.0, 0.0], [10.0, 0.0]]))
                writer.writerow(_candidate_row(rank=2, score=0.85, points=[[20.0, 0.0], [30.0, 0.0]]))
                writer.writerow(_candidate_row(rank=3, score=0.95, points=[[0.0, 20.0], [60.0, 20.0]]))
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
                    ProjectionCompetitionVariant(
                        "projection_comp",
                        min_gap=4.0,
                        top_k=50,
                        union_min_score=0.8,
                        single_min_score=0.9,
                        angle_threshold_deg=8.0,
                        offset_threshold_px=4.0,
                        min_cluster_count=2,
                        projection_gap_px=100.0,
                        rank_feature="length",
                        max_predictions=1,
                    ),
                ),
            )

            self.assertEqual(result["missing_gt_count"], 2)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "fragment_projection_competition_variants.csv").exists())
            self.assertEqual(result["variants"][0]["variant"], "projection_comp")


if __name__ == "__main__":
    unittest.main()
