import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

from tools.probe_pv26_stopline_candidate_pool import (
    _candidate_feature_rows,
    _points_json,
    _projection_competition_variant_fields,
    _scenario_with_dataset_root,
    _write_candidate_features_csv,
)
from tools.pv26_train.config import (
    DatasetConfig,
    MetaTrainScenario,
    PhaseConfig,
    PreviewConfig,
    RunConfig,
    SelectionConfig,
    TrainDefaultsConfig,
)


class StopLineCandidatePoolManifestTest(unittest.TestCase):
    def test_feature_rows_include_projection_replay_manifest_fields(self) -> None:
        rows = _candidate_feature_rows(
            [
                {
                    "proposal_source": "max",
                    "proposal_min_gap": 4.0,
                    "proposal_rank": 1,
                    "proposal_row": 12,
                    "proposal_col": 34,
                    "decoded_center_row": 13,
                    "decoded_center_col": 35,
                    "points_xy": [[1.0, 2.0], [31.0, 42.0]],
                    "score": 0.95,
                    "length": 50.0,
                }
            ],
            batch_index=2,
            sample_index=3,
            meta={
                "sample_id": "sample-123",
                "dataset_key": "aihub_lane",
                "image_path": Path("/data/sample.png"),
            },
            gt_stop_lines=[
                {"points_xy": [[0.0, 0.0], [60.0, 0.0]]},
                {"points_xy": [[5.0, 5.0], [65.0, 5.0]]},
            ],
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["sample_id"], "sample-123")
        self.assertEqual(row["dataset_key"], "aihub_lane")
        self.assertEqual(row["image_path"], "/data/sample.png")
        self.assertEqual(row["gt_stop_line_count"], 2)
        self.assertEqual(row["proposal_min_gap"], 4.0)
        self.assertEqual(json.loads(row["candidate_points_json"]), [[1.0, 2.0], [31.0, 42.0]])
        self.assertEqual(
            json.loads(row["gt_stop_line_points_json"]),
            [[[0.0, 0.0], [60.0, 0.0]], [[5.0, 5.0], [65.0, 5.0]]],
        )
        self.assertEqual(row["component_svd_length"], 50.0)

    def test_points_json_tolerates_bad_input(self) -> None:
        self.assertEqual(json.loads(_points_json(object())), [])

    def test_dataset_root_override_replaces_scenario_dataset_only(self) -> None:
        scenario = MetaTrainScenario(
            dataset=DatasetConfig(root=Path("/old/root"), additional_roots=(Path("/extra/root"),)),
            run=RunConfig(),
            train_defaults=TrainDefaultsConfig(),
            selection=SelectionConfig(),
            preview=PreviewConfig(enabled=False),
            phases=(
                PhaseConfig(
                    name="phase",
                    stage="stage_4_lane_family_finetune",
                    min_epochs=1,
                    max_epochs=1,
                    patience=1,
                ),
            ),
        )

        updated = _scenario_with_dataset_root(scenario, "/new/root")

        self.assertEqual(updated.dataset.root, Path("/new/root").resolve())
        self.assertEqual(updated.dataset.additional_roots, ())
        self.assertEqual(updated.run, scenario.run)
        self.assertEqual(updated.phases, scenario.phases)

    def test_candidate_feature_csv_writes_manifest_header_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_features.csv"

            _write_candidate_features_csv(path, [])

            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertIn("sample_id", header)
        self.assertIn("gt_stop_line_points_json", header)
        self.assertIn("candidate_points_json", header)
        self.assertIn("proposal_min_gap", header)

    def test_projection_competition_fields_are_exported_with_stable_names(self) -> None:
        fields = _projection_competition_variant_fields(
            SimpleNamespace(
                min_gap=4.0,
                top_k=50,
                union_min_score=0.8,
                single_min_score=0.9,
                angle_threshold_deg=16.0,
                offset_threshold_px=48.0,
                min_cluster_count=2,
                projection_gap_px=320.0,
                rank_feature="length",
                max_predictions=2,
                second_min_score=0.0,
                second_min_fragment_count=5,
                second_min_length_ratio=0.0,
            )
        )

        self.assertEqual(fields["fragment_projection_comp_min_gap"], 4.0)
        self.assertEqual(fields["fragment_projection_comp_top_k"], 50)
        self.assertEqual(fields["fragment_projection_comp_rank_feature"], "length")
        self.assertEqual(fields["fragment_projection_comp_second_min_fragment_count"], 5)


if __name__ == "__main__":
    unittest.main()
