from __future__ import annotations

from pathlib import Path
import unittest

from model.engine.loss import build_loss_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
OD_BOOTSTRAP_README = REPO_ROOT / "tools" / "od_bootstrap" / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class DocsSyncTests(unittest.TestCase):
    def test_numbered_docs_set_is_the_active_docs_surface(self) -> None:
        numbered = sorted(path.name for path in DOCS_ROOT.glob("[0-9]*.md"))
        self.assertIn("0_PRD.md", numbered)
        self.assertIn("9_EXECUTION_STATUS.md", numbered)
        self.assertNotIn("yolopv26_main_code_cleanliness_checklists.md", numbered)
        self.assertNotIn("yolopv26_main_code_cleanliness_report.md", numbered)

    def test_no_absolute_repo_links_remain_in_active_docs(self) -> None:
        targets = [
            REPO_ROOT / "README.md",
            OD_BOOTSTRAP_README,
            *sorted(DOCS_ROOT.glob("*.md")),
        ]
        for path in targets:
            content = _read(path)
            self.assertNotIn("/home/user1", content, msg=str(path))
            self.assertNotIn("/YOLOpv26/", content, msg=str(path))

    def test_readme_matches_current_entrypoints_and_package_layout(self) -> None:
        readme = _read(REPO_ROOT / "README.md")
        self.assertIn("python3 tools/check_env.py", readme)
        self.assertIn("python3 tools/run_pv26_train.py --preset default", readme)
        self.assertIn("python3 tools/run_pv26_train.py --derive-run", readme)
        self.assertIn("tools/check_env/", readme)
        self.assertIn("tools/pv26_train/", readme)
        self.assertIn("tools/od_bootstrap/source/aihub/", readme)
        self.assertIn("tools/od_bootstrap/source/shared/", readme)
        self.assertIn("tools/od_bootstrap/teacher/runtime/", readme)
        self.assertIn("tools/pv26_train/cli.py", readme)
        self.assertIn("config/pv26_train_hyperparameters.yaml", readme)

    def test_training_docs_track_resume_and_stage3_direct_entrypoints(self) -> None:
        training_doc = _read(DOCS_ROOT / "6_TRAINING_AND_EVALUATION.md")
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")

        self.assertIn("tools/run_pv26_train.py --resume-run", training_doc)
        self.assertIn("tools/run_pv26_train.py --derive-run", training_doc)
        self.assertIn("tools/run_pv26_train.py --preset default --stage3-vram-stress", training_doc)
        self.assertIn("tools/run_pv26_train.py --resume-run", execution_doc)
        self.assertIn("tools/run_pv26_train.py --derive-run", execution_doc)
        self.assertIn("tools/run_pv26_train.py --preset default --stage3-vram-stress", execution_doc)

    def test_od_bootstrap_readme_tracks_current_package_layout_and_teacher_defaults(self) -> None:
        readme = _read(OD_BOOTSTRAP_README)
        self.assertIn("python -m tools.od_bootstrap prepare-sources", readme)
        self.assertIn("python -m tools.od_bootstrap build-teacher-datasets", readme)
        self.assertIn("python -m tools.od_bootstrap train --teacher mobility", readme)
        self.assertIn("mobility/signal은 `yolo26s.pt`, obstacle은 `yolo26m.pt`", readme)
        self.assertIn("source/aihub/", readme)
        self.assertIn("source/shared/", readme)
        self.assertIn("teacher/runtime/", readme)

    def test_implementation_and_execution_docs_track_package_native_tooling(self) -> None:
        implementation_plan = _read(DOCS_ROOT / "7_IMPLEMENTATION_PLAN.md")
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")
        architecture_doc = _read(DOCS_ROOT / "2_SYSTEM_ARCHITECTURE.md")

        self.assertIn("test/test_run_pv26_train.py", implementation_plan)
        self.assertIn("test/test_portability_runtime.py", implementation_plan)
        self.assertIn("test/test_docs_sync.py", implementation_plan)
        self.assertIn("tools/pv26_train/scenario.py", implementation_plan)
        self.assertIn("tools/pv26_train/runtime.py", implementation_plan)
        self.assertIn("tools/check_env/launch.py", implementation_plan)
        self.assertIn("tools/od_bootstrap/teacher/runtime/trainer.py", implementation_plan)
        self.assertIn("test/test_run_pv26_train.py", execution_doc)
        self.assertIn("test/test_portability_runtime.py", execution_doc)
        self.assertIn("test/test_docs_sync.py", execution_doc)
        self.assertIn("tools/check_env/launch.py", execution_doc)
        self.assertIn("tools/od_bootstrap/teacher/runtime/trainer.py", execution_doc)
        self.assertIn("tools/od_bootstrap/source/shared/io.py", execution_doc)
        self.assertIn("tools/od_bootstrap/source/aihub/pipeline.py", execution_doc)

        self.assertIn("stable thin facade", implementation_plan)
        self.assertIn("shared progress status helper", implementation_plan)
        self.assertIn("tools/check_env/", architecture_doc)
        self.assertIn("tools/pv26_train/", architecture_doc)
        self.assertIn("aihub/", architecture_doc)
        self.assertIn("shared/", architecture_doc)
        self.assertIn("runtime/", architecture_doc)

    def test_execution_status_tracks_current_runtime_and_policy_boundaries(self) -> None:
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")
        self.assertIn("rank-6/7 runtime cleanup도 마감", execution_doc)
        self.assertIn("join_status_segments()", execution_doc)
        self.assertIn("build_progress_status()", execution_doc)
        self.assertIn("framework-specific renderer", execution_doc)
        self.assertIn("local helper implementation residue는 `now_iso` 2곳, `timestamp_token` 1곳, `write_json` 2곳, `append_jsonl` 1곳", execution_doc)
        self.assertIn("`common/io.py`, `source/shared/io.py`, `source/aihub/pipeline.py`, `build/teacher_dataset.py`, `build/final_dataset.py`, `teacher/runtime/artifacts.py`, `teacher/data_yaml.py`", execution_doc)
        self.assertIn("TeacherJobManifestPayload", execution_doc)
        self.assertIn("SourcePrepManifest", execution_doc)
        self.assertIn("FinalDatasetPublishMarker", execution_doc)
        self.assertIn("model/engine/det_geometry.py", execution_doc)
        self.assertIn("model/engine/train_summary.py", execution_doc)
        self.assertIn("model/engine/trainer_progress.py", execution_doc)

    def test_system_architecture_tracks_runtime_not_contract_gap(self) -> None:
        architecture_doc = _read(DOCS_ROOT / "2_SYSTEM_ARCHITECTURE.md")
        self.assertIn("tools.od_bootstrap.source.aihub / bdd100k", architecture_doc)
        self.assertIn("model/data", architecture_doc)
        self.assertIn("model/net", architecture_doc)
        self.assertIn("model/engine", architecture_doc)
        self.assertNotIn("model/preprocess/", architecture_doc)
        self.assertNotIn("model/encoding/", architecture_doc)
        self.assertNotIn("model/loading/", architecture_doc)
        self.assertNotIn("model/training/", architecture_doc)
        self.assertNotIn("model/viz/", architecture_doc)

    def test_sample_contract_doc_exists_and_is_referenced(self) -> None:
        sample_doc = DOCS_ROOT / "4A_SAMPLE_AND_TRANSFORM_CONTRACT.md"
        self.assertTrue(sample_doc.exists())
        content = _read(sample_doc)
        self.assertIn('"image"', content)
        self.assertIn('"det_targets"', content)
        self.assertIn('"tl_attr_targets"', content)
        self.assertIn('"lane_targets"', content)
        self.assertIn('"source_mask"', content)
        self.assertIn('"valid_mask"', content)
        self.assertIn('"meta"', content)

        prd = (DOCS_ROOT / "0_PRD.md").read_text(encoding="utf-8")
        self.assertIn("[4A_SAMPLE_AND_TRANSFORM_CONTRACT.md]", prd)

    def test_contract_terminology_is_locked(self) -> None:
        sample_doc = _read(DOCS_ROOT / "4A_SAMPLE_AND_TRANSFORM_CONTRACT.md")
        self.assertIn("`N_gt_det`", sample_doc)
        self.assertIn("`Q_det`", sample_doc)
        self.assertIn("non_car_traffic_light", sample_doc)
        self.assertIn("bilinear", sample_doc)
        self.assertIn("114", sample_doc)

        loss_doc = _read(DOCS_ROOT / "5_TARGETS_AND_LOSS.md")
        self.assertIn("## raw model output contract", loss_doc)
        self.assertIn("## export / ROS prediction bundle", loss_doc)

    def test_query_counts_are_synced_between_docs_and_spec(self) -> None:
        spec = build_loss_spec()
        architecture_doc = _read(DOCS_ROOT / "4_MODEL_ARCHITECTURE.md")
        self.assertIn(f"fixed query count `{spec['heads']['lane']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['stop_line']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['crosswalk']['query_count']}`", architecture_doc)
        self.assertNotIn("query count 최종값", architecture_doc)

    def test_pv26_docs_track_backbone_and_stage4_direction(self) -> None:
        architecture_doc = _read(DOCS_ROOT / "4_MODEL_ARCHITECTURE.md")
        loss_doc = _read(DOCS_ROOT / "5_TARGETS_AND_LOSS.md")
        training_doc = _read(DOCS_ROOT / "6_TRAINING_AND_EVALUATION.md")

        self.assertIn("`yolo26s.pt`", architecture_doc)
        self.assertIn("`yolo26n.pt`", architecture_doc)
        self.assertIn("stage_4_lane_family_finetune", loss_doc)
        self.assertIn("selection_metrics.phase_objective", loss_doc)
        self.assertIn("stage 4", training_doc)
        self.assertIn("phase-specific selection", training_doc)
        self.assertIn("selection_metrics.phase_objective", training_doc)
        self.assertIn("min_delta_abs", training_doc)

    def test_standardization_doc_tracks_bootstrap_output_roots(self) -> None:
        standardization_doc = _read(DOCS_ROOT / "3_DATA_AND_STANDARDIZATION.md")
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized", standardization_doc)
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_aihub_standardized", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_bdd100k_standardized", standardization_doc)


if __name__ == "__main__":
    unittest.main()
