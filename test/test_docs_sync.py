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

    def test_readme_matches_current_pv26_training_entrypoint(self) -> None:
        readme = _read(REPO_ROOT / "README.md")
        self.assertIn("python3 tools/check_env.py", readme)
        self.assertIn("## 가장 먼저 실행", readme)
        self.assertIn("interactive launcher", readme)
        self.assertIn("이게 기본 진입점이다.", readme)
        self.assertIn("python3 tools/check_env.py --strict --check-yolo-runtime", readme)
        self.assertIn("`H`를 누르면", readme)
        self.assertIn("docs/3A_RAW_DATASET_LAYOUTS.md", readme)
        self.assertIn("config/user_paths.yaml", readme)
        self.assertIn("config/od_bootstrap_hyperparameters.yaml", readme)
        self.assertIn("config/pv26_train_hyperparameters.yaml", readme)
        self.assertIn("tools/od_bootstrap/presets.py", readme)
        self.assertIn("tools/run_pv26_train.py", readme)
        self.assertIn("USER CONFIG", readme)
        self.assertIn("HYPERPARAMETERS", readme)
        self.assertIn("PHASE HYPERPARAMETERS", readme)
        self.assertIn("tools/run_pv26_train.py --preset default", readme)
        self.assertIn("tools/run_pv26_train.py --resume-run", readme)
        self.assertIn("tools/run_pv26_train.py --preset default --stage3-vram-stress", readme)
        self.assertIn("python -m tools.od_bootstrap prepare-sources", readme)
        self.assertIn("python -m tools.od_bootstrap build-teacher-datasets", readme)
        self.assertIn("python -m tools.od_bootstrap build-exhaustive-od", readme)
        self.assertIn("python -m tools.od_bootstrap build-final-dataset", readme)
        self.assertIn("seg_dataset/pv26_exhaustive_od_lane_dataset", readme)
        self.assertIn("runs/pv26_exhaustive_od_lane_train", readme)
        self.assertIn("`2번`부터 `8번`까지", readme)
        self.assertNotIn("runs/pv26_meta_train/", readme)
        self.assertNotIn("tools/run_aihub_standardize.py", readme)
        self.assertNotIn("tools/run_bdd100k_standardize.py", readme)
        self.assertNotIn("tools/od_bootstrap/config/", readme)
        self.assertNotIn("[config/](config/)", readme)

    def test_training_docs_track_resume_and_stage3_direct_entrypoints(self) -> None:
        training_doc = _read(DOCS_ROOT / "6_TRAINING_AND_EVALUATION.md")
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")

        self.assertIn("tools/run_pv26_train.py --resume-run", training_doc)
        self.assertIn("tools/run_pv26_train.py --preset default --stage3-vram-stress", training_doc)
        self.assertIn("tools/run_pv26_train.py --resume-run", execution_doc)
        self.assertIn("tools/run_pv26_train.py --preset default --stage3-vram-stress", execution_doc)

    def test_od_bootstrap_readme_tracks_current_teacher_model_defaults(self) -> None:
        readme = _read(OD_BOOTSTRAP_README)
        self.assertIn("python -m tools.od_bootstrap prepare-sources", readme)
        self.assertIn("python -m tools.od_bootstrap build-teacher-datasets", readme)
        self.assertIn("python -m tools.od_bootstrap train --teacher mobility", readme)
        self.assertIn("python -m tools.od_bootstrap train --teacher signal", readme)
        self.assertIn("python -m tools.od_bootstrap train --teacher obstacle", readme)
        self.assertIn("python -m tools.od_bootstrap build-exhaustive-od", readme)
        self.assertIn("python -m tools.od_bootstrap build-final-dataset", readme)
        self.assertIn("mobility/signal은 `yolo26s.pt`, obstacle은 `yolo26m.pt`", readme)
        self.assertIn("config/user_paths.yaml", readme)
        self.assertIn("config/od_bootstrap_hyperparameters.yaml", readme)

    def test_priority_2b_docs_track_run_pv26_train_split_boundary(self) -> None:
        implementation_plan = _read(DOCS_ROOT / "7_IMPLEMENTATION_PLAN.md")
        cleanliness_checklist = _read(DOCS_ROOT / "yolopv26_main_code_cleanliness_checklists.md")
        cleanliness_report = _read(DOCS_ROOT / "yolopv26_main_code_cleanliness_report.md")

        for content in (implementation_plan, cleanliness_checklist, cleanliness_report):
            self.assertIn("test/test_run_pv26_train.py", content)
            self.assertIn("test/test_portability_runtime.py", content)
            self.assertIn("test/test_docs_sync.py", content)

        self.assertIn("priority-2b extraction review boundary", implementation_plan)
        self.assertIn("stable thin facade", implementation_plan)

        self.assertIn("tools/pv26_train_scenario.py", cleanliness_checklist)
        self.assertIn("tools/pv26_train_runtime.py", cleanliness_checklist)
        self.assertIn("tools/pv26_train_stress.py", cleanliness_checklist)

        self.assertIn("run_pv26_train.py         # stable thin facade / CLI entrypoint", cleanliness_report)
        self.assertIn("pv26_train_scenario.py", cleanliness_report)
        self.assertIn("pv26_train_runtime.py", cleanliness_report)
        self.assertIn("pv26_train_stress.py", cleanliness_report)

    def test_rank3_cleanliness_docs_track_exact_helper_residue_and_link_policy_boundaries(self) -> None:
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")
        cleanliness_checklist = _read(DOCS_ROOT / "yolopv26_main_code_cleanliness_checklists.md")
        cleanliness_report = _read(DOCS_ROOT / "yolopv26_main_code_cleanliness_report.md")

        self.assertIn(
            "local helper implementation residue는 `now_iso` 2곳, `timestamp_token` 1곳, `write_json` 2곳, `append_jsonl` 1곳",
            execution_doc,
        )
        self.assertIn(
            "`common/io.py`, `source/shared_io.py`, `source/aihub.py`, `build/teacher_dataset.py`, `build/final_dataset.py`, `teacher/runtime_artifacts.py`, `teacher/data_yaml.py`",
            execution_doc,
        )

        self.assertIn("`tools/od_bootstrap/source/raw_common.py`의 UTC timestamp contract", cleanliness_checklist)
        self.assertIn("`tools/od_bootstrap/teacher/calibrate.py`의 `default=str` JSON 직렬화 call-site", cleanliness_checklist)
        self.assertIn("`tools/od_bootstrap/build/final_dataset.py`의 overwrite 금지 publish semantics", cleanliness_checklist)
        self.assertIn("direct re-export surface", cleanliness_checklist)

        self.assertIn("`source/raw_common.py`의 UTC timestamp contract", cleanliness_report)
        self.assertIn("`teacher/calibrate.py`의 `default=str` JSON 직렬화 call-site", cleanliness_report)
        self.assertIn("`build/final_dataset.py`의 overwrite 금지 publish semantics", cleanliness_report)
        for fragment in (
            "`common/io.py`",
            "`source/shared_io.py`",
            "`source/aihub.py`",
            "`build/teacher_dataset.py`",
            "`build/final_dataset.py`",
            "`teacher/runtime_artifacts.py`",
            "`teacher/data_yaml.py`",
        ):
            self.assertIn(fragment, cleanliness_report)

    def test_od_bootstrap_readme_mentions_current_debug_vis_and_review_tooling(self) -> None:
        readme = _read(OD_BOOTSTRAP_README)
        self.assertIn("python -m tools.od_bootstrap", readme)
        self.assertIn("debug_vis.py", readme)
        self.assertIn("sample_manifest.py", readme)
        self.assertIn("review.py", readme)
        self.assertIn("checkpoint_audit.py", readme)

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
        self.assertIn("stage 4", training_doc)
        self.assertIn("phase-specific selection", training_doc)

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

    def test_standardization_doc_tracks_bootstrap_output_roots(self) -> None:
        standardization_doc = _read(DOCS_ROOT / "3_DATA_AND_STANDARDIZATION.md")
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized", standardization_doc)
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_aihub_standardized", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_bdd100k_standardized", standardization_doc)


if __name__ == "__main__":
    unittest.main()
