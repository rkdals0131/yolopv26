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
        self.assertIn("00A_CURRENT_STATUS.md", numbered)
        self.assertIn("00B_STATUS_HISTORY.md", numbered)
        self.assertIn("00C_NEXT_GATES.md", numbered)
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
        self.assertIn("tools/modal/", readme)
        self.assertIn("tools/pv26_train/cli.py", readme)
        self.assertIn("config/pv26_train_hyperparameters.yaml", readme)
        self.assertIn("docs/10_CODEBASE_REFACTORING_MAP_AND_PLAN.md", readme)

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
        implementation_plan = _read(DOCS_ROOT / "legacy" / "7_IMPLEMENTATION_PLAN.md")
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
        self.assertIn("tools/modal/", architecture_doc)
        self.assertIn("aihub/", architecture_doc)
        self.assertIn("shared/", architecture_doc)
        self.assertIn("runtime/", architecture_doc)

    def test_modal_runbook_tracks_tools_package_location(self) -> None:
        runbook = _read(DOCS_ROOT / "17_MODAL_A100_TRAINING_RUNBOOK.md")
        refactor_map = _read(DOCS_ROOT / "10_CODEBASE_REFACTORING_MAP_AND_PLAN.md")

        self.assertIn("python -m tools.modal.local_preflight", runbook)
        self.assertIn("python -m tools.modal.prepare_dataset_volume", runbook)
        self.assertIn("modal run tools/modal/check.py", runbook)
        self.assertIn("modal run --detach tools/modal/train.py", runbook)
        self.assertIn("tools/modal/constants.py", runbook)
        self.assertIn("### `tools/modal/`", refactor_map)
        self.assertIn("Modal SDK shadow drift", refactor_map)
        self.assertIn("repo_root/tools", refactor_map)
        retired_spellings = [
            "tools/" + "tools/modal",
            "python " + "modal/",
            "modal run " + "modal/",
        ]
        for spelling in retired_spellings:
            self.assertNotIn(spelling, runbook)

    def test_execution_status_tracks_current_runtime_and_policy_boundaries(self) -> None:
        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")
        self.assertIn("repo-wide codebase refactor map", execution_doc)
        self.assertIn("E2E connection audit", execution_doc)
        self.assertIn("docs/10_CODEBASE_REFACTORING_MAP_AND_PLAN.md", execution_doc)
        self.assertIn("baseline drift 4건", execution_doc)
        self.assertIn("model.engine.batch.merge_raw_batches", execution_doc)
        self.assertIn("model.engine._trainer_epochs._merge_raw_batches", execution_doc)
        self.assertIn("obsolete `tools/probe_pv26_*`", execution_doc)
        self.assertIn("probe-owned helper tests", execution_doc)
        self.assertIn("tools/modal/", execution_doc)
        self.assertIn("raw source standardization부터 final dataset", execution_doc)
        self.assertIn("docs/legacy/", execution_doc)
        self.assertIn("docs/00B_STATUS_HISTORY.md", execution_doc)
        self.assertIn("tools/analyze_pv26_run.py", execution_doc)
        self.assertIn("common.io.write_json(overwrite=False)", execution_doc)
        self.assertIn("exhaustive OD materialization scene/manifest/summary JSON은 `common.io.write_json(...)`", execution_doc)
        self.assertIn("teacher/exhaustive/final label/class-map/data-yaml text는 `common.io.write_text(...)`", execution_doc)
        self.assertIn("common.io.write_json(default=str)", execution_doc)
        self.assertIn("calibration prediction JSONL은 `common.io.write_jsonl(...)`", execution_doc)
        self.assertIn("남은 의도적 helper 차이는 `now_iso` 2곳", execution_doc)
        self.assertIn("teacher summary/report JSON 직렬화는 direct `json.dumps(..., default=str)` call-site가 아니라 common helper 호출 인자로 고정한다", execution_doc)
        self.assertNotIn("default=str` JSON 직렬화 call-site에 남아 있다", execution_doc)
        self.assertIn("common/io.py", execution_doc)
        self.assertIn("tools/od_bootstrap/source/shared/io.py", execution_doc)
        self.assertIn("tools/od_bootstrap/source/aihub/pipeline.py", execution_doc)
        self.assertIn("tools/od_bootstrap/build/teacher_dataset.py", execution_doc)
        self.assertIn("tools/od_bootstrap/build/final_dataset.py", execution_doc)
        self.assertIn("tools/od_bootstrap/teacher/runtime/artifacts.py", execution_doc)
        self.assertIn("tools/od_bootstrap/teacher/data_yaml.py", execution_doc)
        self.assertIn("TeacherJobManifestPayload", execution_doc)
        self.assertIn("SourcePrepManifest", execution_doc)
        self.assertIn("FinalDatasetPublishMarker", execution_doc)
        self.assertIn("model/engine/det_geometry.py", execution_doc)
        self.assertIn("model/engine/train_summary.py", execution_doc)
        self.assertIn("model/engine/trainer_progress.py", execution_doc)
        self.assertIn("roadmark-source P2/P3/P4/P5 pyramid", execution_doc)
        self.assertIn("P2/P3/P4/P5 = 128/128/256/512", execution_doc)
        self.assertNotIn("P3/P4/P5 = 64/128/256 channels", execution_doc)
        self.assertNotIn("detect-source pyramid directly", execution_doc)

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

    def test_refactoring_map_tracks_boundaries_and_probe_taxonomy(self) -> None:
        refactor_map = _read(DOCS_ROOT / "10_CODEBASE_REFACTORING_MAP_AND_PLAN.md")

        self.assertIn("active refactor map", refactor_map)
        self.assertIn("PV26Heads", refactor_map)
        self.assertIn("P2/P3/P4/P5", refactor_map)
        self.assertIn("model.engine.__all__", refactor_map)
        self.assertIn("public `model.engine.spec`", refactor_map)
        self.assertIn("tools/check_env.py", refactor_map)
        self.assertIn("tools/run_pv26_train.py", refactor_map)
        self.assertIn("python -m tools.od_bootstrap", refactor_map)
        self.assertIn("tools/model_export/", refactor_map)
        self.assertIn("tools/analyze_pv26_run.py", refactor_map)
        self.assertIn("Durable analysis drift", refactor_map)
        self.assertIn("trainer-selected `best_epoch`", refactor_map)
        self.assertIn("retired experimental probes", refactor_map)
        self.assertIn("stable runtime이 아니고", refactor_map)
        self.assertIn("docs/legacy/", refactor_map)
        self.assertIn("docs/00B_STATUS_HISTORY.md", refactor_map)
        self.assertIn("retired probe imports must remain absent from tests", refactor_map)
        self.assertIn("model.engine.batch.merge_raw_batches", refactor_map)
        self.assertIn("Known Baseline Drift Before Refactor", refactor_map)
        self.assertNotIn("Do not delete " + "probe files", refactor_map)
        self.assertNotIn("probe mesh " + "reduction", refactor_map)

        execution_doc = _read(DOCS_ROOT / "9_EXECUTION_STATUS.md")
        self.assertIn("repo-wide codebase refactor map", execution_doc)
        self.assertIn("model.engine.batch.merge_raw_batches", execution_doc)

    def test_refactoring_map_tracks_e2e_pipeline_audit(self) -> None:
        refactor_map = _read(DOCS_ROOT / "10_CODEBASE_REFACTORING_MAP_AND_PLAN.md")

        self.assertIn("## E2E Pipeline Connection Audit", refactor_map)
        self.assertIn("Raw source -> canonical source", refactor_map)
        self.assertIn("meta/source_prep_manifest.json", refactor_map)
        self.assertIn("meta/bootstrap_image_list.jsonl", refactor_map)
        self.assertIn("downstream `sample_uid`, `dataset_key`, `split`, `dataset_root`, `source_name`, and optional `det_path`", refactor_map)
        self.assertIn("Worker task-mask policy", refactor_map)
        self.assertIn("AIHUB lane must emit lane/stop/crosswalk scenes with no `labels_det`", refactor_map)
        self.assertIn("Canonical source -> teacher/exhaustive/final dataset", refactor_map)
        self.assertIn("Final dataset -> loader/target encoder", refactor_map)
        self.assertIn("Encoded batch -> model/loss/postprocess/trainer", refactor_map)
        self.assertIn("Regression Risk Register", refactor_map)
        self.assertIn("Lane-only semantic drift", refactor_map)
        self.assertIn("lane_supervised_count` and `lane_valid`/query objectness", refactor_map)
        self.assertIn("semantic-less lane geometry rows as query supervision", refactor_map)
        self.assertIn("model.data.PV26CanonicalDataset", refactor_map)
        self.assertIn("model.net.PV26Heads", refactor_map)
        self.assertIn("model.engine.loss.PV26MultiTaskLoss", refactor_map)
        self.assertIn("loader detector targets are pinned to `labels_det` YOLO rows", refactor_map)
        self.assertIn("`labels_scene.detections` remains descriptive/provenance data", refactor_map)
        self.assertIn("Teacher detector labels also come from `labels_det`", refactor_map)
        self.assertIn("`source_label_path`", refactor_map)
        self.assertIn("source-derived `sample_uid`", refactor_map)
        self.assertIn("source `labels_scene`", refactor_map)
        self.assertIn("optional source `labels_det`", refactor_map)
        self.assertIn("Raw source `detections[].id` must already match row order", refactor_map)
        self.assertIn("Exhaustive row-order drift", refactor_map)
        self.assertIn("row/class-id order", refactor_map)
        self.assertIn("source scene/image/optional det paths", refactor_map)
        self.assertIn("published scene/image/det paths", refactor_map)
        self.assertIn("can be traced back to exhaustive or lane source files", refactor_map)
        self.assertIn("Loader manifest guard", refactor_map)
        self.assertIn("Final-manifest loader drift", refactor_map)
        self.assertIn("source trace metadata", refactor_map)
        self.assertIn("required det paths for detector-supervised sources", refactor_map)
        self.assertIn("Final publish det drift", refactor_map)
        self.assertIn("Scene-det substitution drift", refactor_map)
        self.assertIn("Source det row-order drift", refactor_map)
        self.assertIn("Source resume task drift", refactor_map)
        self.assertIn("Canonical det-label task drift", refactor_map)
        self.assertIn("explicit `tasks.has_det` no longer matches `labels_det` presence", refactor_map)
        self.assertIn("stale canonical scene files whose `tasks`", refactor_map)
        self.assertIn("labels_scene.detections[].id", refactor_map)
        self.assertIn("collate_pv26_encoded_eval_batch()", refactor_map)
        self.assertIn("encoded query/vector shapes against `PV26Heads.describe()`", refactor_map)
        self.assertIn("real `PV26Heads` raw forward shape", refactor_map)
        self.assertIn("Encoded eval collation drift", refactor_map)
        self.assertIn("Detector query count must match `det_feature_shapes`", refactor_map)
        self.assertIn("actual `PV26Heads` raw output through `PV26MultiTaskLoss` and `postprocess_pv26_batch`", refactor_map)
        self.assertIn("Detector feature metadata drift", refactor_map)
        self.assertIn("must match exported `det`/`tl_attr` row counts", refactor_map)
        self.assertIn("Export metadata drift", refactor_map)
        self.assertIn("encoded eval batches with `_raw_batch` can be rehydrated", refactor_map)
        self.assertIn("Prepare-batch raw-bundle drift", refactor_map)
        self.assertIn("same unmoved no-image supervision bundle", refactor_map)
        self.assertIn("Raw head contract drift", refactor_map)
        self.assertIn("emitted `det`/`tl_attr` tensor dims", refactor_map)
        self.assertIn("Trunk/head channel handoff drift", refactor_map)
        self.assertIn("default `yolo26s` remains `(128,128,256,512)`", refactor_map)
        self.assertIn("recorded `head_channels` must come from the constructed `PV26Heads.in_channels`", refactor_map)
        self.assertIn("Checkpoint exact-resume drift", refactor_map)
        self.assertIn("every raw-head query/vector dimension", refactor_map)

    def test_active_docs_route_current_execution_to_stable_entrypoints(self) -> None:
        active_command_docs = [
            REPO_ROOT / "README.md",
            DOCS_ROOT / "2_SYSTEM_ARCHITECTURE.md",
            DOCS_ROOT / "6_TRAINING_AND_EVALUATION.md",
            DOCS_ROOT / "8_TEST_PLAN_AND_CHECKLIST.md",
            DOCS_ROOT / "9_EXECUTION_STATUS.md",
            DOCS_ROOT / "10_CODEBASE_REFACTORING_MAP_AND_PLAN.md",
            DOCS_ROOT / "00C_NEXT_GATES.md",
        ]
        retired_command_tokens = [
            "python3 tools/probe_pv26_",
            "python tools/probe_pv26_",
            "tools/run_pv26_" + "lane60_probe.py --",
            "tools/evaluate_pv26_" + "lane60_checkpoint.py --",
            "tools/replay_pv26_" + "lane_point_repair.py --",
            "tools/interpolate_pv26_" + "checkpoints.py --",
            "tools/merge_pv26_" + "lane_family_task_heads.py --",
        ]
        for path in active_command_docs:
            content = _read(path)
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                for token in retired_command_tokens:
                    self.assertNotIn(token, content)

        refactor_map = _read(DOCS_ROOT / "10_CODEBASE_REFACTORING_MAP_AND_PLAN.md")
        self.assertIn("tools/check_env.py", refactor_map)
        self.assertIn("tools/run_pv26_train.py", refactor_map)
        self.assertIn("python -m tools.od_bootstrap", refactor_map)
        self.assertIn("tools/analyze_pv26_run.py", refactor_map)

    def test_probe_history_is_preserved_in_docs_not_active_code(self) -> None:
        current_status = _read(DOCS_ROOT / "00A_CURRENT_STATUS.md")
        status_history = _read(DOCS_ROOT / "00B_STATUS_HISTORY.md")
        legacy_probe_doc = _read(DOCS_ROOT / "legacy" / "19_PV26_LANE60_PROBES_20260509.md")

        self.assertIn("Probe references below are retained as historical experiment evidence", current_status)
        self.assertIn("tools/run_pv26_" + "lane60_probe.py", status_history)
        self.assertIn("tools/run_pv26_" + "lane60_probe.py", legacy_probe_doc)

    def test_sample_contract_doc_exists_and_is_referenced(self) -> None:
        sample_doc = DOCS_ROOT / "legacy" / "4A_SAMPLE_AND_TRANSFORM_CONTRACT.md"
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
        loss_doc = (DOCS_ROOT / "5_TARGETS_AND_LOSS.md").read_text(encoding="utf-8")
        training_doc = (DOCS_ROOT / "6_TRAINING_AND_EVALUATION.md").read_text(encoding="utf-8")
        self.assertIn("legacy/4A_SAMPLE_AND_TRANSFORM_CONTRACT.md", loss_doc)
        self.assertIn("legacy/4A_SAMPLE_AND_TRANSFORM_CONTRACT.md", training_doc)
        self.assertIn("00A_CURRENT_STATUS.md", prd)

    def test_contract_terminology_is_locked(self) -> None:
        sample_doc = _read(DOCS_ROOT / "legacy" / "4A_SAMPLE_AND_TRANSFORM_CONTRACT.md")
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
        architecture_doc = _read(DOCS_ROOT / "legacy" / "4_MODEL_ARCHITECTURE.md")
        self.assertIn(f"fixed query count `{spec['heads']['lane']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['stop_line']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['crosswalk']['query_count']}`", architecture_doc)
        self.assertNotIn("query count 최종값", architecture_doc)

    def test_pv26_docs_track_backbone_and_stage4_direction(self) -> None:
        architecture_doc = _read(DOCS_ROOT / "legacy" / "4_MODEL_ARCHITECTURE.md")
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
        standardization_doc = _read(DOCS_ROOT / "legacy" / "3_DATA_AND_STANDARDIZATION.md")
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized", standardization_doc)
        self.assertIn("seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_aihub_standardized", standardization_doc)
        self.assertNotIn("seg_dataset/pv26_bdd100k_standardized", standardization_doc)


if __name__ == "__main__":
    unittest.main()
