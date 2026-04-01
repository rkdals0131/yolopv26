from __future__ import annotations

from pathlib import Path
import unittest

from model.loss.spec import build_loss_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
OD_BOOTSTRAP_README = REPO_ROOT / "tools" / "od_bootstrap" / "README.md"
OD_BOOTSTRAP_PREPROCESS_README = REPO_ROOT / "tools" / "od_bootstrap" / "preprocess" / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class DocsSyncTests(unittest.TestCase):
    def test_no_absolute_repo_links_remain_in_active_docs(self) -> None:
        targets = [
            REPO_ROOT / "README.md",
            OD_BOOTSTRAP_README,
            OD_BOOTSTRAP_PREPROCESS_README,
            *sorted(DOCS_ROOT.glob("*.md")),
        ]
        for path in targets:
            content = _read(path)
            self.assertNotIn("/home/user1", content, msg=str(path))
            self.assertNotIn("/YOLOpv26/", content, msg=str(path))

    def test_readme_matches_current_pv26_training_entrypoint(self) -> None:
        readme = _read(REPO_ROOT / "README.md")
        self.assertIn("config/pv26_meta_train.default.yaml", readme)
        self.assertIn("seg_dataset/pv26_exhaustive_od_lane_dataset", readme)
        self.assertIn("runs/pv26_exhaustive_od_lane_train", readme)
        self.assertIn("`4번`부터 `10번`까지", readme)
        self.assertNotIn("runs/pv26_meta_train/", readme)
        self.assertNotIn("`2번`, `3번`으로 전처리한 뒤 `1번`을 실행하면 된다", readme)

    def test_od_bootstrap_readme_tracks_current_teacher_model_defaults(self) -> None:
        readme = _read(OD_BOOTSTRAP_README)
        self.assertIn("tools/od_bootstrap/config/train/mobility_yolo26s.default.yaml", readme)
        self.assertIn("tools/od_bootstrap/config/train/signal_yolo26s.default.yaml", readme)
        self.assertIn("tools/od_bootstrap/config/train/obstacle_yolo26m.default.yaml", readme)
        self.assertIn("mobility/signal은 `yolo26s.pt`, obstacle은 `yolo26m.pt`", readme)

    def test_preprocess_readme_mentions_current_debug_vis_and_smoke_tooling(self) -> None:
        readme = _read(OD_BOOTSTRAP_PREPROCESS_README)
        self.assertIn("run_generate_debug_vis.py", readme)
        self.assertIn("debug_vis.py", readme)
        self.assertIn("run_build_smoke_image_list.py", readme)
        self.assertIn("run_render_smoke_review.py", readme)
        self.assertIn("run_audit_teacher_checkpoints.py", readme)

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

    def test_system_architecture_tracks_runtime_not_contract_gap(self) -> None:
        architecture_doc = _read(DOCS_ROOT / "2_SYSTEM_ARCHITECTURE.md")
        self.assertIn("training sample runtime", architecture_doc)
        self.assertNotIn("training sample contract", architecture_doc)


if __name__ == "__main__":
    unittest.main()
