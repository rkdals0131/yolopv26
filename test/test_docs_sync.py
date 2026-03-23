from __future__ import annotations

from pathlib import Path
import unittest

from model.loss.spec import build_loss_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


class DocsSyncTests(unittest.TestCase):
    def test_no_absolute_repo_links_remain_in_active_docs(self) -> None:
        targets = [REPO_ROOT / "README.md", DOCS_ROOT / "0_PRD.md", DOCS_ROOT / "5_TARGETS_AND_LOSS.md"]
        for path in targets:
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/user1", content, msg=str(path))
            self.assertNotIn("/YOLOpv26/", content, msg=str(path))

    def test_sample_contract_doc_exists_and_is_referenced(self) -> None:
        sample_doc = DOCS_ROOT / "4A_SAMPLE_AND_TRANSFORM_CONTRACT.md"
        self.assertTrue(sample_doc.exists())
        content = sample_doc.read_text(encoding="utf-8")
        self.assertIn('"image"', content)
        self.assertIn('"det_targets"', content)
        self.assertIn('"tl_attr_targets"', content)
        self.assertIn('"lane_targets"', content)
        self.assertIn('"source_mask"', content)
        self.assertIn('"valid_mask"', content)
        self.assertIn('"meta"', content)

        prd = (DOCS_ROOT / "0_PRD.md").read_text(encoding="utf-8")
        self.assertIn("[4A_SAMPLE_AND_TRANSFORM_CONTRACT.md]", prd)

    def test_query_counts_are_synced_between_docs_and_spec(self) -> None:
        spec = build_loss_spec()
        architecture_doc = (DOCS_ROOT / "4_MODEL_ARCHITECTURE.md").read_text(encoding="utf-8")
        self.assertIn(f"fixed query count `{spec['heads']['lane']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['stop_line']['query_count']}`", architecture_doc)
        self.assertIn(f"fixed query count `{spec['heads']['crosswalk']['query_count']}`", architecture_doc)
        self.assertNotIn("query count 최종값", architecture_doc)


if __name__ == "__main__":
    unittest.main()
