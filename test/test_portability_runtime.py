from __future__ import annotations

import unittest
from pathlib import Path


class PV26PortabilityRuntimeTests(unittest.TestCase):
    def test_standardization_defaults_follow_repo_root(self) -> None:
        from model.preprocess import aihub_standardize, bdd100k_standardize

        repo_root = Path(aihub_standardize.__file__).resolve().parents[2]
        self.assertEqual(aihub_standardize.DEFAULT_REPO_ROOT, repo_root)
        self.assertEqual(bdd100k_standardize.DEFAULT_REPO_ROOT, repo_root)
        self.assertTrue(aihub_standardize.DEFAULT_AIHUB_ROOT.is_absolute())
        self.assertTrue(aihub_standardize.DEFAULT_OBSTACLE_ROOT.is_absolute())
        self.assertTrue(bdd100k_standardize.DEFAULT_BDD_ROOT.is_absolute())
        self.assertEqual(aihub_standardize.DEFAULT_AIHUB_ROOT.name, "AIHUB")
        self.assertEqual(aihub_standardize.DEFAULT_OBSTACLE_ROOT.name, "도로장애물·표면 인지 영상(수도권)")
        self.assertEqual(bdd100k_standardize.DEFAULT_BDD_ROOT.name, "BDD100K")

    def test_preflight_report_is_available_without_runtime_side_effects(self) -> None:
        from tools.check_env import check_env

        report = check_env(check_yolo_runtime=False)

        self.assertIn("versions", report)
        self.assertIn("checks", report)
        self.assertIn("torchvision_nms", report["checks"])
        self.assertIn("yolo26", report["checks"])


if __name__ == "__main__":
    unittest.main()
