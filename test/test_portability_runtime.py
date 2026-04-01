from __future__ import annotations

import unittest
from pathlib import Path


class PV26PortabilityRuntimeTests(unittest.TestCase):
    def test_standardization_defaults_follow_repo_root(self) -> None:
        from tools.od_bootstrap.data import aihub, bdd100k

        repo_root = Path(aihub.__file__).resolve().parents[3]
        self.assertEqual(aihub.DEFAULT_REPO_ROOT, repo_root)
        self.assertEqual(bdd100k.DEFAULT_REPO_ROOT, repo_root)
        self.assertTrue(aihub.DEFAULT_AIHUB_ROOT.is_absolute())
        self.assertTrue(aihub.DEFAULT_OBSTACLE_ROOT.is_absolute())
        self.assertTrue(bdd100k.DEFAULT_BDD_ROOT.is_absolute())
        self.assertEqual(aihub.DEFAULT_AIHUB_ROOT.name, "AIHUB")
        self.assertEqual(aihub.DEFAULT_OBSTACLE_ROOT.name, "도로장애물·표면 인지 영상(수도권)")
        self.assertEqual(bdd100k.DEFAULT_BDD_ROOT.name, "BDD100K")

    def test_preflight_report_is_available_without_runtime_side_effects(self) -> None:
        from tools.check_env import check_env

        report = check_env(check_yolo_runtime=False)

        self.assertIn("versions", report)
        self.assertIn("checks", report)
        self.assertIn("torchvision_nms", report["checks"])
        self.assertIn("yolo26", report["checks"])


if __name__ == "__main__":
    unittest.main()
