from __future__ import annotations

import importlib
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class ModuleLayoutCompatTests(unittest.TestCase):
    def test_check_env_package_exports_runtime_surface(self) -> None:
        check_env_pkg = importlib.import_module("tools.check_env")
        scan_new = importlib.import_module("tools.check_env.scan")
        launch_new = importlib.import_module("tools.check_env.launch")

        self.assertTrue(callable(check_env_pkg.check_env))
        self.assertTrue(callable(check_env_pkg.scan_workspace_status))
        self.assertTrue(callable(launch_new._resolve_stage3_stress_action))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tools.check_env_scan")

    def test_pv26_train_package_exports_runtime_surface(self) -> None:
        run_old = importlib.import_module("tools.run_pv26_train")
        run_new = importlib.import_module("tools.pv26_train.cli")
        config_new = importlib.import_module("tools.pv26_train.config")

        self.assertIs(run_old, run_new)
        self.assertTrue(callable(run_old.load_meta_train_scenario))
        self.assertTrue(hasattr(config_new, "MetaTrainScenario"))
        expected_repo_root = REPO_ROOT
        self.assertEqual(run_new.REPO_ROOT, expected_repo_root)
        self.assertEqual(config_new.REPO_ROOT, expected_repo_root)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tools.pv26_train_config")

    def test_od_bootstrap_source_and_teacher_runtime_packages_expose_new_layout(self) -> None:
        source_aihub = importlib.import_module("tools.od_bootstrap.source.aihub")
        shared_new = importlib.import_module("tools.od_bootstrap.source.shared.io")
        runtime_new = importlib.import_module("tools.od_bootstrap.teacher.runtime.progress")
        runtime_trainer = importlib.import_module("tools.od_bootstrap.teacher.runtime.trainer")

        self.assertTrue(callable(source_aihub.run_standardization))
        self.assertTrue(callable(shared_new.link_or_copy))
        self.assertTrue(callable(runtime_new.install_ultralytics_postfix_renderer))
        self.assertTrue(callable(runtime_trainer.make_teacher_trainer))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tools.od_bootstrap.source.shared_io")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tools.od_bootstrap.teacher.runtime_progress")

    def test_check_env_package_uses_repo_root_not_tools_root(self) -> None:
        check_env_pkg = importlib.import_module("tools.check_env")
        scan_new = importlib.import_module("tools.check_env.scan")
        expected_repo_root = REPO_ROOT

        self.assertEqual(check_env_pkg.REPO_ROOT, expected_repo_root)
        self.assertEqual(scan_new.REPO_ROOT, expected_repo_root)


if __name__ == "__main__":
    unittest.main()
