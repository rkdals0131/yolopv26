from __future__ import annotations

import importlib
import unittest


class ModuleLayoutCompatTests(unittest.TestCase):
    def test_check_env_package_and_legacy_shims_share_modules(self) -> None:
        check_env_pkg = importlib.import_module("tools.check_env")
        scan_old = importlib.import_module("tools.check_env_scan")
        scan_new = importlib.import_module("tools.check_env.scan")
        launch_old = importlib.import_module("tools.check_env_launch")
        launch_new = importlib.import_module("tools.check_env.launch")

        self.assertIs(scan_old, scan_new)
        self.assertIs(launch_old, launch_new)
        self.assertTrue(callable(check_env_pkg.check_env))
        self.assertTrue(callable(check_env_pkg.scan_workspace_status))

    def test_pv26_train_package_and_legacy_shims_share_modules(self) -> None:
        run_old = importlib.import_module("tools.run_pv26_train")
        run_new = importlib.import_module("tools.pv26_train.cli")
        config_old = importlib.import_module("tools.pv26_train_config")
        config_new = importlib.import_module("tools.pv26_train.config")

        self.assertIs(run_old, run_new)
        self.assertIs(config_old, config_new)
        self.assertTrue(callable(run_old.load_meta_train_scenario))
        self.assertTrue(hasattr(config_old, "MetaTrainScenario"))

    def test_od_bootstrap_source_and_teacher_runtime_packages_keep_legacy_shims(self) -> None:
        source_aihub = importlib.import_module("tools.od_bootstrap.source.aihub")
        shared_old = importlib.import_module("tools.od_bootstrap.source.shared_io")
        shared_new = importlib.import_module("tools.od_bootstrap.source.shared.io")
        runtime_old = importlib.import_module("tools.od_bootstrap.teacher.runtime_progress")
        runtime_new = importlib.import_module("tools.od_bootstrap.teacher.runtime.progress")

        self.assertTrue(callable(source_aihub.run_standardization))
        self.assertIs(shared_old, shared_new)
        self.assertIs(runtime_old, runtime_new)


if __name__ == "__main__":
    unittest.main()
