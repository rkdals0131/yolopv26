from __future__ import annotations

import importlib
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_TOOL_FILENAMES = {
    "run_pv26_" + "lane60_probe.py",
    "evaluate_pv26_" + "lane60_checkpoint.py",
    "analyze_pv26_" + "lane60_prediction_filters.py",
    "analyze_pv26_lane_" + "repairability_model_replay.py",
    "replay_pv26_" + "lane_point_repair.py",
    "interpolate_pv26_" + "checkpoints.py",
    "merge_pv26_" + "lane_family_task_heads.py",
}
RETIRED_IMPORT_TOKENS = (
    "tools." + "probe_",
    "tools.run_pv26_" + "lane60_probe",
    "tools.evaluate_pv26_" + "lane60_checkpoint",
    "tools.analyze_pv26_" + "lane60_prediction_filters",
    "tools.analyze_pv26_lane_" + "repairability_model_replay",
    "tools.replay_pv26_" + "lane_point_repair",
    "tools.interpolate_pv26_" + "checkpoints",
    "tools.merge_pv26_" + "lane_family_task_heads",
)


class ModuleLayoutCompatTests(unittest.TestCase):
    def test_check_env_package_exports_runtime_surface(self) -> None:
        check_env_pkg = importlib.import_module("tools.check_env")
        scan_new = importlib.import_module("tools.check_env.scan")
        launch_new = importlib.import_module("tools.check_env.launch")

        self.assertTrue(callable(check_env_pkg.check_env))
        self.assertTrue(callable(check_env_pkg.scan_workspace_status))
        self.assertTrue(callable(launch_new._resolve_phase_stress_action))
        self.assertTrue(callable(launch_new._resolve_stage3_stress_action))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tools.check_env_scan")

    def test_pv26_train_package_exports_runtime_surface(self) -> None:
        run_old = importlib.import_module("tools.run_pv26_train")
        run_new = importlib.import_module("tools.pv26_train.cli")
        config_new = importlib.import_module("tools.pv26_train.config")
        scenario_shim = importlib.import_module("tools.pv26_train.scenario")
        scenarios_new = importlib.import_module("tools.pv26_train.scenarios")

        self.assertIs(run_old, run_new)
        self.assertTrue(callable(run_old.load_meta_train_scenario))
        self.assertTrue(hasattr(config_new, "MetaTrainScenario"))
        self.assertIs(scenario_shim.load_meta_train_scenario, scenarios_new.load_meta_train_scenario)
        self.assertIs(scenario_shim.build_meta_train_presets, scenarios_new.build_meta_train_presets)
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

    def test_modal_helpers_live_under_tools_package(self) -> None:
        constants = importlib.import_module("tools.modal.constants")
        dataset_archive = importlib.import_module("tools.modal.dataset_archive")
        local_preflight = importlib.import_module("tools.modal.local_preflight")
        prepare_dataset_volume = importlib.import_module("tools.modal.prepare_dataset_volume")

        self.assertFalse((REPO_ROOT / "modal").exists())
        self.assertEqual(local_preflight.REPO_ROOT, REPO_ROOT)
        self.assertEqual(prepare_dataset_volume.REPO_ROOT, REPO_ROOT)
        self.assertTrue(callable(constants.validate_modal_constants))
        self.assertTrue(callable(dataset_archive.verify_layout))

    def test_check_env_package_uses_repo_root_not_tools_root(self) -> None:
        check_env_pkg = importlib.import_module("tools.check_env")
        scan_new = importlib.import_module("tools.check_env.scan")
        expected_repo_root = REPO_ROOT

        self.assertEqual(check_env_pkg.REPO_ROOT, expected_repo_root)
        self.assertEqual(scan_new.REPO_ROOT, expected_repo_root)

    def test_tools_use_public_raw_batch_merge_helper(self) -> None:
        forbidden = "from model.engine._trainer_epochs import _merge_raw_batches"
        for path in sorted((REPO_ROOT / "tools").rglob("*.py")):
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotIn(forbidden, path.read_text(encoding="utf-8"))

    def test_stable_tools_do_not_import_private_engine_modules(self) -> None:
        forbidden_tokens = (
            "from model.engine._",
            "import model.engine._",
        )
        for path in sorted((REPO_ROOT / "tools").rglob("*.py")):
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                content = path.read_text(encoding="utf-8")
                for token in forbidden_tokens:
                    self.assertNotIn(token, content)

    def test_retired_probe_tool_modules_are_absent(self) -> None:
        tool_paths = sorted((REPO_ROOT / "tools").rglob("*.py"))
        retired_probe_paths = [path for path in tool_paths if path.name.startswith("probe_pv26_")]
        retired_exact_paths = [path for path in tool_paths if path.name in RETIRED_TOOL_FILENAMES]

        self.assertEqual([], [str(path.relative_to(REPO_ROOT)) for path in retired_probe_paths])
        self.assertEqual([], [str(path.relative_to(REPO_ROOT)) for path in retired_exact_paths])

    def test_tests_do_not_import_retired_probe_modules(self) -> None:
        for path in sorted((REPO_ROOT / "test").rglob("*.py")):
            import_lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.lstrip().startswith(("from ", "import "))
            ]
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                for token in RETIRED_IMPORT_TOKENS:
                    self.assertFalse(any(token in line for line in import_lines))

    def test_stable_entrypoint_import_surface_remains_available(self) -> None:
        run_train = importlib.import_module("tools.run_pv26_train")
        check_env = importlib.import_module("tools.check_env")
        od_bootstrap = importlib.import_module("tools.od_bootstrap")

        self.assertTrue(callable(run_train.main))
        self.assertTrue(callable(check_env.check_env))
        self.assertTrue(hasattr(od_bootstrap, "__file__"))


if __name__ == "__main__":
    unittest.main()
