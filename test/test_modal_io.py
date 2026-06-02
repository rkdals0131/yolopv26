from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.modal.constants import REQUIRED_DATASET_DIRS
from tools.modal.local_preflight import _dataset_status
from tools.modal.prepare_dataset_volume import _check_dataset


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def _make_dataset_root(root: Path) -> None:
    for rel_path in REQUIRED_DATASET_DIRS:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def _fake_modal_module() -> types.ModuleType:
    module = types.ModuleType("modal")

    class _Image:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *_args):
            return self

        def pip_install_from_requirements(self, *_args):
            return self

        def add_local_dir(self, *_args, **_kwargs):
            return self

    class _App:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def function(self, **_kwargs):
            return lambda func: func

        def local_entrypoint(self):
            return lambda func: func

    class _Volume:
        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

        def commit(self) -> None:
            pass

    module.Image = _Image
    module.App = _App
    module.Volume = _Volume
    return module


class ModalIOTests(unittest.TestCase):
    def test_local_preflight_dataset_status_loads_stats_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _make_dataset_root(dataset_root)
            _write_json(dataset_root / "meta" / "final_dataset_stats.json", {"sample_count": 42})

            status = _dataset_status(dataset_root)

            self.assertEqual(status["sample_count"], 42)
            self.assertEqual(status["missing_required_dirs"], [])

    def test_local_preflight_dataset_status_rejects_non_mapping_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _make_dataset_root(dataset_root)
            _write_json(dataset_root / "meta" / "final_dataset_stats.json", [])

            with self.assertRaisesRegex(TypeError, "final dataset stats root must be a mapping"):
                _dataset_status(dataset_root)

    def test_prepare_dataset_check_loads_stats_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _make_dataset_root(dataset_root)
            _write_json(dataset_root / "meta" / "final_dataset_stats.json", {"sample_count": 43})

            status = _check_dataset(dataset_root)

            self.assertEqual(status["sample_count"], 43)
            self.assertEqual(status["required_dirs"], len(REQUIRED_DATASET_DIRS))

    def test_prepare_dataset_check_rejects_non_mapping_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _make_dataset_root(dataset_root)
            _write_json(dataset_root / "meta" / "final_dataset_stats.json", [])

            with self.assertRaisesRegex(TypeError, "final dataset stats root must be a mapping"):
                _check_dataset(dataset_root)

    def test_modal_sdk_import_skips_repo_tools_path_shadow(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_sdk_root = Path(temp_dir)
            (fake_sdk_root / "modal.py").write_text("SENTINEL = 'external-modal-sdk'\n", encoding="utf-8")
            path_shadow = str(repo_root / "tools")
            patched_path = [path_shadow, str(repo_root), str(fake_sdk_root), *sys.path]

            with patch.object(sys, "path", patched_path):
                importlib.invalidate_caches()
                sys.modules.pop("modal", None)
                sys.modules.pop("tools.modal.sdk_import", None)
                sdk_import = importlib.import_module("tools.modal.sdk_import")

            self.assertEqual(getattr(sdk_import.modal, "SENTINEL", None), "external-modal-sdk")
            self.assertNotEqual(
                getattr(sdk_import.modal, "__file__", None),
                str(repo_root / "tools" / "modal" / "__init__.py"),
            )
            sys.modules.pop("modal", None)
            sys.modules.pop("tools.modal.sdk_import", None)

    def test_modal_train_writes_and_verifies_path_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_modal = _fake_modal_module()
            with patch.dict(sys.modules, {"modal": fake_modal}):
                sys.modules.pop("tools.modal.sdk_import", None)
                sys.modules.pop("tools.modal.train", None)
                train_module = importlib.import_module("tools.modal.train")

            remote_repo_root = Path(temp_dir) / "remote_repo"
            local_dataset_root = Path(temp_dir) / "local_dataset"
            remote_run_root = Path(temp_dir) / "remote_runs"
            with (
                patch.object(train_module, "REMOTE_REPO_ROOT", remote_repo_root),
                patch.object(train_module, "LOCAL_DATASET_ROOT", local_dataset_root),
                patch.object(train_module, "REMOTE_RUN_ROOT", remote_run_root),
            ):
                train_module._write_modal_path_config()

            self.assertEqual(
                (remote_repo_root / "config" / "user_paths.yaml").read_text(encoding="utf-8"),
                "pv26_train:\n"
                f"  dataset_root: {local_dataset_root}\n"
                f"  run_root: {remote_run_root}\n",
            )


if __name__ == "__main__":
    unittest.main()
