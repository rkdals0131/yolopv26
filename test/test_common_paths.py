from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.paths import resolve_latest_root, resolve_optional_path


class CommonPathsTests(unittest.TestCase):
    def test_resolve_optional_path_preserves_empty_values(self) -> None:
        base_dir = Path("/tmp/example")

        self.assertIsNone(resolve_optional_path(None, base_dir=base_dir))
        self.assertIsNone(resolve_optional_path("", base_dir=base_dir))

    def test_resolve_optional_path_resolves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(resolve_optional_path("nested/file.txt", base_dir=root), (root / "nested" / "file.txt").resolve())

    def test_resolve_latest_root_latest_alias_picks_last_named_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "20260327_120000_old").mkdir()
            newest = root / "20260328_120000_new"
            newest.mkdir()

            self.assertEqual(resolve_latest_root(root / "latest"), newest.resolve())
