from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.io import append_jsonl_sorted, link_or_copy, remove_path, write_json_sorted, write_jsonl_sorted


class CommonIOTests(unittest.TestCase):
    def test_write_json_sorted_orders_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload.json"
            write_json_sorted(path, {"b": 1, "a": 2})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": 2,\n  "b": 1\n}\n')

    def test_append_jsonl_sorted_orders_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            append_jsonl_sorted(path, {"b": 1, "a": 2})
            append_jsonl_sorted(path, {"d": 4, "c": 3})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"a": 2, "b": 1}\n{"c": 3, "d": 4}\n',
            )

    def test_write_jsonl_sorted_orders_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            write_jsonl_sorted(path, [{"b": 1, "a": 2}, {"d": 4, "c": 3}])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"a": 2, "b": 1}, {"c": 3, "d": 4}])

    def test_remove_path_handles_files_symlinks_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "payload.txt"
            file_path.write_text("payload", encoding="utf-8")
            remove_path(file_path)
            self.assertFalse(file_path.exists())

            dir_path = root / "tree"
            (dir_path / "nested.txt").parent.mkdir(parents=True, exist_ok=True)
            (dir_path / "nested.txt").write_text("nested", encoding="utf-8")
            remove_path(dir_path)
            self.assertFalse(dir_path.exists())

            source_path = root / "source.txt"
            source_path.write_text("source", encoding="utf-8")
            symlink_path = root / "source-link.txt"
            symlink_path.symlink_to(source_path)
            remove_path(symlink_path)
            self.assertFalse(symlink_path.exists())

    def test_link_or_copy_replaces_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.txt"
            target_path = root / "target.txt"
            source_path.write_text("fresh", encoding="utf-8")
            target_path.write_text("stale", encoding="utf-8")

            link_or_copy(source_path, target_path)

            self.assertTrue(target_path.exists())
            self.assertEqual(target_path.read_text(encoding="utf-8"), "fresh")


if __name__ == "__main__":
    unittest.main()
