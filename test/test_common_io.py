from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.io import append_jsonl_sorted, write_json_sorted, write_jsonl_sorted


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


if __name__ == "__main__":
    unittest.main()
