from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_env.scan import _json_load


class CheckEnvScanIOTests(unittest.TestCase):
    def test_json_load_rejects_non_object_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_path = Path(temp_dir) / "payload.json"
            payload_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "JSON root must be an object"):
                _json_load(payload_path)


if __name__ == "__main__":
    unittest.main()
