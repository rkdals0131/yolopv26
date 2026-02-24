import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.pv26_qc_report import compute_qc_report


class TestPv26QcReport(unittest.TestCase):
    def _write_u8_png(self, path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype(np.uint8, copy=False), mode="L").save(path, format="PNG")

    def test_seg_nonempty_ratios(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pv26_v1"
            (root / "meta").mkdir(parents=True, exist_ok=True)

            # Two rows in train.
            da1 = Path("labels_seg_da/train/a.png")
            da2 = Path("labels_seg_da/train/b.png")
            sem1 = Path("labels_semantic_id/train/a.png")
            sem2 = Path("labels_semantic_id/train/b.png")
            self._write_u8_png(root / da1, np.array([[0, 1], [0, 0]], dtype=np.uint8))
            self._write_u8_png(root / da2, np.zeros((2, 2), dtype=np.uint8))
            # semantic_id: row a has class 2, row b background only
            self._write_u8_png(root / sem1, np.array([[0, 2], [0, 0]], dtype=np.uint8))
            self._write_u8_png(root / sem2, np.zeros((2, 2), dtype=np.uint8))

            manifest = root / "meta" / "split_manifest.csv"
            header = [
                "split",
                "has_da",
                "da_relpath",
                "has_semantic_id",
                "semantic_relpath",
                "weather_tag",
                "time_tag",
                "scene_tag",
            ]
            rows = [
                {
                    "split": "train",
                    "has_da": "1",
                    "da_relpath": str(da1),
                    "has_semantic_id": "1",
                    "semantic_relpath": str(sem1),
                    "weather_tag": "dry",
                    "time_tag": "day",
                    "scene_tag": "open",
                },
                {
                    "split": "train",
                    "has_da": "1",
                    "da_relpath": str(da2),
                    "has_semantic_id": "1",
                    "semantic_relpath": str(sem2),
                    "weather_tag": "dry",
                    "time_tag": "night",
                    "scene_tag": "open",
                },
            ]
            with open(manifest, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

            report = compute_qc_report(root, splits=None)
            self.assertEqual(report["num_rows"], 2)
            self.assertEqual(report["row_count_per_split"]["train"], 2)
            self.assertAlmostEqual(report["seg_nonempty"]["da"]["nonempty_ratio"], 0.5)
            self.assertAlmostEqual(report["seg_nonempty"]["semantic_id"]["nonempty_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()

