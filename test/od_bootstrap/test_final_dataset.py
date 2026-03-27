from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.finalize.final_dataset import build_pv26_exhaustive_od_lane_dataset


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class FinalDatasetTests(unittest.TestCase):
    def test_build_pv26_exhaustive_od_lane_dataset_merges_lane_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od" / "20260327_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_text(exhaustive_root / "images" / "train" / "od.png", "od")
            _write_text(
                exhaustive_root / "labels_scene" / "train" / "od.json",
                json.dumps(
                    {
                        "image": {"file_name": "od.png", "width": 640, "height": 480},
                        "source": {"dataset": "pv26_exhaustive_bdd100k_det_100k", "split": "train"},
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "")

            _write_text(lane_root / "images" / "train" / "lane.png", "lane")
            _write_text(
                lane_root / "labels_scene" / "train" / "lane.json",
                json.dumps(
                    {
                        "image": {"file_name": "lane.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )

            summary = build_pv26_exhaustive_od_lane_dataset(
                exhaustive_od_root=exhaustive_root.parent / "latest",
                aihub_canonical_root=lane_root,
                output_root=output_root,
                copy_images=True,
            )

            self.assertEqual(summary["sample_count"], 2)
            self.assertTrue((output_root / "labels_scene" / "train" / "od.json").is_file())
            self.assertTrue((output_root / "labels_scene" / "train" / "lane.json").is_file())
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_counts"]["aihub_lane_seoul"], 1)
            self.assertEqual(manifest["dataset_counts"]["pv26_exhaustive_bdd100k_det_100k"], 1)
