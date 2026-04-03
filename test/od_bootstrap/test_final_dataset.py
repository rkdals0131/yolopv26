from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.build.final_dataset import (
    FINAL_DATASET_PUBLISH_MARKER,
    FINAL_DATASET_RERUN_MODE,
    build_pv26_exhaustive_od_lane_dataset,
)


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

            _write_text(exhaustive_root / "images" / "train" / "od_input.png", "od")
            _write_text(
                exhaustive_root / "labels_scene" / "train" / "od.json",
                json.dumps(
                    {
                        "image": {
                            "file_name": "od_input.png",
                            "original_file_name": "bdd_frame.png",
                            "width": 640,
                            "height": 480,
                        },
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "od",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "")

            _write_text(lane_root / "images" / "train" / "lane_source.png", "lane")
            _write_text(
                lane_root / "labels_scene" / "train" / "lane.json",
                json.dumps(
                    {
                        "image": {"file_name": "lane_source.png", "width": 640, "height": 480},
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
            self.assertTrue((output_root / "images" / "train" / "od.png").is_file())
            self.assertTrue((output_root / "images" / "train" / "lane.png").is_file())
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            compact_summary = json.loads((output_root / "meta" / "final_dataset_summary.json").read_text(encoding="utf-8"))
            publish_marker = json.loads((output_root / "meta" / FINAL_DATASET_PUBLISH_MARKER).read_text(encoding="utf-8"))
            od_scene = json.loads((output_root / "labels_scene" / "train" / "od.json").read_text(encoding="utf-8"))
            lane_scene = json.loads((output_root / "labels_scene" / "train" / "lane.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_counts"]["aihub_lane_seoul"], 1)
            self.assertEqual(manifest["dataset_counts"]["pv26_exhaustive_bdd100k_det_100k"], 1)
            self.assertEqual(manifest["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertEqual(len(list((output_root / "labels_scene" / "train").glob("*.json"))), 2)
            self.assertEqual(len(list((output_root / "images" / "train").glob("*.png"))), 2)
            self.assertEqual(manifest["samples"][0]["final_sample_id"], "od")
            self.assertEqual(manifest["samples"][1]["final_sample_id"], "lane")
            self.assertIsNone(manifest["samples"][1]["det_path"])
            self.assertEqual(od_scene["source"]["final_sample_id"], "od")
            self.assertEqual(od_scene["source"]["source_kind"], "exhaustive_od")
            self.assertEqual(od_scene["image"]["file_name"], "od.png")
            self.assertEqual(od_scene["image"]["original_file_name"], "bdd_frame.png")
            self.assertEqual(lane_scene["source"]["final_sample_id"], "lane")
            self.assertEqual(lane_scene["source"]["source_kind"], "lane")
            self.assertEqual(lane_scene["image"]["file_name"], "lane.png")
            self.assertEqual(lane_scene["image"]["original_file_name"], "lane_source.png")
            self.assertEqual(publish_marker["status"], "completed")
            self.assertEqual(publish_marker["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertEqual(summary["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertEqual(summary["exhaustive_od_root"], str(exhaustive_root))
            self.assertEqual(summary["aihub_canonical_root"], str(lane_root.resolve()))
            self.assertEqual(summary["summary_path"], str(output_root / "meta" / "final_dataset_summary.json"))
            self.assertEqual(summary["publish_marker_path"], str(output_root / "meta" / FINAL_DATASET_PUBLISH_MARKER))
            self.assertEqual(compact_summary, summary)
            self.assertNotIn("samples", compact_summary)

    def test_build_pv26_exhaustive_od_lane_dataset_rejects_duplicate_final_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od" / "20260327_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_text(exhaustive_root / "images" / "train" / "shared_input.png", "od")
            _write_text(
                exhaustive_root / "labels_scene" / "train" / "shared.json",
                json.dumps(
                    {
                        "image": {"file_name": "shared_input.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "shared",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "shared.txt", "")

            _write_text(lane_root / "images" / "train" / "lane_input.png", "lane")
            _write_text(
                lane_root / "labels_scene" / "train" / "shared.json",
                json.dumps(
                    {
                        "image": {"file_name": "lane_input.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )

            with self.assertRaisesRegex(ValueError, "duplicate final_sample_id"):
                build_pv26_exhaustive_od_lane_dataset(
                    exhaustive_od_root=exhaustive_root.parent / "latest",
                    aihub_canonical_root=lane_root,
                    output_root=output_root,
                    copy_images=True,
                )

    def test_build_pv26_exhaustive_od_lane_dataset_atomically_replaces_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od" / "20260327_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_text(output_root / "stale.txt", "old")
            _write_text(output_root / "meta" / "final_dataset_summary.json", json.dumps({"sample_count": 1}) + "\n")

            _write_text(exhaustive_root / "images" / "train" / "od_input.png", "od")
            _write_text(
                exhaustive_root / "labels_scene" / "train" / "od.json",
                json.dumps(
                    {
                        "image": {"file_name": "od_input.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "od",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "")

            _write_text(lane_root / "images" / "train" / "lane_source.png", "lane")
            _write_text(
                lane_root / "labels_scene" / "train" / "lane.json",
                json.dumps(
                    {
                        "image": {"file_name": "lane_source.png", "width": 640, "height": 480},
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

            self.assertFalse((output_root / "stale.txt").exists())
            self.assertTrue((output_root / "labels_scene" / "train" / "od.json").is_file())
            self.assertTrue((output_root / "meta" / FINAL_DATASET_PUBLISH_MARKER).is_file())
            self.assertEqual(summary["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertTrue((output_root / "meta" / "final_dataset_summary.json").is_file())
