from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.build.final_dataset import (
    FINAL_DATASET_MANIFEST_NAME,
    FINAL_DATASET_PUBLISH_MARKER,
    FINAL_DATASET_RERUN_MODE,
    FINAL_DATASET_SUMMARY_NAME,
    build_pv26_exhaustive_od_lane_dataset,
)
from tools.od_bootstrap.build.final_dataset_stats import (
    FINAL_DATASET_STATS_MARKDOWN_NAME,
    FINAL_DATASET_STATS_NAME,
    analyze_final_dataset,
)


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class FinalDatasetTests(unittest.TestCase):
    def test_build_final_dataset_selects_latest_exhaustive_run_from_parent_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_parent = root / "exhaustive_od"
            older_root = exhaustive_parent / "20260327_000000_model_centric"
            latest_root = exhaustive_parent / "20260328_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_text(older_root / "images" / "train" / "old_input.png", "old")
            _write_text(
                older_root / "labels_scene" / "train" / "old.json",
                json.dumps(
                    {
                        "image": {"file_name": "old_input.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "old",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(older_root / "labels_det" / "train" / "old.txt", "")

            _write_text(latest_root / "images" / "train" / "latest_input.png", "latest")
            _write_text(
                latest_root / "labels_scene" / "train" / "latest.json",
                json.dumps(
                    {
                        "image": {"file_name": "latest_input.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "latest",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(latest_root / "labels_det" / "train" / "latest.txt", "")

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
                exhaustive_od_root=exhaustive_parent,
                aihub_canonical_root=lane_root,
                output_root=output_root,
                copy_images=True,
            )

            self.assertEqual(summary["exhaustive_od_root"], str(latest_root.resolve()))
            self.assertTrue((output_root / "labels_scene" / "train" / "latest.json").is_file())
            self.assertFalse((output_root / "labels_scene" / "train" / "old.json").exists())

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
            manifest = json.loads((output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
            compact_summary = json.loads((output_root / "meta" / FINAL_DATASET_SUMMARY_NAME).read_text(encoding="utf-8"))
            stats_summary = json.loads((output_root / "meta" / FINAL_DATASET_STATS_NAME).read_text(encoding="utf-8"))
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
            self.assertEqual(
                manifest["samples"][0]["scene_path"],
                str((output_root / "labels_scene" / "train" / "od.json").resolve()),
            )
            self.assertEqual(
                manifest["samples"][0]["image_path"],
                str((output_root / "images" / "train" / "od.png").resolve()),
            )
            self.assertEqual(
                manifest["samples"][0]["det_path"],
                str((output_root / "labels_det" / "train" / "od.txt").resolve()),
            )
            self.assertEqual(
                manifest["samples"][1]["scene_path"],
                str((output_root / "labels_scene" / "train" / "lane.json").resolve()),
            )
            self.assertEqual(
                manifest["samples"][1]["image_path"],
                str((output_root / "images" / "train" / "lane.png").resolve()),
            )
            self.assertEqual(od_scene["source"]["final_sample_id"], "od")
            self.assertEqual(od_scene["source"]["source_kind"], "exhaustive_od")
            self.assertEqual(od_scene["image"]["file_name"], "od.png")
            self.assertEqual(od_scene["image"]["original_file_name"], "bdd_frame.png")
            self.assertEqual(lane_scene["source"]["final_sample_id"], "lane")
            self.assertEqual(lane_scene["source"]["source_kind"], "lane")
            self.assertEqual(lane_scene["image"]["file_name"], "lane.png")
            self.assertEqual(lane_scene["image"]["original_file_name"], "lane_source.png")
            self.assertEqual(publish_marker["status"], "completed")
            self.assertEqual(publish_marker["sample_count"], 2)
            self.assertEqual(
                publish_marker["dataset_counts"],
                {
                    "aihub_lane_seoul": 1,
                    "pv26_exhaustive_bdd100k_det_100k": 1,
                },
            )
            self.assertEqual(publish_marker["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertEqual(summary["rerun_mode"], FINAL_DATASET_RERUN_MODE)
            self.assertEqual(summary["exhaustive_od_root"], str(exhaustive_root))
            self.assertEqual(summary["aihub_canonical_root"], str(lane_root.resolve()))
            self.assertEqual(summary["summary_path"], str(output_root / "meta" / FINAL_DATASET_SUMMARY_NAME))
            self.assertEqual(summary["publish_marker_path"], str(output_root / "meta" / FINAL_DATASET_PUBLISH_MARKER))
            self.assertEqual(summary["stats_path"], str(output_root / "meta" / FINAL_DATASET_STATS_NAME))
            self.assertEqual(summary["stats_markdown_path"], str(output_root / "meta" / FINAL_DATASET_STATS_MARKDOWN_NAME))
            self.assertEqual(compact_summary, summary)
            self.assertNotIn("samples", compact_summary)
            self.assertEqual(stats_summary["sample_count"], 2)
            self.assertEqual(stats_summary["detector"]["classes"]["vehicle"]["instance_count"], 0)
            self.assertEqual(stats_summary["lane"]["classes"]["white_lane"]["instance_count"], 0)

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

    def test_analyze_final_dataset_reports_lane_only_and_stale_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "pv26_exhaustive_od_lane_dataset"
            image_path = dataset_root / "images" / "val" / "lane_001.png"
            scene_path = dataset_root / "labels_scene" / "val" / "lane_001.json"
            _write_text(image_path, "lane-image")
            _write_text(
                scene_path,
                json.dumps(
                    {
                        "image": {"file_name": image_path.name, "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "val", "final_sample_id": "lane_001"},
                        "detections": [],
                        "lanes": [{"class_name": "yellow_lane", "source_style": "solid", "points": [[0, 0], [10, 10]]}],
                        "stop_lines": [{"points": [[0, 0], [20, 0]]}],
                        "crosswalks": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(
                dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME,
                json.dumps(
                    {
                        "version": "test",
                        "sample_count": 1,
                        "dataset_counts": {"aihub_lane_seoul": 1},
                        "samples": [
                            {
                                "final_sample_id": "lane_001",
                                "source_dataset_key": "aihub_lane_seoul",
                                "split": "val",
                                "scene_path": str(root / ".staging" / "labels_scene" / "val" / "lane_001.json"),
                                "image_path": str(root / ".staging" / "images" / "val" / "lane_001.png"),
                                "det_path": None,
                            }
                        ],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )

            stats = analyze_final_dataset(dataset_root=dataset_root, write_artifacts=True)

        self.assertEqual(stats["sample_count"], 1)
        self.assertEqual(stats["dataset_counts"], {"aihub_lane_seoul": 1})
        self.assertEqual(stats["lane"]["classes"]["yellow_lane"]["instance_count"], 1)
        self.assertEqual(stats["stop_line"]["instance_count"], 1)
        self.assertIn("lane_only_final_dataset", stats["warnings"])
        self.assertIn("manifest_paths_stale", stats["warnings"])
        self.assertTrue(stats["audit"]["rebuild_needed"])

    def test_build_pv26_exhaustive_od_lane_dataset_atomically_replaces_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od" / "20260327_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_text(output_root / "stale.txt", "old")
            _write_text(output_root / "meta" / FINAL_DATASET_SUMMARY_NAME, json.dumps({"sample_count": 1}) + "\n")

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
            self.assertTrue((output_root / "meta" / FINAL_DATASET_SUMMARY_NAME).is_file())
