from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.od_bootstrap.build.final_dataset import (
    FINAL_DATASET_MANIFEST_NAME,
    FINAL_DATASET_PUBLISH_MARKER,
    FINAL_DATASET_RERUN_MODE,
    FINAL_DATASET_SUMMARY_NAME,
    _load_scene,
    build_pv26_exhaustive_od_lane_dataset,
)
from tools.od_bootstrap.build.final_dataset_stats import (
    FINAL_DATASET_STATS_MARKDOWN_NAME,
    FINAL_DATASET_STATS_NAME,
    _load_json as _load_stats_json,
    analyze_final_dataset,
)


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_image(path: Path, *, width: int = 640, height: int = 480, color: str = "#303840") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (width, height), color).save(path)


class FinalDatasetTests(unittest.TestCase):
    def test_load_scene_rejects_non_object_json_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.json"
            _write_text(scene_path, "[]\n")

            with self.assertRaisesRegex(TypeError, "scene root must be an object"):
                _load_scene(scene_path)

    def test_load_final_dataset_stats_json_rejects_non_mapping_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_path = Path(temp_dir) / "stats.json"
            _write_text(stats_path, "[]\n")

            with self.assertRaisesRegex(TypeError, "JSON root must be a mapping"):
                _load_stats_json(stats_path)

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
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "0 0.350000 0.350000 0.300000 0.300000\n")

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
            class_map_text = (output_root / "meta" / "class_map_det.yaml").read_text(encoding="utf-8")
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
                manifest["samples"][0]["source_scene_path"],
                str(exhaustive_root / "labels_scene" / "train" / "od.json"),
            )
            self.assertEqual(
                manifest["samples"][0]["source_image_path"],
                str(exhaustive_root / "images" / "train" / "od_input.png"),
            )
            self.assertEqual(
                manifest["samples"][0]["source_det_path"],
                str(exhaustive_root / "labels_det" / "train" / "od.txt"),
            )
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
                (output_root / "labels_det" / "train" / "od.txt").read_text(encoding="utf-8"),
                "0 0.350000 0.350000 0.300000 0.300000\n",
            )
            self.assertEqual(
                manifest["samples"][1]["scene_path"],
                str((output_root / "labels_scene" / "train" / "lane.json").resolve()),
            )
            self.assertEqual(
                manifest["samples"][1]["source_scene_path"],
                str(lane_root / "labels_scene" / "train" / "lane.json"),
            )
            self.assertEqual(
                manifest["samples"][1]["source_image_path"],
                str(lane_root / "images" / "train" / "lane_source.png"),
            )
            self.assertIsNone(manifest["samples"][1]["source_det_path"])
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
            self.assertIn("'0': vehicle", class_map_text)
            self.assertEqual(stats_summary["sample_count"], 2)
            self.assertEqual(stats_summary["detector"]["classes"]["vehicle"]["instance_count"], 0)
            self.assertEqual(stats_summary["lane"]["classes"]["white_lane"]["instance_count"], 0)

    def test_final_dataset_publication_loads_through_pv26_dataset_and_encoder(self) -> None:
        from common.pv26_schema import OD_CLASSES
        from model.data import PV26CanonicalDataset, collate_pv26_encoded_eval_batch
        from model.data.transform import NETWORK_HW
        from model.net import PV26Heads

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od" / "20260327_000000_model_centric"
            lane_root = root / "canonical" / "aihub_standardized"
            output_root = root / "pv26_exhaustive_od_lane_dataset"

            _write_image(exhaustive_root / "images" / "train" / "od_input.png", color="#202830")
            _write_text(
                exhaustive_root / "labels_scene" / "train" / "od.json",
                json.dumps(
                    {
                        "image": {
                            "file_name": "od_input.png",
                            "width": 640,
                            "height": 480,
                        },
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "train",
                            "bootstrap_sample_uid": "od",
                        },
                        "detections": [
                            {"id": 0, "class_name": "vehicle", "bbox": [128.0, 96.0, 320.0, 240.0]}
                        ],
                        "traffic_lights": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "0 0.350000 0.350000 0.300000 0.300000\n")

            _write_image(lane_root / "images" / "train" / "lane_source.png", color="#203020")
            _write_text(
                lane_root / "labels_scene" / "train" / "lane.json",
                json.dumps(
                    {
                        "image": {"file_name": "lane_source.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        "detections": [],
                        "lanes": [
                            {
                                "class_name": "white_lane",
                                "source_style": "solid",
                                "points": [[180.0, 470.0], [210.0, 340.0], [240.0, 220.0]],
                                "visibility": [1.0, 1.0, 1.0],
                            }
                        ],
                        "stop_lines": [{"points": [[160.0, 360.0], [480.0, 360.0]]}],
                        "crosswalks": [
                            {"points": [[250.0, 390.0], [360.0, 390.0], [390.0, 430.0], [230.0, 430.0]]}
                        ],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )

            build_pv26_exhaustive_od_lane_dataset(
                exhaustive_od_root=exhaustive_root.parent / "latest",
                aihub_canonical_root=lane_root,
                output_root=output_root,
                copy_images=True,
            )

            manifest = json.loads((output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
            dataset = PV26CanonicalDataset([output_root])

            self.assertEqual(len(dataset), manifest["sample_count"])
            record_by_id = {record.sample_id: record for record in dataset.records}
            self.assertEqual(sorted(record_by_id), ["lane", "od"])
            manifest_by_id = {str(row["final_sample_id"]): row for row in manifest["samples"]}
            self.assertEqual(
                manifest_by_id["od"]["source_scene_path"],
                str(exhaustive_root / "labels_scene" / "train" / "od.json"),
            )
            self.assertEqual(
                manifest_by_id["od"]["source_image_path"],
                str(exhaustive_root / "images" / "train" / "od_input.png"),
            )
            self.assertEqual(
                manifest_by_id["od"]["source_det_path"],
                str(exhaustive_root / "labels_det" / "train" / "od.txt"),
            )
            self.assertEqual(
                manifest_by_id["lane"]["source_scene_path"],
                str(lane_root / "labels_scene" / "train" / "lane.json"),
            )
            self.assertEqual(
                manifest_by_id["lane"]["source_image_path"],
                str(lane_root / "images" / "train" / "lane_source.png"),
            )
            self.assertIsNone(manifest_by_id["lane"]["source_det_path"])
            for row in manifest["samples"]:
                record = record_by_id[str(row["final_sample_id"])]
                self.assertEqual(str(record.scene_path), row["scene_path"])
                self.assertEqual(str(record.image_path), row["image_path"])
                self.assertEqual(str(record.det_path) if record.det_path is not None else None, row["det_path"])
                self.assertEqual(record.dataset_key, row["source_dataset_key"])
                self.assertEqual(record.split, row["split"])

            samples = [dataset[index] for index in range(len(dataset))]
            sample_by_id = {sample["meta"]["sample_id"]: sample for sample in samples}
            self.assertEqual(
                sample_by_id["od"]["meta"]["final_manifest_path"],
                str(output_root / "meta" / FINAL_DATASET_MANIFEST_NAME),
            )
            self.assertEqual(sample_by_id["od"]["meta"]["source_kind"], "exhaustive_od")
            self.assertEqual(
                sample_by_id["od"]["meta"]["source_scene_path"],
                str((exhaustive_root / "labels_scene" / "train" / "od.json").resolve()),
            )
            self.assertEqual(
                sample_by_id["od"]["meta"]["source_image_path"],
                str((exhaustive_root / "images" / "train" / "od_input.png").resolve()),
            )
            self.assertEqual(
                sample_by_id["od"]["meta"]["source_det_path"],
                str((exhaustive_root / "labels_det" / "train" / "od.txt").resolve()),
            )
            self.assertEqual(sample_by_id["lane"]["meta"]["source_kind"], "lane")
            self.assertEqual(
                sample_by_id["lane"]["meta"]["source_scene_path"],
                str((lane_root / "labels_scene" / "train" / "lane.json").resolve()),
            )
            self.assertEqual(
                sample_by_id["lane"]["meta"]["source_image_path"],
                str((lane_root / "images" / "train" / "lane_source.png").resolve()),
            )
            self.assertIsNone(sample_by_id["lane"]["meta"]["source_det_path"])
            batch = collate_pv26_encoded_eval_batch(samples)

            self.assertEqual(tuple(batch["image"].shape), (2, 3, *NETWORK_HW))
            self.assertEqual(batch["det_gt"]["boxes_xyxy"].shape[0], 2)
            self.assertEqual(batch["det_gt"]["boxes_xyxy"].shape[-1], 4)
            self.assertEqual(batch["lane"].shape[0], 2)
            self.assertEqual(batch["stop_line"].shape[0], 2)
            self.assertEqual(batch["crosswalk"].shape[0], 2)
            self.assertEqual(batch["mask"]["det_source"].tolist(), [False, True])
            self.assertEqual(batch["mask"]["lane_source"].tolist(), [True, False])
            self.assertEqual(batch["mask"]["stop_line_source"].tolist(), [True, False])
            self.assertEqual(batch["mask"]["crosswalk_source"].tolist(), [True, False])
            self.assertEqual(
                batch["mask"]["det_supervised_class_mask"][1].nonzero().flatten().tolist(),
                list(range(len(OD_CLASSES))),
            )
            self.assertGreater(int(batch["mask"]["lane_valid"][0].sum().item()), 0)
            self.assertGreater(int(batch["mask"]["stop_line_valid"][0].sum().item()), 0)
            self.assertGreater(int(batch["mask"]["crosswalk_valid"][0].sum().item()), 0)

            heads = PV26Heads(in_channels=(64, 64, 128, 256)).eval()
            head_summary = heads.describe()
            feature_batch_size = int(batch["image"].shape[0])
            features = [
                torch.zeros((feature_batch_size, 64, 152, 200), dtype=torch.float32),
                torch.zeros((feature_batch_size, 64, 76, 100), dtype=torch.float32),
                torch.zeros((feature_batch_size, 128, 38, 50), dtype=torch.float32),
                torch.zeros((feature_batch_size, 256, 19, 25), dtype=torch.float32),
            ]
            with torch.no_grad():
                head_outputs = heads(features)
            det_query_count = sum(height * width for height, width in head_outputs["det_feature_shapes"])

            self.assertEqual(tuple(batch["lane"].shape[1:]), (head_summary["lane_queries"], head_summary["lane_dim"]))
            self.assertEqual(
                tuple(batch["stop_line"].shape[1:]),
                (head_summary["stop_line_queries"], head_summary["stop_line_dim"]),
            )
            self.assertEqual(
                tuple(batch["crosswalk"].shape[1:]),
                (head_summary["crosswalk_queries"], head_summary["crosswalk_dim"]),
            )
            self.assertEqual(tuple(head_outputs["det"].shape), (feature_batch_size, det_query_count, head_summary["det_dim"]))
            self.assertEqual(
                tuple(head_outputs["tl_attr"].shape),
                (feature_batch_size, det_query_count, head_summary["tl_attr_dim"]),
            )
            self.assertEqual(
                tuple(head_outputs["lane"].shape),
                (feature_batch_size, head_summary["lane_queries"], head_summary["lane_dim"]),
            )
            self.assertEqual(
                tuple(head_outputs["stop_line"].shape),
                (feature_batch_size, head_summary["stop_line_queries"], head_summary["stop_line_dim"]),
            )
            self.assertEqual(
                tuple(head_outputs["crosswalk"].shape),
                (feature_batch_size, head_summary["crosswalk_queries"], head_summary["crosswalk_dim"]),
            )

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

    def test_build_pv26_exhaustive_od_lane_dataset_does_not_publish_lane_det_labels(self) -> None:
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
                        "lanes": [
                            {
                                "class_name": "white_lane",
                                "source_style": "solid",
                                "points": [[180.0, 470.0], [210.0, 340.0], [240.0, 220.0]],
                            }
                        ],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(lane_root / "labels_det" / "train" / "lane.txt", "0 0.5 0.5 0.1 0.1\n")

            build_pv26_exhaustive_od_lane_dataset(
                exhaustive_od_root=exhaustive_root.parent / "latest",
                aihub_canonical_root=lane_root,
                output_root=output_root,
                copy_images=True,
            )

            manifest = json.loads((output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
            lane_rows = [row for row in manifest["samples"] if row["source_dataset_key"] == "aihub_lane_seoul"]

            self.assertEqual(len(lane_rows), 1)
            self.assertIsNone(lane_rows[0]["det_path"])
            self.assertFalse((output_root / "labels_det" / "train" / "lane.txt").exists())

    def test_build_pv26_exhaustive_od_lane_dataset_rejects_missing_exhaustive_det_labels(self) -> None:
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

            with self.assertRaisesRegex(FileNotFoundError, "det label file not found for final dataset sample"):
                build_pv26_exhaustive_od_lane_dataset(
                    exhaustive_od_root=exhaustive_root.parent / "latest",
                    aihub_canonical_root=lane_root,
                    output_root=output_root,
                    copy_images=True,
                )

    def test_build_pv26_exhaustive_od_lane_dataset_rejects_scene_image_path_traversal(self) -> None:
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
                        "image": {"file_name": "../od_input.png", "width": 640, "height": 480},
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

            with self.assertRaisesRegex(ValueError, "scene image.file_name must be a basename"):
                build_pv26_exhaustive_od_lane_dataset(
                    exhaustive_od_root=exhaustive_root.parent / "latest",
                    aihub_canonical_root=lane_root,
                    output_root=output_root,
                    copy_images=True,
                )

    def test_build_pv26_exhaustive_od_lane_dataset_rejects_scene_split_mismatch(self) -> None:
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
                        "image": {"file_name": "od_input.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "pv26_exhaustive_bdd100k_det_100k",
                            "split": "val",
                            "bootstrap_sample_uid": "od",
                        },
                        "detections": [],
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            _write_text(exhaustive_root / "labels_det" / "train" / "od.txt", "")

            with self.assertRaisesRegex(ValueError, "scene source.split must match labels_scene split"):
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
