from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from tools.od_bootstrap.source.aihub import run_standardization as run_aihub_standardization
from tools.od_bootstrap.source.bdd100k import run_standardization as run_bdd_standardization


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_dummy_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%stub\n")


def _rewrite_scene_dataset_keys(root: Path, mapping: dict[str, str]) -> None:
    for scene_path in (root / "labels_scene").rglob("*.json"):
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        source = scene.setdefault("source", {})
        dataset_key = str(source.get("dataset") or "")
        if dataset_key in mapping:
            source["dataset"] = mapping[dataset_key]
            scene_path.write_text(json.dumps(scene, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class PV26LoaderTests(unittest.TestCase):
    def test_loader_rejects_non_object_scene_roots(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_text(root / "labels_scene" / "train" / "bad.json", "[]\n")

            with self.assertRaisesRegex(TypeError, "scene root must be an object"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_missing_scene_image_file_name(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root / "labels_scene" / "train" / "bad.json",
                {
                    "image": {"width": 640, "height": 480},
                    "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                },
            )

            with self.assertRaisesRegex(ValueError, "scene image.file_name must not be empty"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_scene_image_file_name_path_traversal(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root / "labels_scene" / "train" / "bad.json",
                {
                    "image": {"file_name": "../outside.png", "width": 640, "height": 480},
                    "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                },
            )

            with self.assertRaisesRegex(ValueError, "scene image.file_name must be a basename"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_missing_scene_source_dataset(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root / "labels_scene" / "train" / "bad.json",
                {
                    "image": {"file_name": "bad.png", "width": 640, "height": 480},
                    "source": {"split": "train"},
                },
            )

            with self.assertRaisesRegex(ValueError, "scene source.dataset must not be empty"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_unsupported_scene_source_dataset(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root / "labels_scene" / "train" / "bad.json",
                {
                    "image": {"file_name": "bad.png", "width": 640, "height": 480},
                    "source": {"dataset": "unknown_dataset", "split": "train"},
                },
            )

            with self.assertRaisesRegex(KeyError, "unsupported dataset key for loader"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_scene_source_split_mismatch(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root / "labels_scene" / "train" / "bad.json",
                {
                    "image": {"file_name": "bad.png", "width": 640, "height": 480},
                    "source": {"dataset": "aihub_lane_seoul", "split": "val"},
                },
            )

            with self.assertRaisesRegex(ValueError, "scene source.split must match labels_scene split"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_final_dataset_manifest_record_drift(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sample_id in ("kept", "extra"):
                _make_image(root / "images" / "train" / f"{sample_id}.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / f"{sample_id}.json",
                    {
                        "image": {"file_name": f"{sample_id}.png", "width": 640, "height": 480},
                        "source": {
                            "dataset": "aihub_lane_seoul",
                            "split": "train",
                            "final_sample_id": sample_id,
                        },
                        "detections": [],
                        "lanes": [],
                        "stop_lines": [],
                        "crosswalks": [],
                    },
                )
            _write_json(
                root / "meta" / "final_dataset_manifest.json",
                {
                    "version": "pv26-exhaustive-od-lane-v2",
                    "sample_count": 1,
                    "dataset_counts": {"aihub_lane_seoul": 1},
                    "samples": [
                        {
                            "final_sample_id": "kept",
                            "source_kind": "lane",
                            "source_dataset_key": "aihub_lane_seoul",
                            "split": "train",
                            "source_scene_path": str(root / "source" / "kept.json"),
                            "source_image_path": str(root / "source" / "kept.png"),
                            "source_det_path": None,
                            "scene_path": str((root / "labels_scene" / "train" / "kept.json").resolve()),
                            "det_path": None,
                            "image_path": str((root / "images" / "train" / "kept.png").resolve()),
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "final dataset manifest samples must match discovered records"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_final_dataset_manifest_path_drift(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
            _write_json(
                root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.png", "width": 640, "height": 480},
                    "source": {
                        "dataset": "aihub_lane_seoul",
                        "split": "train",
                        "final_sample_id": "sample",
                    },
                    "detections": [],
                    "lanes": [],
                    "stop_lines": [],
                    "crosswalks": [],
                },
            )
            _write_json(
                root / "meta" / "final_dataset_manifest.json",
                {
                    "version": "pv26-exhaustive-od-lane-v2",
                    "sample_count": 1,
                    "dataset_counts": {"aihub_lane_seoul": 1},
                    "samples": [
                        {
                            "final_sample_id": "sample",
                            "source_kind": "lane",
                            "source_dataset_key": "aihub_lane_seoul",
                            "split": "train",
                            "source_scene_path": str(root / "source" / "sample.json"),
                            "source_image_path": str(root / "source" / "sample.png"),
                            "source_det_path": None,
                            "scene_path": str((root / ".staging" / "labels_scene" / "train" / "sample.json").resolve()),
                            "det_path": None,
                            "image_path": str((root / "images" / "train" / "sample.png").resolve()),
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "final dataset manifest scene_path must match discovered record"):
                PV26CanonicalDataset([root])

    def test_loader_rejects_scene_image_size_mismatch(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_image(root / "images" / "train" / "sample.png", 320, 240, "#202020")
            _write_json(
                root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.png", "width": 640, "height": 480},
                    "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                    "detections": [],
                    "lanes": [],
                    "stop_lines": [],
                    "crosswalks": [],
                },
            )

            dataset = PV26CanonicalDataset([root])
            with self.assertRaisesRegex(ValueError, "scene image size must match image file size"):
                dataset[0]

    def test_loader_rejects_invalid_scene_image_dimensions(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            ("missing_height", {"file_name": "sample.png", "width": 640}),
            ("zero_width", {"file_name": "sample.png", "width": 0, "height": 480}),
        ]
        for case_name, image_payload in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": image_payload,
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        "detections": [],
                        "lanes": [],
                        "stop_lines": [],
                        "crosswalks": [],
                    },
                )

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(ValueError, "scene image dimensions must be positive integers"):
                    dataset[0]

    def test_loader_rejects_malformed_scene_geometry_collections(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            ("non_list", {"lanes": {"points": []}}, ValueError, "scene lanes must be a list"),
            ("non_object_item", {"lanes": ["bad"]}, TypeError, "scene lanes\\[0\\] must be an object"),
        ]
        for case_name, geometry_payload, error_type, error_message in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": {"file_name": "sample.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        **geometry_payload,
                    },
                )

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(error_type, error_message):
                    dataset[0]

    def test_loader_rejects_malformed_scene_geometry_points(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            (
                "points_not_list",
                {"lanes": [{"points": "bad"}]},
                "scene lanes\\[0\\].points must be a list",
            ),
            (
                "wrong_point_arity",
                {"stop_lines": [{"points": [[1.0, 2.0, 3.0]]}]},
                "scene stop_lines\\[0\\].points\\[0\\] must be \\[x, y\\]",
            ),
            (
                "non_finite_point",
                {"crosswalks": [{"points": [[1.0, 2.0], [float("nan"), 3.0], [4.0, 5.0]]}]},
                "scene crosswalks\\[0\\].points\\[1\\] coordinates must be finite",
            ),
        ]
        for case_name, geometry_payload, error_message in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": {"file_name": "sample.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        **geometry_payload,
                    },
                )

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(ValueError, error_message):
                    dataset[0]

    def test_loader_rejects_malformed_lane_visibility(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            ("length_mismatch", [1], "scene lanes\\[0\\].visibility length must match points"),
            ("non_finite", [1, float("nan")], "scene lanes\\[0\\].visibility must be finite values"),
            ("wrong_type", "bad", "scene lanes\\[0\\].visibility must be a list"),
        ]
        for case_name, visibility, error_message in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": {"file_name": "sample.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                        "lanes": [
                            {
                                "class_name": "white_lane",
                                "source_style": "solid",
                                "points": [[10.0, 400.0], [120.0, 120.0]],
                                "visibility": visibility,
                            }
                        ],
                        "stop_lines": [],
                        "crosswalks": [],
                    },
                )

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(ValueError, error_message):
                    dataset[0]

    def test_loader_rejects_malformed_scene_traffic_light_payload(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            (
                "traffic_lights_not_list",
                {"traffic_lights": {"detection_id": 0}},
                ValueError,
                "scene traffic_lights must be a list",
            ),
            (
                "traffic_light_not_object",
                {"traffic_lights": ["bad"]},
                TypeError,
                "scene traffic_lights\\[0\\] must be an object",
            ),
            (
                "traffic_light_bad_detection_id",
                {
                    "traffic_lights": [
                        {
                            "detection_id": "bad",
                            "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
                            "tl_attr_valid": True,
                        }
                    ]
                },
                ValueError,
                "scene traffic_lights\\[0\\].detection_id must be a non-negative integer",
            ),
            (
                "traffic_light_non_finite_bit",
                {
                    "traffic_lights": [
                        {
                            "detection_id": 0,
                            "tl_bits": {"red": float("nan"), "yellow": 0, "green": 0, "arrow": 1},
                            "tl_attr_valid": True,
                        }
                    ]
                },
                ValueError,
                "scene traffic_lights\\[0\\].tl_bits.red must be finite 0/1",
            ),
        ]
        for case_name, traffic_payload, error_type, error_message in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": {"file_name": "sample.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_traffic_seoul", "split": "train"},
                        "lanes": [],
                        "stop_lines": [],
                        "crosswalks": [],
                        **traffic_payload,
                    },
                )
                _write_text(root / "labels_det" / "train" / "sample.txt", "5 0.5 0.5 0.1 0.1\n")

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(error_type, error_message):
                    dataset[0]

    def test_loader_rejects_traffic_light_detection_id_drift(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        malformed_cases = [
            (
                "duplicate_detection_id",
                [
                    {
                        "detection_id": 0,
                        "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 0},
                        "tl_attr_valid": True,
                    },
                    {
                        "detection_id": 0,
                        "tl_bits": {"red": 0, "yellow": 1, "green": 0, "arrow": 0},
                        "tl_attr_valid": True,
                    },
                ],
                "scene traffic_lights detection_id must be unique",
            ),
            (
                "out_of_range_detection_id",
                [
                    {
                        "detection_id": 1,
                        "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 0},
                        "tl_attr_valid": True,
                    }
                ],
                "scene traffic_lights detection_id must reference a detection row",
            ),
        ]
        for case_name, traffic_lights, error_message in malformed_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
                _write_json(
                    root / "labels_scene" / "train" / "sample.json",
                    {
                        "image": {"file_name": "sample.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_traffic_seoul", "split": "train"},
                        "lanes": [],
                        "stop_lines": [],
                        "crosswalks": [],
                        "traffic_lights": traffic_lights,
                    },
                )
                _write_text(root / "labels_det" / "train" / "sample.txt", "5 0.5 0.5 0.1 0.1\n")

                dataset = PV26CanonicalDataset([root])
                with self.assertRaisesRegex(ValueError, error_message):
                    dataset[0]

    def test_loader_uses_labels_det_not_scene_detections_for_detector_targets(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_image(root / "images" / "train" / "sample.png", 640, 480, "#202020")
            _write_json(
                root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.png", "width": 640, "height": 480},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    "detections": [
                        {
                            "id": 0,
                            "class_name": "traffic_light",
                            "bbox": [1.0, 1.0, 2.0, 2.0],
                        }
                    ],
                    "lanes": [],
                    "stop_lines": [],
                    "crosswalks": [],
                    "traffic_lights": [],
                },
            )
            _write_text(root / "labels_det" / "train" / "sample.txt", "0 0.5 0.5 0.25 0.5\n")

            sample = PV26CanonicalDataset([root])[0]

            self.assertEqual(sample["det_targets"]["classes"].tolist(), [0])
            self.assertEqual(sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian"])
            self.assertFalse(sample["tl_attr_targets"]["is_traffic_light"][0].item())
            self.assertEqual(sample["tl_attr_targets"]["collapse_reason"][0], "not_traffic_light")
            box = sample["det_targets"]["boxes_xyxy"][0].tolist()
            self.assertAlmostEqual(box[0], 300.0, places=4)
            self.assertAlmostEqual(box[1], 154.0, places=4)
            self.assertAlmostEqual(box[2], 500.0, places=4)
            self.assertAlmostEqual(box[3], 454.0, places=4)

    def test_loader_returns_sample_contract_for_aihub_and_bdd_sources(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            bdd_root = root / "BDD100K"
            aihub_output = root / "pv26_aihub_standardized"
            bdd_output = root / "pv26_bdd100k_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)
            self._create_bdd_fixture(
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
            )

            run_aihub_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=aihub_output,
                workers=1,
                debug_vis_count=0,
            )
            run_bdd_standardization(
                bdd_root=bdd_root,
                images_root=bdd_root / "bdd100k_images_100k" / "100k",
                labels_root=bdd_root / "bdd100k_labels" / "100k",
                output_root=bdd_output,
                workers=1,
                debug_vis_count=0,
            )

            dataset = PV26CanonicalDataset([aihub_output, bdd_output])
            self.assertEqual(len(dataset), 8)

            samples = [dataset[index] for index in range(len(dataset))]
            keyed = {(item["meta"]["dataset_key"], item["meta"]["split"]): item for item in samples}

            traffic_sample = keyed[("aihub_traffic_seoul", "train")]
            self.assertEqual(tuple(traffic_sample["image"].shape), (3, 608, 800))
            self.assertEqual(traffic_sample["image"].dtype, torch.float32)
            self.assertEqual(traffic_sample["meta"]["network_hw"], (608, 800))
            self.assertEqual(traffic_sample["meta"]["raw_hw"], (1080, 1920))
            self.assertAlmostEqual(traffic_sample["meta"]["transform"]["scale"], 800.0 / 1920.0, places=6)
            self.assertEqual(traffic_sample["meta"]["transform"]["pad_top"], 79)
            self.assertEqual(traffic_sample["source_mask"]["det"], True)
            self.assertEqual(traffic_sample["source_mask"]["tl_attr"], True)
            self.assertEqual(traffic_sample["source_mask"]["lane"], False)
            self.assertEqual(traffic_sample["meta"]["det_supervised_classes"], ["traffic_light", "sign"])
            self.assertEqual(traffic_sample["meta"]["det_supervised_class_ids"], [5, 6])
            self.assertFalse(traffic_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(traffic_sample["meta"]["det_allow_unmatched_class_negatives"])
            self.assertEqual(tuple(traffic_sample["det_targets"]["boxes_xyxy"].shape), (4, 4))
            self.assertEqual(tuple(traffic_sample["tl_attr_targets"]["bits"].shape), (4, 4))
            self.assertEqual(traffic_sample["valid_mask"]["tl_attr"].tolist(), [True, True, False, False])
            first_box = traffic_sample["det_targets"]["boxes_xyxy"][0].tolist()
            self.assertAlmostEqual(first_box[0], 50.0, places=4)
            self.assertAlmostEqual(first_box[1], 112.3333, places=3)
            self.assertAlmostEqual(first_box[2], 75.0, places=4)
            self.assertAlmostEqual(first_box[3], 179.0, places=3)
            self.assertEqual(traffic_sample["tl_attr_targets"]["bits"][0].tolist(), [1.0, 0.0, 0.0, 1.0])
            self.assertEqual(traffic_sample["tl_attr_targets"]["collapse_reason"][0], "valid")

            lane_sample = keyed[("aihub_lane_seoul", "train")]
            self.assertEqual(lane_sample["source_mask"]["det"], False)
            self.assertEqual(lane_sample["source_mask"]["lane"], True)
            self.assertEqual(lane_sample["meta"]["det_supervised_classes"], [])
            self.assertEqual(lane_sample["meta"]["det_supervised_class_ids"], [])
            self.assertFalse(lane_sample["meta"]["det_allow_objectness_negatives"])
            self.assertFalse(lane_sample["meta"]["det_allow_unmatched_class_negatives"])
            self.assertEqual(len(lane_sample["lane_targets"]["lanes"]), 2)
            self.assertEqual(len(lane_sample["lane_targets"]["stop_lines"]), 1)
            self.assertEqual(len(lane_sample["lane_targets"]["crosswalks"]), 1)
            self.assertEqual(lane_sample["valid_mask"]["lane"].tolist(), [True, True])
            self.assertEqual(lane_sample["valid_mask"]["stop_line"].tolist(), [True])
            self.assertEqual(lane_sample["valid_mask"]["crosswalk"].tolist(), [True])
            self.assertTrue(torch.all(lane_sample["lane_targets"]["lanes"][0]["visibility"] == 1.0))
            self.assertTrue(torch.all(lane_sample["lane_targets"]["lanes"][1]["visibility"] == 1.0))

            obstacle_sample = keyed[("aihub_obstacle_seoul", "train")]
            self.assertEqual(obstacle_sample["source_mask"]["det"], True)
            self.assertEqual(obstacle_sample["source_mask"]["tl_attr"], False)
            self.assertEqual(obstacle_sample["source_mask"]["lane"], False)
            self.assertEqual(obstacle_sample["meta"]["det_supervised_classes"], ["traffic_cone", "obstacle"])
            self.assertEqual(obstacle_sample["meta"]["det_supervised_class_ids"], [3, 4])
            self.assertFalse(obstacle_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(obstacle_sample["meta"]["det_allow_unmatched_class_negatives"])
            self.assertEqual(tuple(obstacle_sample["det_targets"]["boxes_xyxy"].shape), (2, 4))
            self.assertEqual(obstacle_sample["valid_mask"]["tl_attr"].tolist(), [False, False])
            obstacle_first_box = obstacle_sample["det_targets"]["boxes_xyxy"][0].tolist()
            self.assertAlmostEqual(obstacle_first_box[0], 43.75, places=3)
            self.assertAlmostEqual(obstacle_first_box[1], 140.25, places=2)
            self.assertAlmostEqual(obstacle_first_box[2], 100.625, places=3)
            self.assertAlmostEqual(obstacle_first_box[3], 257.125, places=3)

            bdd_sample = keyed[("bdd100k_det_100k", "train")]
            self.assertEqual(bdd_sample["source_mask"]["det"], True)
            self.assertEqual(bdd_sample["source_mask"]["tl_attr"], False)
            self.assertEqual(bdd_sample["source_mask"]["lane"], False)
            self.assertEqual(bdd_sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian"])
            self.assertEqual(bdd_sample["meta"]["det_supervised_class_ids"], [0, 1, 2])
            self.assertFalse(bdd_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(bdd_sample["meta"]["det_allow_unmatched_class_negatives"])
            self.assertEqual(tuple(bdd_sample["det_targets"]["boxes_xyxy"].shape), (1, 4))
            self.assertEqual(bdd_sample["valid_mask"]["tl_attr"].tolist(), [False])
            self.assertEqual(bdd_sample["tl_attr_targets"]["collapse_reason"][0], "not_traffic_light")
            self.assertEqual(bdd_sample["meta"]["raw_hw"], (720, 1280))
            self.assertEqual(bdd_sample["meta"]["transform"]["pad_top"], 79)

    def test_loader_supports_exhaustive_od_dataset_keys(self) -> None:
        from model.data.dataset import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            bdd_root = root / "BDD100K"
            aihub_output = root / "pv26_aihub_standardized"
            bdd_output = root / "pv26_bdd100k_standardized"
            exhaustive_aihub_output = root / "pv26_exhaustive_aihub_standardized"
            exhaustive_bdd_output = root / "pv26_exhaustive_bdd100k_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)
            self._create_bdd_fixture(
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
            )

            run_aihub_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=aihub_output,
                workers=1,
                debug_vis_count=0,
            )
            run_bdd_standardization(
                bdd_root=bdd_root,
                images_root=bdd_root / "bdd100k_images_100k" / "100k",
                labels_root=bdd_root / "bdd100k_labels" / "100k",
                output_root=bdd_output,
                workers=1,
                debug_vis_count=0,
            )

            shutil.copytree(aihub_output, exhaustive_aihub_output)
            shutil.copytree(bdd_output, exhaustive_bdd_output)
            _rewrite_scene_dataset_keys(
                exhaustive_aihub_output,
                {
                    "aihub_traffic_seoul": "pv26_exhaustive_aihub_traffic_seoul",
                    "aihub_obstacle_seoul": "pv26_exhaustive_aihub_obstacle_seoul",
                },
            )
            _rewrite_scene_dataset_keys(
                exhaustive_bdd_output,
                {"bdd100k_det_100k": "pv26_exhaustive_bdd100k_det_100k"},
            )

            dataset = PV26CanonicalDataset([exhaustive_aihub_output, exhaustive_bdd_output])
            keyed = {(item["meta"]["dataset_key"], item["meta"]["split"]): item for item in (dataset[index] for index in range(len(dataset)))}

            traffic_sample = keyed[("pv26_exhaustive_aihub_traffic_seoul", "train")]
            self.assertEqual(traffic_sample["source_mask"]["det"], True)
            self.assertEqual(traffic_sample["source_mask"]["tl_attr"], True)
            self.assertEqual(traffic_sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"])
            self.assertEqual(traffic_sample["meta"]["det_supervised_class_ids"], [0, 1, 2, 3, 4, 5, 6])
            self.assertTrue(traffic_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(traffic_sample["meta"]["det_allow_unmatched_class_negatives"])

            obstacle_sample = keyed[("pv26_exhaustive_aihub_obstacle_seoul", "train")]
            self.assertEqual(obstacle_sample["source_mask"]["det"], True)
            self.assertEqual(obstacle_sample["source_mask"]["tl_attr"], False)
            self.assertEqual(obstacle_sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"])
            self.assertEqual(obstacle_sample["meta"]["det_supervised_class_ids"], [0, 1, 2, 3, 4, 5, 6])
            self.assertTrue(obstacle_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(obstacle_sample["meta"]["det_allow_unmatched_class_negatives"])

            bdd_sample = keyed[("pv26_exhaustive_bdd100k_det_100k", "train")]
            self.assertEqual(bdd_sample["source_mask"]["det"], True)
            self.assertEqual(bdd_sample["source_mask"]["tl_attr"], False)
            self.assertEqual(bdd_sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"])
            self.assertEqual(bdd_sample["meta"]["det_supervised_class_ids"], [0, 1, 2, 3, 4, 5, 6])
            self.assertTrue(bdd_sample["meta"]["det_allow_objectness_negatives"])
            self.assertTrue(bdd_sample["meta"]["det_allow_unmatched_class_negatives"])

            lane_sample = keyed[("aihub_lane_seoul", "train")]
            self.assertEqual(lane_sample["source_mask"]["det"], False)
            self.assertEqual(lane_sample["source_mask"]["lane"], True)
            self.assertEqual(lane_sample["meta"]["det_supervised_classes"], [])
            self.assertEqual(lane_sample["meta"]["det_supervised_class_ids"], [])
            self.assertFalse(lane_sample["meta"]["det_allow_objectness_negatives"])
            self.assertFalse(lane_sample["meta"]["det_allow_unmatched_class_negatives"])

    def test_collate_stacks_images_and_preserves_ragged_targets(self) -> None:
        from model.data.dataset import PV26CanonicalDataset, collate_pv26_samples

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            bdd_root = root / "BDD100K"
            aihub_output = root / "pv26_aihub_standardized"
            bdd_output = root / "pv26_bdd100k_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)
            self._create_bdd_fixture(
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
            )

            run_aihub_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=aihub_output,
                workers=1,
                debug_vis_count=0,
            )
            run_bdd_standardization(
                bdd_root=bdd_root,
                images_root=bdd_root / "bdd100k_images_100k" / "100k",
                labels_root=bdd_root / "bdd100k_labels" / "100k",
                output_root=bdd_output,
                workers=1,
                debug_vis_count=0,
            )

            dataset = PV26CanonicalDataset([aihub_output, bdd_output])
            batch = collate_pv26_samples([dataset[0], dataset[1], dataset[2]])

            self.assertEqual(tuple(batch["image"].shape), (3, 3, 608, 800))
            self.assertEqual(len(batch["det_targets"]), 3)
            self.assertEqual(len(batch["tl_attr_targets"]), 3)
            self.assertEqual(len(batch["lane_targets"]), 3)
            self.assertEqual(len(batch["source_mask"]), 3)
            self.assertEqual(len(batch["valid_mask"]), 3)
            self.assertEqual(len(batch["meta"]), 3)

    def test_collate_rejects_empty_sample_list(self) -> None:
        from model.data.dataset import collate_pv26_samples

        with self.assertRaisesRegex(ValueError, "cannot collate zero PV26 samples"):
            collate_pv26_samples([])

    def test_collate_rejects_sample_image_shape_drift(self) -> None:
        from model.data.dataset import collate_pv26_samples

        sample = self._minimal_collate_sample()
        sample["image"] = torch.zeros((3, 607, 800), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "PV26 sample image must be float32"):
            collate_pv26_samples([sample])

    def test_collate_rejects_sample_image_dtype_drift(self) -> None:
        from model.data.dataset import collate_pv26_samples

        sample = self._minimal_collate_sample()
        sample["image"] = torch.zeros((3, 608, 800), dtype=torch.uint8)

        with self.assertRaisesRegex(ValueError, "PV26 sample image must be float32"):
            collate_pv26_samples([sample])

    def _minimal_collate_sample(self) -> dict:
        empty_bool = torch.zeros((0,), dtype=torch.bool)
        return {
            "image": torch.zeros((3, 608, 800), dtype=torch.float32),
            "det_targets": {
                "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
                "classes": torch.zeros((0,), dtype=torch.long),
            },
            "tl_attr_targets": {
                "bits": torch.zeros((0, 4), dtype=torch.float32),
                "is_traffic_light": empty_bool,
                "collapse_reason": [],
            },
            "lane_targets": {"lanes": [], "stop_lines": [], "crosswalks": []},
            "source_mask": {
                "det": False,
                "tl_attr": False,
                "lane": False,
                "stop_line": False,
                "crosswalk": False,
            },
            "valid_mask": {
                "det": empty_bool,
                "tl_attr": empty_bool,
                "lane": empty_bool,
                "stop_line": empty_bool,
                "crosswalk": empty_bool,
            },
            "meta": {"sample_id": "synthetic_sample"},
        }

    def _create_docs_fixture(self, docs_root: Path) -> None:
        _write_dummy_pdf(docs_root / "차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf")
        _write_dummy_pdf(docs_root / "수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf")

    def _create_obstacle_fixture(self, obstacle_root: Path) -> None:
        train_image = obstacle_root / "Training" / "Images" / "TOA" / "1.Frontback_A01" / "obstacle_train_001.png"
        val_image = obstacle_root / "Validation" / "Images" / "TOA" / "1.Frontback_F01" / "obstacle_val_001.png"
        train_label = obstacle_root / "Training" / "Annotations" / "TOA" / "1.Frontback_A01" / "obstacle_train_001_BBOX.json"
        val_label = obstacle_root / "Validation" / "Annotations" / "TOA" / "1.Frontback_F01" / "obstacle_val_001_BBOX.json"

        _make_image(train_image, 1280, 720, "#303030")
        _make_image(val_image, 1280, 720, "#505050")
        categories = [
            {"id": 1, "name": "Animals(Dolls)"},
            {"id": 2, "name": "Person"},
            {"id": 3, "name": "Garbage bag & sacks"},
            {"id": 4, "name": "Construction signs & Parking prohibited board"},
            {"id": 5, "name": "Traffic cone"},
            {"id": 6, "name": "Box"},
            {"id": 7, "name": "Stones on road"},
            {"id": 8, "name": "Pothole on road"},
            {"id": 9, "name": "Filled pothole"},
            {"id": 10, "name": "Manhole"},
        ]
        _write_json(
            train_label,
            {
                "images": {"file_name": "obstacle_train_001.png", "width": 1280, "height": 720, "id": 1},
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [70.0, 98.0, 91.0, 187.0], "category_id": 5},
                    {"id": 2, "image_id": 1, "bbox": [802.0, 181.0, 21.0, 46.0], "category_id": 6},
                    {"id": 3, "image_id": 1, "bbox": [420.0, 250.0, 48.0, 130.0], "category_id": 2},
                ],
                "categories": categories,
            },
        )
        _write_json(
            val_label,
            {
                "images": {"file_name": "obstacle_val_001.png", "width": 1280, "height": 720, "id": 2},
                "annotations": [
                    {"id": 1, "image_id": 2, "bbox": [180.0, 220.0, 60.0, 80.0], "category_id": 3},
                    {"id": 2, "image_id": 2, "bbox": [620.0, 260.0, 120.0, 90.0], "category_id": 4},
                    {"id": 3, "image_id": 2, "bbox": [920.0, 310.0, 100.0, 70.0], "category_id": 9},
                ],
                "categories": categories,
            },
        )

    def _create_lane_fixture(self, lane_root: Path) -> None:
        image_path = lane_root / "Training" / "[원천]c_lane_train_1" / "c_lane_train_1" / "lane_train_001.jpg"
        label_path = lane_root / "Training" / "[라벨]c_lane_train_1" / "lane_train_001.json"

        _make_image(image_path, 1280, 720, "#202020")
        _write_json(
            label_path,
            {
                "image": {"file_name": "lane_train_001.jpg", "image_size": [720, 1280]},
                "annotations": [
                    {
                        "class": "traffic_lane",
                        "attributes": [
                            {"code": "lane_color", "value": "white"},
                            {"code": "lane_type", "value": "solid"},
                        ],
                        "category": "polyline",
                        "data": [{"x": 220, "y": 690}, {"x": 240, "y": 520}, {"x": 260, "y": 360}],
                    },
                    {
                        "class": "traffic_lane",
                        "attributes": [
                            {"code": "lane_color", "value": "blue"},
                            {"code": "lane_type", "value": "dotted"},
                        ],
                        "category": "polyline",
                        "data": [{"x": 980, "y": 700}, {"x": 960, "y": 540}, {"x": 940, "y": 380}],
                    },
                    {
                        "class": "stop_line",
                        "attributes": [],
                        "category": "polyline",
                        "data": [{"x": 260, "y": 620}, {"x": 1000, "y": 620}],
                    },
                    {
                        "class": "crosswalk",
                        "attributes": [],
                        "category": "polygon",
                        "data": [
                            {"x": 330, "y": 650},
                            {"x": 470, "y": 650},
                            {"x": 500, "y": 710},
                            {"x": 300, "y": 710},
                        ],
                    },
                ],
            },
        )

    def _create_traffic_fixture(self, traffic_root: Path) -> None:
        train_image = traffic_root / "Training" / "[원천]c_train_1" / "traffic_train_001.jpg"
        train_label = traffic_root / "Training" / "[라벨]c_train_1" / "c_train_1" / "traffic_train_001.json"
        val_image = traffic_root / "Validation" / "[원천]c_val_1" / "traffic_val_001.jpg"
        val_label = traffic_root / "Validation" / "[라벨]c_val_1" / "c_val_1" / "traffic_val_001.json"
        crop_image = traffic_root / "Training" / "표지판코드분류crop데이터1" / "result_1" / "crop_only.jpg"

        _make_image(train_image, 1920, 1080, "#101010")
        _make_image(val_image, 1920, 1080, "#404040")
        _make_image(crop_image, 80, 80, "#808080")

        _write_json(
            train_label,
            {
                "image": {"filename": "traffic_train_001.jpg", "imsize": {"width": 1920, "height": 1080}},
                "annotation": [
                    {
                        "class": "traffic_light",
                        "box": [120, 80, 180, 240],
                        "light_count": 3,
                        "attribute": [{"red": "on", "green": "off", "yellow": "off", "left_arrow": "on"}],
                        "type": "car",
                        "direction": "horizontal",
                    },
                    {
                        "class": "traffic_light",
                        "box": [240, 90, 300, 250],
                        "light_count": 1,
                        "attribute": [{"red": "off", "green": "off", "yellow": "off", "others_arrow": "on"}],
                        "type": "car",
                        "direction": "vertical",
                    },
                    {
                        "class": "traffic_light",
                        "box": [360, 90, 420, 250],
                        "light_count": 2,
                        "attribute": [{"red": "on", "green": "off", "yellow": "off"}],
                        "type": "pedestrian",
                    },
                    {
                        "class": "traffic_sign",
                        "box": {"x1": 480, "y1": 240, "x2": 560, "y2": 320},
                        "shape": "triangle",
                        "color": "yellow",
                        "kind": "normal",
                        "type": "warning",
                        "text": 30,
                    },
                    {
                        "class": "traffic_information",
                        "box": [900, 400, 980, 480],
                        "type": "construction",
                    },
                ],
            },
        )
        _write_json(
            val_label,
            {
                "image": {"filename": "traffic_val_001.jpg", "imsize": {"width": 1920, "height": 1080}},
                "annotation": [
                    {
                        "class": "traffic_light",
                        "box": [120, 80, 180, 240],
                        "light_count": 3,
                        "attribute": [{"red": "off", "green": "off", "yellow": "off"}],
                        "type": "car",
                    },
                    {
                        "class": "traffic_light",
                        "box": [240, 90, 300, 250],
                        "light_count": 3,
                        "attribute": [{"red": "on", "green": "off", "yellow": "on"}],
                        "type": "car",
                    },
                ],
            },
        )

    def _create_bdd_fixture(self, images_root: Path, labels_root: Path) -> None:
        samples = {
            "train": ("5bf43587-94432457", "#222222"),
            "val": ("c4dbd719-26df8369", "#444444"),
            "test": ("e301d643-216af5d9", "#666666"),
        }
        for split, (stem, color) in samples.items():
            _make_image(images_root / split / f"{stem}.jpg", 1280, 720, color)

        _write_json(
            labels_root / "train" / "5bf43587-94432457.json",
            {
                "name": "5bf43587-94432457.jpg",
                "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
                "frames": [
                    {
                        "timestamp": 1000,
                        "objects": [
                            {
                                "id": 1,
                                "category": "car",
                                "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                                "attributes": {"occluded": False, "truncated": False, "trafficLightColor": "none"},
                            },
                            {
                                "id": 2,
                                "category": "traffic light",
                                "box2d": {"x1": 500, "y1": 100, "x2": 540, "y2": 200},
                                "attributes": {"trafficLightColor": "red"},
                            },
                            {
                                "id": 3,
                                "category": "traffic sign",
                                "box2d": {"x1": 700, "y1": 180, "x2": 760, "y2": 260},
                                "attributes": {"occluded": False},
                            },
                            {
                                "id": 4,
                                "category": "lane/single white",
                                "box2d": None,
                            },
                        ],
                    }
                ],
            },
        )
        _write_json(
            labels_root / "val" / "c4dbd719-26df8369.json",
            {
                "name": "c4dbd719-26df8369.jpg",
                "attributes": {"weather": "partly cloudy", "scene": "highway", "timeofday": "daytime"},
                "frames": [
                    {
                        "timestamp": 2000,
                        "objects": [
                            {
                                "id": 1,
                                "category": "truck",
                                "box2d": {"x1": 140, "y1": 240, "x2": 420, "y2": 520},
                                "attributes": {"occluded": False},
                            },
                            {
                                "id": 2,
                                "category": "person",
                                "box2d": {"x1": 860, "y1": 240, "x2": 900, "y2": 380},
                                "attributes": {"occluded": False},
                            },
                            {
                                "id": 3,
                                "category": "bike",
                                "box2d": {"x1": 940, "y1": 300, "x2": 1020, "y2": 420},
                                "attributes": {"occluded": False},
                            },
                        ],
                    }
                ],
            },
        )
        _write_json(
            labels_root / "test" / "e301d643-216af5d9.json",
            {
                "name": "e301d643-216af5d9.jpg",
                "attributes": {"weather": "night", "scene": "residential", "timeofday": "night"},
                "frames": [
                    {
                        "timestamp": 3000,
                        "objects": [
                            {
                                "id": 1,
                                "category": "bus",
                                "box2d": {"x1": 180, "y1": 250, "x2": 480, "y2": 560},
                                "attributes": {"occluded": True},
                            },
                            {
                                "id": 2,
                                "category": "rider",
                                "box2d": {"x1": 840, "y1": 280, "x2": 900, "y2": 430},
                                "attributes": {"occluded": False},
                            },
                            {
                                "id": 3,
                                "category": "motor",
                                "box2d": {"x1": 920, "y1": 330, "x2": 1040, "y2": 470},
                                "attributes": {"occluded": False},
                            },
                        ],
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
