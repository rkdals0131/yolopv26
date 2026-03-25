from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from model.preprocess.aihub_standardize import run_standardization as run_aihub_standardization
from model.preprocess.bdd100k_standardize import run_standardization as run_bdd_standardization


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["convert", "-size", f"{width}x{height}", f"xc:{color}", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_dummy_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%stub\n")


class PV26LoaderTests(unittest.TestCase):
    def test_loader_returns_sample_contract_for_aihub_and_bdd_sources(self) -> None:
        from model.loading.pv26_loader import PV26CanonicalDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            traffic_root = root / "traffic"
            bdd_root = root / "BDD100K"
            aihub_output = root / "pv26_aihub_standardized"
            bdd_output = root / "pv26_bdd100k_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_traffic_fixture(traffic_root)
            self._create_bdd_fixture(
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
            )

            run_aihub_standardization(
                lane_root=lane_root,
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
            self.assertEqual(len(dataset), 6)

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
            self.assertFalse(traffic_sample["meta"]["det_allow_background_negatives"])
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
            self.assertFalse(lane_sample["meta"]["det_allow_background_negatives"])
            self.assertEqual(len(lane_sample["lane_targets"]["lanes"]), 2)
            self.assertEqual(len(lane_sample["lane_targets"]["stop_lines"]), 1)
            self.assertEqual(len(lane_sample["lane_targets"]["crosswalks"]), 1)
            self.assertEqual(lane_sample["valid_mask"]["lane"].tolist(), [True, True])
            self.assertEqual(lane_sample["valid_mask"]["stop_line"].tolist(), [True])
            self.assertEqual(lane_sample["valid_mask"]["crosswalk"].tolist(), [True])

            bdd_sample = keyed[("bdd100k_det_100k", "train")]
            self.assertEqual(bdd_sample["source_mask"]["det"], True)
            self.assertEqual(bdd_sample["source_mask"]["tl_attr"], False)
            self.assertEqual(bdd_sample["source_mask"]["lane"], False)
            self.assertEqual(bdd_sample["meta"]["det_supervised_classes"], ["vehicle", "bike", "pedestrian"])
            self.assertEqual(bdd_sample["meta"]["det_supervised_class_ids"], [0, 1, 2])
            self.assertFalse(bdd_sample["meta"]["det_allow_background_negatives"])
            self.assertEqual(tuple(bdd_sample["det_targets"]["boxes_xyxy"].shape), (1, 4))
            self.assertEqual(bdd_sample["valid_mask"]["tl_attr"].tolist(), [False])
            self.assertEqual(bdd_sample["tl_attr_targets"]["collapse_reason"][0], "not_traffic_light")
            self.assertEqual(bdd_sample["meta"]["raw_hw"], (720, 1280))
            self.assertEqual(bdd_sample["meta"]["transform"]["pad_top"], 79)

    def test_collate_stacks_images_and_preserves_ragged_targets(self) -> None:
        from model.loading.pv26_loader import PV26CanonicalDataset, collate_pv26_samples

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            traffic_root = root / "traffic"
            bdd_root = root / "BDD100K"
            aihub_output = root / "pv26_aihub_standardized"
            bdd_output = root / "pv26_bdd100k_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_traffic_fixture(traffic_root)
            self._create_bdd_fixture(
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
            )

            run_aihub_standardization(
                lane_root=lane_root,
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

    def _create_docs_fixture(self, docs_root: Path) -> None:
        _write_dummy_pdf(docs_root / "차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf")
        _write_dummy_pdf(docs_root / "수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf")

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
