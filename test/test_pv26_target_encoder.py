from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from model.loading import PV26CanonicalDataset, collate_pv26_samples
from model.preprocess.aihub_standardize import LANE_CLASSES, LANE_TYPES
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


class PV26TargetEncoderTests(unittest.TestCase):
    def test_encode_batch_builds_fixed_shape_targets(self) -> None:
        from model.encoding import encode_pv26_batch

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._build_dataset(Path(temp_dir))
            selected = self._select_mixed_task_samples(dataset)

            batch = collate_pv26_samples(selected)
            encoded = encode_pv26_batch(batch)

            self.assertEqual(tuple(encoded["image"].shape), (3, 3, 608, 800))
            self.assertEqual(tuple(encoded["det_gt"]["boxes_xyxy"].shape), (3, 4, 4))
            self.assertEqual(tuple(encoded["det_gt"]["classes"].shape), (3, 4))
            self.assertEqual(tuple(encoded["det_gt"]["valid_mask"].shape), (3, 4))
            self.assertEqual(tuple(encoded["tl_attr_gt_bits"].shape), (3, 4, 4))
            self.assertEqual(tuple(encoded["tl_attr_gt_mask"].shape), (3, 4))
            self.assertEqual(tuple(encoded["lane"].shape), (3, 12, 54))
            self.assertEqual(tuple(encoded["stop_line"].shape), (3, 6, 9))
            self.assertEqual(tuple(encoded["crosswalk"].shape), (3, 4, 17))

            keyed_index = {
                (item["dataset_key"], item["split"]): index
                for index, item in enumerate(encoded["meta"])
            }
            traffic_index = keyed_index[("aihub_traffic_seoul", "train")]
            lane_index = keyed_index[("aihub_lane_seoul", "train")]
            bdd_index = keyed_index[("bdd100k_det_100k", "train")]

            self.assertEqual(
                encoded["tl_attr_gt_bits"][traffic_index, 0].tolist(),
                [1.0, 0.0, 0.0, 1.0],
            )
            self.assertEqual(
                encoded["tl_attr_gt_mask"][traffic_index].tolist(),
                [True, True, False, False],
            )
            self.assertEqual(
                encoded["det_gt"]["valid_mask"][lane_index].tolist(),
                [False, False, False, False],
            )
            self.assertEqual(
                encoded["tl_attr_gt_mask"][bdd_index].tolist(),
                [False, False, False, False],
            )

            first_lane = encoded["lane"][lane_index, 0]
            self.assertEqual(float(first_lane[0]), 1.0)
            self.assertEqual(first_lane[1:4].tolist().count(1.0), 1)
            self.assertEqual(first_lane[4:6].tolist().count(1.0), 1)
            self.assertTrue(torch.all(first_lane[38:54] == 1.0))
            self.assertEqual(float(encoded["stop_line"][lane_index, 0, 0]), 1.0)
            self.assertEqual(float(encoded["crosswalk"][lane_index, 0, 0]), 1.0)

            self.assertEqual(encoded["mask"]["det_source"].tolist(), [False, True, True])
            self.assertEqual(encoded["mask"]["lane_source"].tolist(), [True, False, False])

    def test_encode_batch_pads_unused_queries_with_zeros(self) -> None:
        from model.encoding import encode_pv26_batch

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._build_dataset(Path(temp_dir))
            selected = self._select_mixed_task_samples(dataset)

            encoded = encode_pv26_batch(collate_pv26_samples(selected))
            keyed_index = {
                (item["dataset_key"], item["split"]): index
                for index, item in enumerate(encoded["meta"])
            }
            lane_index = keyed_index[("aihub_lane_seoul", "train")]

            self.assertEqual(encoded["mask"]["lane_valid"][lane_index][:2].tolist(), [True, True])
            self.assertEqual(encoded["mask"]["lane_valid"][lane_index][2:].tolist(), [False] * 10)
            self.assertTrue(torch.all(encoded["lane"][lane_index, 2:, :] == 0.0))
            self.assertEqual(encoded["mask"]["stop_line_valid"][lane_index].tolist(), [True, False, False, False, False, False])
            self.assertEqual(encoded["mask"]["crosswalk_valid"][lane_index].tolist(), [True, False, False, False])
            self.assertTrue(torch.all(encoded["stop_line"][lane_index, 1:, :] == 0.0))
            self.assertTrue(torch.all(encoded["crosswalk"][lane_index, 1:, :] == 0.0))

    def _build_dataset(self, root: Path) -> PV26CanonicalDataset:
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
        return PV26CanonicalDataset([aihub_output, bdd_output])

    def _select_mixed_task_samples(self, dataset: PV26CanonicalDataset) -> list[dict]:
        samples = [dataset[index] for index in range(len(dataset))]
        keyed = {(item["meta"]["dataset_key"], item["meta"]["split"]): item for item in samples}
        return [
            keyed[("aihub_lane_seoul", "train")],
            keyed[("aihub_traffic_seoul", "train")],
            keyed[("bdd100k_det_100k", "train")],
        ]

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
                        "type": "car",
                        "attribute": {"red": "on", "yellow": "off", "green": "off", "left_arrow": "on"},
                    },
                    {
                        "class": "traffic_light",
                        "box": [260, 120, 320, 280],
                        "light_count": 3,
                        "type": "car",
                        "attribute": {"red": "off", "yellow": "off", "green": "on", "others_arrow": "off"},
                    },
                    {
                        "class": "traffic_light",
                        "box": [400, 130, 460, 290],
                        "light_count": 3,
                        "type": "pedestrian",
                        "attribute": {"red": "on", "yellow": "off", "green": "off"},
                    },
                    {"class": "traffic_sign", "box": [620, 140, 740, 300]},
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
                        "box": [140, 90, 200, 250],
                        "light_count": 3,
                        "type": "car",
                        "attribute": {"red": "off", "yellow": "on", "green": "off"},
                    },
                    {"class": "traffic_sign", "box": [660, 160, 780, 310]},
                ],
            },
        )

    def _create_bdd_fixture(self, images_root: Path, labels_root: Path) -> None:
        _make_image(images_root / "train" / "bdd_train_001.jpg", 1280, 720, "#222244")
        _make_image(images_root / "val" / "bdd_val_001.jpg", 1280, 720, "#444422")
        _make_image(images_root / "test" / "bdd_test_001.jpg", 1280, 720, "#224422")

        train_label = {
            "name": "bdd_train_001.jpg",
            "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "city street"},
            "frames": [
                {
                    "objects": [
                        {"category": "car", "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
                        {
                            "category": "traffic light",
                            "box2d": {"x1": 500, "y1": 120, "x2": 560, "y2": 260},
                            "attributes": {"trafficLightColor": "green"},
                        },
                        {"category": "traffic sign", "box2d": {"x1": 700, "y1": 160, "x2": 780, "y2": 280}},
                        {"category": "lane", "poly2d": [{"vertices": [[0, 0], [100, 100]], "closed": False}]},
                    ]
                }
            ],
        }
        val_label = {
            "name": "bdd_val_001.jpg",
            "attributes": {"weather": "overcast", "timeofday": "daytime", "scene": "highway"},
            "frames": [
                {
                    "objects": [
                        {"category": "truck", "box2d": {"x1": 140, "y1": 220, "x2": 380, "y2": 430}},
                        {"category": "person", "box2d": {"x1": 620, "y1": 240, "x2": 680, "y2": 420}},
                    ]
                }
            ],
        }
        test_label = {
            "name": "bdd_test_001.jpg",
            "attributes": {"weather": "rainy", "timeofday": "night", "scene": "residential"},
            "frames": [
                {
                    "objects": [
                        {"category": "bus", "box2d": {"x1": 220, "y1": 180, "x2": 520, "y2": 420}},
                    ]
                }
            ],
        }

        _write_json(labels_root / "train" / "bdd_train_001.json", train_label)
        _write_json(labels_root / "val" / "bdd_val_001.json", val_label)
        _write_json(labels_root / "test" / "bdd_test_001.json", test_label)


if __name__ == "__main__":
    unittest.main()
