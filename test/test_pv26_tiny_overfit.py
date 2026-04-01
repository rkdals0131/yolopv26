from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from model.preprocess.aihub_standardize import run_standardization as run_aihub_standardization
from runtime_support import has_yolo26_runtime


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_dummy_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%stub\n")


class PV26TinyOverfitTests(unittest.TestCase):
    @unittest.skipUnless(has_yolo26_runtime(), "requires ultralytics yolo26 runtime")
    def test_tiny_overfit_reduces_best_loss_on_real_loader_batch(self) -> None:
        from model.net import PV26Heads
        from model.data import PV26CanonicalDataset, collate_pv26_samples
        from model.engine.trainer import PV26Trainer, run_pv26_tiny_overfit
        from model.net import build_yolo26n_trunk

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            output_root = root / "pv26_aihub_standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)

            run_aihub_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=output_root,
                workers=1,
                debug_vis_count=0,
            )

            dataset = PV26CanonicalDataset([output_root])
            selected = []
            seen = set()
            for sample in dataset:
                dataset_key = str(sample["meta"]["dataset_key"])
                split = str(sample["meta"]["split"])
                if dataset_key in {"aihub_traffic_seoul", "aihub_obstacle_seoul", "aihub_lane_seoul"} and split == "train" and dataset_key not in seen:
                    selected.append(sample)
                    seen.add(dataset_key)
            self.assertEqual(seen, {"aihub_traffic_seoul", "aihub_obstacle_seoul", "aihub_lane_seoul"})

            batch = collate_pv26_samples(selected)
            adapter = build_yolo26n_trunk()
            heads = PV26Heads(in_channels=(64, 128, 256))
            trainer = PV26Trainer(
                adapter,
                heads,
                stage="stage_1_frozen_trunk_warmup",
                head_lr=5e-3,
            )

            summary = run_pv26_tiny_overfit(trainer, batch, steps=6)

            self.assertEqual(summary["steps"], 6)
            self.assertEqual(len(summary["history"]), 6)
            self.assertEqual(len(summary["sample_ids"]), 3)
            self.assertTrue(any(sample_id.startswith("aihub_lane_seoul_train_") for sample_id in summary["sample_ids"]))
            self.assertTrue(any(sample_id.startswith("aihub_obstacle_seoul_train_") for sample_id in summary["sample_ids"]))
            self.assertTrue(any(sample_id.startswith("aihub_traffic_seoul_train_") for sample_id in summary["sample_ids"]))
            self.assertGreater(summary["first_total"], summary["best_total"])
            self.assertGreater(summary["improvement"], 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(summary["final_total"])))

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

        _make_image(train_image, 1920, 1080, "#101010")
        _make_image(val_image, 1920, 1080, "#404040")

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
                        "box": [520, 160, 620, 320],
                        "code": "W1-1",
                    },
                    {
                        "class": "vehicle",
                        "box": [700, 500, 980, 840],
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
                        "box": [100, 100, 150, 220],
                        "light_count": 3,
                        "attribute": [{"red": "off", "green": "on", "yellow": "off"}],
                        "type": "car",
                    },
                    {
                        "class": "traffic_sign",
                        "box": [600, 180, 670, 290],
                        "code": "R2-1",
                    },
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
