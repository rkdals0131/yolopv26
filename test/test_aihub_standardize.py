from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.data.aihub import (
    TL_BITS,
    _prepare_debug_scene_for_overlay,
    _select_debug_vis_summaries,
    run_standardization,
)


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


class AIHubStandardizationTests(unittest.TestCase):
    def test_debug_vis_prefers_positive_obstacle_samples(self) -> None:
        selected = _select_debug_vis_summaries(
            [
                {
                    "dataset_key": "aihub_obstacle_seoul",
                    "split": "train",
                    "sample_id": "empty",
                    "scene_path": "empty.json",
                    "image_path": "empty.png",
                    "det_count": 0,
                    "traffic_light_count": 0,
                    "traffic_sign_count": 0,
                    "lane_count": 0,
                    "stop_line_count": 0,
                    "crosswalk_count": 0,
                },
                {
                    "dataset_key": "aihub_obstacle_seoul",
                    "split": "train",
                    "sample_id": "positive",
                    "scene_path": "positive.json",
                    "image_path": "positive.png",
                    "det_count": 2,
                    "traffic_light_count": 0,
                    "traffic_sign_count": 0,
                    "lane_count": 0,
                    "stop_line_count": 0,
                    "crosswalk_count": 0,
                },
            ],
            count=1,
            seed=26,
        )

        self.assertEqual([item["sample_id"] for item in selected], ["positive"])

    def test_debug_overlay_marks_excluded_obstacle_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            output_root = root / "standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)

            run_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=output_root,
                workers=1,
                debug_vis_count=0,
            )

            obstacle_scene_path = sorted((output_root / "labels_scene").rglob("aihub_obstacle_seoul*.json"))[0]
            obstacle_scene = json.loads(obstacle_scene_path.read_text(encoding="utf-8"))
            overlay_scene = _prepare_debug_scene_for_overlay(obstacle_scene)

            self.assertEqual(overlay_scene["detections"][0]["class_name"], "traffic_cone")
            self.assertIn("debug_rectangles", overlay_scene)
            self.assertEqual(len(overlay_scene["debug_rectangles"]), 1)
            self.assertEqual(overlay_scene["debug_rectangles"][0]["bbox"], [420.0, 250.0, 468.0, 380.0])
            self.assertEqual(overlay_scene["debug_rectangles"][0]["color"], "#00e5ff")
            self.assertEqual(overlay_scene["debug_rectangles"][0]["label"], "excluded:person")

    def test_standardization_generates_readmes_reports_and_tl_bits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            output_root = root / "standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)

            outputs = run_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=output_root,
                workers=1,
            )

            lane_readme = (lane_root / "README.md").read_text(encoding="utf-8")
            obstacle_readme = (obstacle_root / "README.md").read_text(encoding="utf-8")
            traffic_readme = (traffic_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("# 차선-횡단보도 인지 영상(수도권)", lane_readme)
            self.assertIn("문서 기준 수도권 json 수량", lane_readme)
            self.assertIn("# 도로장애물·표면 인지 영상(수도권)", obstacle_readme)
            self.assertIn("traffic_cone", obstacle_readme)
            self.assertIn("# 신호등-도로표지판 인지 영상(수도권)", traffic_readme)
            self.assertIn("4-bit supervision", traffic_readme)

            conversion_report = json.loads(outputs["conversion_json"].read_text(encoding="utf-8"))
            source_inventory = json.loads(outputs["inventory_json"].read_text(encoding="utf-8"))

            inventory_by_key = {item["dataset_key"]: item for item in source_inventory["datasets"]}
            self.assertEqual(inventory_by_key["aihub_lane_seoul"]["local_inventory"]["splits"]["Training"]["images"], 1)
            self.assertEqual(inventory_by_key["aihub_obstacle_seoul"]["local_inventory"]["splits"]["Training"]["images"], 1)
            self.assertEqual(inventory_by_key["aihub_obstacle_seoul"]["local_inventory"]["splits"]["Training"]["json_files"], 1)
            traffic_inventory = inventory_by_key["aihub_traffic_seoul"]["local_inventory"]["splits"]
            self.assertEqual(traffic_inventory["Training"]["raw_images"], 1)
            self.assertEqual(traffic_inventory["Training"]["crop_images"], 1)

            datasets = {item["dataset_key"]: item for item in conversion_report["datasets"]}
            lane_dataset = datasets["aihub_lane_seoul"]
            obstacle_dataset = datasets["aihub_obstacle_seoul"]
            traffic_dataset = datasets["aihub_traffic_seoul"]

            self.assertEqual(lane_dataset["lane_class_counts"]["blue_lane"], 1)
            self.assertEqual(lane_dataset["lane_class_counts"]["white_lane"], 1)
            self.assertEqual(lane_dataset["stop_line_count"], 1)
            self.assertEqual(lane_dataset["crosswalk_count"], 1)

            self.assertEqual(obstacle_dataset["det_class_counts"]["traffic_cone"], 1)
            self.assertEqual(obstacle_dataset["det_class_counts"]["obstacle"], 3)
            self.assertEqual(obstacle_dataset["detection_count"], 4)
            self.assertEqual(obstacle_dataset["held_reason_counts"]["excluded_obstacle_person_policy"], 1)
            self.assertEqual(obstacle_dataset["held_reason_counts"]["excluded_obstacle_filled_pothole_policy"], 1)
            self.assertEqual(obstacle_dataset["traffic_light_count"], 0)
            self.assertEqual(obstacle_dataset["traffic_sign_count"], 0)

            self.assertEqual(traffic_dataset["det_class_counts"]["sign"], 1)
            self.assertEqual(traffic_dataset["det_class_counts"]["traffic_light"], 5)
            self.assertEqual(traffic_dataset["tl_attr_valid_count"], 3)
            self.assertEqual(traffic_dataset["tl_attr_invalid_count"], 2)
            self.assertEqual(traffic_dataset["tl_combo_counts"]["red+arrow"], 1)
            self.assertEqual(traffic_dataset["tl_combo_counts"]["arrow"], 1)
            self.assertEqual(traffic_dataset["tl_combo_counts"]["off"], 1)
            self.assertEqual(traffic_dataset["tl_invalid_reason_counts"]["multi_color_active"], 1)
            self.assertEqual(traffic_dataset["tl_invalid_reason_counts"]["non_car_traffic_light"], 1)

            lane_scenes = sorted((output_root / "labels_scene").rglob("aihub_lane_seoul*.json"))
            obstacle_scenes = sorted((output_root / "labels_scene").rglob("aihub_obstacle_seoul*.json"))
            traffic_scenes = sorted((output_root / "labels_scene").rglob("aihub_traffic_seoul*.json"))
            det_labels = sorted((output_root / "labels_det").rglob("*.txt"))
            debug_vis = sorted((output_root / "meta" / "debug_vis").rglob("*.png"))
            self.assertEqual(len(lane_scenes), 1)
            self.assertEqual(len(obstacle_scenes), 2)
            self.assertEqual(len(traffic_scenes), 2)
            self.assertEqual(len(det_labels), 4)
            self.assertEqual(len(debug_vis), 5)

            lane_scene = json.loads(lane_scenes[0].read_text(encoding="utf-8"))
            self.assertEqual(lane_scene["tasks"]["has_lane"], 1)
            self.assertEqual(lane_scene["tasks"]["has_stop_line"], 1)
            self.assertEqual(lane_scene["tasks"]["has_crosswalk"], 1)
            self.assertIn("blue_lane", {item["class_name"] for item in lane_scene["lanes"]})

            obstacle_scene = json.loads(obstacle_scenes[0].read_text(encoding="utf-8"))
            self.assertEqual(obstacle_scene["tasks"]["has_det"], 1)
            self.assertEqual(obstacle_scene["tasks"]["has_tl_attr"], 0)
            self.assertEqual(obstacle_scene["traffic_lights"], [])
            self.assertEqual(obstacle_scene["traffic_signs"], [])
            self.assertEqual([item["class_name"] for item in obstacle_scene["detections"]], ["traffic_cone", "obstacle"])

            traffic_bits = []
            for traffic_scene_path in traffic_scenes:
                traffic_scene = json.loads(traffic_scene_path.read_text(encoding="utf-8"))
                traffic_bits.extend(traffic_scene["traffic_lights"])
            combo_to_valid = {
                "+".join(bit for bit in TL_BITS if item["tl_bits"].get(bit)) or "off": item["tl_attr_valid"]
                for item in traffic_bits
            }
            self.assertEqual(combo_to_valid["red+arrow"], 1)
            self.assertEqual(combo_to_valid["arrow"], 1)
            self.assertEqual(combo_to_valid["off"], 1)

            det_map_yaml = outputs["det_map_yaml"].read_text(encoding="utf-8")
            self.assertIn("traffic_light", det_map_yaml)
            self.assertNotIn("yellow_arrow_not_supported", det_map_yaml)

            debug_vis_index = json.loads(outputs["debug_vis_index"].read_text(encoding="utf-8"))
            self.assertEqual(debug_vis_index["selection_count"], 5)
            failure_manifest = json.loads(outputs["failure_json"].read_text(encoding="utf-8"))
            qa_summary = json.loads(outputs["qa_json"].read_text(encoding="utf-8"))
            self.assertEqual(failure_manifest["failure_count"], 0)
            self.assertEqual(qa_summary["debug_vis"]["selection_count"], 5)
            self.assertEqual({item["dataset_key"] for item in qa_summary["datasets"]}, {"aihub_lane_seoul", "aihub_obstacle_seoul", "aihub_traffic_seoul"})

    def test_standardization_supports_resume_and_failure_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_root = root / "docs"
            lane_root = root / "lane"
            obstacle_root = root / "obstacle"
            traffic_root = root / "traffic"
            output_root = root / "standardized"

            self._create_docs_fixture(docs_root)
            self._create_lane_fixture(lane_root)
            self._create_obstacle_fixture(obstacle_root)
            self._create_traffic_fixture(traffic_root)
            broken_image = traffic_root / "Validation" / "[원천]c_val_1" / "traffic_val_001.jpg"
            broken_label = traffic_root / "Validation" / "[라벨]c_val_1" / "c_val_1" / "traffic_val_001.json"
            broken_image.write_text("not an image", encoding="utf-8")
            broken_label.write_text(
                json.dumps(
                    {
                        "image": {"filename": "traffic_val_001.jpg"},
                        "annotation": [
                            {
                                "class": "traffic_light",
                                "box": [120, 80, 180, 240],
                                "attribute": [{"red": "off", "green": "off", "yellow": "off"}],
                                "type": "car",
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            first_outputs = run_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=output_root,
                workers=1,
                debug_vis_count=0,
            )
            first_report = json.loads(first_outputs["conversion_json"].read_text(encoding="utf-8"))
            first_failures = json.loads(first_outputs["failure_json"].read_text(encoding="utf-8"))
            first_datasets = {item["dataset_key"]: item for item in first_report["datasets"]}

            self.assertEqual(first_failures["failure_count"], 1)
            self.assertEqual(first_datasets["aihub_lane_seoul"]["fresh_processed_count"], 1)
            self.assertEqual(first_datasets["aihub_lane_seoul"]["resume_skipped_count"], 0)
            self.assertEqual(first_datasets["aihub_obstacle_seoul"]["fresh_processed_count"], 2)
            self.assertEqual(first_datasets["aihub_obstacle_seoul"]["failure_count"], 0)
            self.assertEqual(first_datasets["aihub_traffic_seoul"]["failure_count"], 1)
            self.assertEqual(first_datasets["aihub_traffic_seoul"]["processed_samples"], 1)

            _make_image(broken_image, 1920, 1080, "#505050")
            broken_label.write_text(
                json.dumps(
                    {
                        "image": {"filename": "traffic_val_001.jpg", "imsize": {"width": 1920, "height": 1080}},
                        "annotation": [
                            {
                                "class": "traffic_light",
                                "box": [32, 40, 80, 120],
                                "attribute": [{"green": "on"}],
                                "type": "car",
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            second_outputs = run_standardization(
                lane_root=lane_root,
                obstacle_root=obstacle_root,
                traffic_root=traffic_root,
                docs_root=docs_root,
                output_root=output_root,
                workers=1,
                debug_vis_count=0,
            )
            second_report = json.loads(second_outputs["conversion_json"].read_text(encoding="utf-8"))
            second_failures = json.loads(second_outputs["failure_json"].read_text(encoding="utf-8"))
            second_datasets = {item["dataset_key"]: item for item in second_report["datasets"]}

            self.assertEqual(second_failures["failure_count"], 0)
            self.assertEqual(second_datasets["aihub_lane_seoul"]["resume_skipped_count"], 1)
            self.assertEqual(second_datasets["aihub_lane_seoul"]["fresh_processed_count"], 0)
            self.assertEqual(second_datasets["aihub_obstacle_seoul"]["resume_skipped_count"], 2)
            self.assertEqual(second_datasets["aihub_obstacle_seoul"]["fresh_processed_count"], 0)
            self.assertEqual(second_datasets["aihub_traffic_seoul"]["resume_skipped_count"], 1)
            self.assertEqual(second_datasets["aihub_traffic_seoul"]["fresh_processed_count"], 1)

    def _create_docs_fixture(self, docs_root: Path) -> None:
        _write_dummy_pdf(docs_root / "차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf")
        _write_dummy_pdf(docs_root / "수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf")

    def _create_obstacle_fixture(self, obstacle_root: Path) -> None:
        train_image = obstacle_root / "Training" / "Images" / "TOA" / "1.Frontback_A01" / "obstacle_train_001.png"
        train_label = obstacle_root / "Training" / "Annotations" / "TOA" / "1.Frontback_A01" / "obstacle_train_001_BBOX.json"
        val_image = obstacle_root / "Validation" / "Images" / "TOA" / "1.Frontback_F01" / "obstacle_val_001.png"
        val_label = obstacle_root / "Validation" / "Annotations" / "TOA" / "1.Frontback_F01" / "obstacle_val_001_BBOX.json"

        _write_dummy_pdf(obstacle_root / "061.도로장애물_표면_인지_영상(수도권)_데이터_구축_가이드라인.pdf")
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
                    {"id": 1, "image_id": 1, "bbox": [70.0, 98.0, 91.0, 187.0], "category_id": 5, "area": 17017.0, "is_crowd": 0},
                    {"id": 2, "image_id": 1, "bbox": [802.0, 181.0, 21.0, 46.0], "category_id": 6, "area": 966.0, "is_crowd": 0},
                    {"id": 3, "image_id": 1, "bbox": [420.0, 250.0, 48.0, 130.0], "category_id": 2, "area": 6240.0, "is_crowd": 0},
                ],
                "categories": categories,
            },
        )
        _write_json(
            val_label,
            {
                "images": {"file_name": "obstacle_val_001.png", "width": 1280, "height": 720, "id": 2},
                "annotations": [
                    {"id": 1, "image_id": 2, "bbox": [180.0, 220.0, 60.0, 80.0], "category_id": 3, "area": 4800.0, "is_crowd": 0},
                    {"id": 2, "image_id": 2, "bbox": [620.0, 260.0, 120.0, 90.0], "category_id": 4, "area": 10800.0, "is_crowd": 0},
                    {"id": 3, "image_id": 2, "bbox": [920.0, 310.0, 100.0, 70.0], "category_id": 9, "area": 7000.0, "is_crowd": 0},
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


if __name__ == "__main__":
    unittest.main()
