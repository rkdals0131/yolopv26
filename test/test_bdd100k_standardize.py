from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from model.preprocess.bdd100k_standardize import TL_BITS, run_standardization


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


class BDD100KStandardizationTests(unittest.TestCase):
    def test_standardization_generates_det_only_canonical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "BDD100K"
            images_root = bdd_root / "bdd100k_images_100k" / "100k"
            labels_root = bdd_root / "bdd100k_labels" / "100k"
            output_root = root / "pv26_bdd100k_standardized"

            self._create_bdd_fixture(images_root, labels_root)

            outputs = run_standardization(
                bdd_root=bdd_root,
                images_root=images_root,
                labels_root=labels_root,
                output_root=output_root,
                workers=1,
            )

            readme = (bdd_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("# BDD100K", readme)
            self.assertIn("PV26 클래스 collapse 규칙", readme)

            conversion_report = json.loads(outputs["conversion_json"].read_text(encoding="utf-8"))
            inventory = json.loads(outputs["inventory_json"].read_text(encoding="utf-8"))
            dataset = conversion_report["dataset"]

            self.assertEqual(inventory["dataset"]["local_inventory"]["splits"]["train"]["images"], 1)
            self.assertEqual(inventory["dataset"]["local_inventory"]["splits"]["val"]["json_files"], 1)
            self.assertEqual(dataset["processed_samples"], 3)
            self.assertEqual(dataset["det_class_counts"]["vehicle"], 3)
            self.assertEqual(dataset["det_class_counts"]["pedestrian"], 2)
            self.assertEqual(dataset["det_class_counts"]["bike"], 2)
            self.assertEqual(dataset["det_class_counts"]["traffic_light"], 1)
            self.assertEqual(dataset["det_class_counts"]["sign"], 1)
            self.assertEqual(dataset["tl_state_hint_counts"]["red"], 1)
            self.assertEqual(dataset["held_reason_counts"]["ignored_non_pv26_category"], 3)

            det_labels = sorted((output_root / "labels_det").rglob("*.txt"))
            scene_labels = sorted((output_root / "labels_scene").rglob("*.json"))
            debug_vis = sorted((output_root / "meta" / "debug_vis").rglob("*.png"))
            self.assertEqual(len(det_labels), 3)
            self.assertEqual(len(scene_labels), 3)
            self.assertEqual(len(debug_vis), 3)

            scene_by_split = {
                path.parent.name: json.loads(path.read_text(encoding="utf-8"))
                for path in scene_labels
            }
            train_scene = scene_by_split["train"]
            self.assertEqual(train_scene["tasks"]["has_det"], 1)
            self.assertEqual(train_scene["tasks"]["has_tl_attr"], 0)
            self.assertEqual(train_scene["context"]["weather"], "clear")
            self.assertEqual(train_scene["context"]["scene"], "city street")
            self.assertEqual(train_scene["context"]["timeofday"], "daytime")
            self.assertEqual(train_scene["traffic_lights"][0]["state_hint"], "red")
            self.assertEqual(train_scene["traffic_lights"][0]["tl_attr_valid"], 0)
            self.assertEqual(train_scene["traffic_lights"][0]["collapse_reason"], "bdd_det_only_source")
            self.assertEqual(train_scene["traffic_lights"][0]["tl_bits"], {bit: 0 for bit in TL_BITS})

            det_map_yaml = outputs["det_map_yaml"].read_text(encoding="utf-8")
            scene_map_yaml = outputs["scene_map_yaml"].read_text(encoding="utf-8")
            self.assertIn("traffic_light", det_map_yaml)
            self.assertIn("has_tl_attr: false", scene_map_yaml)

            debug_vis_index = json.loads(outputs["debug_vis_index"].read_text(encoding="utf-8"))
            self.assertEqual(debug_vis_index["selection_count"], 3)

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
                            {
                                "id": 4,
                                "category": "area/drivable",
                                "box2d": None,
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
                            {
                                "id": 4,
                                "category": "lane/crosswalk",
                                "box2d": None,
                            },
                        ],
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
