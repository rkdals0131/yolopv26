from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.od_bootstrap.build.debug_vis import (
    generate_canonical_debug_vis,
    generate_final_dataset_debug_vis,
    generate_final_lane_label_audit,
)
from tools.od_bootstrap.build.final_dataset import FINAL_DATASET_MANIFEST_NAME
from tools.od_bootstrap.build.image_list import ImageListEntry, build_sample_uid, write_image_list


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class ODBootstrapBuildDebugVisTests(unittest.TestCase):
    def test_generate_canonical_debug_vis_writes_per_dataset_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_root = root / "canonical"
            bdd_root = canonical_root / "bdd100k_det_100k"
            aihub_root = canonical_root / "aihub_standardized"

            bdd_entry = self._make_canonical_entry(
                dataset_root=bdd_root,
                dataset_key="bdd100k_det_100k",
                split="train",
                sample_id="bdd_train_001",
                image_name="bdd_train_001.jpg",
                color="#111111",
            )
            aihub_entry = self._make_canonical_entry(
                dataset_root=aihub_root,
                dataset_key="aihub_traffic_seoul",
                split="val",
                sample_id="aihub_traffic_val_001",
                image_name="aihub_traffic_val_001.png",
                color="#222222",
            )

            image_list_path = root / "image_list.jsonl"
            write_image_list(image_list_path, (bdd_entry, aihub_entry))

            outputs = generate_canonical_debug_vis(
                image_list_manifest_path=image_list_path,
                canonical_root=canonical_root,
                debug_vis_count=1,
                debug_vis_seed=7,
            )

            self.assertEqual(sorted(outputs), ["aihub_standardized", "bdd100k_det_100k"])
            for dataset_name, dataset_key in (
                ("bdd100k_det_100k", "bdd100k_det_100k"),
                ("aihub_standardized", "aihub_traffic_seoul"),
            ):
                manifest = json.loads(outputs[dataset_name]["debug_vis_manifest"].read_text(encoding="utf-8"))
                self.assertEqual(outputs[dataset_name]["selection_count"], 1)
                self.assertEqual(manifest["selection_count"], 1)
                self.assertEqual(manifest["items"][0]["dataset_key"], dataset_key)
                self.assertTrue(Path(manifest["items"][0]["overlay_path"]).is_file())

    def test_generate_final_dataset_debug_vis_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "pv26_exhaustive_od_lane_dataset"
            image_path = dataset_root / "images" / "train" / "final_001.png"
            scene_path = dataset_root / "labels_scene" / "train" / "final_001.json"
            _make_image(image_path, 640, 360, "#333333")
            _write_json(
                scene_path,
                {
                    "version": "test",
                    "image": {
                        "file_name": image_path.name,
                        "width": 640,
                        "height": 360,
                    },
                    "source": {
                        "dataset": "aihub_lane_seoul",
                        "split": "train",
                        "final_sample_id": "final_001",
                    },
                    "detections": [{"class_name": "vehicle", "bbox": [10, 20, 120, 160]}],
                    "lanes": [{"class_name": "white_lane", "points": [[0, 0], [100, 100]]}],
                },
            )
            manifest_rows = [
                {
                    "final_sample_id": "final_001",
                    "source_dataset_key": "aihub_lane_seoul",
                    "split": "train",
                    "scene_path": str(scene_path),
                    "image_path": str(image_path),
                }
            ]
            _write_json(
                dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME,
                {
                    "version": "test",
                    "output_root": str(dataset_root),
                    "sample_count": 1,
                    "dataset_counts": {"aihub_lane_seoul": 1},
                    "samples": manifest_rows,
                },
            )

            outputs = generate_final_dataset_debug_vis(
                dataset_root=dataset_root,
                manifest_rows=manifest_rows,
                debug_vis_count=1,
                debug_vis_seed=7,
            )

            manifest = json.loads(outputs["debug_vis_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(outputs["selection_count"], 1)
            self.assertEqual(manifest["selection_count"], 1)
            self.assertEqual(manifest["items"][0]["dataset_key"], "aihub_lane_seoul")
            self.assertTrue(Path(manifest["items"][0]["overlay_path"]).is_file())

    def test_generate_final_lane_label_audit_writes_shallow_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "pv26_exhaustive_od_lane_dataset"
            manifest_rows = [
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="aihub_lane_seoul",
                    split="train",
                    sample_id="aihub_lane_seoul_train_Training_c_1280_720_daylight_train_1_100",
                    color="#101010",
                    source_image_path="/raw/lane/train/c_1280_720_daylight_train_1/100.jpg",
                    source_raw_id="Training_c_1280_720_daylight_train_1_100",
                    lane_count=2,
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="aihub_lane_seoul",
                    split="train",
                    sample_id="aihub_lane_seoul_train_Training_c_1280_720_daylight_train_1_200",
                    color="#111111",
                    source_image_path="/raw/lane/train/c_1280_720_daylight_train_1/200.jpg",
                    source_raw_id="Training_c_1280_720_daylight_train_1_200",
                    lane_count=0,
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="aihub_lane_seoul",
                    split="val",
                    sample_id="aihub_lane_seoul_val_Validation_c_1280_720_daylight_validation_1_c_1280_720_daylight_val_1_300",
                    color="#121212",
                    source_image_path="/raw/lane/val/c_1280_720_daylight_val_1/300.jpg",
                    source_raw_id="Validation_c_1280_720_daylight_validation_1_c_1280_720_daylight_val_1_300",
                    lane_count=1,
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="aihub_lane_seoul",
                    split="val",
                    sample_id="aihub_lane_seoul_val_Validation_c_1280_720_daylight_validation_1_c_1280_720_daylight_val_1_400",
                    color="#131313",
                    source_image_path="/raw/lane/val/c_1280_720_daylight_val_1/400.jpg",
                    source_raw_id="Validation_c_1280_720_daylight_validation_1_c_1280_720_daylight_val_1_400",
                    lane_count=3,
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="pv26_exhaustive_aihub_traffic_seoul",
                    split="train",
                    sample_id="traffic_train_001",
                    color="#202020",
                    source_image_path="/raw/traffic/train/[원천]c_train_1280_720_daylight_2/001.jpg",
                    source_raw_id="traffic_train_001",
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="pv26_exhaustive_aihub_obstacle_seoul",
                    split="val",
                    sample_id="obstacle_val_001",
                    color="#303030",
                    source_image_path="/raw/obstacle/Validation/Images/TOA/4.Kidzone_F01/001.png",
                    source_raw_id="obstacle_val_001",
                ),
                self._make_final_dataset_sample(
                    dataset_root=dataset_root,
                    dataset_key="pv26_exhaustive_bdd100k_det_100k",
                    split="test",
                    sample_id="bdd_test_001",
                    color="#404040",
                    source_image_path="/raw/bdd/100k/test/bdd_test_001.jpg",
                    source_raw_id="bdd_test_001",
                ),
            ]
            _write_json(
                dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME,
                {
                    "version": "test",
                    "output_root": str(dataset_root),
                    "sample_count": len(manifest_rows),
                    "samples": manifest_rows,
                },
            )

            result = generate_final_lane_label_audit(
                dataset_root=dataset_root,
                manifest_rows=manifest_rows,
                output_root=dataset_root / "debug_vis_lane_audit",
                overview_count=4,
                lane_bin_count=2,
                lane_samples_per_bin=1,
                debug_vis_seed=7,
                workers=2,
            )

            output_root = dataset_root / "debug_vis_lane_audit"
            index_payload = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
            summary_payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            overview_files = sorted(path.name for path in (output_root / "overview").glob("*.png"))
            lane_train_files = sorted(path.name for path in (output_root / "lane_train").glob("*.png"))
            lane_val_files = sorted(path.name for path in (output_root / "lane_val").glob("*.png"))

            self.assertEqual(result["selection_count"], len(index_payload["items"]))
            self.assertTrue(overview_files)
            self.assertTrue(lane_train_files)
            self.assertTrue(lane_val_files)
            self.assertTrue(all(name.startswith("overview__") for name in overview_files))
            self.assertTrue(all(name.startswith("lane__train__") for name in lane_train_files))
            self.assertTrue(all(name.startswith("lane__val__") for name in lane_val_files))
            self.assertEqual(summary_payload["lane_zero_counts"]["total"], 1)
            self.assertEqual(
                sorted(path.name for path in output_root.iterdir()),
                ["index.json", "lane_train", "lane_val", "overview", "summary.json"],
            )

    def _make_canonical_entry(
        self,
        *,
        dataset_root: Path,
        dataset_key: str,
        split: str,
        sample_id: str,
        image_name: str,
        color: str,
    ) -> ImageListEntry:
        image_path = dataset_root / "images" / split / image_name
        scene_path = dataset_root / "labels_scene" / split / f"{sample_id}.json"
        _make_image(image_path, 640, 360, color)
        _write_json(
            scene_path,
            {
                "version": "test",
                "image": {
                    "file_name": image_name,
                    "width": 640,
                    "height": 360,
                },
                "source": {
                    "dataset": dataset_key,
                    "split": split,
                },
                "detections": [{"class_name": "vehicle", "bbox": [10, 20, 120, 160]}],
                "lanes": [{"class_name": "white_lane", "points": [[0, 0], [100, 100]]}],
            },
        )
        return ImageListEntry(
            sample_id=sample_id,
            sample_uid=build_sample_uid(dataset_key=dataset_key, split=split, sample_id=sample_id),
            image_path=image_path,
            scene_path=scene_path,
            dataset_root=dataset_root,
            dataset_key=dataset_key,
            split=split,
        )

    def _make_final_dataset_sample(
        self,
        *,
        dataset_root: Path,
        dataset_key: str,
        split: str,
        sample_id: str,
        color: str,
        source_image_path: str,
        source_raw_id: str,
        lane_count: int = 0,
    ) -> dict[str, str]:
        image_path = dataset_root / "images" / split / f"{sample_id}.png"
        scene_path = dataset_root / "labels_scene" / split / f"{sample_id}.json"
        _make_image(image_path, 320, 180, color)
        _write_json(
            scene_path,
            {
                "version": "test",
                "image": {
                    "file_name": image_path.name,
                    "width": 320,
                    "height": 180,
                },
                "source": {
                    "dataset": dataset_key,
                    "split": split,
                    "final_sample_id": sample_id,
                    "image_path": source_image_path,
                    "raw_id": source_raw_id,
                    "source_kind": "lane" if dataset_key == "aihub_lane_seoul" else "exhaustive_od",
                },
                "detections": [{"class_name": "vehicle", "bbox": [10, 20, 120, 160]}],
                "lanes": [
                    {"class_name": "white_lane", "points": [[0, 0], [20, 20]]}
                    for _ in range(lane_count)
                ],
                "stop_lines": [],
                "crosswalks": [],
                "traffic_lights": [],
            },
        )
        return {
            "final_sample_id": sample_id,
            "source_dataset_key": dataset_key,
            "split": split,
            "scene_path": str(scene_path),
            "image_path": str(image_path),
        }


if __name__ == "__main__":
    unittest.main()
