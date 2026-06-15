from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from common.io import read_json, read_jsonl, write_json
from tools.od_bootstrap.signal_attr import (
    DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    SIGNAL_ATTR_CANONICAL_INVALID_REASON,
    SIGNAL_ATTR_DATASET_REJECT_REASONS,
    SIGNAL_ATTR_INVALID_BBOX_REASON,
    materialize_aihub_signal_attr_crop_dataset,
    materialize_aihub_signal_attr_crop_dataset_from_canonical_root,
)
from tools.od_bootstrap.source.raw_common import PairRecord, TRAFFIC_DATASET_KEY


def _traffic_light(
    bbox: list[float],
    *,
    light_type: str = "car",
    attribute: Any | None = None,
    raw_class: str = "traffic_light",
) -> dict[str, Any]:
    if attribute is None:
        attribute = {
            "red": "off",
            "yellow": "off",
            "green": "off",
            "left_arrow": "off",
            "others_arrow": "off",
            "x_light": "off",
        }
    return {
        "class": raw_class,
        "type": light_type,
        "bbox": bbox,
        "attribute": attribute,
    }


def _write_pair(
    root: Path,
    *,
    split: str,
    relative_id: str,
    annotations: list[dict[str, Any]],
    image_size: tuple[int, int] = (80, 60),
) -> PairRecord:
    image_path = root / split / "images" / f"{relative_id}.jpg"
    label_path = root / split / "labels" / f"{relative_id}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, "black").save(image_path)
    write_json(
        label_path,
        {
            "image": {
                "filename": image_path.name,
                "imsize": [image_size[0], image_size[1]],
            },
            "annotations": annotations,
        },
    )
    return PairRecord(
        dataset_key=TRAFFIC_DATASET_KEY,
        dataset_root=root,
        split=split,
        image_path=image_path,
        label_path=label_path,
        image_file_name=image_path.name,
        relative_id=relative_id,
    )


class SignalAttrDatasetMaterializationTests(unittest.TestCase):
    def test_signal_attr_dataset_materializes_from_canonical_scene_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            canonical_root = Path(temp_dir) / "canonical" / "aihub_standardized"
            output_root = Path(temp_dir) / "signal_attr"
            image_path = canonical_root / "images" / "train" / "traffic_001.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 60), "black").save(image_path)
            scene_path = canonical_root / "labels_scene" / "train" / "traffic_001.json"
            write_json(
                scene_path,
                {
                    "source": {"dataset": "aihub_traffic_seoul", "split": "train"},
                    "image": {"file_name": image_path.name, "width": 80, "height": 60},
                    "detections": [
                        {"id": 0, "class_name": "traffic_light", "bbox": [10.0, 10.0, 30.0, 40.0]},
                        {"id": 1, "class_name": "traffic_light", "bbox": [20.0, 5.0, 42.0, 35.0]},
                        {"id": 2, "class_name": "traffic_light", "bbox": [1.0, 1.0, 3.0, 10.0]},
                    ],
                    "traffic_lights": [
                        {
                            "id": 0,
                            "detection_id": 0,
                            "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
                            "tl_attr_valid": 1,
                            "collapse_reason": "valid",
                        },
                        {
                            "id": 1,
                            "detection_id": 1,
                            "tl_bits": {"red": 1, "yellow": 1, "green": 0, "arrow": 0},
                            "tl_attr_valid": 1,
                            "collapse_reason": "valid",
                        },
                        {
                            "id": 2,
                            "detection_id": 2,
                            "tl_bits": {"red": 0, "yellow": 1, "green": 0, "arrow": 0},
                            "tl_attr_valid": 0,
                            "collapse_reason": "missing_attribute_map",
                        },
                    ],
                },
            )
            skipped_image = canonical_root / "images" / "train" / "lane_001.jpg"
            Image.new("RGB", (80, 60), "black").save(skipped_image)
            write_json(
                canonical_root / "labels_scene" / "train" / "lane_001.json",
                {
                    "source": {"dataset": "aihub_lane_seoul", "split": "train"},
                    "image": {"file_name": skipped_image.name},
                    "traffic_lights": [
                        {
                            "detection_id": 0,
                            "bbox": [10.0, 10.0, 30.0, 30.0],
                            "tl_bits": {"red": 1},
                            "tl_attr_valid": 1,
                        }
                    ],
                },
            )

            logs: list[str] = []
            manifest = materialize_aihub_signal_attr_crop_dataset_from_canonical_root(
                canonical_root,
                output_root,
                workers=2,
                log_every=1,
                log_fn=logs.append,
            )

            rows = read_jsonl(output_root / "labels" / "train.jsonl")
            rejected_rows = read_jsonl(output_root / "meta" / "rejected_rows.jsonl")

            self.assertEqual(manifest["input_format"], "canonical_scene")
            self.assertEqual(manifest["input_root"], str(canonical_root))
            self.assertEqual(manifest["accepted_count_by_split"], {"train": 1, "val": 0})
            self.assertEqual(manifest["rejected_count_by_split"], {"train": 2, "val": 0})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_scene_path"], str(scene_path))
            self.assertEqual(rows[0]["source_label_path"], str(scene_path))
            self.assertEqual(rows[0]["bbox"], [10.0, 10.0, 30.0, 40.0])
            self.assertEqual(rows[0]["tl_bits"], {"arrow": 1, "green": 0, "red": 1, "yellow": 0})
            self.assertEqual(rows[0]["base_color"], "red")
            self.assertEqual(rows[0]["arrow"], 1)
            self.assertEqual(rows[0]["sample_id"], "traffic_001_tl0000")
            self.assertTrue((output_root / rows[0]["crop_path"]).is_file())
            self.assertEqual(
                [row["reject_reason"] for row in rejected_rows],
                [SIGNAL_ATTR_CANONICAL_INVALID_REASON, "missing_attribute_map"],
            )
            joined_logs = "\n".join(logs)
            self.assertIn("[teacher:signal_attr] dataset start", joined_logs)
            self.assertIn("[teacher:signal_attr] dataset progress", joined_logs)
            self.assertIn("[teacher:signal_attr] dataset done", joined_logs)

    def test_signal_attr_dataset_uses_only_traffic_worker_valid_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            output_root = Path(temp_dir) / "signal_attr"
            train_pair = _write_pair(
                root,
                split="train",
                relative_id="sample_b",
                annotations=[
                    _traffic_light(
                        [10.0, 10.0, 30.0, 40.0],
                        attribute={"red": "on", "yellow": "off", "green": "off", "left_arrow": "on"},
                    ),
                    _traffic_light(
                        [12.0, 12.0, 32.0, 42.0],
                        light_type="pedestrian",
                        attribute={"red": "on", "yellow": "off", "green": "off"},
                    ),
                    _traffic_light([14.0, 14.0, 34.0, 44.0], attribute="red"),
                    _traffic_light(
                        [16.0, 16.0, 36.0, 46.0],
                        attribute={"red": "on", "yellow": "off", "green": "off", "x_light": "on"},
                    ),
                    _traffic_light(
                        [18.0, 18.0, 38.0, 48.0],
                        attribute={"red": "on", "yellow": "on", "green": "off"},
                    ),
                    _traffic_light(
                        [1.0, 1.0, 3.0, 10.0],
                        attribute={"red": "off", "yellow": "on", "green": "off"},
                    ),
                    _traffic_light(
                        [20.0, 20.0, 0.0, 10.0],
                        attribute={"red": "off", "yellow": "off", "green": "on"},
                    ),
                    _traffic_light(
                        [5.0, 5.0, 25.0, 25.0],
                        attribute={"red": "on", "yellow": "off", "green": "off"},
                        raw_class="traffic_sign",
                    ),
                ],
            )
            val_pair = _write_pair(
                root,
                split="val",
                relative_id="sample_a",
                annotations=[
                    _traffic_light(
                        [12.0, 5.0, 40.0, 35.0],
                        attribute={"red": "off", "yellow": "off", "green": "on", "others_arrow": "on"},
                    )
                ],
            )

            manifest = materialize_aihub_signal_attr_crop_dataset([val_pair, train_pair], output_root)

            train_rows = read_jsonl(output_root / "labels" / "train.jsonl")
            val_rows = read_jsonl(output_root / "labels" / "val.jsonl")
            rejected_rows = read_jsonl(output_root / "meta" / "rejected_rows.jsonl")
            crop_config = read_json(output_root / "meta" / "crop_config.json")
            manifest_on_disk = read_json(output_root / "meta" / "signal_attr_dataset_manifest.json")

            self.assertEqual(manifest, manifest_on_disk)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["accepted_count_by_split"], {"train": 1, "val": 1})
            self.assertEqual(manifest["rejected_count_by_split"], {"train": 6, "val": 0})
            self.assertEqual(tuple(manifest["closed_reject_reasons"]), SIGNAL_ATTR_DATASET_REJECT_REASONS)
            self.assertEqual(crop_config, manifest["crop_config"])
            self.assertEqual(crop_config["input_size"], DEFAULT_SIGNAL_ATTR_CROP_CONFIG.input_size)

            self.assertEqual(len(train_rows), 1)
            self.assertEqual(len(val_rows), 1)
            train_row = train_rows[0]
            self.assertEqual(train_row["source_image_path"], str(train_pair.image_path))
            self.assertEqual(train_row["source_label_path"], str(train_pair.label_path))
            self.assertEqual(train_row["bbox"], [10.0, 10.0, 30.0, 40.0])
            self.assertEqual(train_row["tl_bits"], {"arrow": 1, "green": 0, "red": 1, "yellow": 0})
            self.assertEqual(train_row["base_color"], "red")
            self.assertEqual(train_row["arrow"], 1)
            self.assertEqual(train_row["collapse_reason"], "valid")
            self.assertEqual(train_row["crop_path"], f"images/train/{train_row['sample_id']}.jpg")
            self.assertEqual(train_row["sample_id"], f"{train_row['source_sample_id']}_tl0000")
            self.assertTrue((output_root / train_row["crop_path"]).is_file())
            with Image.open(output_root / train_row["crop_path"]) as crop:
                self.assertEqual(crop.size, (128, 128))
                self.assertEqual(crop.mode, "RGB")

            val_row = val_rows[0]
            self.assertEqual(val_row["base_color"], "green")
            self.assertEqual(val_row["arrow"], 1)
            self.assertEqual(val_row["collapse_reason"], "valid")

            reject_reasons = [row["reject_reason"] for row in rejected_rows]
            self.assertEqual(
                reject_reasons,
                [
                    "non_car_traffic_light",
                    "missing_attribute_map",
                    "x_light_active",
                    "multi_color_active",
                    "signal_attr_teacher_invalid_roi",
                    SIGNAL_ATTR_INVALID_BBOX_REASON,
                ],
            )
            self.assertEqual({row["raw_class"] for row in rejected_rows}, {"traffic_light"})
            self.assertEqual(rejected_rows[-1]["bbox"], None)
            self.assertEqual(rejected_rows[-1]["collapse_reason"], "valid")
            self.assertEqual(manifest["reject_reason_counts"]["signal_attr_teacher_invalid_roi"], 1)
            self.assertEqual(manifest["reject_reason_counts"][SIGNAL_ATTR_INVALID_BBOX_REASON], 1)
            self.assertEqual(manifest["tl_combo_counts_by_split"]["train"], {"red+arrow": 1})
            self.assertEqual(manifest["tl_combo_counts_by_split"]["val"], {"green+arrow": 1})

    def test_signal_attr_dataset_order_is_deterministic(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            output_a = Path(temp_dir) / "out_a"
            output_b = Path(temp_dir) / "out_b"
            pair_b = _write_pair(
                root,
                split="train",
                relative_id="sample_b",
                annotations=[_traffic_light([20.0, 10.0, 45.0, 45.0])],
            )
            pair_a = _write_pair(
                root,
                split="train",
                relative_id="sample_a",
                annotations=[_traffic_light([10.0, 10.0, 30.0, 30.0])],
            )

            materialize_aihub_signal_attr_crop_dataset([pair_b, pair_a], output_a, workers=2, log_every=1)
            materialize_aihub_signal_attr_crop_dataset([pair_a, pair_b], output_b, workers=1, log_every=1)

            rows_a = read_jsonl(output_a / "labels" / "train.jsonl")
            rows_b = read_jsonl(output_b / "labels" / "train.jsonl")

            self.assertEqual(rows_a, rows_b)
            self.assertEqual([row["source_relative_id"] for row in rows_a], ["sample_a", "sample_b"])


if __name__ == "__main__":
    unittest.main()
