from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.data.image_list import build_sample_uid, load_image_list


class ODBootstrapImageListTests(unittest.TestCase):
    def test_load_image_list_resolves_relative_paths_and_sorts_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "b.png").write_bytes(b"b")
            (images_dir / "a.png").write_bytes(b"a")
            (root / "labels_scene").mkdir(parents=True, exist_ok=True)
            (root / "labels_scene" / "scene_a.json").write_text("{}", encoding="utf-8")
            (root / "labels_scene" / "scene_b.json").write_text("{}", encoding="utf-8")

            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "scene_b",
                    "sample_uid": build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="scene_b"),
                    "image_path": "images/b.png",
                    "scene_path": "labels_scene/scene_b.json",
                    "dataset_root": ".",
                    "dataset_key": "aihub_traffic_seoul",
                    "split": "train",
                },
                {
                    "sample_id": "scene_a",
                    "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="val", sample_id="scene_a"),
                    "image_path": "images/a.png",
                    "scene_path": "labels_scene/scene_a.json",
                    "dataset_root": ".",
                    "dataset_key": "bdd100k_det_100k",
                    "split": "val",
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            entries = load_image_list(manifest_path)

            self.assertEqual([entry.sample_uid for entry in entries], ["aihub_traffic_seoul__train__scene_b", "bdd100k_det_100k__val__scene_a"])
            self.assertEqual(entries[0].image_path, (images_dir / "b.png").resolve())
            self.assertEqual(entries[1].dataset_key, "bdd100k_det_100k")

    def test_load_image_list_rejects_duplicate_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "dup.png"
            image_path.write_bytes(b"dup")
            scene_path = root / "scene.json"
            scene_path.write_text("{}", encoding="utf-8")
            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "one",
                    "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="one"),
                    "image_path": str(image_path),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                },
                {
                    "sample_id": "two",
                    "sample_uid": build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="two"),
                    "image_path": str(image_path),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate image_path"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_duplicate_sample_uids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "image_a.png").write_bytes(b"a")
            (root / "image_b.png").write_bytes(b"b")
            scene_path = root / "scene.json"
            scene_path.write_text("{}", encoding="utf-8")
            sample_uid = build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="dup")
            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "dup",
                    "sample_uid": sample_uid,
                    "image_path": str(root / "image_a.png"),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                },
                {
                    "sample_id": "dup",
                    "sample_uid": sample_uid,
                    "image_path": str(root / "image_b.png"),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate sample_uid"):
                load_image_list(manifest_path)
