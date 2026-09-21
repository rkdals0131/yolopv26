from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.build.image_list import (
    ImageListEntry,
    build_sample_uid,
    discover_image_list_entries,
    load_image_list,
    write_image_list,
)


def _scene_payload(*, file_name: str, dataset_key: str, split: str, tasks: dict | None = None) -> dict:
    payload = {
        "image": {"file_name": file_name},
        "source": {"dataset": dataset_key, "split": split},
    }
    if tasks is not None:
        payload["tasks"] = tasks
    return payload


class ODBootstrapImageListTests(unittest.TestCase):
    def test_discover_image_list_entries_rejects_non_object_scene_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "bad.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "scene root must be an object"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_discover_image_list_entries_rejects_source_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path = root / "images" / "val" / "sample.jpg"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "val"},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scene source.split must match labels_scene split"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_discover_image_list_entries_rejects_image_file_name_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "../sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image.file_name must be a file name"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_discover_image_list_entries_rejects_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "missing.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "image_list image not found"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_discover_image_list_entries_rejects_missing_det_label_when_scene_requires_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                        "tasks": {"has_det": 1},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "image_list det label not found"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_discover_image_list_entries_rejects_stale_det_label_when_scene_has_no_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path = root / "images" / "train" / "sample.jpg"
            det_path = root / "labels_det" / "train" / "sample.txt"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            det_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            det_path.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                        "tasks": {"has_det": 0},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_list stale det label"):
                discover_image_list_entries([root], allowed_dataset_keys={"bdd100k_det_100k"})

    def test_load_image_list_resolves_relative_paths_and_sorts_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            (images_dir / "train").mkdir(parents=True, exist_ok=True)
            (images_dir / "val").mkdir(parents=True, exist_ok=True)
            (images_dir / "train" / "b.png").write_bytes(b"b")
            (images_dir / "val" / "a.png").write_bytes(b"a")
            (root / "labels_scene" / "val").mkdir(parents=True, exist_ok=True)
            (root / "labels_scene" / "train").mkdir(parents=True, exist_ok=True)
            (root / "labels_scene" / "val" / "scene_a.json").write_text(
                json.dumps(
                    _scene_payload(file_name="a.png", dataset_key="bdd100k_det_100k", split="val"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "labels_scene" / "train" / "scene_b.json").write_text(
                json.dumps(
                    _scene_payload(file_name="b.png", dataset_key="aihub_traffic_seoul", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "scene_b",
                    "sample_uid": build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="scene_b"),
                    "image_path": "images/train/b.png",
                    "scene_path": "labels_scene/train/scene_b.json",
                    "dataset_root": ".",
                    "dataset_key": "aihub_traffic_seoul",
                    "split": "train",
                },
                {
                    "sample_id": "scene_a",
                    "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="val", sample_id="scene_a"),
                    "image_path": "images/val/a.png",
                    "scene_path": "labels_scene/val/scene_a.json",
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
            self.assertEqual(entries[0].image_path, (images_dir / "train" / "b.png").resolve())
            self.assertEqual(entries[1].dataset_key, "bdd100k_det_100k")

    def test_load_image_list_rejects_scene_dataset_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="sample.jpg", dataset_key="aihub_traffic_seoul", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scene source.dataset must match image list dataset_key"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_scene_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="sample.jpg", dataset_key="bdd100k_det_100k", split="val"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scene source.split must match image list split"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_image_path_mismatch_with_scene_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "manifest.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="scene.jpg", dataset_key="bdd100k_det_100k", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_list image_path must match scene image.file_name"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_scene_path_mismatch_with_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "other.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="sample.jpg", dataset_key="bdd100k_det_100k", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_list scene_path must match dataset_root/split/sample_id"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_sample_uid_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="sample.jpg", dataset_key="bdd100k_det_100k", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="val",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_list sample_uid must match dataset_key/split/sample_id"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_missing_det_label_when_scene_requires_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                        "tasks": {"has_det": 1},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "image_list det label not found"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "labels_scene" / "train" / "sample.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(root / "images" / "train" / "sample.jpg"),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "image_list image not found"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_stale_det_label_when_scene_has_no_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "sample.jpg"
            scene_path = root / "labels_scene" / "train" / "sample.json"
            det_path = root / "labels_det" / "train" / "sample.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            det_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            det_path.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
            scene_path.write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sample.jpg"},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                        "tasks": {"has_det": 0},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "sample_uid": build_sample_uid(
                            dataset_key="bdd100k_det_100k",
                            split="train",
                            sample_id="sample",
                        ),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(root),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "train",
                        "det_path": str(det_path),
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_list stale det label"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_non_object_rows_with_physical_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "image_list.jsonl"
            manifest_path.write_text("\n[]\n", encoding="utf-8")

            with self.assertRaisesRegex(TypeError, r"image_list\[2\] must be a JSON object"):
                load_image_list(manifest_path)

    def test_load_image_list_rejects_duplicate_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "train" / "dup.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"dup")
            scene_path = root / "labels_scene" / "train" / "one.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="dup.png", dataset_key="bdd100k_det_100k", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "one",
                    "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="one"),
                    "image_path": str(image_path),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                    "dataset_key": "bdd100k_det_100k",
                    "split": "train",
                },
                {
                    "sample_id": "two",
                    "sample_uid": build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="two"),
                    "image_path": str(image_path),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                    "dataset_key": "aihub_traffic_seoul",
                    "split": "train",
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
            (root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (root / "images" / "train" / "image_a.png").write_bytes(b"a")
            (root / "images" / "train" / "image_b.png").write_bytes(b"b")
            scene_path = root / "labels_scene" / "train" / "dup.json"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                json.dumps(
                    _scene_payload(file_name="image_a.png", dataset_key="bdd100k_det_100k", split="train"),
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            sample_uid = build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="dup")
            manifest_path = root / "image_list.jsonl"
            rows = [
                {
                    "sample_id": "dup",
                    "sample_uid": sample_uid,
                    "image_path": str(root / "images" / "train" / "image_a.png"),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                    "dataset_key": "bdd100k_det_100k",
                    "split": "train",
                },
                {
                    "sample_id": "dup",
                    "sample_uid": sample_uid,
                    "image_path": str(root / "images" / "train" / "image_b.png"),
                    "scene_path": str(scene_path),
                    "dataset_root": str(root),
                    "dataset_key": "bdd100k_det_100k",
                    "split": "train",
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate sample_uid"):
                load_image_list(manifest_path)

    def test_write_image_list_serializes_entries_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "image_list.jsonl"
            entries = (
                ImageListEntry(
                    sample_id="scene_a",
                    sample_uid=build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="scene_a"),
                    image_path=root / "images" / "scene_a.png",
                    scene_path=root / "labels_scene" / "scene_a.json",
                    dataset_root=root,
                    dataset_key="aihub_traffic_seoul",
                    split="train",
                    source_name="aihub",
                ),
            )

            written_path = write_image_list(manifest_path, entries)

            self.assertEqual(written_path, manifest_path)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                json.dumps(entries[0].to_dict(), ensure_ascii=True) + "\n",
            )
