from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.source.types import CanonicalSourceBundle
from tools.od_bootstrap.build.teacher_dataset import (
    TeacherDatasetBuildConfig,
    _load_scene,
    build_teacher_dataset,
    build_teacher_datasets,
)


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


class ODBootstrapTeacherDatasetTests(unittest.TestCase):
    def test_load_scene_rejects_non_object_json_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.json"
            _write_text(scene_path, "[]\n")

            with self.assertRaisesRegex(TypeError, "scene root must be an object"):
                _load_scene(scene_path)

    def test_build_teacher_dataset_rejects_source_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _make_image(bdd_root / "images" / "val" / "sample.jpg", 1280, 720, "#222222")
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.jpg"},
                    "source": {"dataset": "bdd100k_det_100k", "split": "val"},
                },
            )
            _write_text(bdd_root / "labels_det" / "val" / "sample.txt", "0 0.5 0.5 0.1 0.1\n")
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            with self.assertRaisesRegex(ValueError, "scene source.split must match labels_scene split"):
                build_teacher_dataset(
                    bundle,
                    "mobility",
                    config=TeacherDatasetBuildConfig(output_root=root / "teacher_datasets"),
                )

    def test_build_teacher_dataset_rejects_image_file_name_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "../sample.jpg"},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                },
            )
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            with self.assertRaisesRegex(ValueError, "image.file_name must be a file name"):
                build_teacher_dataset(
                    bundle,
                    "mobility",
                    config=TeacherDatasetBuildConfig(output_root=root / "teacher_datasets"),
                )

    def test_build_teacher_dataset_uses_labels_det_not_scene_detections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _make_image(bdd_root / "images" / "train" / "sample.jpg", 1280, 720, "#222222")
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.jpg", "width": 1280, "height": 720},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    "detections": [
                        {
                            "id": 0,
                            "class_name": "bike",
                            "bbox": [1.0, 1.0, 2.0, 2.0],
                        }
                    ],
                },
            )
            _write_text(
                bdd_root / "labels_det" / "train" / "sample.txt",
                "0 0.500000 0.500000 0.250000 0.250000\n",
            )
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            result = build_teacher_dataset(
                bundle,
                "mobility",
                config=TeacherDatasetBuildConfig(
                    output_root=root / "pv26_od_bootstrap" / "teacher_datasets",
                    workers=1,
                    debug_vis_count=0,
                ),
            )

            label_rows = (result.dataset_root / "labels" / "train" / "sample.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(label_rows, ["0 0.500000 0.500000 0.250000 0.250000"])
            self.assertEqual(result.detection_count, 1)

    def test_build_teacher_dataset_rejects_missing_det_label_when_scene_requires_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _make_image(bdd_root / "images" / "train" / "sample.jpg", 1280, 720, "#222222")
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.jpg", "width": 1280, "height": 720},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    "tasks": {"has_det": 1},
                    "detections": [
                        {
                            "id": 0,
                            "class_name": "vehicle",
                            "bbox": [100.0, 100.0, 300.0, 300.0],
                        }
                    ],
                },
            )
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            with self.assertRaisesRegex(FileNotFoundError, "teacher dataset det label missing"):
                build_teacher_dataset(
                    bundle,
                    "mobility",
                    config=TeacherDatasetBuildConfig(
                        output_root=root / "pv26_od_bootstrap" / "teacher_datasets",
                        workers=1,
                        debug_vis_count=0,
                    ),
                )

    def test_build_teacher_dataset_rejects_stale_det_label_when_scene_has_no_det(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _make_image(bdd_root / "images" / "train" / "sample.jpg", 1280, 720, "#222222")
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.jpg", "width": 1280, "height": 720},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    "tasks": {"has_det": 0},
                    "detections": [],
                },
            )
            _write_text(
                bdd_root / "labels_det" / "train" / "sample.txt",
                "0 0.500000 0.500000 0.250000 0.250000\n",
            )
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            with self.assertRaisesRegex(ValueError, "teacher dataset stale det label"):
                build_teacher_dataset(
                    bundle,
                    "mobility",
                    config=TeacherDatasetBuildConfig(
                        output_root=root / "pv26_od_bootstrap" / "teacher_datasets",
                        workers=1,
                        debug_vis_count=0,
                    ),
                )

    def test_build_teacher_dataset_rejects_malformed_det_label_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "canonical" / "bdd100k_det_100k"
            aihub_root = root / "canonical" / "aihub_standardized"
            _make_image(bdd_root / "images" / "train" / "sample.jpg", 1280, 720, "#222222")
            _write_json(
                bdd_root / "labels_scene" / "train" / "sample.json",
                {
                    "image": {"file_name": "sample.jpg", "width": 1280, "height": 720},
                    "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                    "tasks": {"has_det": 1},
                    "detections": [
                        {
                            "id": 0,
                            "class_name": "vehicle",
                            "bbox": [100.0, 100.0, 300.0, 300.0],
                        }
                    ],
                },
            )
            _write_text(
                bdd_root / "labels_det" / "train" / "sample.txt",
                "0 0.500000 0.500000 0.250000\n",
            )
            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            with self.assertRaisesRegex(ValueError, "teacher dataset malformed det label row"):
                build_teacher_dataset(
                    bundle,
                    "mobility",
                    config=TeacherDatasetBuildConfig(
                        output_root=root / "pv26_od_bootstrap" / "teacher_datasets",
                        workers=1,
                        debug_vis_count=0,
                    ),
                )

    def test_build_teacher_datasets_filters_lane_and_remaps_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_root = root / "pv26_od_bootstrap" / "canonical"
            bdd_root = canonical_root / "bdd100k_det_100k"
            aihub_root = canonical_root / "aihub_standardized"

            self._create_bdd_canonical_fixture(bdd_root)
            self._create_aihub_canonical_fixture(aihub_root)

            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )
            logs: list[str] = []
            results = build_teacher_datasets(
                bundle,
                root / "pv26_od_bootstrap" / "teacher_datasets",
                workers=2,
                log_every=1,
                debug_vis_count=1,
                debug_vis_seed=26,
                log_fn=logs.append,
            )

            self.assertEqual(sorted(results), ["mobility", "obstacle", "signal"])
            mobility = results["mobility"]
            signal = results["signal"]
            obstacle = results["obstacle"]

            mobility_label = (mobility.dataset_root / "labels" / "train" / "bdd_train_001.txt").read_text(encoding="utf-8").splitlines()
            signal_label = (signal.dataset_root / "labels" / "train" / "aihub_traffic_train_001.txt").read_text(encoding="utf-8").splitlines()
            obstacle_label = (obstacle.dataset_root / "labels" / "train" / "aihub_obstacle_train_001.txt").read_text(encoding="utf-8").splitlines()

            self.assertEqual(
                signal_label,
                [
                    "0 0.100000 0.100000 0.200000 0.200000",
                    "1 0.300000 0.300000 0.200000 0.200000",
                ],
            )
            self.assertEqual(mobility_label[0].split()[0], "0")
            self.assertEqual(mobility_label[1].split()[0], "1")
            self.assertEqual(mobility_label[2].split()[0], "2")
            self.assertEqual(signal_label[0].split()[0], "0")
            self.assertEqual(signal_label[1].split()[0], "1")
            self.assertEqual(obstacle_label[0].split()[0], "0")
            self.assertEqual(obstacle_label[1].split()[0], "1")
            self.assertFalse((signal.dataset_root / "data.yaml").exists())
            self.assertTrue((obstacle.dataset_root / "meta" / "teacher_dataset_manifest.json").is_file())
            self.assertEqual(mobility.sample_count, 1)
            self.assertEqual(signal.sample_count, 1)
            self.assertEqual(obstacle.sample_count, 1)
            self.assertEqual(signal.detection_count, 2)
            self.assertEqual(obstacle.detection_count, 2)
            self.assertIn("[teacher:mobility] progress 1/1 samples", " ".join(logs))
            self.assertIn("[teacher:signal] done samples=1 detections=2", " ".join(logs))

            signal_manifest = json.loads(
                (signal.dataset_root / "meta" / "teacher_dataset_manifest.json").read_text(encoding="utf-8")
            )
            signal_sample = signal_manifest["samples"][0]
            self.assertEqual(signal_manifest["workers"], 2)
            self.assertEqual(signal_manifest["log_every"], 1)
            self.assertEqual(signal_manifest["debug_vis_count"], 1)
            self.assertEqual(signal_manifest["debug_vis_seed"], 26)
            self.assertEqual(signal_manifest["source_dataset_keys"], ["aihub_traffic_seoul"])
            self.assertEqual(signal_manifest["class_names"], ["traffic_light", "sign"])
            self.assertEqual(signal_sample["teacher_name"], "signal")
            self.assertEqual(signal_sample["source_dataset_key"], "aihub_traffic_seoul")
            self.assertEqual(signal_sample["split"], "train")
            self.assertEqual(signal_sample["sample_id"], "aihub_traffic_train_001")
            self.assertEqual(signal_sample["sample_uid"], "aihub_traffic_seoul__train__aihub_traffic_train_001")
            self.assertEqual(signal_sample["image_action"], "hardlink")
            self.assertEqual(signal_sample["detection_count"], len(signal_label))
            self.assertEqual(
                signal_sample["source_scene_path"],
                str(aihub_root / "labels_scene" / "train" / "aihub_traffic_train_001.json"),
            )
            self.assertEqual(
                signal_sample["source_label_path"],
                str(aihub_root / "labels_det" / "train" / "aihub_traffic_train_001.txt"),
            )
            self.assertEqual(
                signal_sample["source_image_path"],
                str(aihub_root / "images" / "train" / "aihub_traffic_train_001.png"),
            )
            self.assertEqual(
                Path(signal_sample["output_label_path"]).read_text(encoding="utf-8").splitlines(),
                signal_label,
            )
            self.assertTrue(Path(signal_sample["output_image_path"]).is_file())
            self.assertTrue(signal.debug_vis_manifest_path.is_file())
            debug_vis_manifest = json.loads(signal.debug_vis_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(debug_vis_manifest["selection_count"], 1)
            debug_vis_dir = signal.dataset_root / "meta" / "debug_vis"
            overlay_files = sorted(debug_vis_dir.glob("*.png"))
            self.assertEqual(len(overlay_files), 1)
            self.assertEqual(sorted(path.name for path in debug_vis_dir.iterdir()), [overlay_files[0].name])
            self.assertTrue(Path(debug_vis_manifest["items"][0]["overlay_path"]).is_file())

            lane_labels = list((signal.dataset_root / "labels").rglob("aihub_lane_train_001.txt"))
            self.assertEqual(lane_labels, [])

    def test_build_teacher_dataset_keeps_same_sample_id_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_root = root / "pv26_od_bootstrap" / "canonical"
            bdd_root = canonical_root / "bdd100k_det_100k"
            aihub_root = canonical_root / "aihub_standardized"
            for split, color in (("train", "#222222"), ("val", "#333333")):
                _make_image(bdd_root / "images" / split / "shared.jpg", 1280, 720, color)
                _write_json(
                    bdd_root / "labels_scene" / split / "shared.json",
                    {
                        "version": "test",
                        "image": {
                            "file_name": "shared.jpg",
                            "width": 1280,
                            "height": 720,
                        },
                        "source": {
                            "dataset": "bdd100k_det_100k",
                            "split": split,
                        },
                    },
                )
                _write_text(
                    bdd_root / "labels_det" / split / "shared.txt",
                    "0 0.500000 0.500000 0.250000 0.250000\n",
                )

            bundle = CanonicalSourceBundle(
                bdd_root=bdd_root,
                aihub_root=aihub_root,
                output_root=root / "pv26_od_bootstrap",
            )

            result = build_teacher_dataset(
                bundle,
                "mobility",
                config=TeacherDatasetBuildConfig(
                    output_root=root / "pv26_od_bootstrap" / "teacher_datasets",
                    workers=1,
                    debug_vis_count=0,
                ),
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.sample_count, 2)
            self.assertEqual(
                [(row["split"], row["sample_uid"]) for row in manifest["samples"]],
                [
                    ("train", "bdd100k_det_100k__train__shared"),
                    ("val", "bdd100k_det_100k__val__shared"),
                ],
            )
            self.assertTrue((result.dataset_root / "labels" / "train" / "shared.txt").is_file())
            self.assertTrue((result.dataset_root / "labels" / "val" / "shared.txt").is_file())

    def _create_bdd_canonical_fixture(self, bdd_root: Path) -> None:
        _make_image(bdd_root / "images" / "train" / "bdd_train_001.jpg", 1280, 720, "#222222")
        _write_json(
            bdd_root / "labels_scene" / "train" / "bdd_train_001.json",
            {
                "version": "test",
                "image": {
                    "file_name": "bdd_train_001.jpg",
                    "width": 1280,
                    "height": 720,
                },
                "source": {
                    "dataset": "bdd100k_det_100k",
                    "split": "train",
                },
            },
        )
        _write_text(
            bdd_root / "labels_det" / "train" / "bdd_train_001.txt",
            "\n".join(
                [
                    "0 0.500000 0.500000 0.250000 0.250000",
                    "1 0.300000 0.300000 0.100000 0.100000",
                    "2 0.700000 0.700000 0.100000 0.100000",
                ]
            )
            + "\n",
        )

    def _create_aihub_canonical_fixture(self, aihub_root: Path) -> None:
        _make_image(aihub_root / "images" / "train" / "aihub_traffic_train_001.png", 1920, 1080, "#444444")
        _make_image(aihub_root / "images" / "train" / "aihub_obstacle_train_001.png", 1920, 1080, "#666666")
        _make_image(aihub_root / "images" / "train" / "aihub_lane_train_001.png", 1920, 1080, "#888888")

        _write_json(
            aihub_root / "labels_scene" / "train" / "aihub_traffic_train_001.json",
            {
                "version": "test",
                "image": {
                    "file_name": "aihub_traffic_train_001.png",
                    "width": 1920,
                    "height": 1080,
                },
                "source": {
                    "dataset": "aihub_traffic_seoul",
                    "split": "train",
                },
            },
        )
        _write_json(
            aihub_root / "labels_scene" / "train" / "aihub_obstacle_train_001.json",
            {
                "version": "test",
                "image": {
                    "file_name": "aihub_obstacle_train_001.png",
                    "width": 1920,
                    "height": 1080,
                },
                "source": {
                    "dataset": "aihub_obstacle_seoul",
                    "split": "train",
                },
            },
        )
        _write_json(
            aihub_root / "labels_scene" / "train" / "aihub_lane_train_001.json",
            {
                "version": "test",
                "image": {
                    "file_name": "aihub_lane_train_001.png",
                    "width": 1920,
                    "height": 1080,
                },
                "source": {
                    "dataset": "aihub_lane_seoul",
                    "split": "train",
                },
            },
        )
        _write_text(
            aihub_root / "labels_det" / "train" / "aihub_traffic_train_001.txt",
            "\n".join(
                [
                    "5 0.100000 0.100000 0.200000 0.200000",
                    "6 0.300000 0.300000 0.200000 0.200000",
                    "0 0.900000 0.900000 0.100000 0.100000",
                ]
            )
            + "\n",
        )
        _write_text(
            aihub_root / "labels_det" / "train" / "aihub_obstacle_train_001.txt",
            "\n".join(
                [
                    "3 0.200000 0.200000 0.200000 0.200000",
                    "4 0.400000 0.400000 0.200000 0.200000",
                ]
            )
            + "\n",
        )
        _write_text(
            aihub_root / "labels_det" / "train" / "aihub_lane_train_001.txt",
            "0 0.500000 0.500000 0.100000 0.100000\n",
        )
