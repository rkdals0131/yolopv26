from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.od_bootstrap.preprocess.sources import CanonicalSourceBundle
from tools.od_bootstrap.preprocess.teacher_dataset import build_teacher_datasets


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        ["convert", "-size", f"{width}x{height}", f"xc:{color}", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class ODBootstrapTeacherDatasetTests(unittest.TestCase):
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
                log_fn=logs.append,
            )

            self.assertEqual(sorted(results), ["mobility", "obstacle", "signal"])
            mobility = results["mobility"]
            signal = results["signal"]
            obstacle = results["obstacle"]

            mobility_label = (mobility.dataset_root / "labels" / "train" / "bdd_train_001.txt").read_text(encoding="utf-8").splitlines()
            signal_label = (signal.dataset_root / "labels" / "train" / "aihub_traffic_train_001.txt").read_text(encoding="utf-8").splitlines()
            obstacle_label = (obstacle.dataset_root / "labels" / "train" / "aihub_obstacle_train_001.txt").read_text(encoding="utf-8").splitlines()

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
            self.assertEqual(signal_manifest["workers"], 2)
            self.assertEqual(signal_manifest["log_every"], 1)
            self.assertEqual(signal_manifest["samples"][0]["image_action"], "hardlink")

            lane_labels = list((signal.dataset_root / "labels").rglob("aihub_lane_train_001.txt"))
            self.assertEqual(lane_labels, [])

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
