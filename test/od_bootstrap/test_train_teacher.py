from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.od_bootstrap.train.data_yaml import build_teacher_data_yaml, resolve_teacher_dataset_root
from tools.od_bootstrap.train.run_train_teacher import run_teacher_train_scenario
from tools.od_bootstrap.train.scenario import load_teacher_train_scenario


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _FakeYOLO:
    last_instance: "_FakeYOLO | None" = None

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.trainer = None
        self.last_train_kwargs: dict | None = None
        _FakeYOLO.last_instance = self

    def train(self, **kwargs):
        self.last_train_kwargs = dict(kwargs)
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_text("best", encoding="utf-8")
        (weights_dir / "last.pt").write_text("last", encoding="utf-8")
        self.trainer = SimpleNamespace(save_dir=save_dir)
        return SimpleNamespace(save_dir=save_dir)


class TeacherTrainTests(unittest.TestCase):
    def test_resolve_teacher_dataset_root_and_build_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")

            dataset_root = resolve_teacher_dataset_root(
                source_root=source_root,
                image_dir="images",
                label_dir="labels",
            )
            data_yaml = build_teacher_data_yaml(
                dataset_root=dataset_root,
                class_names=("vehicle", "bike", "pedestrian"),
                output_path=root / "data.yaml",
            )

            self.assertEqual(dataset_root, source_root.resolve())
            self.assertTrue((dataset_root / "images" / "train").exists())
            self.assertTrue((dataset_root / "labels" / "train").exists())
            content = data_yaml.read_text(encoding="utf-8")
            self.assertIn("vehicle", content)
            self.assertIn("images/train", content)

    def test_run_teacher_train_scenario_writes_summary_and_uses_ultralytics_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = root / "train.yaml"
            source_root = root / "teacher_source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    teacher_name: mobility
                    run:
                      output_root: runs
                      exist_ok: true
                    dataset:
                      root: teacher_source
                      image_dir: images
                      label_dir: labels
                    model:
                      model_size: n
                      weights: yolo26n.pt
                      class_names: [vehicle, bike, pedestrian]
                    train:
                      epochs: 2
                      imgsz: 640
                      batch: 2
                      device: cpu
                      workers: 1
                      pin_memory: true
                      persistent_workers: true
                      prefetch_factor: 3
                      patience: 1
                      cache: false
                      amp: false
                      optimizer: auto
                      seed: 7
                      resume: false
                      val: true
                      save_period: 1
                      log_every_n_steps: 5
                      profile_window: 7
                      profile_device_sync: false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_teacher_train_scenario(scenario_path)
            with patch("tools.od_bootstrap.train.ultralytics_runner.YOLO", _FakeYOLO):
                summary = run_teacher_train_scenario(scenario, scenario_path=scenario_path)

            self.assertEqual(summary["teacher_name"], "mobility")
            self.assertEqual(summary["dataset_root"], str(source_root.resolve()))
            self.assertEqual(summary["train_summary"]["weights"], "yolo26n.pt")
            self.assertEqual(summary["train"]["prefetch_factor"], 3)
            self.assertEqual(summary["train_summary"]["runtime"]["prefetch_factor"], 3)
            self.assertEqual(summary["train_summary"]["runtime"]["profile_window"], 7)
            run_dir = Path(summary["train_summary"]["run_dir"])
            self.assertEqual(run_dir.parent, root / "runs" / "mobility")
            self.assertRegex(run_dir.name, r"^\d{8}_\d{6}$")
            self.assertTrue(summary["train_summary"]["tensorboard_dir"].endswith(f"/runs/mobility/{run_dir.name}/tensorboard"))
            self.assertTrue(Path(summary["train_summary"]["best_checkpoint"]).is_file())
            self.assertTrue((root / "runs" / "mobility" / "weights" / "best.pt").is_file())
            self.assertTrue((root / "runs" / "mobility" / "train_summary.json").is_file())
            latest_run_payload = json.loads((root / "runs" / "mobility" / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_run_payload["run_dir"], str(run_dir))
            self.assertTrue(Path(summary["data_yaml_path"]).is_file())
            self.assertFalse((root / "runs" / "mobility" / "dataset").exists())
            self.assertTrue((root / "runs" / "mobility" / "run_summary.json").is_file())
            self.assertNotIn("pin_memory", _FakeYOLO.last_instance.last_train_kwargs)
            self.assertNotIn("prefetch_factor", _FakeYOLO.last_instance.last_train_kwargs)
            self.assertEqual(_FakeYOLO.last_instance.last_train_kwargs["project"], str(root / "runs" / "mobility"))
            self.assertRegex(_FakeYOLO.last_instance.last_train_kwargs["name"], r"^\d{8}_\d{6}$")

    def test_run_teacher_train_scenario_scopes_resume_to_teacher_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = root / "train.yaml"
            source_root = root / "teacher_source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")

            scenario_path.write_text(
                textwrap.dedent(
                    """
                    teacher_name: signal
                    run:
                      output_root: runs
                      exist_ok: true
                    dataset:
                      root: teacher_source
                      image_dir: images
                      label_dir: labels
                    model:
                      model_size: n
                      weights: yolo26n.pt
                      class_names: [traffic_light, sign]
                    train:
                      epochs: 72
                      imgsz: 640
                      batch: 2
                      device: cpu
                      workers: 1
                      pin_memory: true
                      persistent_workers: true
                      prefetch_factor: 3
                      patience: 1
                      cache: false
                      amp: false
                      optimizer: auto
                      seed: 7
                      resume: true
                      val: true
                      save_period: 1
                      log_every_n_steps: 5
                      profile_window: 7
                      profile_device_sync: false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            teacher_root = root / "runs" / "signal"
            older_signal = teacher_root / "20260328_040148" / "weights" / "last.pt"
            newer_signal = teacher_root / "20260328_050148" / "weights" / "last.pt"
            unrelated_newest = root / "runs" / "obstacle" / "20260328_060148" / "weights" / "last.pt"
            for path in (older_signal, newer_signal, unrelated_newest):
                _write_text(path, "last")
            os.utime(older_signal, (100, 100))
            os.utime(newer_signal, (200, 200))
            os.utime(unrelated_newest, (300, 300))

            scenario = load_teacher_train_scenario(scenario_path)
            with patch("tools.od_bootstrap.train.ultralytics_runner.YOLO", _FakeYOLO):
                run_teacher_train_scenario(scenario, scenario_path=scenario_path)

            self.assertEqual(_FakeYOLO.last_instance.weights, str(newer_signal))
            self.assertEqual(_FakeYOLO.last_instance.last_train_kwargs["resume"], str(newer_signal))

    def test_run_teacher_train_scenario_resume_requires_local_teacher_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = root / "train.yaml"
            source_root = root / "teacher_source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")

            scenario_path.write_text(
                textwrap.dedent(
                    """
                    teacher_name: signal
                    run:
                      output_root: runs
                      exist_ok: true
                    dataset:
                      root: teacher_source
                      image_dir: images
                      label_dir: labels
                    model:
                      model_size: n
                      weights: yolo26n.pt
                      class_names: [traffic_light, sign]
                    train:
                      epochs: 72
                      imgsz: 640
                      batch: 2
                      device: cpu
                      workers: 1
                      pin_memory: true
                      persistent_workers: true
                      prefetch_factor: 3
                      patience: 1
                      cache: false
                      amp: false
                      optimizer: auto
                      seed: 7
                      resume: true
                      val: true
                      save_period: 1
                      log_every_n_steps: 5
                      profile_window: 7
                      profile_device_sync: false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_teacher_train_scenario(scenario_path)
            with patch("tools.od_bootstrap.train.ultralytics_runner.YOLO", _FakeYOLO):
                with self.assertRaisesRegex(FileNotFoundError, "no last.pt exists under"):
                    run_teacher_train_scenario(scenario, scenario_path=scenario_path)
