from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.od_bootstrap.train.data_yaml import TeacherDatasetLayout, build_teacher_data_yaml, stage_teacher_dataset_layout
from tools.od_bootstrap.train.run_train_teacher import run_teacher_train_scenario
from tools.od_bootstrap.train.scenario import load_teacher_train_scenario


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _FakeYOLO:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.trainer = None
        self.last_train_kwargs: dict | None = None

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
    def test_stage_teacher_dataset_layout_and_build_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels_det" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")

            staged_root = stage_teacher_dataset_layout(
                TeacherDatasetLayout(
                    source_root=source_root,
                    staging_root=root / "staged",
                    image_dir="images",
                    label_dir="labels_det",
                )
            )
            data_yaml = build_teacher_data_yaml(
                dataset_root=staged_root,
                class_names=("vehicle", "bike", "pedestrian"),
                output_path=root / "staged" / "data.yaml",
            )

            self.assertTrue((staged_root / "images" / "train").exists())
            self.assertTrue((staged_root / "labels" / "train").exists())
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
                _write_text(source_root / "labels_det" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    teacher_name: mobility
                    run:
                      output_root: runs
                      run_name: mobility_yolo26n
                      exist_ok: true
                    dataset:
                      root: teacher_source
                      image_dir: images
                      label_dir: labels_det
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
                      patience: 1
                      cache: false
                      amp: false
                      optimizer: auto
                      seed: 7
                      resume: false
                      val: true
                      save_period: 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_teacher_train_scenario(scenario_path)
            with patch("tools.od_bootstrap.train.ultralytics_runner.YOLO", _FakeYOLO):
                summary = run_teacher_train_scenario(scenario, scenario_path=scenario_path)

            self.assertEqual(summary["teacher_name"], "mobility")
            self.assertEqual(summary["train_summary"]["weights"], "yolo26n.pt")
            self.assertTrue(Path(summary["train_summary"]["best_checkpoint"]).is_file())
            self.assertTrue(Path(summary["data_yaml_path"]).is_file())
            self.assertTrue((root / "runs" / "mobility" / "run_summary.json").is_file())
            self.assertTrue((root / "runs" / "mobility" / "train_summary.json").is_file())
