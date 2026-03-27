from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.od_bootstrap.eval.checkpoint_eval import eval_teacher_checkpoint
from tools.od_bootstrap.eval.scenario import load_teacher_checkpoint_eval_scenario


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _FakeYOLO:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def predict(self, **kwargs):
        del kwargs
        result_1 = SimpleNamespace(
            path="/tmp/sample_1.jpg",
            names={0: "vehicle", 1: "bike"},
            boxes=SimpleNamespace(
                xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
                cls=torch.tensor([0, 1]),
                conf=torch.tensor([0.91, 0.77]),
            ),
        )
        result_2 = SimpleNamespace(
            path="/tmp/sample_2.jpg",
            names={0: "vehicle", 1: "bike"},
            boxes=SimpleNamespace(
                xyxy=torch.tensor([[15.0, 25.0, 35.0, 45.0]]),
                cls=torch.tensor([0]),
                conf=torch.tensor([0.66]),
            ),
        )
        return [result_1, result_2]

    def val(self, **kwargs):
        del kwargs
        return {"results_dict": {"box.map50": 0.42, "box.map": 0.31}}


class CheckpointEvalTests(unittest.TestCase):
    def test_eval_teacher_checkpoint_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "teacher_source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels_det" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")
            checkpoint_path = root / "weights" / "best.pt"
            _write_text(checkpoint_path, "checkpoint")
            scenario_path = root / "eval.yaml"
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    teacher_name: mobility
                    run:
                      output_root: runs
                      run_name: mobility_checkpoint_eval
                      exist_ok: true
                    dataset:
                      root: teacher_source
                      image_dir: images
                      label_dir: labels_det
                      split: val
                      sample_limit: 2
                      stage_dataset: true
                    model:
                      checkpoint_path: weights/best.pt
                      model_size: n
                      class_names: [vehicle, bike, pedestrian]
                    eval:
                      imgsz: 640
                      batch: 1
                      device: cpu
                      conf: 0.25
                      iou: 0.7
                      predict: true
                      val: true
                      save_conf: false
                      verbose: false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_teacher_checkpoint_eval_scenario(scenario_path)
            with patch("tools.od_bootstrap.eval.checkpoint_eval.YOLO", _FakeYOLO):
                summary = eval_teacher_checkpoint(scenario=scenario, scenario_path=scenario_path)

            self.assertEqual(summary["teacher_name"], "mobility")
            self.assertEqual(summary["prediction_summary"]["prediction_count"], 3)
            self.assertEqual(summary["prediction_summary"]["class_counts"]["vehicle"], 2)
            self.assertAlmostEqual(summary["prediction_summary"]["confidence"]["max"], 0.91, places=6)
            self.assertIn("box.map50", summary["val_summary"]["results_dict"])
            self.assertTrue(Path(summary["predictions_path"]).is_file())
            self.assertTrue((root / "runs" / "mobility" / "checkpoint_eval_summary.json").is_file())
