from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch

import torch

from tools.od_bootstrap.presets import build_teacher_eval_preset
from tools.od_bootstrap.teacher.eval import eval_teacher_checkpoint


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
    def test_shipped_eval_default_uses_teacher_name_checkpoint_path(self) -> None:
        scenario = build_teacher_eval_preset("mobility")

        self.assertEqual(scenario.teacher_name, "mobility")
        self.assertEqual(
            scenario.model.checkpoint_path,
            (Path(__file__).resolve().parents[2] / "runs" / "od_bootstrap" / "train" / "mobility" / "weights" / "best.pt").resolve(),
        )

    def test_eval_teacher_checkpoint_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "teacher_source"
            for split in ("train", "val"):
                _write_text(source_root / "images" / split / f"{split}.jpg", "img")
                _write_text(source_root / "labels" / split / f"{split}.txt", "0 0.5 0.5 0.2 0.2\n")
            checkpoint_path = root / "weights" / "best.pt"
            _write_text(checkpoint_path, "checkpoint")
            scenario = replace(
                build_teacher_eval_preset("mobility"),
                run=replace(build_teacher_eval_preset("mobility").run, output_root=root / "runs"),
                dataset=replace(
                    build_teacher_eval_preset("mobility").dataset,
                    root=source_root,
                    sample_limit=2,
                ),
                model=replace(
                    build_teacher_eval_preset("mobility").model,
                    checkpoint_path=checkpoint_path,
                    class_names=("vehicle", "bike", "pedestrian"),
                ),
            )
            with patch("tools.od_bootstrap.teacher.eval.YOLO", _FakeYOLO):
                summary = eval_teacher_checkpoint(scenario=scenario, scenario_path=root / "preset_eval")

            self.assertEqual(summary["teacher_name"], "mobility")
            self.assertEqual(summary["prediction_summary"]["prediction_count"], 3)
            self.assertEqual(summary["prediction_summary"]["class_counts"]["vehicle"], 2)
            self.assertAlmostEqual(summary["prediction_summary"]["confidence"]["max"], 0.91, places=6)
            self.assertIn("box.map50", summary["val_summary"]["results_dict"])
            self.assertTrue(Path(summary["predictions_path"]).is_file())
            self.assertEqual(summary["dataset_root"], str(source_root.resolve()))
            self.assertFalse((root / "runs" / "mobility" / "dataset").exists())
            self.assertTrue((root / "runs" / "mobility" / "checkpoint_eval_summary.json").is_file())
