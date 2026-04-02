from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from common.io import read_yaml
from tools.od_bootstrap.teacher.calibration_types import (
    CalibrationDatasetConfig,
    CalibrationScenario,
    CalibrationTeacherConfig,
    HardNegativeConfig,
)
from tools.od_bootstrap.presets import build_calibration_preset
from tools.od_bootstrap.teacher.calibrate import calibrate_class_policy_scenario
from tools.od_bootstrap.build.sweep_types import ClassPolicy


class _FakeYOLO:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def predict(self, **kwargs):
        results = []
        for item in kwargs["source"]:
            image_path = Path(item)
            if image_path.stem == "frame_001":
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]]),
                    cls=torch.tensor([0, 0]),
                    conf=torch.tensor([0.95, 0.85]),
                )
            else:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.55]),
                )
            results.append(
                SimpleNamespace(
                    path=str(image_path),
                    names={0: "vehicle"},
                    boxes=boxes,
                    orig_shape=(100, 100),
                )
            )
        return results


class _HardNegativeAwareFakeYOLO:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def predict(self, **kwargs):
        results = []
        for item in kwargs["source"]:
            image_path = Path(item)
            if image_path.stem == "frame_001":
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.95]),
                )
            elif image_path.stem == "frame_002":
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.85]),
                )
            else:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[65.0, 65.0, 85.0, 85.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.65]),
                )
            results.append(
                SimpleNamespace(
                    path=str(image_path),
                    names={0: "vehicle"},
                    boxes=boxes,
                    orig_shape=(100, 100),
                )
            )
        return results


class ODBootstrapCalibrationTests(unittest.TestCase):
    def test_build_calibration_preset_allows_teacher_specific_imgsz_override(self) -> None:
        hyperparameters = {
            "od_bootstrap": {
                "calibration": {
                    "run": {"imgsz": 640},
                    "teachers": {
                        "signal": {"imgsz": 960},
                    },
                }
            }
        }
        with patch("tools.od_bootstrap.presets.load_user_paths_config", return_value={}), patch(
            "tools.od_bootstrap.presets.load_user_hyperparameters_config",
            return_value=hyperparameters,
        ):
            scenario = build_calibration_preset()

        teacher_map = {teacher.name: teacher for teacher in scenario.teachers}
        self.assertIsNone(teacher_map["mobility"].imgsz)
        self.assertEqual(teacher_map["signal"].imgsz, 960)

    def _build_scenario(
        self,
        *,
        output_root: Path,
        teachers: tuple[CalibrationTeacherConfig, ...],
        search: dict[str, object] | None = None,
        policy_template_path: Path | None = None,
        policy_template: dict[str, object] | None = None,
        hard_negative: HardNegativeConfig | None = None,
    ) -> CalibrationScenario:
        preset = build_calibration_preset()
        return replace(
            preset,
            run=replace(preset.run, output_root=output_root, device="cpu", imgsz=640, batch_size=2, predict_conf=0.001, predict_iou=0.99),
            search=replace(preset.search, **(search or {})),
            teachers=teachers,
            policy_template_path=policy_template_path,
            policy_template=policy_template,
            hard_negative=hard_negative,
        )

    def test_calibration_selects_precision_constrained_policy_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "teacher_dataset"
            for sample_id in ("frame_001", "frame_002"):
                (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
                (dataset_root / "images" / "val" / f"{sample_id}.jpg").write_bytes(b"img")
            (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "val" / "frame_001.txt").write_text("0 0.200000 0.200000 0.200000 0.200000\n", encoding="utf-8")
            (dataset_root / "labels" / "val" / "frame_002.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            checkpoint_path = root / "weights" / "mobility.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            scenario = self._build_scenario(
                output_root=root / "calibration",
                teachers=(
                    CalibrationTeacherConfig(
                        name="mobility",
                        checkpoint_path=root / "weights" / "mobility.pt",
                        model_version="mobility_v1",
                        dataset=CalibrationDatasetConfig(
                            root=dataset_root,
                            source_dataset_key="bdd100k_det_100k",
                            image_dir="images",
                            label_dir="labels",
                            split="val",
                        ),
                        classes=("vehicle",),
                        imgsz=960,
                    ),
                ),
                search={
                    "match_iou": 0.5,
                    "min_precision": 0.90,
                    "score_thresholds": (0.50, 0.90),
                    "nms_iou_thresholds": (0.50,),
                    "min_box_sizes": (4,),
                },
                policy_template_path=root / "template.yaml",
                policy_template={
                    "vehicle": ClassPolicy(
                        score_threshold=0.50,
                        nms_iou_threshold=0.50,
                        min_box_size=4,
                        center_y_range=(0.0, 0.8),
                    )
                },
            )
            with patch("tools.od_bootstrap.teacher.calibrate.YOLO", _FakeYOLO):
                summary = calibrate_class_policy_scenario(scenario, scenario_path=root / "preset_calibration")

            class_policy = read_yaml(summary["class_policy_path"])
            report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(class_policy["vehicle"]["score_threshold"], 0.9)
            self.assertEqual(class_policy["vehicle"]["center_y_range"], [0.0, 0.8])
            self.assertTrue(report["classes"]["vehicle"]["meets_precision_floor"])
            self.assertAlmostEqual(report["classes"]["vehicle"]["metrics"]["precision"], 1.0, places=6)
            self.assertTrue((Path(summary["output_root"]) / "teachers" / "mobility" / "predictions.jsonl").is_file())
            self.assertTrue(Path(summary["hard_negative_manifest_path"]).is_file())
            self.assertEqual(summary["teachers"][0]["resolved_runtime"]["imgsz"], 960)

    def test_calibration_loads_policy_template_from_yaml_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "teacher_dataset"
            for sample_id in ("frame_001", "frame_002"):
                (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
                (dataset_root / "images" / "val" / f"{sample_id}.jpg").write_bytes(b"img")
            (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "val" / "frame_001.txt").write_text("0 0.200000 0.200000 0.200000 0.200000\n", encoding="utf-8")
            (dataset_root / "labels" / "val" / "frame_002.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            checkpoint_path = root / "weights" / "mobility.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            template_path = root / "template.yaml"
            template_path.write_text(
                "\n".join(
                    [
                        "vehicle:",
                        "  score_threshold: 0.50",
                        "  nms_iou_threshold: 0.50",
                        "  min_box_size: 4",
                        "  center_y_range: [0.0, 0.8]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scenario = self._build_scenario(
                output_root=root / "calibration",
                teachers=(
                    CalibrationTeacherConfig(
                        name="mobility",
                        checkpoint_path=checkpoint_path,
                        model_version="mobility_v1",
                        dataset=CalibrationDatasetConfig(
                            root=dataset_root,
                            source_dataset_key="bdd100k_det_100k",
                            image_dir="images",
                            label_dir="labels",
                            split="val",
                        ),
                        classes=("vehicle",),
                    ),
                ),
                search={
                    "match_iou": 0.5,
                    "min_precision": 0.90,
                    "score_thresholds": (0.50, 0.90),
                    "nms_iou_thresholds": (0.50,),
                    "min_box_sizes": (4,),
                },
                policy_template_path=template_path,
            )
            with patch("tools.od_bootstrap.teacher.calibrate.YOLO", _FakeYOLO):
                summary = calibrate_class_policy_scenario(scenario, scenario_path=root / "preset_calibration")

            class_policy = read_yaml(summary["class_policy_path"])
            self.assertEqual(class_policy["vehicle"]["score_threshold"], 0.9)
            self.assertEqual(class_policy["vehicle"]["center_y_range"], [0.0, 0.8])

    def test_calibration_uses_hard_negative_regression_set_as_tie_breaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "teacher_dataset"
            image_dir = dataset_root / "images" / "val"
            label_dir = dataset_root / "labels" / "val"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in ("frame_001", "frame_002", "frame_003"):
                (image_dir / f"{sample_id}.jpg").write_bytes(b"img")
            (label_dir / "frame_001.txt").write_text("0 0.200000 0.200000 0.200000 0.200000\n", encoding="utf-8")
            (label_dir / "frame_002.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            (label_dir / "frame_003.txt").write_text("0 0.800000 0.800000 0.200000 0.200000\n", encoding="utf-8")

            hard_negative_image = root / "hard_negatives" / "hn_001.jpg"
            hard_negative_image.parent.mkdir(parents=True, exist_ok=True)
            hard_negative_image.write_bytes(b"hn")

            checkpoint_path = root / "weights" / "mobility.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            hard_negative_manifest = root / "hard_negative_manifest.json"
            hard_negative_manifest.write_text(
                json.dumps(
                    {
                        "version": "od-bootstrap-hard-negative-v1",
                        "classes": {
                            "vehicle": [
                                {
                                    "sample_id": "hn_001",
                                    "image_path": "hard_negatives/hn_001.jpg",
                                    "dataset_key": "bdd100k_det_100k",
                                    "teacher_name": "mobility",
                                }
                            ]
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            scenario = self._build_scenario(
                output_root=root / "calibration",
                teachers=(
                    CalibrationTeacherConfig(
                        name="mobility",
                        checkpoint_path=checkpoint_path,
                        model_version="mobility_v1",
                        dataset=CalibrationDatasetConfig(
                            root=dataset_root,
                            source_dataset_key="bdd100k_det_100k",
                            image_dir="images",
                            label_dir="labels",
                            split="val",
                        ),
                        classes=("vehicle",),
                    ),
                ),
                search={
                    "match_iou": 0.5,
                    "min_precision": 0.90,
                    "score_thresholds": (0.50, 0.80),
                    "nms_iou_thresholds": (0.50,),
                    "min_box_sizes": (4,),
                },
                hard_negative=HardNegativeConfig(
                    manifest_path=hard_negative_manifest,
                    top_k_per_class=5,
                    focus_classes=("vehicle",),
                ),
            )
            with patch("tools.od_bootstrap.teacher.calibrate.YOLO", _HardNegativeAwareFakeYOLO):
                summary = calibrate_class_policy_scenario(scenario, scenario_path=root / "preset_calibration")

            class_policy = read_yaml(summary["class_policy_path"])
            report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(class_policy["vehicle"]["score_threshold"], 0.8)
            self.assertEqual(report["classes"]["vehicle"]["metrics"]["hard_negative_sample_count"], 1)
            self.assertEqual(report["classes"]["vehicle"]["metrics"]["hard_negative_failures"], 0)
            self.assertEqual(report["hard_negative"]["input_sample_count_by_class"]["vehicle"], 1)

    def test_calibration_allows_class_specific_precision_floor_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "teacher_dataset"
            image_dir = dataset_root / "images" / "val"
            label_dir = dataset_root / "labels" / "val"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in ("frame_001", "frame_002", "frame_003"):
                (image_dir / f"{sample_id}.jpg").write_bytes(b"img")
            (label_dir / "frame_001.txt").write_text("0 0.200000 0.200000 0.200000 0.200000\n", encoding="utf-8")
            (label_dir / "frame_002.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            (label_dir / "frame_003.txt").write_text("0 0.800000 0.800000 0.200000 0.200000\n", encoding="utf-8")

            checkpoint_path = root / "weights" / "signal.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            class _SignalPrecisionOverrideFakeYOLO:
                def __init__(self, checkpoint_path: str) -> None:
                    self.checkpoint_path = checkpoint_path

                def predict(self, **kwargs):
                    results = []
                    for item in kwargs["source"]:
                        image_path = Path(item)
                        if image_path.stem == "frame_001":
                            boxes = SimpleNamespace(
                                xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                                cls=torch.tensor([0]),
                                conf=torch.tensor([0.80]),
                            )
                        elif image_path.stem == "frame_002":
                            boxes = SimpleNamespace(
                                xyxy=torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
                                cls=torch.tensor([0]),
                                conf=torch.tensor([0.80]),
                            )
                        else:
                            boxes = SimpleNamespace(
                                xyxy=torch.tensor([[70.0, 70.0, 90.0, 90.0], [10.0, 10.0, 20.0, 20.0]]),
                                cls=torch.tensor([0, 0]),
                                conf=torch.tensor([0.50, 0.50]),
                            )
                        results.append(
                            SimpleNamespace(
                                path=str(image_path),
                                names={0: "traffic_light"},
                                boxes=boxes,
                                orig_shape=(100, 100),
                            )
                        )
                    return results

            scenario = self._build_scenario(
                output_root=root / "calibration",
                teachers=(
                    CalibrationTeacherConfig(
                        name="signal",
                        checkpoint_path=checkpoint_path,
                        model_version="signal_v1",
                        dataset=CalibrationDatasetConfig(
                            root=dataset_root,
                            source_dataset_key="aihub_traffic_seoul",
                            image_dir="images",
                            label_dir="labels",
                            split="val",
                        ),
                        classes=("traffic_light",),
                    ),
                ),
                search={
                    "match_iou": 0.5,
                    "min_precision": 0.90,
                    "min_precision_by_class": {"traffic_light": 0.75},
                    "score_thresholds": (0.50, 0.80),
                    "nms_iou_thresholds": (0.50,),
                    "min_box_sizes": (4,),
                },
            )
            with patch("tools.od_bootstrap.teacher.calibrate.YOLO", _SignalPrecisionOverrideFakeYOLO):
                summary = calibrate_class_policy_scenario(scenario, scenario_path=root / "preset_calibration")

            class_policy = read_yaml(summary["class_policy_path"])
            report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(class_policy["traffic_light"]["score_threshold"], 0.5)
            self.assertTrue(report["classes"]["traffic_light"]["meets_precision_floor"])
            self.assertAlmostEqual(report["classes"]["traffic_light"]["min_precision_target"], 0.75, places=6)
