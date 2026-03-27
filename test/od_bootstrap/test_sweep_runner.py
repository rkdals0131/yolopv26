from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.od_bootstrap.sweep.run_model_centric_sweep import run_model_centric_sweep_scenario
from tools.od_bootstrap.sweep.scenario import load_sweep_scenario


class _FakeYOLO:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def predict(self, **kwargs):
        source = [Path(item) for item in kwargs["source"]]
        names = {0: "vehicle", 1: "bike", 2: "pedestrian"}
        checkpoint_name = Path(self.checkpoint_path).name
        if "signal" in checkpoint_name:
            names = {0: "traffic_light", 1: "sign"}
        elif "obstacle" in checkpoint_name:
            names = {0: "traffic_cone", 1: "obstacle"}
        results = []
        for image_path in source:
            if "mobility" in checkpoint_name:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[305.0, 305.0, 365.0, 385.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.95]),
                )
            elif "signal" in checkpoint_name:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[20.0, 20.0, 40.0, 60.0]]),
                    cls=torch.tensor([0]),
                    conf=torch.tensor([0.91]),
                )
            else:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[200.0, 200.0, 250.0, 250.0]]),
                    cls=torch.tensor([1]),
                    conf=torch.tensor([0.89]),
                )
            results.append(SimpleNamespace(path=str(image_path), names=names, boxes=boxes))
        return results


class ODBootstrapRunnerTests(unittest.TestCase):
    def test_run_model_centric_sweep_writes_manifests_and_materializes_exhaustive_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "frame_001.png").write_bytes(b"001")
            (images_dir / "frame_002.png").write_bytes(b"002")
            (root / "weights").mkdir(parents=True, exist_ok=True)
            for name in ("mobility", "signal", "obstacle"):
                (root / "weights" / f"{name}.pt").write_text(name, encoding="utf-8")

            scene_dir = root / "canonical" / "bdd100k_det_100k" / "labels_scene" / "train"
            det_dir = root / "canonical" / "bdd100k_det_100k" / "labels_det" / "train"
            image_root = root / "canonical" / "bdd100k_det_100k" / "images" / "train"
            scene_dir.mkdir(parents=True, exist_ok=True)
            det_dir.mkdir(parents=True, exist_ok=True)
            image_root.mkdir(parents=True, exist_ok=True)
            for frame in ("frame_001.png", "frame_002.png"):
                (image_root / frame).write_bytes(b"image")
            for sample_id in ("frame_001", "frame_002"):
                (scene_dir / f"{sample_id}.json").write_text(
                    json.dumps(
                        {
                            "version": "test",
                            "image": {"file_name": f"{sample_id}.png", "width": 640, "height": 480},
                            "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                            "tasks": {"has_det": 1},
                            "detections": [
                                {
                                    "id": 0,
                                    "class_name": "vehicle",
                                    "bbox": {"x1": 300.0, "y1": 300.0, "x2": 360.0, "y2": 380.0},
                                    "score": None,
                                    "meta": {},
                                }
                            ],
                            "traffic_lights": [],
                            "traffic_signs": [],
                            "lanes": [],
                            "stop_lines": [],
                            "crosswalks": [],
                        },
                        ensure_ascii=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (det_dir / f"{sample_id}.txt").write_text("0 0.515625 0.708333 0.093750 0.166667\n", encoding="utf-8")

            image_list_manifest = root / "image_list.jsonl"
            image_list_manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "image_id": "frame_002",
                                "image_path": str(image_root / "frame_002.png"),
                                "scene_path": str(scene_dir / "frame_002.json"),
                                "dataset_root": str(root / "canonical" / "bdd100k_det_100k"),
                                "dataset_key": "bdd100k_det_100k",
                                "split": "train",
                                "det_path": str(det_dir / "frame_002.txt"),
                            }
                        ),
                        json.dumps(
                            {
                                "image_id": "frame_001",
                                "image_path": str(image_root / "frame_001.png"),
                                "scene_path": str(scene_dir / "frame_001.json"),
                                "dataset_root": str(root / "canonical" / "bdd100k_det_100k"),
                                "dataset_key": "bdd100k_det_100k",
                                "split": "train",
                                "det_path": str(det_dir / "frame_001.txt"),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scenario_path = root / "bootstrap.yaml"
            scenario_path.write_text(
                textwrap.dedent(
                    """
                    run:
                      output_root: runs/od_bootstrap
                      execution_mode: model-centric
                      dry_run: false
                      device: cpu
                      imgsz: 640
                      batch_size: 2
                    image_list:
                      manifest_path: image_list.jsonl
                    materialization:
                      output_root: exhaustive_od
                      copy_images: false
                    teachers:
                      - name: mobility
                        base_model: yolov26n
                        checkpoint_path: weights/mobility.pt
                        model_version: mobility_v1
                        classes: [vehicle, bike, pedestrian]
                      - name: signal
                        base_model: yolov26n
                        checkpoint_path: weights/signal.pt
                        model_version: signal_v1
                        classes: [traffic_light, sign]
                      - name: obstacle
                        base_model: yolov26n
                        checkpoint_path: weights/obstacle.pt
                        model_version: obstacle_v1
                        classes: [traffic_cone, obstacle]
                    class_policy:
                      vehicle: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                      bike: {score_threshold: 0.30, nms_iou_threshold: 0.55, min_box_size: 8}
                      pedestrian: {score_threshold: 0.35, nms_iou_threshold: 0.50, min_box_size: 8}
                      traffic_light: {score_threshold: 0.40, nms_iou_threshold: 0.45, min_box_size: 6}
                      sign: {score_threshold: 0.40, nms_iou_threshold: 0.50, min_box_size: 8}
                      traffic_cone: {score_threshold: 0.45, nms_iou_threshold: 0.45, min_box_size: 8}
                      obstacle: {score_threshold: 0.55, nms_iou_threshold: 0.40, min_box_size: 12}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            scenario = load_sweep_scenario(scenario_path)
            with patch("tools.od_bootstrap.sweep.run_model_centric_sweep.YOLO", _FakeYOLO):
                summary = run_model_centric_sweep_scenario(scenario, scenario_path=scenario_path)
            run_dir = Path(summary["run_dir"])

            self.assertEqual(summary["teacher_names"], ["mobility", "signal", "obstacle"])
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "image_list.jsonl").is_file())
            self.assertTrue((run_dir / "teachers" / "mobility" / "job_manifest.json").is_file())
            self.assertTrue((run_dir / "teachers" / "signal" / "job_manifest.json").is_file())
            self.assertTrue((run_dir / "teachers" / "obstacle" / "job_manifest.json").is_file())
            self.assertTrue((run_dir / "teachers" / "mobility" / "predictions.jsonl").is_file())

            snapshot_lines = (run_dir / "image_list.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(snapshot_lines), 2)
            self.assertIn("frame_001", snapshot_lines[0])
            materialized_root = Path(summary["materialization"]["dataset_root"])
            materialized_scene = json.loads(
                (materialized_root / "labels_scene" / "train" / "frame_001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(materialized_scene["source"]["dataset"], "pv26_exhaustive_bdd100k_det_100k")
            self.assertEqual(len(materialized_scene["detections"]), 3)
            self.assertEqual(materialized_scene["detections"][0]["provenance"]["label_origin"], "raw_source")
            self.assertEqual(materialized_scene["detections"][1]["provenance"]["label_origin"], "bootstrap")
