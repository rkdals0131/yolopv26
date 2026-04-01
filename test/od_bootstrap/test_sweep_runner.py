from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.od_bootstrap.data.image_list import ImageListEntry, build_sample_uid
from tools.od_bootstrap.data.sweep import ClassPolicy, TeacherConfig, run_model_centric_sweep_scenario
from tools.od_bootstrap.presets import build_sweep_preset
from tools.od_bootstrap.data._sweep_impl import _extract_teacher_rows


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
                    xyxy=torch.tensor([[20.0, 20.0, 40.0, 60.0], [18.0, 18.0, 42.0, 62.0]]),
                    cls=torch.tensor([0, 1]),
                    conf=torch.tensor([0.91, 0.88]),
                )
            else:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[200.0, 200.0, 250.0, 250.0], [202.0, 202.0, 248.0, 248.0]]),
                    cls=torch.tensor([1, 0]),
                    conf=torch.tensor([0.89, 0.87]),
                )
            results.append(SimpleNamespace(path=str(image_path), names=names, boxes=boxes, orig_shape=(480, 640)))
        return results


class ODBootstrapRunnerTests(unittest.TestCase):
    def test_extract_teacher_rows_falls_back_to_batch_position_when_result_path_is_rewritten(self) -> None:
        entry = ImageListEntry(
            sample_id="frame_001",
            sample_uid="bdd100k_det_100k__train__frame_001",
            image_path=Path("/tmp/source.png"),
            scene_path=Path("/tmp/source.json"),
            dataset_root=Path("/tmp"),
            dataset_key="bdd100k_det_100k",
            split="train",
            det_path=None,
            source_name="canonical",
        )
        teacher = TeacherConfig(
            name="mobility",
            base_model="yolov26s",
            checkpoint_path=Path("/tmp/mobility.pt"),
            model_version="mobility_yolov26s_bootstrap_v1",
            classes=("vehicle", "bike", "pedestrian"),
        )
        result = SimpleNamespace(
            path="image0.jpg",
            names={0: "vehicle"},
            boxes=SimpleNamespace(
                xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                cls=torch.tensor([0]),
                conf=torch.tensor([0.95]),
            ),
            orig_shape=(480, 640),
        )

        rows = _extract_teacher_rows(
            teacher=teacher,
            batch_entries=[entry],
            results=[result],
            class_policy={
                "vehicle": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.55, min_box_size=8),
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_uid"], entry.sample_uid)
        self.assertEqual(rows[0]["class_name"], "vehicle")

    def test_run_model_centric_sweep_writes_manifests_and_materializes_exhaustive_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "weights").mkdir(parents=True, exist_ok=True)
            for name in ("mobility", "signal", "obstacle"):
                (root / "weights" / f"{name}.pt").write_text(name, encoding="utf-8")

            bdd_root = root / "canonical" / "bdd100k_det_100k"
            traffic_root = root / "canonical" / "aihub_standardized"
            bdd_scene_dir = bdd_root / "labels_scene" / "train"
            bdd_det_dir = bdd_root / "labels_det" / "train"
            bdd_image_root = bdd_root / "images" / "train"
            traffic_scene_dir = traffic_root / "labels_scene" / "train"
            traffic_det_dir = traffic_root / "labels_det" / "train"
            traffic_image_root = traffic_root / "images" / "train"
            for path in (bdd_scene_dir, bdd_det_dir, bdd_image_root, traffic_scene_dir, traffic_det_dir, traffic_image_root):
                path.mkdir(parents=True, exist_ok=True)

            (bdd_image_root / "frame_001.png").write_bytes(b"bdd")
            (traffic_image_root / "frame_001.png").write_bytes(b"traffic")

            (bdd_scene_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "version": "test",
                        "image": {"file_name": "frame_001.png", "width": 640, "height": 480},
                        "source": {"dataset": "bdd100k_det_100k", "split": "train"},
                        "tasks": {"has_det": 1},
                        "detections": [
                            {
                                "id": 0,
                                "class_name": "vehicle",
                                "bbox": [300.0, 300.0, 360.0, 380.0],
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
            (bdd_det_dir / "frame_001.txt").write_text("0 0.515625 0.708333 0.093750 0.166667\n", encoding="utf-8")

            (traffic_scene_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "version": "test",
                        "image": {"file_name": "frame_001.png", "width": 640, "height": 480},
                        "source": {"dataset": "aihub_traffic_seoul", "split": "train"},
                        "tasks": {"has_det": 1},
                        "detections": [
                            {
                                "id": 0,
                                "class_name": "traffic_light",
                                "bbox": [20.0, 20.0, 40.0, 60.0],
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
            (traffic_det_dir / "frame_001.txt").write_text("3 0.046875 0.083333 0.031250 0.083333\n", encoding="utf-8")

            image_list_manifest = root / "image_list.jsonl"
            image_list_manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "frame_001",
                                "sample_uid": build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="frame_001"),
                                "image_path": str(traffic_image_root / "frame_001.png"),
                                "scene_path": str(traffic_scene_dir / "frame_001.json"),
                                "dataset_root": str(traffic_root),
                                "dataset_key": "aihub_traffic_seoul",
                                "split": "train",
                                "det_path": str(traffic_det_dir / "frame_001.txt"),
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "frame_001",
                                "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="frame_001"),
                                "image_path": str(bdd_image_root / "frame_001.png"),
                                "scene_path": str(bdd_scene_dir / "frame_001.json"),
                                "dataset_root": str(bdd_root),
                                "dataset_key": "bdd100k_det_100k",
                                "split": "train",
                                "det_path": str(bdd_det_dir / "frame_001.txt"),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            preset = build_sweep_preset()
            scenario = replace(
                preset,
                run=replace(preset.run, output_root=root / "runs" / "od_bootstrap"),
                image_list=replace(preset.image_list, manifest_path=image_list_manifest),
                materialization=replace(preset.materialization, output_root=root / "exhaustive_od", copy_images=False),
                teachers=tuple(
                    replace(
                        teacher,
                        checkpoint_path=root / "weights" / f"{teacher.name}.pt",
                    )
                    for teacher in preset.teachers
                ),
                class_policy_path=root / "class_policy.yaml",
            )
            with patch("tools.od_bootstrap.data.sweep.YOLO", _FakeYOLO):
                summary = run_model_centric_sweep_scenario(scenario, scenario_path=root / "preset_sweep")
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
            self.assertIn("sample_uid", snapshot_lines[0])
            materialized_root = Path(summary["materialization"]["dataset_root"])
            bdd_uid = build_sample_uid(dataset_key="bdd100k_det_100k", split="train", sample_id="frame_001")
            traffic_uid = build_sample_uid(dataset_key="aihub_traffic_seoul", split="train", sample_id="frame_001")
            bdd_scene = json.loads(
                (materialized_root / "labels_scene" / "train" / f"{bdd_uid}.json").read_text(encoding="utf-8")
            )
            traffic_scene = json.loads(
                (materialized_root / "labels_scene" / "train" / f"{traffic_uid}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(bdd_scene["source"]["dataset"], "pv26_exhaustive_bdd100k_det_100k")
            self.assertEqual(traffic_scene["source"]["dataset"], "pv26_exhaustive_aihub_traffic_seoul")
            self.assertEqual(bdd_scene["source"]["bootstrap_sample_uid"], bdd_uid)
            self.assertEqual(traffic_scene["source"]["bootstrap_sample_uid"], traffic_uid)
            self.assertEqual(len(bdd_scene["detections"]), 5)
            self.assertEqual(len(traffic_scene["detections"]), 5)
            self.assertEqual(bdd_scene["detections"][0]["provenance"]["label_origin"], "raw_source")
            self.assertEqual(bdd_scene["detections"][1]["provenance"]["label_origin"], "bootstrap")
            traffic_classes = [item["class_name"] for item in traffic_scene["detections"]]
            self.assertIn("traffic_light", traffic_classes)
            self.assertIn("obstacle", traffic_classes)
            self.assertIn("vehicle", traffic_classes)
            self.assertTrue((materialized_root / "labels_det" / "train" / f"{bdd_uid}.txt").is_file())
            self.assertTrue((materialized_root / "labels_det" / "train" / f"{traffic_uid}.txt").is_file())
