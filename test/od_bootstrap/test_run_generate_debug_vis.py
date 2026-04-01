from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from tools.od_bootstrap import main as od_bootstrap_main
from tools.od_bootstrap.data.image_list import build_sample_uid


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class GenerateDebugVisEntrypointTests(unittest.TestCase):
    def test_canonical_mode_renders_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_root = root / "pv26_od_bootstrap"
            canonical_root = bootstrap_root / "canonical"
            image_path = canonical_root / "bdd100k_det_100k" / "images" / "val" / "bdd_val_001.jpg"
            scene_path = canonical_root / "bdd100k_det_100k" / "labels_scene" / "val" / "bdd_val_001.json"
            det_path = canonical_root / "bdd100k_det_100k" / "labels_det" / "val" / "bdd_val_001.txt"
            _make_image(image_path, 64, 48, "#222222")
            _write_json(
                scene_path,
                {
                    "image": {"file_name": image_path.name, "width": 64, "height": 48},
                    "source": {"dataset": "bdd100k_det_100k", "split": "val"},
                    "detections": [{"class_name": "vehicle", "bbox": [10, 10, 30, 30]}],
                },
            )
            _write_text(det_path, "0 0.312500 0.416667 0.312500 0.416667\n")
            image_list_path = bootstrap_root / "meta" / "bootstrap_image_list.jsonl"
            image_list_path.parent.mkdir(parents=True, exist_ok=True)
            image_list_path.write_text(
                json.dumps(
                    {
                        "sample_id": "bdd_val_001",
                        "sample_uid": build_sample_uid(dataset_key="bdd100k_det_100k", split="val", sample_id="bdd_val_001"),
                        "image_path": str(image_path),
                        "scene_path": str(scene_path),
                        "dataset_root": str(canonical_root / "bdd100k_det_100k"),
                        "dataset_key": "bdd100k_det_100k",
                        "split": "val",
                        "det_path": str(det_path),
                        "source_name": "bdd100k_det_100k",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = od_bootstrap_main(
                    [
                        "generate-debug-vis",
                        "--mode",
                        "canonical",
                        "--bootstrap-root",
                        str(bootstrap_root),
                        "--count",
                        "1",
                        "--seed",
                        "26",
                    ]
                )

            self.assertEqual(exit_code, 0)
            debug_vis_dir = canonical_root / "bdd100k_det_100k" / "meta" / "debug_vis"
            manifest_path = canonical_root / "bdd100k_det_100k" / "meta" / "debug_vis_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["selection_count"], 1)
            overlay_files = sorted(debug_vis_dir.glob("*.png"))
            self.assertEqual(len(overlay_files), 1)
            self.assertEqual(sorted(path.name for path in debug_vis_dir.iterdir()), [overlay_files[0].name])
            self.assertTrue(Path(manifest_payload["items"][0]["overlay_path"]).is_file())

    def test_teacher_mode_renders_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_root = root / "teacher_datasets"
            dataset_root = teacher_root / "signal"
            image_path = dataset_root / "images" / "val" / "signal_val_001.png"
            label_path = dataset_root / "labels" / "val" / "signal_val_001.txt"
            _make_image(image_path, 64, 48, "#444444")
            _write_text(label_path, "0 0.375000 0.239583 0.125000 0.270833\n")
            _write_json(
                dataset_root / "meta" / "teacher_dataset_manifest.json",
                {
                    "teacher_name": "signal",
                    "class_names": ["traffic_light", "sign"],
                    "samples": [
                        {
                            "source_dataset_key": "aihub_traffic_seoul",
                            "split": "val",
                            "sample_id": "signal_val_001",
                            "sample_uid": build_sample_uid(
                                dataset_key="aihub_traffic_seoul",
                                split="val",
                                sample_id="signal_val_001",
                            ),
                            "source_image_path": str(image_path),
                            "output_label_path": str(label_path),
                        }
                    ],
                },
            )

            with redirect_stdout(io.StringIO()):
                exit_code = od_bootstrap_main(
                    [
                        "generate-debug-vis",
                        "--mode",
                        "teacher",
                        "--teacher-root",
                        str(teacher_root),
                        "--teacher",
                        "signal",
                        "--count",
                        "1",
                        "--seed",
                        "26",
                    ]
                )

            self.assertEqual(exit_code, 0)
            debug_vis_dir = dataset_root / "meta" / "debug_vis"
            manifest_path = dataset_root / "meta" / "debug_vis_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["selection_count"], 1)
            overlay_files = sorted(debug_vis_dir.glob("*.png"))
            self.assertEqual(len(overlay_files), 1)
            self.assertEqual(sorted(path.name for path in debug_vis_dir.iterdir()), [overlay_files[0].name])
            self.assertTrue(Path(manifest_payload["items"][0]["overlay_path"]).is_file())

    def test_exhaustive_mode_renders_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exhaustive_root = root / "exhaustive_od"
            run_root = exhaustive_root / "20260401_000000_model_centric_default"
            image_path = run_root / "images" / "val" / "scene_001.jpg"
            scene_path = run_root / "labels_scene" / "val" / "scene_001.json"
            det_path = run_root / "labels_det" / "val" / "scene_001.txt"
            _make_image(image_path, 64, 48, "#222222")
            _write_json(
                scene_path,
                {
                    "image": {"file_name": image_path.name, "width": 64, "height": 48},
                    "source": {"dataset": "bdd100k_det_100k", "split": "val"},
                    "detections": [{"class_name": "vehicle", "bbox": [10, 10, 30, 30]}],
                },
            )
            _write_text(det_path, "0 0.312500 0.416667 0.312500 0.416667\n")
            _write_json(
                run_root / "meta" / "materialization_manifest.json",
                {
                    "run_id": run_root.name,
                    "samples": [
                        {
                            "sample_id": "scene_001",
                            "sample_uid": build_sample_uid(
                                dataset_key="bdd100k_det_100k",
                                split="val",
                                sample_id="scene_001",
                            ),
                            "source_dataset_key": "bdd100k_det_100k",
                            "split": "val",
                            "scene_path": str(scene_path),
                            "det_path": str(det_path),
                            "image_path": str(image_path),
                        }
                    ],
                },
            )

            with redirect_stdout(io.StringIO()):
                exit_code = od_bootstrap_main(
                    [
                        "generate-debug-vis",
                        "--mode",
                        "exhaustive",
                        "--exhaustive-root",
                        str(exhaustive_root),
                        "--run",
                        run_root.name,
                        "--count",
                        "1",
                        "--seed",
                        "26",
                    ]
                )

            self.assertEqual(exit_code, 0)
            debug_vis_dir = run_root / "meta" / "debug_vis"
            manifest_path = run_root / "meta" / "debug_vis_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["selection_count"], 1)
            overlay_files = sorted(debug_vis_dir.glob("*.png"))
            self.assertEqual(len(overlay_files), 1)
            self.assertEqual(sorted(path.name for path in debug_vis_dir.iterdir()), [overlay_files[0].name])
            self.assertTrue(Path(manifest_payload["items"][0]["overlay_path"]).is_file())
