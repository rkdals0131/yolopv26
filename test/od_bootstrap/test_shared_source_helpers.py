from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.od_bootstrap.source import __all__ as SOURCE_EXPORTS
from tools.od_bootstrap.source.constants import DEFAULT_AIHUB_OUTPUT_ROOT, DEFAULT_BDD_ROOT
from tools.od_bootstrap.source.defaults import build_default_source_prep_config, resolve_source_path
from tools.od_bootstrap.source.raw_common import PairRecord
from tools.od_bootstrap.source.shared_debug import build_debug_vis_manifest
from tools.od_bootstrap.source.shared_io import link_or_copy, load_json, write_json, write_text
from tools.od_bootstrap.source.shared_parallel import LiveLogger, default_workers, iter_task_chunks, parallel_chunk_size
from tools.od_bootstrap.source.shared_raw import extract_annotations, normalize_text, safe_slug
from tools.od_bootstrap.source.shared_resume import count_held_annotation_reasons, load_existing_scene_output
from tools.od_bootstrap.source.shared_scene import bbox_to_yolo_line, build_base_scene, sample_id
from tools.od_bootstrap.source.shared_source_meta import build_bdd_inventory, build_bdd_source_inventory, tree_markdown
from tools.od_bootstrap.source.shared_summary import counter_to_dict


class SharedSourceHelpersTests(unittest.TestCase):
    def test_source_package_exports_shared_support_modules(self) -> None:
        self.assertTrue(
            {
                "constants",
                "defaults",
                "shared_io",
                "shared_parallel",
                "shared_raw",
                "shared_resume",
                "shared_scene",
                "shared_source_meta",
                "shared_summary",
                "types",
            }.issubset(set(SOURCE_EXPORTS))
        )

    def test_defaults_build_expected_config_and_resolve_relative_paths(self) -> None:
        config = build_default_source_prep_config()
        self.assertEqual(config.roots.bdd_root, DEFAULT_BDD_ROOT)
        self.assertEqual(config.output_root, (DEFAULT_AIHUB_OUTPUT_ROOT.parent / "pv26_od_bootstrap").resolve())

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_output = Path(temp_dir) / "custom_output"
            base_dir = Path(temp_dir) / "workspace"
            base_dir.mkdir()

            overridden = build_default_source_prep_config(output_root=custom_output)
            self.assertEqual(overridden.output_root, custom_output.resolve())
            self.assertEqual(resolve_source_path("raw/bdd", base_dir=base_dir), (base_dir / "raw/bdd").resolve())

    def test_shared_io_round_trips_and_reuses_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.txt"
            source_path.write_text("payload\n", encoding="utf-8")

            target_path = root / "linked.txt"
            materialization = link_or_copy(source_path, target_path)
            self.assertIn(materialization, {"hardlink", "copy"})
            self.assertEqual(target_path.read_text(encoding="utf-8"), "payload\n")
            self.assertEqual(link_or_copy(source_path, target_path), "existing")

            text_path = root / "nested" / "note.txt"
            json_path = root / "nested" / "payload.json"
            payload = {"b": 2, "a": 1}
            write_text(text_path, "hello\n")
            write_json(json_path, payload)

            self.assertEqual(text_path.read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(load_json(json_path), payload)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), payload)

    def test_shared_parallel_helpers_log_progress_and_chunk_work(self) -> None:
        stream = io.StringIO()
        logger = LiveLogger(stream=stream, throttle_seconds=0.0)

        logger.stage("sync", "checking shared helpers", total=3)
        logger.progress(1)
        logger.heartbeat("pending")
        logger.progress(3, detail="done")

        log_output = stream.getvalue()
        self.assertIn("stage=sync", log_output)
        self.assertIn("checking shared helpers", log_output)
        self.assertIn("[sync] 1/3", log_output)
        self.assertIn("still running", log_output)
        self.assertIn("done", log_output)
        self.assertGreaterEqual(default_workers(), 1)
        self.assertEqual(parallel_chunk_size(0, 4), 1)
        self.assertEqual(list(iter_task_chunks([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_shared_debug_helper_builds_sorted_manifest_payload(self) -> None:
        manifest = build_debug_vis_manifest(
            generated_at="2026-04-03T00:00:00Z",
            debug_vis_seed=7,
            items=[
                {
                    "dataset_key": "bdd",
                    "split": "val",
                    "sample_id": "b",
                    "scene_path": "/tmp/b.json",
                    "image_path": "/tmp/b.jpg",
                    "output_path": "/tmp/b.png",
                },
                {
                    "dataset_key": "aihub",
                    "split": "train",
                    "sample_id": "a",
                    "scene_path": "/tmp/a.json",
                    "image_path": "/tmp/a.jpg",
                    "output_path": "/tmp/a.png",
                },
            ],
        )

        self.assertEqual(manifest["selection_count"], 2)
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual([item["dataset_key"] for item in manifest["items"]], ["aihub", "bdd"])

    def test_shared_raw_scene_and_summary_helpers_preserve_scene_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "val" / "source-name.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640, 480), "#123456").save(image_path)

            pair = PairRecord(
                dataset_key="aihub_traffic_source",
                dataset_root=DEFAULT_BDD_ROOT,
                split="val",
                image_path=image_path,
                label_path=root / "labels" / "val" / "source-name.json",
                image_file_name="fallback-name.png",
                relative_id="folder/source name",
            )
            raw = {
                "image": {
                    "file_name": "nested/source-name.png",
                    "imsize": {"width": 640, "height": 480},
                },
                "annotations": {
                    "lane": [{"id": 1, "x": [1, 2], "y": [3, 4]}],
                    "sign": {"id": 2, "bbox": [5, 6, 7, 8]},
                },
            }
            output_image_path = root / "canonical" / "val" / "sample.png"

            width, height, scene = build_base_scene("aihub_traffic_seoul", pair, raw, output_image_path)

            self.assertEqual((width, height), (640, 480))
            self.assertEqual(scene["image"]["file_name"], "sample.png")
            self.assertEqual(scene["image"]["original_file_name"], "source-name.png")
            self.assertEqual(scene["source"]["raw_id"], "folder/source name")
            self.assertEqual(sample_id("aihub_traffic_seoul", pair, safe_slug=safe_slug), "aihub_traffic_seoul_val_folder_source_name")
            self.assertEqual(bbox_to_yolo_line(2, [10, 20, 110, 220], 200, 400), "2 0.300000 0.300000 0.500000 0.500000")
            self.assertEqual(normalize_text("Traffic Light"), "traffic_light")
            self.assertEqual(safe_slug("bdd val/sample 001"), "bdd_val_sample_001")
            self.assertEqual(
                extract_annotations(raw),
                [
                    {"id": 1, "x": [1, 2], "y": [3, 4], "class": "lane"},
                    {"id": 2, "bbox": [5, 6, 7, 8], "class": "sign"},
                ],
            )
        self.assertEqual(counter_to_dict(Counter({"z": 1, "a": 2})), {"a": 2, "z": 1})

    def test_shared_resume_and_bdd_source_meta_helpers_cover_public_source_cleanup_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "BDD100K"
            images_root = bdd_root / "bdd100k_images_100k" / "100k" / "train"
            labels_root = bdd_root / "bdd100k_labels" / "100k" / "train"
            images_root.mkdir(parents=True, exist_ok=True)
            labels_root.mkdir(parents=True, exist_ok=True)
            (images_root / "sample.jpg").write_bytes(b"jpg")
            (labels_root / "sample.json").write_text("{}", encoding="utf-8")
            (bdd_root / "bdd100k-gh").mkdir(parents=True, exist_ok=True)

            inventory = build_bdd_inventory(
                bdd_root,
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
                splits=("train", "val"),
                official_split_sizes={"train": 70000, "val": 10000},
            )
            source_inventory = build_bdd_source_inventory(
                pipeline_version="pv26-test",
                readme_path=str(bdd_root / "README.md"),
                inventory=inventory,
            )

            self.assertEqual(source_inventory["dataset"]["local_inventory"]["splits"]["train"]["images"], 1)
            self.assertIn("bdd100k_images_100k", tree_markdown(bdd_root))

            output_root = root / "canonical"
            sample_id_value = "bdd_train_sample"
            image_path = output_root / "images" / "train" / f"{sample_id_value}.jpg"
            scene_path = output_root / "labels_scene" / "train" / f"{sample_id_value}.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            payload = {
                "version": "scene-v1",
                "detections": [],
                "held_annotations": [{"reason": "Traffic Light"}],
            }
            write_json(scene_path, payload)

            bundle = load_existing_scene_output(
                output_root=output_root,
                split="train",
                sample_id=sample_id_value,
                image_suffix=".jpg",
                load_json_fn=load_json,
                scene_version="scene-v1",
            )

            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(bundle["scene"]["version"], "scene-v1")
            self.assertEqual(count_held_annotation_reasons(payload["held_annotations"]), {"traffic light": 1})
            self.assertEqual(
                count_held_annotation_reasons(payload["held_annotations"], normalize_reason=normalize_text),
                {"traffic_light": 1},
            )


if __name__ == "__main__":
    unittest.main()
