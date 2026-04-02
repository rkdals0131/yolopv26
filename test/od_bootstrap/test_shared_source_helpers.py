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
from tools.od_bootstrap.source.shared_io import link_or_copy, load_json, write_json, write_text
from tools.od_bootstrap.source.shared_parallel import LiveLogger, default_workers, iter_task_chunks, parallel_chunk_size
from tools.od_bootstrap.source.shared_raw import extract_annotations, normalize_text, safe_slug
from tools.od_bootstrap.source.shared_scene import bbox_to_yolo_line, build_base_scene, sample_id
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
                "shared_scene",
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


if __name__ == "__main__":
    unittest.main()
