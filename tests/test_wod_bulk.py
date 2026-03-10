import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from pv26.dataset.classmap import render_class_map_yaml
from pv26.dataset.wod_bulk import (
    WOD_STATUS_BLOCKED,
    WOD_STATUS_COMPLETED,
    WOD_STATUS_FAILED,
    WOD_STATUS_PENDING,
    completed_shard_roots_from_state,
    reconcile_wod_bulk_state,
)
from tools.data_analysis.wod.process_wod_pv26_bulk import main as wod_bulk_main


def _touch(path: Path, *, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _write_rgb_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 1] = 127
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def _write_u8_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((4, 4), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _write_fake_wod_shard(shard_root: Path, *, context_name: str) -> None:
    sample_id = f"waymo__{context_name}__1000__front"
    split = "train"
    image_rel = f"images/{split}/{sample_id}.jpg"
    det_rel = f"labels_det/{split}/{sample_id}.txt"
    da_rel = f"labels_seg_da/{split}/{sample_id}.png"
    lane_rel = f"labels_seg_rm_lane_marker/{split}/{sample_id}.png"
    road_rel = f"labels_seg_rm_road_marker_non_lane/{split}/{sample_id}.png"
    stop_rel = f"labels_seg_rm_stop_line/{split}/{sample_id}.png"
    lane_sub_rel = f"labels_seg_rm_lane_subclass/{split}/{sample_id}.png"

    _write_rgb_jpg(shard_root / image_rel)
    (shard_root / det_rel).parent.mkdir(parents=True, exist_ok=True)
    (shard_root / det_rel).write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    _write_u8_png(shard_root / da_rel, 1)
    _write_u8_png(shard_root / lane_rel, 1)
    _write_u8_png(shard_root / road_rel, 1)
    _write_u8_png(shard_root / stop_rel, 255)
    _write_u8_png(shard_root / lane_sub_rel, 255)

    meta = shard_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "source",
                "sequence",
                "frame",
                "camera_id",
                "timestamp_ns",
                "has_det",
                "has_da",
                "has_rm_lane_marker",
                "has_rm_road_marker_non_lane",
                "has_rm_stop_line",
                "has_rm_lane_subclass",
                "has_semantic_id",
                "det_label_scope",
                "det_annotated_class_ids",
                "image_relpath",
                "det_relpath",
                "da_relpath",
                "rm_lane_marker_relpath",
                "rm_road_marker_non_lane_relpath",
                "rm_stop_line_relpath",
                "rm_lane_subclass_relpath",
                "semantic_relpath",
                "width",
                "height",
                "weather_tag",
                "time_tag",
                "scene_tag",
                "source_group_key",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "split": split,
                "source": "waymo",
                "sequence": context_name,
                "frame": "1000",
                "camera_id": "front",
                "timestamp_ns": "1000000",
                "has_det": "1",
                "has_da": "1",
                "has_rm_lane_marker": "1",
                "has_rm_road_marker_non_lane": "1",
                "has_rm_stop_line": "0",
                "has_rm_lane_subclass": "0",
                "has_semantic_id": "0",
                "det_label_scope": "full",
                "det_annotated_class_ids": "",
                "image_relpath": image_rel,
                "det_relpath": det_rel,
                "da_relpath": da_rel,
                "rm_lane_marker_relpath": lane_rel,
                "rm_road_marker_non_lane_relpath": road_rel,
                "rm_stop_line_relpath": stop_rel,
                "rm_lane_subclass_relpath": lane_sub_rel,
                "semantic_relpath": "",
                "width": "4",
                "height": "4",
                "weather_tag": "unknown",
                "time_tag": "unknown",
                "scene_tag": "unknown",
                "source_group_key": f"waymo::{context_name}",
            }
        )
    (meta / "class_map.yaml").write_text(render_class_map_yaml(), encoding="utf-8")


class TestWodBulkState(unittest.TestCase):
    def test_reconcile_scan_tracks_component_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_root = root / "training"
            shards_root = root / "shards"
            _touch(training_root / "camera_image" / "ctx_a.parquet", size=10)
            _touch(training_root / "camera_segmentation" / "ctx_a.parquet", size=20)
            _touch(training_root / "camera_box" / "ctx_a.parquet", size=30)
            _touch(training_root / "camera_image" / "ctx_b.parquet", size=40)
            _touch(training_root / "camera_segmentation" / "ctx_c.parquet", size=50)

            state = reconcile_wod_bulk_state(training_root=training_root, shards_root=shards_root)
            by_name = {ctx["context_name"]: ctx for ctx in state["contexts"]}

            self.assertEqual(set(by_name.keys()), {"ctx_a", "ctx_b", "ctx_c"})
            self.assertEqual(by_name["ctx_a"]["status"], WOD_STATUS_PENDING)
            self.assertTrue(by_name["ctx_a"]["processable_now"])
            self.assertEqual(by_name["ctx_a"]["image_bytes"], 10)
            self.assertEqual(by_name["ctx_a"]["segmentation_bytes"], 20)
            self.assertEqual(by_name["ctx_a"]["box_bytes"], 30)
            self.assertEqual(by_name["ctx_b"]["status"], WOD_STATUS_BLOCKED)
            self.assertFalse(by_name["ctx_b"]["processable_now"])
            self.assertTrue(by_name["ctx_b"]["has_image"])
            self.assertFalse(by_name["ctx_b"]["has_segmentation"])
            self.assertEqual(by_name["ctx_c"]["status"], WOD_STATUS_BLOCKED)

    def test_reconcile_preserves_completed_context_after_raw_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_root = root / "training"
            shards_root = root / "shards"
            prior_state = {
                "created_at": "2026-03-10T00:00:00Z",
                "contexts": [
                    {
                        "context_name": "ctx_done",
                        "status": WOD_STATUS_COMPLETED,
                        "image_relpath": "camera_image/ctx_done.parquet",
                        "segmentation_relpath": "camera_segmentation/ctx_done.parquet",
                        "box_relpath": "",
                        "image_bytes": 111,
                        "segmentation_bytes": 222,
                        "box_bytes": 0,
                        "attempt_count": 1,
                        "output_num_rows": 95,
                        "output_rows_by_split": {"train": 95},
                        "output_has_det_rows": 95,
                        "shard_root": str(shards_root / "pv26_wod_ctx_done"),
                    }
                ],
            }

            state = reconcile_wod_bulk_state(
                training_root=training_root,
                shards_root=shards_root,
                prior_state=prior_state,
            )
            ctx = state["contexts"][0]
            self.assertEqual(ctx["context_name"], "ctx_done")
            self.assertEqual(ctx["status"], WOD_STATUS_COMPLETED)
            self.assertFalse(ctx["processable_now"])
            self.assertEqual(ctx["output_num_rows"], 95)

    def test_completed_shard_roots_are_collected(self):
        state = {
            "contexts": [
                {"context_name": "a", "status": WOD_STATUS_COMPLETED, "shard_root": "/tmp/a"},
                {"context_name": "b", "status": WOD_STATUS_FAILED, "shard_root": "/tmp/b"},
            ]
        }
        roots = completed_shard_roots_from_state(state)
        self.assertEqual([str(p) for p in roots], ["/tmp/a"])


class TestWodBulkTool(unittest.TestCase):
    def test_run_marks_completed_and_deletes_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_root = root / "training"
            shards_root = root / "shards"
            _touch(training_root / "camera_image" / "ctx_a.parquet", size=10)
            _touch(training_root / "camera_segmentation" / "ctx_a.parquet", size=20)

            def _fake_run(cmd, cwd=None, check=False):
                self.assertIn("--context-name", cmd)
                context_name = cmd[cmd.index("--context-name") + 1]
                shard_root = Path(cmd[cmd.index("--out-root") + 1])
                _write_fake_wod_shard(shard_root, context_name=context_name)
                cp = subprocess.CompletedProcess(cmd, 0)  # type: ignore[name-defined]
                return cp

            import subprocess

            with mock.patch("tools.data_analysis.wod.process_wod_pv26_bulk.subprocess.run", side_effect=_fake_run):
                rc = wod_bulk_main(
                    [
                        "run",
                        "--training-root",
                        str(training_root),
                        "--shards-root",
                        str(shards_root),
                        "--delete-raw-on-success",
                    ]
                )

            self.assertEqual(rc, 0)
            state_path = shards_root / "meta" / "wod_bulk_state.json"
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            ctx = state["contexts"][0]
            self.assertEqual(ctx["context_name"], "ctx_a")
            self.assertEqual(ctx["status"], WOD_STATUS_COMPLETED)
            self.assertEqual(ctx["output_num_rows"], 1)
            self.assertEqual(ctx["output_has_det_rows"], 1)
            self.assertFalse((training_root / "camera_image" / "ctx_a.parquet").exists())
            self.assertFalse((training_root / "camera_segmentation" / "ctx_a.parquet").exists())
            self.assertTrue((shards_root / "meta" / "wod_bulk_state.csv").exists())


if __name__ == "__main__":
    unittest.main()
