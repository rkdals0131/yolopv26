import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pv26.class_map import render_class_map_yaml
from pv26.manifest import ManifestRow, write_manifest_csv
from pv26.masks import IGNORE_VALUE
from pv26.validate_dataset import validate_pv26_dataset


class TestDatasetValidatorSmoke(unittest.TestCase):
    def test_validate_minimal_dataset_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pv26_v1"
            # dirs
            (root / "images/train").mkdir(parents=True, exist_ok=True)
            (root / "labels_det/train").mkdir(parents=True, exist_ok=True)
            (root / "labels_seg_da/train").mkdir(parents=True, exist_ok=True)
            (root / "labels_seg_rm_lane_marker/train").mkdir(parents=True, exist_ok=True)
            (root / "labels_seg_rm_road_marker_non_lane/train").mkdir(parents=True, exist_ok=True)
            (root / "labels_seg_rm_stop_line/train").mkdir(parents=True, exist_ok=True)
            (root / "meta").mkdir(parents=True, exist_ok=True)

            sample_id = "bdd100k__seq__000001__cam0"
            # image 2x2
            img = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
            img.save(root / "images/train" / f"{sample_id}.jpg", format="JPEG")
            # det empty
            (root / "labels_det/train" / f"{sample_id}.txt").write_text("", encoding="utf-8")
            # masks all 255
            mask = np.full((2, 2), IGNORE_VALUE, dtype=np.uint8)
            for sub in [
                "labels_seg_da/train",
                "labels_seg_rm_lane_marker/train",
                "labels_seg_rm_road_marker_non_lane/train",
                "labels_seg_rm_stop_line/train",
            ]:
                Image.fromarray(mask, mode="L").save(root / sub / f"{sample_id}.png", format="PNG")

            (root / "meta" / "class_map.yaml").write_text(render_class_map_yaml(), encoding="utf-8")
            row = ManifestRow(
                sample_id=sample_id,
                split="train",
                source="bdd100k",
                sequence="seq",
                frame="000001",
                camera_id="cam0",
                timestamp_ns="",
                has_det=0,
                has_da=0,
                has_rm_lane_marker=0,
                has_rm_road_marker_non_lane=0,
                has_rm_stop_line=0,
                has_semantic_id=0,
                det_label_scope="none",
                det_annotated_class_ids="",
                image_relpath=f"images/train/{sample_id}.jpg",
                det_relpath=f"labels_det/train/{sample_id}.txt",
                da_relpath=f"labels_seg_da/train/{sample_id}.png",
                rm_lane_marker_relpath=f"labels_seg_rm_lane_marker/train/{sample_id}.png",
                rm_road_marker_non_lane_relpath=f"labels_seg_rm_road_marker_non_lane/train/{sample_id}.png",
                rm_stop_line_relpath=f"labels_seg_rm_stop_line/train/{sample_id}.png",
                semantic_relpath="",
                width=2,
                height=2,
                weather_tag="dry",
                time_tag="day",
                scene_tag="open",
                source_group_key="bdd100k::seq",
            )
            write_manifest_csv(root / "meta" / "split_manifest.csv", [row])

            summary = validate_pv26_dataset(root)
            self.assertEqual(summary.errors, [])


if __name__ == "__main__":
    unittest.main()

