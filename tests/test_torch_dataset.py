import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pv26.torch_dataset import LetterboxSpec, Pv26ManifestDataset


class TestPv26ManifestDataset(unittest.TestCase):
    def test_load_one_sample_letterbox(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Layout
            (root / "meta").mkdir(parents=True, exist_ok=True)
            for p in [
                "images/train",
                "labels_det/train",
                "labels_seg_da/train",
                "labels_seg_rm_lane_marker/train",
                "labels_seg_rm_road_marker_non_lane/train",
                "labels_seg_rm_stop_line/train",
            ]:
                (root / p).mkdir(parents=True, exist_ok=True)

            sample_id = "bdd100k__seq__000000__cam0"
            img_rel = f"images/train/{sample_id}.jpg"
            det_rel = f"labels_det/train/{sample_id}.txt"
            da_rel = f"labels_seg_da/train/{sample_id}.png"
            rm_lane_rel = f"labels_seg_rm_lane_marker/train/{sample_id}.png"
            rm_road_rel = f"labels_seg_rm_road_marker_non_lane/train/{sample_id}.png"
            rm_stop_rel = f"labels_seg_rm_stop_line/train/{sample_id}.png"

            # Image 4x2
            img = Image.fromarray(np.full((2, 4, 3), 10, dtype=np.uint8), mode="RGB")
            img.save(root / img_rel)

            # 1 bbox in YOLO normalized (orig 4x2)
            (root / det_rel).write_text("0 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")

            # DA mask {0,1,255}
            da = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)
            Image.fromarray(da, mode="L").save(root / da_rel)

            # RM lane/road masks
            rm_lane = np.array([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)
            rm_road = np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
            Image.fromarray(rm_lane, mode="L").save(root / rm_lane_rel)
            Image.fromarray(rm_road, mode="L").save(root / rm_road_rel)
            # stop_line file exists but will be ignored (has_rm_stop_line=0)
            Image.fromarray(np.full((2, 4), 255, dtype=np.uint8), mode="L").save(root / rm_stop_rel)

            # Manifest
            manifest_path = root / "meta" / "split_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
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
                        "has_semantic_id",
                        "det_label_scope",
                        "det_annotated_class_ids",
                        "image_relpath",
                        "det_relpath",
                        "da_relpath",
                        "rm_lane_marker_relpath",
                        "rm_road_marker_non_lane_relpath",
                        "rm_stop_line_relpath",
                        "semantic_relpath",
                        "width",
                        "height",
                        "weather_tag",
                        "time_tag",
                        "scene_tag",
                        "source_group_key",
                    ]
                )
                w.writerow(
                    [
                        sample_id,
                        "train",
                        "bdd100k",
                        "seq",
                        "000000",
                        "cam0",
                        "",
                        "1",
                        "1",
                        "1",
                        "1",
                        "0",
                        "0",
                        "full",
                        "",
                        img_rel,
                        det_rel,
                        da_rel,
                        rm_lane_rel,
                        rm_road_rel,
                        rm_stop_rel,
                        "",
                        "4",
                        "2",
                        "dry",
                        "day",
                        "open",
                        "bdd100k::seq",
                    ]
                )

            ds = Pv26ManifestDataset(
                dataset_root=root,
                splits=("train",),
                letterbox=LetterboxSpec(out_width=6, out_height=6),
            )
            self.assertEqual(len(ds), 1)
            s = ds[0]

            self.assertEqual(tuple(s.image.shape), (3, 6, 6))
            self.assertEqual(tuple(s.da_mask.shape), (6, 6))
            self.assertEqual(tuple(s.rm_mask.shape), (3, 6, 6))
            self.assertEqual(tuple(s.det_yolo.shape), (1, 5))

            # Check bbox transforms roughly (see letterbox math in pv26.torch_dataset)
            _, cx, cy, w, h = s.det_yolo[0].tolist()
            self.assertAlmostEqual(cx, 0.5, places=5)
            self.assertAlmostEqual(cy, 2.5 / 6.0, places=5)  # 0.416666...
            self.assertAlmostEqual(w, 0.5, places=5)
            self.assertAlmostEqual(h, 0.25, places=5)

            # stop_line is unsupervised -> ignore mask
            self.assertTrue(bool((s.rm_mask[2] == 255).all().item()))


if __name__ == "__main__":
    unittest.main()

