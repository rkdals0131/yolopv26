import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pv26.dataset.classmap import render_class_map_yaml
from pv26.dataset.validation import validate_pv26_dataset
from tools.data_analysis.merge_pv26_dataset import merge_pv26_datasets


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_u8_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _write_rgb_jpg(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, format="JPEG")


def _make_row(*, sample_id: str, source: str, split: str) -> dict[str, str]:
    base = f"{sample_id}"
    return {
        "sample_id": sample_id,
        "split": split,
        "source": source,
        "sequence": f"{source}_seq0",
        "frame": "000000",
        "camera_id": "cam0",
        "timestamp_ns": "0",
        "has_det": "0",
        "has_da": "1",
        "has_rm_lane_marker": "1",
        "has_rm_road_marker_non_lane": "1",
        "has_rm_stop_line": "1",
        "has_rm_lane_subclass": "1",
        "has_semantic_id": "0",
        "det_label_scope": "none",
        "det_annotated_class_ids": "",
        "image_relpath": f"images/{split}/{base}.jpg",
        "det_relpath": f"labels_det/{split}/{base}.txt",
        "da_relpath": f"labels_seg_da/{split}/{base}.png",
        "rm_lane_marker_relpath": f"labels_seg_rm_lane_marker/{split}/{base}.png",
        "rm_road_marker_non_lane_relpath": f"labels_seg_rm_road_marker_non_lane/{split}/{base}.png",
        "rm_stop_line_relpath": f"labels_seg_rm_stop_line/{split}/{base}.png",
        "rm_lane_subclass_relpath": f"labels_seg_rm_lane_subclass/{split}/{base}.png",
        "semantic_relpath": "",
        "width": "4",
        "height": "4",
        "weather_tag": "day",
        "time_tag": "day",
        "scene_tag": "urban",
        "source_group_key": f"{source}:{sample_id}",
    }


def _create_fake_dataset(root: Path, *, sample_id: str, source: str, split: str) -> None:
    row = _make_row(sample_id=sample_id, source=source, split=split)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 1] = 127
    bin_mask = np.zeros((4, 4), dtype=np.uint8)
    bin_mask[1:3, 1:3] = 1
    lane_sub = np.zeros((4, 4), dtype=np.uint8)
    lane_sub[1:3, 1:3] = 2

    _write_rgb_jpg(root / row["image_relpath"], rgb)
    _write_text(root / row["det_relpath"], "")
    _write_u8_png(root / row["da_relpath"], bin_mask)
    _write_u8_png(root / row["rm_lane_marker_relpath"], bin_mask)
    _write_u8_png(root / row["rm_road_marker_non_lane_relpath"], bin_mask)
    _write_u8_png(root / row["rm_stop_line_relpath"], bin_mask)
    _write_u8_png(root / row["rm_lane_subclass_relpath"], lane_sub)

    with (meta / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    _write_text(meta / "class_map.yaml", render_class_map_yaml())
    _write_text(meta / "source_stats.csv", f"source,num_samples\n{source},1\n")
    _write_text(meta / "checksums.sha256", "")
    (meta / "conversion_report.json").write_text(json.dumps({"source": source}, indent=2), encoding="utf-8")
    (meta / "qc_report.json").write_text(json.dumps({"source": source, "num_rows": 1}, indent=2), encoding="utf-8")


class TestMergePv26TypeA(unittest.TestCase):
    def test_merge_two_roots_produces_valid_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root_a = tmp_path / "pv26_v1_bdd_full"
            root_b = tmp_path / "pv26_v1_etri"
            out_root = tmp_path / "pv26_v1_merged"

            _create_fake_dataset(root_a, sample_id="bdd100k__a__000000__cam0", source="bdd100k", split="train")
            _create_fake_dataset(root_b, sample_id="etri__b__000000__cam0", source="etri", split="val")

            report = merge_pv26_datasets(
                input_roots=[root_a, root_b],
                out_root=out_root,
                materialize_mode="copy",
                workers=1,
                validate=True,
                validate_workers=1,
                argv=["merge-test"],
            )

            self.assertEqual(int(report["num_input_roots"]), 2)
            self.assertEqual(int(report["num_rows"]), 2)
            self.assertEqual(report["rows_by_source"]["bdd100k"], 1)
            self.assertEqual(report["rows_by_source"]["etri"], 1)

            with (out_root / "meta" / "split_manifest.csv").open("r", encoding="utf-8", newline="") as f:
                manifest_rows = list(csv.DictReader(f))
            self.assertEqual(len(manifest_rows), 2)
            self.assertTrue((out_root / manifest_rows[0]["image_relpath"]).exists())
            self.assertTrue((out_root / "meta" / "merge_report.json").exists())
            self.assertTrue((out_root / "meta" / "input_datasets.json").exists())
            self.assertTrue((out_root / "meta" / "input_datasets" / "pv26_v1_bdd_full" / "conversion_report.json").exists())

            summary = validate_pv26_dataset(out_root, workers=1)
            self.assertEqual(summary.errors, [])


if __name__ == "__main__":
    unittest.main()
