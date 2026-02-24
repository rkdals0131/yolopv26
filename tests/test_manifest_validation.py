import unittest

from pv26.manifest import MANIFEST_COLUMNS, validate_manifest_row_basic


class TestManifestValidation(unittest.TestCase):
    def _base_row(self):
        row = {k: "" for k in MANIFEST_COLUMNS}
        row.update(
            {
                "sample_id": "bdd100k__seq__000001__cam0",
                "split": "train",
                "source": "bdd100k",
                "sequence": "seq",
                "frame": "000001",
                "camera_id": "cam0",
                "timestamp_ns": "",
                "has_det": "0",
                "has_da": "0",
                "has_rm_lane_marker": "0",
                "has_rm_road_marker_non_lane": "0",
                "has_rm_stop_line": "0",
                "has_semantic_id": "0",
                "det_label_scope": "none",
                "det_annotated_class_ids": "",
                "image_relpath": "images/train/x.jpg",
                "det_relpath": "labels_det/train/x.txt",
                "da_relpath": "labels_seg_da/train/x.png",
                "rm_lane_marker_relpath": "labels_seg_rm_lane_marker/train/x.png",
                "rm_road_marker_non_lane_relpath": "labels_seg_rm_road_marker_non_lane/train/x.png",
                "rm_stop_line_relpath": "labels_seg_rm_stop_line/train/x.png",
                "semantic_relpath": "",
                "width": "2",
                "height": "2",
                "weather_tag": "dry",
                "time_tag": "day",
                "scene_tag": "open",
                "source_group_key": "bdd100k::seq",
            }
        )
        return row

    def test_subset_requires_annotated_ids(self):
        r = self._base_row()
        r["det_label_scope"] = "subset"
        errs = validate_manifest_row_basic(r)
        self.assertIn("subset_missing_det_annotated_class_ids", errs)


if __name__ == "__main__":
    unittest.main()

