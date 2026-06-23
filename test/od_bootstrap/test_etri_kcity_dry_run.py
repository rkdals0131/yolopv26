from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path

from PIL import Image

from common.pv26_schema import (
    ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY,
    ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY,
)
from tools.od_bootstrap.source.etri_kcity import (
    ATTRPSEUDO_DATASET_KEY,
    BLOCKED_STATUS,
    CANDIDATE_REASON_MISSING_SEMANTIC_LABEL,
    CANDIDATE_REASON_SAMPLE_ID_MISMATCH,
    RELEASE_BLOCKER_ZERO_SAMPLES,
    RAW_SCAN_REASON_LIDAR,
    RAW_SCAN_REASON_MONO_CAMERA,
    RAW_SCAN_REASON_PV26_OUTPUT,
    RAW_SCAN_REASON_RIGHT_IMG,
    READY_STATUS,
    EtriCandidateError,
    EtriDryRunNotReadyError,
    EtriMaterializationError,
    EtriDryRunResult,
    EtriDryRunSample,
    HELD_LABEL_REASON_UNMAPPED,
    HELD_LABELS_NAME,
    MULTICAMERA_ATTRPSEUDO_DATASET_KEY,
    MULTICAMERA_ATTRPSEUDO_SOURCE_KIND,
    MULTICAMERA_DATASET_KEY,
    build_dry_run_sample,
    is_dry_run_ready,
    materialize_kcity_val_release,
    require_dry_run_ready,
    scan_dry_run,
    write_ready_dry_run_manifest,
)
from model.data.dataset import PV26CanonicalDataset
from tools.od_bootstrap.source.etri_kcity.dry_run import main as etri_dry_run_main


class _FakeSignalAttrSidecar:
    def __init__(self, expected_image_path: Path) -> None:
        self.expected_image_path = expected_image_path.resolve()
        self.checkpoint_path = Path("/tmp/fake_best_signal_attr.pt")

    def apply_to_scene(self, scene: dict, image_path: Path, *, run_id: str, created_at: str) -> SimpleNamespace:
        self.assertEqual(Path(image_path).resolve(), self.expected_image_path)
        rows = []
        for index, detection in enumerate(scene["detections"]):
            self.assertEqual(detection["id"], index)
            if detection["class_name"] != "traffic_light":
                continue
            rows.append(
                {
                    "id": len(rows),
                    "detection_id": index,
                    "bbox": [
                        detection["bbox"]["x1"],
                        detection["bbox"]["y1"],
                        detection["bbox"]["x2"],
                        detection["bbox"]["y2"],
                    ],
                    "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
                    "tl_attr_valid": 1,
                    "collapse_reason": "valid",
                    "meta": {
                        "label_origin": "signal_attr_sidecar",
                        "run_id": run_id,
                        "created_at": created_at,
                    },
                }
            )
        scene["traffic_lights"] = rows
        scene["tasks"]["has_tl_attr"] = int(bool(rows))
        return SimpleNamespace(
            traffic_light_count=len(rows),
            valid_count=len(rows),
            invalid_count=0,
            reason_counts={"valid": len(rows)},
        )

    def assertEqual(self, actual: object, expected: object) -> None:
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")


class EtriKCityDryRunTests(unittest.TestCase):
    def test_etri_dry_run_includes_only_leftimg_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "leftImg" / "kc_001_leftImg.png"
            right_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "rightImg" / "kc_001_rightImg.png"
            mono_image = root / "ETRI" / "MonoCamera" / "KCity" / "train" / "leftImg" / "mono_001_leftImg.png"
            non_kcity_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "Seoul" / "train" / "leftImg" / "seoul_001_leftImg.png"
            generated_image = (
                root
                / "ETRI"
                / "Multi Camera Semantic Segmentation"
                / "pv26_etri_kcity_leftimg_attrpseudo_v1"
                / "images"
                / "train"
                / "kc_001_leftImg.png"
            )
            semantic_label = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "semantic" / "kc_001_semantic.json"
            self._make_image(left_image)
            self._make_image(right_image)
            self._make_image(mono_image)
            self._make_image(non_kcity_image)
            self._make_image(generated_image)
            self._write_json(
                semantic_label,
                {
                    "image": {"file_name": left_image.name, "image_size": {"width": 8, "height": 6}},
                    "annotations": [{"class_name": "vehicle"}],
                },
            )

            result = scan_dry_run(root)
            manifest = result.to_manifest()

            self.assertEqual(result.status, READY_STATUS)
            self.assertTrue(result.is_ready)
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["release_blockers"], [])
            self.assertEqual(manifest["samples"][0]["image_path"], str(left_image.resolve()))
            self.assertEqual(manifest["samples"][0]["semantic_label_path"], str(semantic_label.resolve()))
            self.assertEqual(
                manifest["raw_scan_ignored_count_by_reason"],
                {
                    RAW_SCAN_REASON_MONO_CAMERA: 1,
                    RAW_SCAN_REASON_PV26_OUTPUT: 1,
                    RAW_SCAN_REASON_RIGHT_IMG: 1,
                },
            )
            self.assertEqual(manifest["candidate_excluded_count_by_reason"], {})
            self.assertEqual(manifest["raw_class_inventory"], {"vehicle": 1})
            output_path = root / "release" / "conversion_manifest.json"
            self.assertEqual(write_ready_dry_run_manifest(root, output_path), output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["status"], READY_STATUS)

    def test_etri_converter_requires_semantic_label_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "KCity" / "train" / "leftImg" / "kc_002_leftImg.png"
            self._make_image(image_path)

            result = scan_dry_run(root)
            manifest = result.to_manifest()

            self.assertEqual(result.status, BLOCKED_STATUS)
            self.assertFalse(result.is_ready)
            self.assertEqual(manifest["sample_count"], 0)
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(
                manifest["candidate_excluded_count_by_reason"],
                {CANDIDATE_REASON_MISSING_SEMANTIC_LABEL: 1},
            )
            self.assertEqual(
                [blocker["reason"] for blocker in manifest["release_blockers"]],
                [RELEASE_BLOCKER_ZERO_SAMPLES, CANDIDATE_REASON_MISSING_SEMANTIC_LABEL],
            )
            output_path = root / "release" / "conversion_manifest.json"
            with self.assertRaises(EtriDryRunNotReadyError):
                write_ready_dry_run_manifest(root, output_path)
            self.assertFalse(output_path.exists())
            with self.assertRaisesRegex(EtriCandidateError, CANDIDATE_REASON_MISSING_SEMANTIC_LABEL):
                build_dry_run_sample(
                    image_path=image_path,
                    semantic_label_path=None,
                    dataset_root=root,
                )

    def test_etri_converter_rejects_image_label_sample_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "KCity" / "val" / "leftImg" / "kc_003_leftImg.png"
            label_path = root / "KCity" / "val" / "semantic" / "kc_003_semantic.json"
            self._make_image(image_path)
            self._write_json(
                label_path,
                {
                    "sample_id": "kc_999",
                    "image_size": [8, 6],
                    "annotations": [{"class_name": "vehicle"}],
                },
            )

            result = scan_dry_run(root)
            manifest = result.to_manifest()

            self.assertEqual(manifest["sample_count"], 0)
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(
                manifest["candidate_excluded_count_by_reason"],
                {CANDIDATE_REASON_SAMPLE_ID_MISMATCH: 1},
            )
            self.assertEqual(manifest["failures"][0]["semantic_label_path"], str(label_path.resolve()))
            with self.assertRaisesRegex(EtriCandidateError, CANDIDATE_REASON_SAMPLE_ID_MISMATCH):
                build_dry_run_sample(
                    image_path=image_path,
                    semantic_label_path=root / "KCity" / "val" / "semantic" / "other_semantic.json",
                    dataset_root=root,
                )

    def test_etri_dry_run_manifest_separates_raw_scan_ignored_from_candidate_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good_image = root / "KCity" / "leftImg" / "good_leftImg.png"
            missing_label_image = root / "KCity" / "leftImg" / "missing_leftImg.png"
            right_image = root / "KCity" / "rightImg" / "ignored_rightImg.png"
            lidar_label = root / "KCity" / "LiDAR" / "ignored_lidar.json"
            good_label = root / "KCity" / "labels" / "good_semantic.json"
            self._make_image(good_image)
            self._make_image(missing_label_image)
            self._make_image(right_image)
            self._write_json(lidar_label, {"annotations": [{"class_name": "vehicle"}]})
            self._write_json(
                good_label,
                {
                    "image": {"file_name": good_image.name, "image_size": {"width": 8, "height": 6}},
                    "annotations": [{"class_name": "vehicle"}, {"class_name": "crosswalk"}],
                },
            )

            manifest = scan_dry_run(root, default_split="train").to_manifest()

            self.assertEqual(manifest["status"], BLOCKED_STATUS)
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(
                manifest["raw_scan_ignored_count_by_reason"],
                {
                    RAW_SCAN_REASON_LIDAR: 1,
                    RAW_SCAN_REASON_RIGHT_IMG: 1,
                },
            )
            self.assertEqual(
                manifest["candidate_excluded_count_by_reason"],
                {CANDIDATE_REASON_MISSING_SEMANTIC_LABEL: 1},
            )
            self.assertNotIn(RAW_SCAN_REASON_RIGHT_IMG, manifest["candidate_excluded_count_by_reason"])
            self.assertNotIn(CANDIDATE_REASON_MISSING_SEMANTIC_LABEL, manifest["raw_scan_ignored_count_by_reason"])
            self.assertEqual(manifest["raw_class_inventory"], {"crosswalk": 1, "vehicle": 1})

    def test_etri_dry_run_reads_objects_label_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "KCity" / "val" / "20221124_kcity" / "leftImg" / "kc_005_leftImg.png"
            label_path = root / "KCity" / "val" / "20221124_kcity" / "semantic" / "kc_005_semantic.json"
            self._make_image(image_path)
            self._write_json(
                label_path,
                {
                    "image": {"file_name": image_path.name, "image_size": {"width": 8, "height": 6}},
                    "objects": [{"label": "car"}, {"label": "rubber cone"}],
                },
            )

            manifest = scan_dry_run(root).to_manifest()

            self.assertEqual(manifest["status"], READY_STATUS)
            self.assertEqual(manifest["raw_class_inventory"], {"car": 1, "rubber cone": 1})
            self.assertEqual(manifest["samples"][0]["raw_class_counts"], {"car": 1, "rubber cone": 1})

    def test_etri_general_scan_can_include_non_kcity_leftimg_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kcity_image = root / "leftImg" / "val" / "20221124_kcity" / "kc_001_leftImg8bit.png"
            kcity_label = root / "labels" / "val" / "20221124_kcity" / "kc_001_gtFine_polygons.json"
            highway_image = root / "leftImg" / "train" / "20220926_134648_highway" / "hw_001_leftImg8bit.png"
            highway_label = root / "labels" / "train" / "20220926_134648_highway" / "hw_001_gtFine_polygons.json"
            self._make_image(kcity_image)
            self._make_image(highway_image)
            self._write_json(
                kcity_label,
                {
                    "imgWidth": 8,
                    "imgHeight": 6,
                    "objects": [{"label": "car", "polygon": [[1, 1], [4, 1], [4, 3], [1, 3]]}],
                },
            )
            self._write_json(
                highway_label,
                {
                    "imgWidth": 8,
                    "imgHeight": 6,
                    "objects": [{"label": "rubber cone", "polygon": [[1, 1], [4, 1], [4, 3], [1, 3]]}],
                },
            )

            kcity_manifest = scan_dry_run(root).to_manifest()
            all_manifest = scan_dry_run(root, required_path_token=None).to_manifest()

            self.assertEqual(kcity_manifest["sample_count"], 1)
            self.assertEqual(kcity_manifest["raw_class_inventory"], {"car": 1})
            self.assertEqual(all_manifest["sample_count"], 2)
            self.assertEqual(all_manifest["raw_class_inventory"], {"car": 1, "rubber cone": 1})

    def test_etri_eval_release_can_collapse_train_and_val_source_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "release"
            train_image = root / "leftImg" / "train" / "20221124_155206_kcity" / "kc_train_001_leftImg8bit.png"
            train_label = root / "labels" / "train" / "20221124_155206_kcity" / "kc_train_001_gtFine_polygons.json"
            val_image = root / "leftImg" / "val" / "20221124_kcity" / "kc_val_001_leftImg8bit.png"
            val_label = root / "labels" / "val" / "20221124_kcity" / "kc_val_001_gtFine_polygons.json"
            self._make_image(train_image)
            self._make_image(val_image)
            label_payload = {
                "imgWidth": 8,
                "imgHeight": 6,
                "objects": [{"label": "car", "polygon": [[1, 1], [4, 1], [4, 3], [1, 3]]}],
            }
            self._write_json(train_label, label_payload)
            self._write_json(val_label, label_payload)

            summary = materialize_kcity_val_release(
                root,
                output_root,
                expected_sample_count=2,
                release_path_token="kcity",
                allowed_splits=("train", "val"),
            )
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            raw_splits = {row["source_raw_split"] for row in manifest["samples"]}
            scene_splits = {
                json.loads(Path(row["scene_path"]).read_text(encoding="utf-8"))["source"]["raw_split"]
                for row in manifest["samples"]
            }
            dataset = PV26CanonicalDataset([output_root])

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(manifest["split"], "val")
            self.assertEqual(manifest["allowed_source_splits"], ["train", "val"])
            self.assertEqual({row["split"] for row in manifest["samples"]}, {"val"})
            self.assertEqual(raw_splits, {"train", "val"})
            self.assertEqual(scene_splits, {"train", "val"})
            self.assertEqual(len(dataset), 2)

    def test_etri_general_release_can_exclude_kcity_into_separate_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "release"
            kcity_image = root / "leftImg" / "val" / "20221124_kcity" / "kc_001_leftImg8bit.png"
            kcity_label = root / "labels" / "val" / "20221124_kcity" / "kc_001_gtFine_polygons.json"
            highway_image = root / "leftImg" / "train" / "20220926_134648_highway" / "hw_001_leftImg8bit.png"
            highway_label = root / "labels" / "train" / "20220926_134648_highway" / "hw_001_gtFine_polygons.json"
            self._make_image(kcity_image)
            self._make_image(highway_image)
            label_payload = {
                "imgWidth": 8,
                "imgHeight": 6,
                "objects": [{"label": "traffic sign", "polygon": [[1, 1], [4, 1], [4, 3], [1, 3]]}],
            }
            self._write_json(kcity_label, label_payload)
            self._write_json(highway_label, label_payload)

            summary = materialize_kcity_val_release(
                root,
                output_root,
                expected_sample_count=1,
                release_path_token=None,
                exclude_path_tokens=("kcity",),
                allowed_splits=("train", "val"),
                scan_required_path_token=None,
                signal_attr_sidecar=_FakeSignalAttrSidecar(highway_image),
                attrpseudo_dataset_key_override=MULTICAMERA_ATTRPSEUDO_DATASET_KEY,
                attrpseudo_source_kind_override=MULTICAMERA_ATTRPSEUDO_SOURCE_KIND,
                sample_id_dataset_key=MULTICAMERA_DATASET_KEY,
            )
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            dataset = PV26CanonicalDataset([output_root])

            self.assertEqual(MULTICAMERA_ATTRPSEUDO_DATASET_KEY, ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(manifest["dataset_key"], ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertEqual(manifest["samples"][0]["source_image_path"], str(highway_image.resolve()))
            self.assertTrue(manifest["samples"][0]["final_sample_id"].startswith("etri_multicamera_leftimg_"))
            self.assertNotIn("20221124_kcity", json.dumps(manifest))
            self.assertEqual(dataset[0]["meta"]["dataset_key"], ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY)

    def test_etri_kcity_val_release_filters_path_holds_unmapped_and_loader_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "release"
            image_path = root / "KCity" / "val" / "20221124_kcity" / "leftImg" / "kc_006_leftImg.png"
            right_image = root / "KCity" / "val" / "20221124_kcity" / "rightImg" / "kc_006_rightImg.png"
            other_val_image = root / "KCity" / "val" / "20221125_kcity" / "leftImg" / "kc_007_leftImg.png"
            label_path = root / "KCity" / "val" / "20221124_kcity" / "semantic" / "kc_006_semantic.json"
            other_label = root / "KCity" / "val" / "20221125_kcity" / "semantic" / "kc_007_semantic.json"
            self._make_image(image_path)
            self._make_image(right_image)
            self._make_image(other_val_image)
            self._write_json(
                label_path,
                {
                    "image": {"file_name": image_path.name, "image_size": {"width": 8, "height": 6}},
                    "objects": [
                        {"label": "car", "box": {"x": 1, "y": 1, "w": 3, "h": 2}},
                        {"label": "whsol", "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]]},
                        {"label": "stop line", "polygon": [[1, 3], [6, 3], [6, 5], [1, 5]]},
                        {"label": "crosswalk", "polygon": [[1, 1], [6, 1], [6, 5], [1, 5]]},
                        {"label": "unknown barrier", "box": {"x": 1, "y": 1, "w": 2, "h": 2}},
                    ],
                },
            )
            self._write_json(
                other_label,
                {
                    "image": {"file_name": other_val_image.name, "image_size": {"width": 8, "height": 6}},
                    "objects": [{"label": "car", "box": {"x": 1, "y": 1, "w": 3, "h": 2}}],
                },
            )

            summary = materialize_kcity_val_release(root, output_root, expected_sample_count=1)
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            scene = json.loads(Path(manifest["samples"][0]["scene_path"]).read_text(encoding="utf-8"))
            held_rows = [
                json.loads(line)
                for line in (output_root / "meta" / HELD_LABELS_NAME).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            dataset = PV26CanonicalDataset([output_root])
            sample = dataset[0]

            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["samples"][0]["source_image_path"], str(image_path.resolve()))
            self.assertNotIn(str(right_image.resolve()), json.dumps(manifest))
            self.assertNotIn(str(other_val_image.resolve()), json.dumps(manifest))
            self.assertEqual(held_rows[0]["raw_label"], "unknown barrier")
            self.assertEqual(held_rows[0]["reason"], HELD_LABEL_REASON_UNMAPPED)
            self.assertEqual(sample["meta"]["dataset_key"], "etri_kcity_multicamera_leftimg")
            self.assertTrue(sample["source_mask"]["det"])
            self.assertTrue(sample["source_mask"]["lane"])
            self.assertTrue(sample["source_mask"]["stop_line"])
            self.assertTrue(sample["source_mask"]["crosswalk"])
            self.assertFalse(sample["source_mask"]["tl_attr"])
            self.assertEqual(sample["det_targets"]["classes"].tolist(), [0])
            self.assertEqual(len(sample["lane_targets"]["lanes"]), 1)
            self.assertEqual(len(sample["lane_targets"]["stop_lines"]), 1)
            self.assertEqual(len(sample["lane_targets"]["crosswalks"]), 1)
            lane_points = scene["lanes"][0]["points"]
            self.assertNotEqual(lane_points, [[1, 1], [5, 1], [5, 5], [1, 5]])
            self.assertTrue(all(abs(point[0] - 3.0) < 1.0e-3 for point in lane_points))
            self.assertEqual(scene["lanes"][0]["meta"]["geometry_policy"], "etri_lane_polygon_row_slice_centerline_v1")
            self.assertEqual(scene["stop_lines"][0]["points"], [[1.0, 4.0], [6.0, 4.0]])
            self.assertEqual(scene["stop_lines"][0]["meta"]["geometry_policy"], "etri_stop_line_polygon_centerline_v1")
            self.assertEqual(scene["crosswalks"][0]["points"], [[1.0, 1.0], [6.0, 1.0], [6.0, 5.0], [1.0, 5.0]])
            self.assertEqual(scene["crosswalks"][0]["meta"]["geometry_policy"], "etri_crosswalk_area_polygon_v1")

    def test_etri_attrpseudo_release_uses_signal_attr_sidecar_on_traffic_light_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "release"
            image_path = root / "KCity" / "val" / "20221124_kcity" / "leftImg" / "kc_010_leftImg.png"
            label_path = root / "KCity" / "val" / "20221124_kcity" / "semantic" / "kc_010_semantic.json"
            self._make_image(image_path)
            self._write_json(
                label_path,
                {
                    "image": {"file_name": image_path.name, "image_size": {"width": 8, "height": 6}},
                    "objects": [
                        {"label": "car", "box": {"x": 1, "y": 1, "w": 2, "h": 2}},
                        {"label": "traffic light", "polygon": [[3, 1], [5, 1], [5, 4], [3, 4]]},
                        {"label": "traffic sign", "polygon": [[6, 1], [7, 1], [7, 3], [6, 3]]},
                    ],
                },
            )

            summary = materialize_kcity_val_release(
                root,
                output_root,
                expected_sample_count=1,
                signal_attr_sidecar=_FakeSignalAttrSidecar(image_path),
            )
            manifest = json.loads((output_root / "meta" / "final_dataset_manifest.json").read_text(encoding="utf-8"))
            scene_path = Path(manifest["samples"][0]["scene_path"])
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            dataset = PV26CanonicalDataset([output_root])
            sample = dataset[0]

            self.assertEqual(ATTRPSEUDO_DATASET_KEY, ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertEqual(summary["signal_attr_sidecar"]["enabled"], True)
            self.assertEqual(summary["signal_attr_sidecar"]["valid_count"], 1)
            self.assertEqual(manifest["dataset_key"], ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertEqual(manifest["signal_attr_sidecar"]["reason_counts"], {"valid": 1})
            self.assertEqual(scene["source"]["dataset"], ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertEqual(scene["tasks"]["has_tl_attr"], 1)
            self.assertEqual([item["class_name"] for item in scene["detections"]], ["vehicle", "traffic_light", "sign"])
            self.assertEqual([item["detection_id"] for item in scene["traffic_lights"]], [1])
            self.assertEqual(scene["traffic_lights"][0]["tl_bits"], {"red": 1, "yellow": 0, "green": 0, "arrow": 1})
            self.assertEqual(sample["meta"]["dataset_key"], ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY)
            self.assertTrue(sample["source_mask"]["tl_attr"])
            self.assertEqual(sample["valid_mask"]["tl_attr"].tolist(), [False, True, False])
            self.assertEqual(sample["tl_attr_targets"]["bits"][1].tolist(), [1.0, 0.0, 0.0, 1.0])

    def test_etri_materialization_rejects_invalid_or_nonfinite_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "release"
            image_path = root / "KCity" / "val" / "20221124_kcity" / "leftImg" / "kc_008_leftImg.png"
            label_path = root / "KCity" / "val" / "20221124_kcity" / "semantic" / "kc_008_semantic.json"
            self._make_image(image_path)
            self._write_json(
                label_path,
                {
                    "image": {"file_name": image_path.name, "image_size": {"width": 8, "height": 6}},
                    "objects": [{"label": "whsol", "points": [[1, 2], [float("nan"), 3]]}],
                },
            )

            with self.assertRaises(EtriMaterializationError):
                materialize_kcity_val_release(root, output_root, expected_sample_count=1)
            self.assertFalse((output_root / "meta" / "final_dataset_manifest.json").exists())

    def test_etri_materialization_fails_release_on_zero_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "release" / "conversion_manifest.json"

            result = scan_dry_run(root, default_split="train")
            manifest = result.to_manifest()

            self.assertEqual(result.status, BLOCKED_STATUS)
            self.assertFalse(is_dry_run_ready(result))
            self.assertEqual(manifest["sample_count"], 0)
            self.assertEqual(manifest["candidate_excluded_count_by_reason"], {})
            self.assertEqual(
                manifest["release_blockers"],
                [
                    {
                        "reason": RELEASE_BLOCKER_ZERO_SAMPLES,
                        "count": 0,
                        "detail": "dry-run accepted no KCity leftImg samples",
                    }
                ],
            )
            with self.assertRaises(EtriDryRunNotReadyError) as error:
                require_dry_run_ready(result)
            self.assertEqual(error.exception.result, result)
            with self.assertRaises(EtriDryRunNotReadyError):
                write_ready_dry_run_manifest(root, output_path, default_split="train")
            self.assertFalse(output_path.exists())

    def test_etri_release_ready_requires_candidate_exclusions_to_be_empty(self) -> None:
        sample = EtriDryRunSample(
            sample_id="sample",
            split="train",
            image_path=Path("sample_leftImg.png"),
            semantic_label_path=Path("sample_semantic.json"),
            width=8,
            height=6,
            raw_class_counts={},
        )
        result = EtriDryRunResult(
            dataset_key="etri_kcity_multicamera_leftimg",
            dataset_root=Path("raw"),
            default_split=None,
            samples=(sample,),
            raw_scan_ignored_count_by_reason={},
            candidate_excluded_count_by_reason={"stale_zero_count": 0},
            raw_class_inventory={},
            failures=(),
            generated_at="2026-06-15T00:00:00+00:00",
        )

        self.assertEqual(result.failure_count, 0)
        self.assertFalse(is_dry_run_ready(result))
        self.assertEqual(result.status, BLOCKED_STATUS)
        self.assertEqual(result.to_manifest()["release_blockers"], [{"reason": "stale_zero_count", "count": 0}])
        with self.assertRaises(EtriDryRunNotReadyError):
            require_dry_run_ready(result)

    def test_etri_dry_run_cli_blocks_without_writing_release_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "KCity" / "train" / "leftImg" / "kc_004_leftImg.png"
            output_path = root / "release" / "conversion_manifest.json"
            self._make_image(image_path)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = etri_dry_run_main([str(root), "--output", str(output_path)])
            manifest = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 1)
            self.assertEqual(manifest["status"], BLOCKED_STATUS)
            self.assertEqual(
                manifest["candidate_excluded_count_by_reason"],
                {CANDIDATE_REASON_MISSING_SEMANTIC_LABEL: 1},
            )
            self.assertFalse(output_path.exists())

    def _make_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "#223344").save(path)

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
