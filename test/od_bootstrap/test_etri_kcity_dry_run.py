from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from tools.od_bootstrap.source.etri_kcity import (
    BLOCKED_STATUS,
    CANDIDATE_REASON_MISSING_SEMANTIC_LABEL,
    CANDIDATE_REASON_SAMPLE_ID_MISMATCH,
    RELEASE_BLOCKER_ZERO_SAMPLES,
    RAW_SCAN_REASON_LIDAR,
    RAW_SCAN_REASON_MONO_CAMERA,
    RAW_SCAN_REASON_RIGHT_IMG,
    READY_STATUS,
    EtriCandidateError,
    EtriDryRunNotReadyError,
    EtriDryRunResult,
    EtriDryRunSample,
    build_dry_run_sample,
    is_dry_run_ready,
    require_dry_run_ready,
    scan_dry_run,
    write_ready_dry_run_manifest,
)
from tools.od_bootstrap.source.etri_kcity.dry_run import main as etri_dry_run_main


class EtriKCityDryRunTests(unittest.TestCase):
    def test_etri_dry_run_includes_only_leftimg_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "leftImg" / "kc_001_leftImg.png"
            right_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "rightImg" / "kc_001_rightImg.png"
            mono_image = root / "ETRI" / "MonoCamera" / "KCity" / "train" / "leftImg" / "mono_001_leftImg.png"
            non_kcity_image = root / "ETRI" / "Multi Camera Semantic Segmentation" / "Seoul" / "train" / "leftImg" / "seoul_001_leftImg.png"
            semantic_label = root / "ETRI" / "Multi Camera Semantic Segmentation" / "KCity" / "train" / "semantic" / "kc_001_semantic.json"
            self._make_image(left_image)
            self._make_image(right_image)
            self._make_image(mono_image)
            self._make_image(non_kcity_image)
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
