from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "od_bootstrap" / "build" / "lane_val_odpseudo.py"
_SPEC = importlib.util.spec_from_file_location("lane_val_odpseudo_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lane_val_odpseudo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = lane_val_odpseudo
_SPEC.loader.exec_module(lane_val_odpseudo)

DATASET_KEY = lane_val_odpseudo.DATASET_KEY
FINAL_DATASET_MANIFEST_NAME = lane_val_odpseudo.FINAL_DATASET_MANIFEST_NAME
METRIC_SEMANTICS = lane_val_odpseudo.METRIC_SEMANTICS
REJECTED_DETECTIONS_NAME = lane_val_odpseudo.REJECTED_DETECTIONS_NAME
TL_ATTR_METRIC_DISABLED_REASON = lane_val_odpseudo.TL_ATTR_METRIC_DISABLED_REASON
TEACHER_NAMES = lane_val_odpseudo.TEACHER_NAMES
LaneValBaseRecord = lane_val_odpseudo.LaneValBaseRecord
TeacherFailureError = lane_val_odpseudo.TeacherFailureError
build_lane_val_odpseudo_eval_root = lane_val_odpseudo.build_lane_val_odpseudo_eval_root
completed_empty_sample_result = lane_val_odpseudo.completed_empty_sample_result
discover_base_lane_val_records = lane_val_odpseudo.discover_base_lane_val_records
guard_lane_val_odpseudo_eval_report = lane_val_odpseudo.guard_lane_val_odpseudo_eval_report
normalize_rejected_candidate_row = lane_val_odpseudo.normalize_rejected_candidate_row
preflight_lane_val_odpseudo_records = lane_val_odpseudo.preflight_lane_val_odpseudo_records
validate_base_records = lane_val_odpseudo.validate_base_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _create_base_sample(
    root: Path,
    sample_id: str,
    *,
    split: str = "val",
    dataset_key: str = "aihub_lane_seoul",
    image_name: str | None = None,
) -> LaneValBaseRecord:
    resolved_image_name = image_name or f"{sample_id}.jpg"
    image_path = root / "images" / split / resolved_image_name
    scene_path = root / "labels_scene" / split / f"{sample_id}.json"
    _write_text(image_path, f"fake image {sample_id}\n")
    _write_json(
        scene_path,
        {
            "image": {"file_name": resolved_image_name, "width": 640, "height": 480},
            "source": {"dataset": dataset_key, "split": split},
            "tasks": {
                "has_det": 0,
                "has_lane": 1,
                "has_stop_line": 0,
                "has_crosswalk": 1,
                "has_tl_attr": 1,
            },
            "detections": [{"class_name": "vehicle", "bbox": [1, 2, 3, 4]}],
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
            "traffic_lights": [],
        },
    )
    return LaneValBaseRecord(
        sample_id=sample_id,
        split=split,
        source_dataset_key=dataset_key,
        scene_path=scene_path,
        image_path=image_path,
    )


def _teacher_checkpoints(root: Path) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for teacher_name in TEACHER_NAMES:
        checkpoint_path = root / "checkpoints" / f"best_{teacher_name}.pt"
        _write_text(checkpoint_path, f"fake checkpoint {teacher_name}\n")
        checkpoints[teacher_name] = checkpoint_path
    return checkpoints


def _empty_results(sample_ids: list[str] | tuple[str, ...]) -> dict[str, dict]:
    return {sample_id: completed_empty_sample_result(sample_id) for sample_id in sample_ids}


def _read_manifest(output_root: Path) -> dict:
    return json.loads(
        (output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
    )


def _read_dataset_group_by_key_literal() -> dict[str, str]:
    sampler_path = Path(__file__).resolve().parents[2] / "model" / "data" / "sampler.py"
    module = ast.parse(sampler_path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "DATASET_GROUP_BY_KEY" for target in node.targets):
            payload = ast.literal_eval(node.value)
            if not isinstance(payload, dict):
                raise TypeError("DATASET_GROUP_BY_KEY must be a dict literal")
            return {str(key): str(value) for key, value in payload.items()}
    raise AssertionError("DATASET_GROUP_BY_KEY literal not found")


class LaneValODPseudoEvalRootTests(unittest.TestCase):
    def test_lane_val_odpseudo_source_is_eval_only_and_not_train_sampler_group(self) -> None:
        from common.pv26_schema import LANE_VAL_ODPSEUDO_DATASET_KEY, SOURCE_MASK_BY_DATASET

        self.assertEqual(LANE_VAL_ODPSEUDO_DATASET_KEY, DATASET_KEY)
        self.assertEqual(
            SOURCE_MASK_BY_DATASET[DATASET_KEY],
            {
                "det": True,
                "tl_attr": False,
                "lane": True,
                "stop_line": True,
                "crosswalk": True,
            },
        )
        DATASET_GROUP_BY_KEY = _read_dataset_group_by_key_literal()
        self.assertNotIn(DATASET_KEY, DATASET_GROUP_BY_KEY)

    def test_lane_val_odpseudo_disables_tl_attr_metrics_in_evaluator_report(self) -> None:
        report = {
            "dataset_key": DATASET_KEY,
            "split": "val",
            "source_mask": {
                "det": True,
                "tl_attr": False,
                "lane": True,
                "stop_line": True,
                "crosswalk": True,
            },
            "metrics": {
                "detector": {"map50": 0.75},
                "traffic_light": {"combo_accuracy": 0.50, "mean_f1": 0.25},
                "lane": {"f1": 0.90},
            },
        }

        guarded = guard_lane_val_odpseudo_eval_report(report)

        self.assertEqual(
            guarded["metrics"]["traffic_light"],
            {"disabled": True, "reason": TL_ATTR_METRIC_DISABLED_REASON},
        )
        self.assertNotIn("combo_accuracy", guarded["metrics"]["traffic_light"])
        self.assertEqual(guarded["metrics"]["detector"], {"map50": 0.75})
        self.assertEqual(guarded["metrics"]["lane"], {"f1": 0.90})
        self.assertEqual(report["metrics"]["traffic_light"]["combo_accuracy"], 0.50)

    def test_lane_val_odpseudo_metric_report_declares_teacher_pseudo_agreement(self) -> None:
        guarded = guard_lane_val_odpseudo_eval_report(
            {
                "metrics": {
                    "detector": {"map50_95": 0.30},
                    "traffic_light": {"combo_accuracy": 1.0},
                }
            },
            source_mask={"det": True, "tl_attr": False},
        )

        self.assertEqual(guarded["dataset_key"], DATASET_KEY)
        self.assertEqual(guarded["split"], "val")
        self.assertEqual(guarded["metric_semantics"], METRIC_SEMANTICS)
        self.assertEqual(guarded["source_mask"]["tl_attr"], False)

        with self.assertRaisesRegex(ValueError, "source_mask.tl_attr=false"):
            guard_lane_val_odpseudo_eval_report(
                {"dataset_key": DATASET_KEY, "split": "val", "metrics": {}},
                source_mask={"det": True, "tl_attr": True},
            )

        with self.assertRaisesRegex(ValueError, "val-only"):
            guard_lane_val_odpseudo_eval_report(
                {"dataset_key": DATASET_KEY, "split": "train", "metrics": {}},
                source_mask={"det": True, "tl_attr": False},
            )

    def test_lane_val_odpseudo_preserves_base_val_sample_ids_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            _create_base_sample(base_root, "lane_b")
            _create_base_sample(base_root, "lane_a")

            base_records = discover_base_lane_val_records(base_root)
            sample_ids = tuple(record.sample_id for record in base_records)
            summary = build_lane_val_odpseudo_eval_root(
                base_lane_root=base_root,
                output_root=output_root,
                sample_results=_empty_results(sample_ids),
                teacher_checkpoints=_teacher_checkpoints(root),
                copy_images=True,
            )

            manifest = _read_manifest(output_root)
            manifest_ids = [row["final_sample_id"] for row in manifest["samples"]]
            self.assertEqual(summary["sample_count"], len(sample_ids))
            self.assertEqual(manifest["dataset_key"], DATASET_KEY)
            self.assertEqual(manifest["sample_count"], len(sample_ids))
            self.assertEqual(manifest_ids, list(sample_ids))
            self.assertEqual(manifest["failure_count"], 0)

            for row in manifest["samples"]:
                sample_id = row["final_sample_id"]
                self.assertEqual(row["teacher_run_status"], "completed")
                self.assertEqual(row["accepted_detection_count"], 0)
                self.assertEqual(row["det_file_status"], "empty")
                self.assertTrue((output_root / "labels_det" / "val" / f"{sample_id}.txt").is_file())
                self.assertEqual(
                    (output_root / "labels_det" / "val" / f"{sample_id}.txt").read_text(encoding="utf-8"),
                    "",
                )
                scene = json.loads(
                    (output_root / "labels_scene" / "val" / f"{sample_id}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(scene["source"]["dataset"], DATASET_KEY)
                self.assertEqual(scene["source"]["split"], "val")
                self.assertEqual(scene["source"]["source_kind"], "lane_val_odpseudo")
                self.assertEqual(scene["tasks"]["has_det"], 0)
                self.assertEqual(scene["tasks"]["has_tl_attr"], 0)
                self.assertEqual(scene["detections"], [])

    def test_lane_val_odpseudo_rejects_non_val_or_foreign_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            train_record = _create_base_sample(base_root, "train_sample", split="train")

            with self.assertRaisesRegex(ValueError, "val-only"):
                validate_base_records((train_record,), base_lane_root=base_root)

            val_record = _create_base_sample(base_root, "val_sample")
            foreign_image = root / "foreign" / "val_sample.jpg"
            _write_text(foreign_image, "foreign image\n")
            with self.assertRaisesRegex(ValueError, "source_image_path must stay under base lane val root"):
                validate_base_records(
                    (
                        LaneValBaseRecord(
                            sample_id=val_record.sample_id,
                            split=val_record.split,
                            source_dataset_key=val_record.source_dataset_key,
                            scene_path=val_record.scene_path,
                            image_path=foreign_image,
                        ),
                    ),
                    base_lane_root=base_root,
                )

            foreign_dataset_root = root / "foreign_dataset"
            _create_base_sample(
                foreign_dataset_root,
                "etri_sample",
                dataset_key="etri_kcity_camera",
            )
            with self.assertRaisesRegex(ValueError, "unsupported base lane dataset"):
                discover_base_lane_val_records(foreign_dataset_root)

    def test_lane_val_odpseudo_manifest_order_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            for sample_id in ("lane_c", "lane_a", "lane_b"):
                _create_base_sample(base_root, sample_id)

            sample_results = [
                completed_empty_sample_result("lane_c"),
                completed_empty_sample_result("lane_b"),
                completed_empty_sample_result("lane_a"),
            ]
            build_lane_val_odpseudo_eval_root(
                base_lane_root=base_root,
                output_root=output_root,
                sample_results=sample_results,
                teacher_checkpoints=_teacher_checkpoints(root),
                copy_images=True,
            )

            manifest = _read_manifest(output_root)
            self.assertEqual(
                [row["final_sample_id"] for row in manifest["samples"]],
                ["lane_a", "lane_b", "lane_c"],
            )
            self.assertEqual(
                [Path(row["scene_path"]).name for row in manifest["samples"]],
                ["lane_a.json", "lane_b.json", "lane_c.json"],
            )

    def test_lane_val_odpseudo_nonfinite_candidate_is_rejected_but_teacher_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            _create_base_sample(base_root, "lane_a")
            checkpoints = _teacher_checkpoints(root)
            nonfinite_candidate = {
                "sample_id": "lane_a",
                "teacher_name": "signal",
                "class_name": "traffic_light",
                "score": math.nan,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "reason": "nonfinite_prediction",
            }

            normalized = normalize_rejected_candidate_row(
                nonfinite_candidate,
                known_sample_ids={"lane_a"},
            )
            self.assertEqual(normalized["sample_id"], "lane_a")
            self.assertEqual(normalized["teacher_name"], "signal")
            self.assertIsNone(normalized["score"])
            self.assertEqual(normalized["reason"], "nonfinite_prediction")
            self.assertEqual(normalized["nonfinite_fields"], ["score"])

            result = completed_empty_sample_result("lane_a")
            result["rejected_candidates"] = [nonfinite_candidate]
            summary = build_lane_val_odpseudo_eval_root(
                base_lane_root=base_root,
                output_root=output_root,
                sample_results={"lane_a": result},
                teacher_checkpoints=checkpoints,
                copy_images=True,
            )
            manifest = _read_manifest(output_root)
            rejected_rows = [
                json.loads(line)
                for line in (output_root / "meta" / REJECTED_DETECTIONS_NAME).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["nonfinite_candidate_count"], 1)
            self.assertEqual(manifest["nonfinite_candidate_count"], 1)
            self.assertEqual(rejected_rows[0]["sample_id"], "lane_a")
            self.assertEqual(rejected_rows[0]["teacher_name"], "signal")
            self.assertIsNone(rejected_rows[0]["score"])

            failed_result = completed_empty_sample_result("lane_a")
            failed_result["rejected_candidates"] = [
                {
                    "sample_id": "lane_a",
                    "teacher_name": "signal",
                    "class_name": "traffic_light",
                    "score": 0.0,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "reason": "teacher_failure",
                }
            ]
            with self.assertRaises(TeacherFailureError):
                preflight_lane_val_odpseudo_records(
                    base_records=discover_base_lane_val_records(base_root),
                    base_lane_root=base_root,
                    output_root=root / "failed_eval_root",
                    sample_results={"lane_a": failed_result},
                    teacher_checkpoints=checkpoints,
                )
            self.assertFalse(
                (root / "failed_eval_root" / "meta" / FINAL_DATASET_MANIFEST_NAME).exists()
            )

            missing_det_result = completed_empty_sample_result("lane_a")
            missing_det_result["det_file_status"] = "missing"
            with self.assertRaises(TeacherFailureError):
                preflight_lane_val_odpseudo_records(
                    base_records=discover_base_lane_val_records(base_root),
                    base_lane_root=base_root,
                    output_root=root / "missing_det_eval_root",
                    sample_results={"lane_a": missing_det_result},
                    teacher_checkpoints=checkpoints,
                )
            self.assertFalse(
                (root / "missing_det_eval_root" / "meta" / FINAL_DATASET_MANIFEST_NAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
