from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.od_bootstrap.build.sweep_types import ClassPolicy, RunConfig, TeacherConfig


_MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "od_bootstrap" / "build" / "lane_val_odpseudo.py"
_SPEC = importlib.util.spec_from_file_location("lane_val_odpseudo_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lane_val_odpseudo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = lane_val_odpseudo
_SPEC.loader.exec_module(lane_val_odpseudo)

DATASET_KEY = lane_val_odpseudo.DATASET_KEY
ATTR_DATASET_KEY = lane_val_odpseudo.ATTR_DATASET_KEY
ATTR_METRIC_SEMANTICS = lane_val_odpseudo.ATTR_METRIC_SEMANTICS
DEFAULT_EXPECTED_BASE_VAL_COUNT = lane_val_odpseudo.DEFAULT_EXPECTED_BASE_VAL_COUNT
FINAL_DATASET_MANIFEST_NAME = lane_val_odpseudo.FINAL_DATASET_MANIFEST_NAME
METRIC_SEMANTICS = lane_val_odpseudo.METRIC_SEMANTICS
REJECTED_DETECTIONS_NAME = lane_val_odpseudo.REJECTED_DETECTIONS_NAME
TL_ATTR_METRIC_DISABLED_REASON = lane_val_odpseudo.TL_ATTR_METRIC_DISABLED_REASON
TEACHER_NAMES = lane_val_odpseudo.TEACHER_NAMES
LaneValBaseRecord = lane_val_odpseudo.LaneValBaseRecord
TeacherFailureError = lane_val_odpseudo.TeacherFailureError
build_lane_val_odpseudo_eval_root = lane_val_odpseudo.build_lane_val_odpseudo_eval_root
build_lane_val_odpseudo_sample_results_from_predictions = (
    lane_val_odpseudo.build_lane_val_odpseudo_sample_results_from_predictions
)
completed_empty_sample_result = lane_val_odpseudo.completed_empty_sample_result
discover_base_lane_val_records = lane_val_odpseudo.discover_base_lane_val_records
guard_lane_val_odpseudo_eval_report = lane_val_odpseudo.guard_lane_val_odpseudo_eval_report
normalize_rejected_candidate_row = lane_val_odpseudo.normalize_rejected_candidate_row
preflight_lane_val_odpseudo_records = lane_val_odpseudo.preflight_lane_val_odpseudo_records
run_lane_val_odpseudo_teacher_sample_results = lane_val_odpseudo.run_lane_val_odpseudo_teacher_sample_results
validate_base_records = lane_val_odpseudo.validate_base_records


class _FakeSignalAttrStats:
    traffic_light_count = 1
    valid_count = 1
    invalid_count = 0
    reason_counts = {"valid": 1}


class _FakeSignalAttrSidecar:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.applied_scenes: list[dict] = []

    def apply_to_scene(self, scene: dict, image_path: Path, *, run_id: str, created_at: str) -> _FakeSignalAttrStats:
        self.applied_scenes.append(
            {
                "image_path": str(image_path),
                "run_id": run_id,
                "created_at": created_at,
                "class_names": [row["class_name"] for row in scene["detections"]],
            }
        )
        traffic_lights = []
        for detection_index, detection in enumerate(scene["detections"]):
            if detection["class_name"] != "traffic_light":
                continue
            traffic_lights.append(
                {
                    "id": len(traffic_lights),
                    "detection_id": detection_index,
                    "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 0},
                    "tl_attr_valid": 1,
                    "collapse_reason": "valid",
                }
            )
        scene["traffic_lights"] = traffic_lights
        scene.setdefault("tasks", {})["has_tl_attr"] = int(bool(traffic_lights))
        return _FakeSignalAttrStats()


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


def _class_policy() -> dict[str, ClassPolicy]:
    return {
        "vehicle": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "bike": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "pedestrian": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "traffic_cone": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "obstacle": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "traffic_light": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
        "sign": ClassPolicy(score_threshold=0.30, nms_iou_threshold=0.50, min_box_size=4),
    }


def _prediction(
    record: LaneValBaseRecord,
    *,
    teacher_name: str,
    class_name: str,
    confidence: float,
    xyxy: list[float],
    box_index: int = 0,
) -> dict:
    return {
        "sample_id": record.sample_id,
        "sample_uid": record.sample_id,
        "image_path": str(record.image_path),
        "scene_path": str(record.scene_path),
        "dataset_key": record.source_dataset_key,
        "split": record.split,
        "teacher_name": teacher_name,
        "model_version": f"{teacher_name}_test",
        "class_name": class_name,
        "confidence": confidence,
        "xyxy": xyxy,
        "box_index": box_index,
        "image_width": 640,
        "image_height": 480,
    }


def _teacher_configs(root: Path) -> tuple[TeacherConfig, ...]:
    configs = []
    for teacher_name in TEACHER_NAMES:
        checkpoint = root / "teacher_weights" / teacher_name / "best.pt"
        _write_text(checkpoint, f"{teacher_name}\n")
        configs.append(
            TeacherConfig(
                name=teacher_name,
                base_model=f"{teacher_name}_model",
                checkpoint_path=checkpoint,
                model_version=f"{teacher_name}_test",
                classes=(),
            )
        )
    return tuple(configs)


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
        from common.pv26_schema import (
            DET_SUPERVISION_BY_DATASET,
            LANE_VAL_ODPSEUDO_ATTR_DATASET_KEY,
            LANE_VAL_ODPSEUDO_DATASET_KEY,
            OD_CLASSES,
            SOURCE_MASK_BY_DATASET,
        )

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
        self.assertEqual(LANE_VAL_ODPSEUDO_ATTR_DATASET_KEY, ATTR_DATASET_KEY)
        self.assertEqual(
            SOURCE_MASK_BY_DATASET[ATTR_DATASET_KEY],
            {
                "det": True,
                "tl_attr": True,
                "lane": True,
                "stop_line": True,
                "crosswalk": True,
            },
        )
        self.assertEqual(tuple(DET_SUPERVISION_BY_DATASET[ATTR_DATASET_KEY]["class_names"]), OD_CLASSES)
        self.assertFalse(DET_SUPERVISION_BY_DATASET[ATTR_DATASET_KEY]["allow_objectness_negatives"])
        self.assertFalse(DET_SUPERVISION_BY_DATASET[ATTR_DATASET_KEY]["allow_unmatched_class_negatives"])
        DATASET_GROUP_BY_KEY = _read_dataset_group_by_key_literal()
        self.assertNotIn(DATASET_KEY, DATASET_GROUP_BY_KEY)
        self.assertNotIn(ATTR_DATASET_KEY, DATASET_GROUP_BY_KEY)

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

    def test_lane_val_odpseudo_attr_v2_report_declares_mixed_metric_semantics(self) -> None:
        guarded = guard_lane_val_odpseudo_eval_report(
            {
                "metrics": {
                    "detector": {"map50_95": 0.30},
                    "traffic_light": {"combo_accuracy": 0.90},
                    "lane": {"f1": 0.95},
                }
            },
            source_mask={"det": True, "tl_attr": True, "lane": True, "stop_line": True, "crosswalk": True},
            variant="attr_v2",
        )

        self.assertEqual(guarded["dataset_key"], ATTR_DATASET_KEY)
        self.assertEqual(guarded["source_mask"]["tl_attr"], True)
        self.assertEqual(guarded["metric_semantics"], ATTR_METRIC_SEMANTICS)
        self.assertEqual(guarded["metric_semantics"]["lane"], "human_gt")
        self.assertEqual(guarded["metric_semantics"]["det"], "teacher_pseudo_agreement")
        self.assertEqual(guarded["metric_semantics"]["tl_attr"], "signal_attr_teacher_pseudo_agreement")
        self.assertEqual(guarded["metrics"]["traffic_light"], {"combo_accuracy": 0.90})

        with self.assertRaisesRegex(ValueError, "source_mask.tl_attr=true"):
            guard_lane_val_odpseudo_eval_report(
                {"dataset_key": ATTR_DATASET_KEY, "split": "val", "metrics": {}},
                source_mask={"det": True, "tl_attr": False},
                variant="attr_v2",
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

    def test_lane_val_odpseudo_discovers_only_lane_records_from_mixed_aihub_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            _create_base_sample(base_root, "traffic_a", dataset_key="aihub_traffic_seoul")
            _create_base_sample(base_root, "lane_b", dataset_key="aihub_lane_seoul")
            _create_base_sample(base_root, "obstacle_a", dataset_key="aihub_obstacle_seoul")
            _create_base_sample(base_root, "lane_a", dataset_key="aihub_lane_seoul")

            base_records = discover_base_lane_val_records(base_root)

            self.assertEqual([record.sample_id for record in base_records], ["lane_a", "lane_b"])
            self.assertEqual({record.source_dataset_key for record in base_records}, {"aihub_lane_seoul"})

    def test_lane_val_odpseudo_sample_results_adapter_preserves_identity_order_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            lane_a = _create_base_sample(base_root, "lane_a")
            lane_b = _create_base_sample(base_root, "lane_b")
            base_records = discover_base_lane_val_records(base_root)
            predictions = {
                "mobility": [
                    _prediction(lane_a, teacher_name="mobility", class_name="vehicle", confidence=0.90, xyxy=[10, 20, 110, 70], box_index=0),
                    _prediction(lane_a, teacher_name="mobility", class_name="vehicle", confidence=0.80, xyxy=[12, 22, 108, 68], box_index=1),
                    _prediction(lane_a, teacher_name="mobility", class_name="vehicle", confidence=0.10, xyxy=[200, 20, 260, 80], box_index=2),
                ],
                "signal": [
                    _prediction(lane_a, teacher_name="signal", class_name="traffic_light", confidence=0.95, xyxy=[300, 40, 320, 90], box_index=0),
                    _prediction(lane_a, teacher_name="signal", class_name="sign", confidence=0.70, xyxy=[400, 45, 450, 95], box_index=1),
                ],
                "obstacle": [
                    _prediction(lane_b, teacher_name="obstacle", class_name="traffic_cone", confidence=0.85, xyxy=[50, 300, 80, 360], box_index=0),
                ],
            }

            sample_results = build_lane_val_odpseudo_sample_results_from_predictions(
                base_records=base_records,
                base_lane_root=base_root,
                predictions_by_teacher=predictions,
                class_policy=_class_policy(),
                expected_base_count=2,
                run_id="test_run",
                created_at="2026-01-01T00:00:00",
            )
            result_by_id = {row["sample_id"]: row for row in sample_results}
            lane_a_result = result_by_id["lane_a"]
            lane_b_result = result_by_id["lane_b"]

            self.assertEqual(lane_a_result["accepted_detection_count"], 3)
            self.assertEqual([row["id"] for row in lane_a_result["accepted_detections"]], [0, 1, 2])
            self.assertEqual(
                [row["class_name"] for row in lane_a_result["accepted_detections"]],
                ["vehicle", "traffic_light", "sign"],
            )
            self.assertEqual([row.split()[0] for row in lane_a_result["accepted_yolo_rows"]], ["0", "5", "6"])
            self.assertEqual(
                lane_a_result["candidate_count_by_teacher"],
                {"mobility": 3, "signal": 2, "obstacle": 0},
            )
            self.assertEqual(
                sorted(row["reason"] for row in lane_a_result["rejected_candidates"]),
                ["teacher_nms_suppressed", "teacher_score_below_threshold"],
            )
            self.assertEqual(lane_b_result["accepted_detection_count"], 1)
            self.assertEqual(lane_b_result["accepted_detections"][0]["class_name"], "traffic_cone")

            build_lane_val_odpseudo_eval_root(
                base_lane_root=base_root,
                output_root=output_root,
                sample_results=sample_results,
                teacher_checkpoints=_teacher_checkpoints(root),
                copy_images=True,
            )
            scene = json.loads((output_root / "labels_scene" / "val" / "lane_a.json").read_text(encoding="utf-8"))
            det_lines = (output_root / "labels_det" / "val" / "lane_a.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual([row["id"] for row in scene["detections"]], [0, 1, 2])
            self.assertEqual(len(det_lines), len(scene["detections"]))

    def test_lane_val_odpseudo_sample_results_adapter_rejects_foreign_prediction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            lane_a = _create_base_sample(base_root, "lane_a")
            foreign = _prediction(
                lane_a,
                teacher_name="mobility",
                class_name="vehicle",
                confidence=0.9,
                xyxy=[1, 2, 20, 30],
            )
            foreign["sample_id"] = "not_in_lane_val"

            with self.assertRaisesRegex(ValueError, "not in base lane val set"):
                build_lane_val_odpseudo_sample_results_from_predictions(
                    base_records=discover_base_lane_val_records(base_root),
                    base_lane_root=base_root,
                    predictions_by_teacher={"mobility": [foreign], "signal": [], "obstacle": []},
                    class_policy=_class_policy(),
                )

    def test_lane_val_odpseudo_teacher_sample_results_writer_is_atomic_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            lane_a = _create_base_sample(base_root, "lane_a")
            teachers = _teacher_configs(root)

            def _fake_infer(*, teacher, base_records, run_config, log_fn=None):
                if teacher.name != "signal":
                    return []
                return [
                    _prediction(
                        lane_a,
                        teacher_name="signal",
                        class_name="traffic_light",
                        confidence=0.95,
                        xyxy=[20, 30, 50, 80],
                    )
                ]

            with patch.object(lane_val_odpseudo, "_run_lane_val_teacher_inference", side_effect=_fake_infer):
                summary = run_lane_val_odpseudo_teacher_sample_results(
                    base_lane_root=base_root,
                    output_root=output_root,
                    teachers=teachers,
                    class_policy=_class_policy(),
                    run_config=RunConfig(output_root=root / "runs", device="cpu", batch_size=2),
                    expected_base_count=1,
                    run_id="test_run",
                    created_at="2026-01-01T00:00:00",
                )

            sample_results_path = Path(summary["sample_results_path"])
            manifest_path = Path(summary["sample_results_manifest_path"])
            rows = [json.loads(line) for line in sample_results_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(rows[0]["sample_id"], "lane_a")
            self.assertEqual(rows[0]["accepted_detection_count"], 1)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["sample_count"], 1)
            self.assertFalse(sample_results_path.with_name(sample_results_path.name + ".tmp").exists())

    def test_lane_val_odpseudo_expected_base_count_is_enforced_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            _create_base_sample(base_root, "lane_a")

            with self.assertRaisesRegex(ValueError, f"exactly {DEFAULT_EXPECTED_BASE_VAL_COUNT}"):
                lane_val_odpseudo.preflight_lane_val_odpseudo_eval_root(
                    base_lane_root=base_root,
                    output_root=root / "eval_root",
                    sample_results={"lane_a": completed_empty_sample_result("lane_a")},
                    teacher_checkpoints=_teacher_checkpoints(root),
                    expected_base_count=DEFAULT_EXPECTED_BASE_VAL_COUNT,
                )

    def test_lane_val_odpseudo_attr_v2_requires_signal_attr_checkpoint_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            _create_base_sample(base_root, "lane_a")

            with self.assertRaises(TeacherFailureError):
                lane_val_odpseudo.preflight_lane_val_odpseudo_eval_root(
                    base_lane_root=base_root,
                    output_root=output_root,
                    sample_results={"lane_a": completed_empty_sample_result("lane_a")},
                    teacher_checkpoints=_teacher_checkpoints(root),
                    signal_attr_checkpoint=root / "missing" / "best_signal_attr.pt",
                    variant="attr_v2",
                )
            self.assertFalse((output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).exists())

    def test_lane_val_odpseudo_missing_od_checkpoint_fails_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            _create_base_sample(base_root, "lane_a")
            checkpoints = _teacher_checkpoints(root)
            checkpoints["signal"] = root / "missing" / "best_signal.pt"

            with self.assertRaises(TeacherFailureError):
                lane_val_odpseudo.preflight_lane_val_odpseudo_eval_root(
                    base_lane_root=base_root,
                    output_root=output_root,
                    sample_results={"lane_a": completed_empty_sample_result("lane_a")},
                    teacher_checkpoints=checkpoints,
                )
            self.assertFalse((output_root / "meta" / FINAL_DATASET_MANIFEST_NAME).exists())

    def test_lane_val_odpseudo_attr_v2_sidecar_uses_final_traffic_light_row_order_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_root = root / "base_lane"
            output_root = root / "eval_root"
            _create_base_sample(base_root, "lane_a")
            signal_attr_checkpoint = root / "checkpoints" / "best_signal_attr.pt"
            _write_text(signal_attr_checkpoint, "fake signal attr checkpoint\n")
            sidecar = _FakeSignalAttrSidecar(signal_attr_checkpoint)
            result = completed_empty_sample_result("lane_a")
            result.update(
                {
                    "accepted_yolo_rows": [
                        "5 0.500000 0.500000 0.100000 0.100000",
                        "6 0.250000 0.250000 0.100000 0.100000",
                    ],
                    "accepted_detections": [
                        {"class_name": "traffic_light", "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}},
                        {"class_name": "sign", "bbox": {"x1": 40, "y1": 10, "x2": 60, "y2": 30}},
                    ],
                    "accepted_detection_count": 2,
                    "det_file_status": "nonempty",
                    "candidate_count_by_teacher": {"mobility": 0, "signal": 2, "obstacle": 0},
                }
            )

            build_lane_val_odpseudo_eval_root(
                base_lane_root=base_root,
                output_root=output_root,
                sample_results={"lane_a": result},
                teacher_checkpoints=_teacher_checkpoints(root),
                signal_attr_sidecar=sidecar,
                copy_images=True,
                variant="attr_v2",
            )

            scene = json.loads((output_root / "labels_scene" / "val" / "lane_a.json").read_text(encoding="utf-8"))
            manifest = _read_manifest(output_root)
            self.assertEqual(scene["source"]["dataset"], ATTR_DATASET_KEY)
            self.assertEqual([row["id"] for row in scene["detections"]], [0, 1])
            self.assertEqual([row["class_name"] for row in scene["detections"]], ["traffic_light", "sign"])
            self.assertEqual([row["detection_id"] for row in scene["traffic_lights"]], [0])
            self.assertEqual(scene["tasks"]["has_tl_attr"], 1)
            self.assertEqual(sidecar.applied_scenes[0]["class_names"], ["traffic_light", "sign"])
            self.assertEqual(manifest["dataset_key"], ATTR_DATASET_KEY)
            self.assertEqual(manifest["metric_semantics"], ATTR_METRIC_SEMANTICS)
            self.assertEqual(manifest["teacher_checkpoints"]["signal_attr"]["path"], str(signal_attr_checkpoint.resolve()))
            self.assertEqual(manifest["signal_attr_sidecar"]["traffic_light_count"], 1)

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
