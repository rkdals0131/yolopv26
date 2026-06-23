from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence, TypedDict

from common.boxes import iou as _box_iou
from common.boxes import nms_rows as _nms_rows
from common.io import now_iso as _now_iso
from common.io import read_json as _read_json
from common.io import timestamp_token as _timestamp_token
from common.io import write_json as _write_json
from common.io import write_jsonl as _write_jsonl
from common.pv26_schema import LANE_VAL_ODPSEUDO_ATTR_DATASET_KEY, LANE_VAL_ODPSEUDO_DATASET_KEY, OD_CLASS_TO_ID
from tools.od_bootstrap.build.sweep_types import ClassPolicy, RunConfig, TeacherConfig, TeacherPredictionRow
from tools.od_bootstrap.teacher.policy import row_passes_geometry_priors as _row_passes_geometry_priors

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


DATASET_KEY = LANE_VAL_ODPSEUDO_DATASET_KEY
ATTR_DATASET_KEY = LANE_VAL_ODPSEUDO_ATTR_DATASET_KEY
SOURCE_KIND = "lane_val_odpseudo"
FINAL_DATASET_MANIFEST_NAME = "final_dataset_manifest.json"
REJECTED_DETECTIONS_NAME = "rejected_detections.jsonl"
MANIFEST_VERSION = "pv26-eval-lane-val-odpseudo-v1"
ATTR_MANIFEST_VERSION = "pv26-eval-lane-val-odpseudo-attr-v2"
METRIC_SEMANTICS = "teacher_pseudo_agreement"
ATTR_METRIC_SEMANTICS = {
    "det": "teacher_pseudo_agreement",
    "tl_attr": "signal_attr_teacher_pseudo_agreement",
    "lane": "human_gt",
    "stop_line": "human_gt",
    "crosswalk": "human_gt",
    "summary": "mixed: lane/stop_line/crosswalk are human GT; OD and TL attr are teacher-pseudo agreement, not human GT OD/TL metrics",
}
AUDIT_POLICY = "audit_gated_v1"
TL_ATTR_METRIC_DISABLED_REASON = "source_mask.tl_attr=false"
TEACHER_NAMES = ("mobility", "signal", "obstacle")
SIGNAL_ATTR_TEACHER_NAME = "signal_attr"
ALLOWED_BASE_LANE_DATASET_KEYS = ("aihub_lane_seoul",)
DEFAULT_EXPECTED_BASE_VAL_COUNT = 27700
DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE = 16
REJECTED_CANDIDATE_REASONS = {
    "teacher_score_below_threshold",
    "teacher_nms_suppressed",
    "manual_audit_rejected",
    "class_policy_rejected",
    "invalid_bbox",
    "nonfinite_prediction",
    "unsupported_teacher_class",
    "teacher_failure",
}
REJECTED_CANDIDATE_KEEP_PRIORITY = {
    "teacher_failure": 0,
    "nonfinite_prediction": 1,
    "unsupported_teacher_class": 2,
    "class_policy_rejected": 3,
    "teacher_nms_suppressed": 4,
    "teacher_score_below_threshold": 5,
    "manual_audit_rejected": 6,
}


@dataclass(frozen=True)
class LaneValODPseudoVariant:
    name: str
    dataset_key: str
    manifest_version: str
    tl_attr_enabled: bool
    metric_semantics: str | dict[str, str]


VARIANT_V1 = LaneValODPseudoVariant(
    name="v1",
    dataset_key=DATASET_KEY,
    manifest_version=MANIFEST_VERSION,
    tl_attr_enabled=False,
    metric_semantics=METRIC_SEMANTICS,
)
VARIANT_ATTR_V2 = LaneValODPseudoVariant(
    name="attr_v2",
    dataset_key=ATTR_DATASET_KEY,
    manifest_version=ATTR_MANIFEST_VERSION,
    tl_attr_enabled=True,
    metric_semantics=ATTR_METRIC_SEMANTICS,
)
VARIANTS_BY_NAME = {
    VARIANT_V1.name: VARIANT_V1,
    VARIANT_ATTR_V2.name: VARIANT_ATTR_V2,
}


class LaneValODPseudoError(ValueError):
    """Raised when the eval-root preflight cannot produce a ready artifact."""


class TeacherFailureError(LaneValODPseudoError):
    """Raised for teacher-level failures that must not be materialized as empty det files."""


def resolve_lane_val_odpseudo_variant(variant: str | LaneValODPseudoVariant | None = None) -> LaneValODPseudoVariant:
    if variant is None:
        return VARIANT_V1
    if isinstance(variant, LaneValODPseudoVariant):
        return variant
    variant_name = str(variant).strip()
    try:
        return VARIANTS_BY_NAME[variant_name]
    except KeyError as exc:
        raise ValueError(f"unsupported lane-val OD pseudo variant: {variant_name}") from exc


@dataclass(frozen=True)
class LaneValBaseRecord:
    sample_id: str
    split: str
    source_dataset_key: str
    scene_path: Path
    image_path: Path


@dataclass(frozen=True)
class LaneValODPseudoPreflightResult:
    base_records: tuple[LaneValBaseRecord, ...]
    sample_results: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    rejected_candidates: tuple[dict[str, Any], ...]


class LaneValODPseudoBuildSummary(TypedDict):
    output_root: str
    manifest_path: str
    rejected_candidates_path: str
    sample_count: int
    nonfinite_candidate_count: int


class LaneValODPseudoSampleResultsSummary(TypedDict):
    output_root: str
    sample_results_path: str
    sample_results_manifest_path: str
    run_id: str
    sample_count: int
    accepted_detection_count: int
    rejected_candidate_count: int
    emitted_rejected_candidate_count: int
    prediction_count_by_teacher: dict[str, int]


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _coerce_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_scene(scene_path: Path) -> dict[str, Any]:
    payload = _read_json(scene_path)
    if not isinstance(payload, dict):
        raise TypeError(f"scene root must be an object: {scene_path}")
    return payload


def _scene_image_file_name(scene: Mapping[str, Any], *, scene_path: Path) -> str:
    image = scene.get("image")
    if not isinstance(image, Mapping):
        raise ValueError(f"scene image must be an object: {scene_path}")
    image_name = _coerce_str(image.get("file_name"), field_name=f"{scene_path}.image.file_name")
    image_path = Path(image_name)
    if image_path.is_absolute() or image_path.name != image_name:
        raise ValueError(f"{scene_path}.image.file_name must be a file name, not a path")
    return image_name


def _scene_source(scene: Mapping[str, Any], *, scene_path: Path) -> tuple[str, str]:
    source = scene.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"scene source must be an object: {scene_path}")
    dataset_key = _coerce_str(source.get("dataset"), field_name=f"{scene_path}.source.dataset")
    split = _coerce_str(source.get("split"), field_name=f"{scene_path}.source.split")
    return dataset_key, split


def discover_base_lane_val_records(
    base_lane_root: Path,
    *,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
) -> tuple[LaneValBaseRecord, ...]:
    """Discover the base lane validation set in deterministic `(split, sample_id)` order."""

    root = Path(base_lane_root).resolve()
    val_scene_root = root / "labels_scene" / "val"
    if not val_scene_root.is_dir():
        raise FileNotFoundError(f"base lane validation labels_scene/val not found: {val_scene_root}")

    allowed = {str(item) for item in allowed_dataset_keys}
    records: list[LaneValBaseRecord] = []
    skipped_source_counts: dict[str, int] = {}
    for scene_path in sorted(val_scene_root.glob("*.json"), key=lambda item: ("val", item.stem)):
        scene = _load_scene(scene_path)
        dataset_key, split = _scene_source(scene, scene_path=scene_path)
        if dataset_key not in allowed:
            skipped_source_counts[dataset_key] = skipped_source_counts.get(dataset_key, 0) + 1
            continue
        image_name = _scene_image_file_name(scene, scene_path=scene_path)
        records.append(
            LaneValBaseRecord(
                sample_id=scene_path.stem,
                split=split,
                source_dataset_key=dataset_key,
                scene_path=scene_path,
                image_path=root / "images" / split / image_name,
            )
        )
    if not records and skipped_source_counts:
        skipped = ", ".join(f"{key}={count}" for key, count in sorted(skipped_source_counts.items()))
        raise ValueError(f"unsupported base lane dataset for {DATASET_KEY}: no allowed records found; skipped {skipped}")
    return validate_base_records(records, base_lane_root=root, allowed_dataset_keys=allowed_dataset_keys)


def validate_base_records(
    records: Iterable[LaneValBaseRecord],
    *,
    base_lane_root: Path,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
) -> tuple[LaneValBaseRecord, ...]:
    root = Path(base_lane_root).resolve()
    allowed = {str(item) for item in allowed_dataset_keys}
    normalized: list[LaneValBaseRecord] = []
    seen_sample_ids: set[str] = set()

    for record in records:
        sample_id = _coerce_str(record.sample_id, field_name="base_record.sample_id")
        split = _coerce_str(record.split, field_name=f"{sample_id}.split")
        if split != "val":
            raise ValueError(f"base records for {DATASET_KEY} must be val-only: {sample_id} has split={split}")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate base lane validation sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        source_dataset_key = _coerce_str(
            record.source_dataset_key,
            field_name=f"{sample_id}.source_dataset_key",
        )
        if source_dataset_key not in allowed:
            raise ValueError(
                f"unsupported base lane dataset for {DATASET_KEY}: {source_dataset_key} ({sample_id})"
            )

        scene_path = Path(record.scene_path).resolve()
        image_path = Path(record.image_path).resolve()
        expected_scene_root = root / "labels_scene" / "val"
        expected_image_root = root / "images" / "val"
        if not scene_path.is_file():
            raise FileNotFoundError(f"base lane scene not found: {scene_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"base lane image not found: {image_path}")
        if not _is_relative_to(scene_path, expected_scene_root):
            raise ValueError(f"source_scene_path must stay under base lane val root: {scene_path}")
        if not _is_relative_to(image_path, expected_image_root):
            raise ValueError(f"source_image_path must stay under base lane val root: {image_path}")
        if scene_path.parent.name != "val" or scene_path.stem != sample_id:
            raise ValueError(f"base scene path must match val/sample_id: {scene_path}")

        scene = _load_scene(scene_path)
        scene_dataset_key, scene_split = _scene_source(scene, scene_path=scene_path)
        if scene_dataset_key != source_dataset_key:
            raise ValueError(
                f"base scene source.dataset mismatch for {sample_id}: "
                f"{scene_dataset_key} != {source_dataset_key}"
            )
        if scene_split != "val":
            raise ValueError(f"base scene source.split must be val-only for {sample_id}: {scene_split}")
        image_name = _scene_image_file_name(scene, scene_path=scene_path)
        expected_image_path = (root / "images" / "val" / image_name).resolve()
        if image_path != expected_image_path:
            raise ValueError(
                f"source_image_path must match base scene image.file_name for {sample_id}: "
                f"{image_path} != {expected_image_path}"
            )

        normalized.append(
            LaneValBaseRecord(
                sample_id=sample_id,
                split="val",
                source_dataset_key=source_dataset_key,
                scene_path=scene_path,
                image_path=image_path,
            )
        )

    if not normalized:
        raise ValueError(f"{DATASET_KEY} requires at least one base lane validation sample")
    return tuple(normalized)


def _normalize_teacher_checkpoints(teacher_checkpoints: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(teacher_checkpoints, Mapping):
        raise TypeError("teacher_checkpoints must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for teacher_name in TEACHER_NAMES:
        if teacher_name not in teacher_checkpoints:
            raise TeacherFailureError(f"teacher checkpoint missing for {teacher_name}")
        raw_payload = teacher_checkpoints[teacher_name]
        if isinstance(raw_payload, (str, Path)):
            checkpoint_path = Path(raw_payload).resolve()
            resolved = checkpoint_path.is_file()
            extra: dict[str, Any] = {}
        elif isinstance(raw_payload, Mapping):
            checkpoint_path = Path(
                _coerce_str(raw_payload.get("path"), field_name=f"teacher_checkpoints.{teacher_name}.path")
            ).resolve()
            resolved = bool(raw_payload.get("resolved", checkpoint_path.is_file()))
            extra = {
                str(key): value
                for key, value in raw_payload.items()
                if str(key) not in {"path", "resolved"}
            }
        else:
            raise TypeError(f"teacher_checkpoints.{teacher_name} must be a path or mapping")

        if not resolved or not checkpoint_path.is_file():
            raise TeacherFailureError(f"teacher checkpoint is not resolved for {teacher_name}: {checkpoint_path}")
        normalized[teacher_name] = {
            "path": str(checkpoint_path),
            "resolved": True,
            **extra,
        }

    unexpected = sorted(str(key) for key in teacher_checkpoints if str(key) not in TEACHER_NAMES)
    if unexpected:
        raise ValueError(f"unexpected teacher checkpoint keys for {DATASET_KEY}: {unexpected}")
    return normalized


def _normalize_signal_attr_checkpoint(signal_attr_checkpoint: Any) -> dict[str, Any]:
    if isinstance(signal_attr_checkpoint, (str, Path)):
        checkpoint_path = Path(signal_attr_checkpoint).resolve()
        resolved = checkpoint_path.is_file()
        extra: dict[str, Any] = {}
    elif isinstance(signal_attr_checkpoint, Mapping):
        checkpoint_path = Path(
            _coerce_str(signal_attr_checkpoint.get("path"), field_name="signal_attr_checkpoint.path")
        ).resolve()
        resolved = bool(signal_attr_checkpoint.get("resolved", checkpoint_path.is_file()))
        extra = {
            str(key): value
            for key, value in signal_attr_checkpoint.items()
            if str(key) not in {"path", "resolved"}
        }
    else:
        raise TypeError("signal_attr_checkpoint must be a path or mapping")
    if not resolved or not checkpoint_path.is_file():
        raise TeacherFailureError(f"signal_attr checkpoint is not resolved: {checkpoint_path}")
    return {
        "path": str(checkpoint_path),
        "resolved": True,
        **extra,
    }


def _finite_number(value: Any, *, field_name: str) -> tuple[float | None, bool]:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        return None, True
    return resolved, False


def rejected_candidate_semantics(row: Mapping[str, Any]) -> str:
    reason = _coerce_str(row.get("reason"), field_name="rejected_candidate.reason")
    if reason == "teacher_failure":
        return "fatal_teacher_failure"
    if reason == "nonfinite_prediction":
        return "candidate_nonfinite"
    return "candidate_rejected"


def normalize_rejected_candidate_row(
    row: Mapping[str, Any],
    *,
    known_sample_ids: Iterable[str] | None = None,
    allow_teacher_failure: bool = False,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("rejected candidate row must be an object")

    known = set(known_sample_ids or ())
    sample_id = _coerce_str(row.get("sample_id"), field_name="rejected_candidate.sample_id")
    if known and sample_id not in known:
        raise ValueError(f"rejected candidate sample_id is not in base validation set: {sample_id}")
    teacher_name = _coerce_str(row.get("teacher_name"), field_name=f"{sample_id}.teacher_name")
    if teacher_name not in TEACHER_NAMES:
        raise ValueError(f"unsupported rejected candidate teacher_name: {teacher_name}")
    class_name = _coerce_str(row.get("class_name"), field_name=f"{sample_id}.class_name")
    reason = _coerce_str(row.get("reason"), field_name=f"{sample_id}.reason")
    if reason not in REJECTED_CANDIDATE_REASONS:
        raise ValueError(f"unsupported rejected candidate reason: {reason}")
    if reason == "teacher_failure" and not allow_teacher_failure:
        raise TeacherFailureError(
            f"teacher_failure rejected candidate is fatal for a ready {DATASET_KEY} manifest: {sample_id}"
        )

    score, score_nonfinite = _finite_number(row.get("score"), field_name=f"{sample_id}.score")
    raw_bbox = row.get("bbox")
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        raise TypeError(f"{sample_id}.bbox must be a 4-item sequence")

    bbox: list[float | None] = []
    nonfinite_fields: list[str] = []
    if score_nonfinite:
        nonfinite_fields.append("score")
    for index, value in enumerate(raw_bbox):
        resolved, nonfinite = _finite_number(value, field_name=f"{sample_id}.bbox[{index}]")
        bbox.append(resolved)
        if nonfinite:
            nonfinite_fields.append(f"bbox[{index}]")

    if nonfinite_fields and reason != "nonfinite_prediction":
        raise ValueError(
            f"nonfinite rejected candidate values require reason=nonfinite_prediction: {sample_id}"
        )
    if reason == "nonfinite_prediction" and not nonfinite_fields:
        raise ValueError(f"reason=nonfinite_prediction requires a nonfinite score or bbox value: {sample_id}")

    normalized = {
        "sample_id": sample_id,
        "teacher_name": teacher_name,
        "class_name": class_name,
        "score": score,
        "bbox": bbox,
        "reason": reason,
    }
    if nonfinite_fields:
        normalized["nonfinite_fields"] = nonfinite_fields
    return normalized


def completed_empty_sample_result(sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": _coerce_str(sample_id, field_name="sample_id"),
        "teacher_run_status": "completed",
        "accepted_yolo_rows": [],
        "accepted_detections": [],
        "accepted_detection_count": 0,
        "det_file_status": "empty",
        "candidate_count_by_teacher": {teacher_name: 0 for teacher_name in TEACHER_NAMES},
        "failure_count": 0,
        "nonfinite_fatal_count": 0,
        "oom_count": 0,
        "rejected_candidates": [],
    }


def _teacher_order_index(teacher_name: str) -> int:
    try:
        return TEACHER_NAMES.index(teacher_name)
    except ValueError:
        return len(TEACHER_NAMES)


def _prediction_sample_id(row: Mapping[str, Any]) -> str:
    return _coerce_str(row.get("sample_id"), field_name="teacher_prediction.sample_id")


def _prediction_teacher_name(row: Mapping[str, Any]) -> str:
    teacher_name = _coerce_str(row.get("teacher_name"), field_name="teacher_prediction.teacher_name")
    if teacher_name not in TEACHER_NAMES:
        raise ValueError(f"unsupported lane-val teacher prediction teacher_name: {teacher_name}")
    return teacher_name


def _prediction_class_name(row: Mapping[str, Any]) -> str:
    return _coerce_str(row.get("class_name"), field_name="teacher_prediction.class_name")


def _prediction_score(row: Mapping[str, Any]) -> float:
    value = row.get("confidence", row.get("score"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("teacher_prediction.confidence must be numeric")
    return float(value)


def _prediction_bbox(row: Mapping[str, Any]) -> list[float]:
    raw_bbox = row.get("xyxy", row.get("bbox"))
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        raise TypeError("teacher_prediction.xyxy must be a 4-item sequence")
    return [float(value) for value in raw_bbox]


def _prediction_image_size(row: Mapping[str, Any], *, record: LaneValBaseRecord) -> tuple[int, int]:
    raw_width = row.get("image_width")
    raw_height = row.get("image_height")
    if raw_width is not None and raw_height is not None:
        return int(raw_width), int(raw_height)
    scene = _load_scene(record.scene_path)
    image = scene.get("image")
    if not isinstance(image, Mapping):
        raise ValueError(f"scene image must be an object: {record.scene_path}")
    return int(image.get("width")), int(image.get("height"))


def _assert_prediction_identity(row: Mapping[str, Any], *, record: LaneValBaseRecord) -> None:
    dataset_key = row.get("dataset_key")
    if dataset_key is not None and str(dataset_key) != record.source_dataset_key:
        raise ValueError(
            f"teacher prediction dataset_key mismatch for {record.sample_id}: "
            f"{dataset_key} != {record.source_dataset_key}"
        )
    split = row.get("split")
    if split is not None and str(split) != record.split:
        raise ValueError(f"teacher prediction split mismatch for {record.sample_id}: {split} != {record.split}")
    image_path = row.get("image_path")
    if image_path is not None and Path(str(image_path)).resolve() != record.image_path:
        raise ValueError(f"teacher prediction image_path mismatch for {record.sample_id}")
    scene_path = row.get("scene_path")
    if scene_path is not None and Path(str(scene_path)).resolve() != record.scene_path:
        raise ValueError(f"teacher prediction scene_path mismatch for {record.sample_id}")


def _prediction_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    class_name = str(row.get("class_name") or "")
    score = float(row.get("confidence", row.get("score", 0.0)))
    bbox = tuple(float(value) for value in row.get("xyxy", row.get("bbox", (0.0, 0.0, 0.0, 0.0))))
    return (
        _teacher_order_index(str(row.get("teacher_name") or "")),
        OD_CLASS_TO_ID.get(class_name, len(OD_CLASS_TO_ID)),
        -score,
        bbox,
    )


def _cross_class_iou_threshold(policy: ClassPolicy, other_policy: ClassPolicy | None = None) -> float:
    if policy.cross_class_iou_threshold is not None:
        return float(policy.cross_class_iou_threshold)
    if other_policy is not None and other_policy.cross_class_iou_threshold is not None:
        return float(other_policy.cross_class_iou_threshold)
    if other_policy is not None:
        return min(float(policy.nms_iou_threshold), float(other_policy.nms_iou_threshold))
    return float(policy.nms_iou_threshold)


def _reject_prediction(row: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return normalize_rejected_candidate_row(
        {
            "sample_id": _prediction_sample_id(row),
            "teacher_name": _prediction_teacher_name(row),
            "class_name": _prediction_class_name(row),
            "score": _prediction_score(row),
            "bbox": _prediction_bbox(row),
            "reason": reason,
        },
        allow_teacher_failure=False,
    )


def _prediction_policy_rejection_reason(
    row: Mapping[str, Any],
    *,
    record: LaneValBaseRecord,
    class_policy: Mapping[str, ClassPolicy],
) -> str | None:
    class_name = _prediction_class_name(row)
    score = _prediction_score(row)
    bbox = _prediction_bbox(row)
    if not math.isfinite(score) or not all(math.isfinite(value) for value in bbox):
        return "nonfinite_prediction"
    if class_name not in OD_CLASS_TO_ID or class_name not in class_policy:
        return "unsupported_teacher_class"
    policy = class_policy[class_name]
    if score < float(policy.score_threshold):
        return "teacher_score_below_threshold"
    image_width, image_height = _prediction_image_size(row, record=record)
    if policy.allowed_source_datasets and record.source_dataset_key not in policy.allowed_source_datasets:
        return "class_policy_rejected"
    if not _row_passes_geometry_priors(
        row={
            "class_name": class_name,
            "confidence": score,
            "xyxy": bbox,
        },
        policy=policy,
        image_width=image_width,
        image_height=image_height,
    ):
        return "class_policy_rejected"
    return None


def _bbox_to_mapping(box: Sequence[float]) -> dict[str, float]:
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
    }


def _bbox_to_yolo_row(class_name: str, box: Sequence[float], *, image_width: int, image_height: int) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive for YOLO row encoding")
    x1, y1, x2, y2 = [float(value) for value in box]
    center_x = ((x1 + x2) * 0.5) / float(image_width)
    center_y = ((y1 + y2) * 0.5) / float(image_height)
    box_width = max(0.0, x2 - x1) / float(image_width)
    box_height = max(0.0, y2 - y1) / float(image_height)
    return f"{OD_CLASS_TO_ID[class_name]} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def _accepted_detection_from_prediction(
    row: Mapping[str, Any],
    *,
    detection_id: int,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "id": detection_id,
        "class_name": _prediction_class_name(row),
        "bbox": _bbox_to_mapping(_prediction_bbox(row)),
        "score": _prediction_score(row),
        "meta": {
            "label_origin": "teacher_pseudo",
            "teacher_name": _prediction_teacher_name(row),
            "model_version": str(row.get("model_version") or ""),
            "bootstrap_run_id": run_id,
            "created_at": created_at,
        },
    }


def _apply_lane_val_teacher_policy(
    *,
    record: LaneValBaseRecord,
    rows: Sequence[Mapping[str, Any]],
    class_policy: Mapping[str, ClassPolicy],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    eligible_by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for row in rows:
        reason = _prediction_policy_rejection_reason(row, record=record, class_policy=class_policy)
        if reason is not None:
            rejected.append(_reject_prediction(row, reason=reason))
            continue
        eligible_by_class[_prediction_class_name(row)].append(row)

    after_class_nms: list[Mapping[str, Any]] = []
    for class_name, class_rows in sorted(eligible_by_class.items(), key=lambda item: OD_CLASS_TO_ID.get(item[0], 999)):
        kept = _nms_rows(class_rows, iou_threshold=float(class_policy[class_name].nms_iou_threshold))
        kept_ids = {id(row) for row in kept}
        after_class_nms.extend(kept)
        for row in class_rows:
            if id(row) not in kept_ids:
                rejected.append(_reject_prediction(row, reason="teacher_nms_suppressed"))

    accepted: list[Mapping[str, Any]] = []
    for row in sorted(after_class_nms, key=_prediction_score, reverse=True):
        class_name = _prediction_class_name(row)
        policy = class_policy[class_name]
        candidate_box = _prediction_bbox(row)
        suppressed = False
        for accepted_row in accepted:
            accepted_class = _prediction_class_name(accepted_row)
            accepted_policy = class_policy[accepted_class]
            if accepted_class not in policy.suppress_with_classes and class_name not in accepted_policy.suppress_with_classes:
                continue
            if _box_iou(candidate_box, _prediction_bbox(accepted_row)) >= _cross_class_iou_threshold(policy, accepted_policy):
                suppressed = True
                break
        if suppressed:
            rejected.append(_reject_prediction(row, reason="teacher_nms_suppressed"))
            continue
        accepted.append(row)

    return sorted(accepted, key=_prediction_sort_key), sorted(rejected, key=_rejected_candidate_sort_key)


def _rejected_candidate_reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = _coerce_str(row.get("reason"), field_name="rejected_candidate.reason")
        if reason not in REJECTED_CANDIDATE_REASONS:
            raise ValueError(f"unsupported rejected candidate reason: {reason}")
        counts[reason] += 1
    return dict(sorted(counts.items()))


def _limited_rejected_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    max_rows: int | None,
) -> tuple[dict[str, Any], ...]:
    if max_rows is None:
        return tuple(rows)
    limit = int(max_rows)
    if limit < 0:
        raise ValueError("max_rejected_candidates_per_sample must be >= 0")
    if limit == 0:
        return ()
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                REJECTED_CANDIDATE_KEEP_PRIORITY.get(str(row.get("reason")), 999),
                _rejected_candidate_sort_key(row),
            ),
        )[:limit]
    )


def build_lane_val_odpseudo_sample_results_from_predictions(
    *,
    base_records: Iterable[LaneValBaseRecord],
    base_lane_root: Path,
    predictions_by_teacher: Mapping[str, Sequence[Mapping[str, Any]]],
    class_policy: Mapping[str, ClassPolicy],
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
    expected_base_count: int | None = None,
    run_id: str = "lane_val_odpseudo_teacher_sweep",
    created_at: str | None = None,
    max_rejected_candidates_per_sample: int | None = DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE,
) -> tuple[dict[str, Any], ...]:
    normalized_records = validate_base_records(
        base_records,
        base_lane_root=base_lane_root,
        allowed_dataset_keys=allowed_dataset_keys,
    )
    normalized_records = tuple(sorted(normalized_records, key=lambda item: (item.split, item.sample_id)))
    if expected_base_count is not None and len(normalized_records) != int(expected_base_count):
        raise ValueError(f"{DATASET_KEY} base lane val count must be exactly {int(expected_base_count)}: {len(normalized_records)}")

    missing_teachers = [teacher_name for teacher_name in TEACHER_NAMES if teacher_name not in predictions_by_teacher]
    if missing_teachers:
        raise TeacherFailureError(f"missing lane-val teacher predictions for: {missing_teachers}")
    unexpected_teachers = sorted(str(key) for key in predictions_by_teacher if str(key) not in TEACHER_NAMES)
    if unexpected_teachers:
        raise ValueError(f"unexpected lane-val teacher prediction keys: {unexpected_teachers}")

    record_by_sample_id = {record.sample_id: record for record in normalized_records}
    rows_by_sample_id: dict[str, list[Mapping[str, Any]]] = {record.sample_id: [] for record in normalized_records}
    candidate_count_by_sample_teacher: dict[str, Counter[str]] = {
        record.sample_id: Counter({teacher_name: 0 for teacher_name in TEACHER_NAMES})
        for record in normalized_records
    }

    for teacher_name in TEACHER_NAMES:
        for raw_row in predictions_by_teacher[teacher_name]:
            if not isinstance(raw_row, Mapping):
                raise TypeError(f"lane-val teacher prediction for {teacher_name} must be an object")
            row = deepcopy(dict(raw_row))
            row_teacher_name = _prediction_teacher_name(row)
            if row_teacher_name != teacher_name:
                raise ValueError(f"teacher prediction key/name mismatch: {teacher_name} != {row_teacher_name}")
            sample_id = _prediction_sample_id(row)
            try:
                record = record_by_sample_id[sample_id]
            except KeyError as exc:
                raise ValueError(f"teacher prediction sample_id is not in base lane val set: {sample_id}") from exc
            _assert_prediction_identity(row, record=record)
            candidate_count_by_sample_teacher[sample_id][teacher_name] += 1
            rows_by_sample_id[sample_id].append(row)

    resolved_created_at = created_at or _now_iso()
    sample_results: list[dict[str, Any]] = []
    for record in normalized_records:
        accepted, rejected = _apply_lane_val_teacher_policy(
            record=record,
            rows=rows_by_sample_id[record.sample_id],
            class_policy=class_policy,
        )
        rejected_reason_counts = _rejected_candidate_reason_counts(rejected)
        emitted_rejected = _limited_rejected_candidates(
            rejected,
            max_rows=max_rejected_candidates_per_sample,
        )
        image_width, image_height = _prediction_image_size({}, record=record)
        accepted_detections = [
            _accepted_detection_from_prediction(
                row,
                detection_id=row_index,
                run_id=run_id,
                created_at=resolved_created_at,
            )
            for row_index, row in enumerate(accepted)
        ]
        accepted_yolo_rows = [
            _bbox_to_yolo_row(
                _prediction_class_name(row),
                _prediction_bbox(row),
                image_width=image_width,
                image_height=image_height,
            )
            for row in accepted
        ]
        sample_results.append(
            {
                "sample_id": record.sample_id,
                "teacher_run_status": "completed",
                "accepted_yolo_rows": accepted_yolo_rows,
                "accepted_detections": accepted_detections,
                "accepted_detection_count": len(accepted_detections),
                "det_file_status": "nonempty" if accepted_detections else "empty",
                "candidate_count_by_teacher": {
                    teacher_name: int(candidate_count_by_sample_teacher[record.sample_id][teacher_name])
                    for teacher_name in TEACHER_NAMES
                },
                "failure_count": 0,
                "nonfinite_fatal_count": 0,
                "oom_count": 0,
                "rejected_candidate_count": sum(rejected_reason_counts.values()),
                "emitted_rejected_candidate_count": len(emitted_rejected),
                "rejected_candidate_count_by_reason": rejected_reason_counts,
                "rejected_candidates": emitted_rejected,
            }
        )

    return _normalize_sample_results(sample_results, sample_ids=tuple(record.sample_id for record in normalized_records))


def _batched_records(records: Sequence[LaneValBaseRecord], batch_size: int) -> Iterable[tuple[LaneValBaseRecord, ...]]:
    batch: list[LaneValBaseRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= max(1, int(batch_size)):
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _result_image_size(result: Any, *, fallback_record: LaneValBaseRecord) -> tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
        return int(orig_shape[1]), int(orig_shape[0])
    return _prediction_image_size({}, record=fallback_record)


def _extract_lane_val_teacher_result_rows(
    *,
    teacher: TeacherConfig,
    records: Sequence[LaneValBaseRecord],
    results: Sequence[Any],
) -> list[TeacherPredictionRow]:
    if len(results) != len(records):
        raise TeacherFailureError(
            f"teacher={teacher.name} returned {len(results)} results for {len(records)} lane-val images"
        )
    rows: list[TeacherPredictionRow] = []
    for record, result in zip(records, results, strict=True):
        image_width, image_height = _result_image_size(result, fallback_record=record)
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required to adapt Ultralytics prediction rows") from exc
        xyxy_rows = torch.as_tensor(getattr(boxes, "xyxy", None)).tolist()
        cls_rows = torch.as_tensor(getattr(boxes, "cls", None)).tolist()
        conf_rows = torch.as_tensor(getattr(boxes, "conf", None)).tolist()
        for box_index, (box, cls_index, confidence) in enumerate(zip(xyxy_rows, cls_rows, conf_rows)):
            class_name = str(names.get(int(cls_index), str(int(cls_index))))
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "sample_uid": record.sample_id,
                    "image_path": str(record.image_path),
                    "scene_path": str(record.scene_path),
                    "dataset_key": record.source_dataset_key,
                    "split": record.split,
                    "teacher_name": teacher.name,
                    "model_version": teacher.model_version,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in box],
                    "box_index": box_index,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )
    return rows


def _run_lane_val_teacher_inference(
    *,
    teacher: TeacherConfig,
    base_records: Sequence[LaneValBaseRecord],
    run_config: RunConfig,
    log_fn: Any | None = None,
) -> list[TeacherPredictionRow]:
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed")
    if not teacher.checkpoint_path.is_file():
        raise TeacherFailureError(f"teacher checkpoint not found: {teacher.checkpoint_path}")
    model = YOLO(str(teacher.checkpoint_path))
    rows: list[TeacherPredictionRow] = []
    batches = tuple(_batched_records(base_records, int(run_config.batch_size)))
    total = len(base_records)
    processed = 0
    if log_fn is not None:
        log_fn(f"[lane-val:{teacher.name}] inference start images={total} batch={run_config.batch_size}")
    for batch in batches:
        results = model.predict(
            source=[str(record.image_path) for record in batch],
            imgsz=run_config.imgsz,
            device=run_config.device,
            conf=run_config.predict_conf,
            iou=run_config.predict_iou,
            verbose=False,
            save=False,
            stream=False,
        )
        rows.extend(
            _extract_lane_val_teacher_result_rows(
                teacher=teacher,
                records=batch,
                results=list(results),
            )
        )
        processed += len(batch)
        if log_fn is not None and (processed == total or processed == len(batch) or processed % max(320, int(run_config.batch_size) * 10) == 0):
            log_fn(f"[lane-val:{teacher.name}] inference progress {processed}/{total} images predictions={len(rows)}")
    if log_fn is not None:
        log_fn(f"[lane-val:{teacher.name}] inference done predictions={len(rows)}")
    return rows


def _atomic_write_json(path: Path, payload: Any, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"target path already exists: {output_path}")
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    _write_json(tmp_path, payload, sort_keys=True)
    tmp_path.replace(output_path)
    return output_path


def _atomic_write_jsonl(path: Path, rows: Iterable[Any], *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"target path already exists: {output_path}")
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    _write_jsonl(tmp_path, rows, sort_keys=True)
    tmp_path.replace(output_path)
    return output_path


def run_lane_val_odpseudo_teacher_sample_results(
    *,
    base_lane_root: Path,
    output_root: Path,
    teachers: Sequence[TeacherConfig],
    class_policy: Mapping[str, ClassPolicy],
    run_config: RunConfig,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
    expected_base_count: int | None = None,
    sample_results_path: Path | None = None,
    max_rejected_candidates_per_sample: int | None = DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE,
    run_id: str | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
    log_fn: Any | None = None,
) -> LaneValODPseudoSampleResultsSummary:
    resolved_output_root = Path(output_root).resolve()
    resolved_sample_results_path = (
        Path(sample_results_path).resolve()
        if sample_results_path is not None
        else resolved_output_root / "meta" / "sample_results.jsonl"
    )
    resolved_manifest_path = resolved_sample_results_path.with_name("sample_results_manifest.json")
    resolved_run_id = run_id or f"{SOURCE_KIND}_sample_results_{_timestamp_token()}"
    resolved_created_at = created_at or _now_iso()
    base_records = discover_base_lane_val_records(base_lane_root, allowed_dataset_keys=allowed_dataset_keys)
    if expected_base_count is not None and len(base_records) != int(expected_base_count):
        raise ValueError(f"{DATASET_KEY} base lane val count must be exactly {int(expected_base_count)}: {len(base_records)}")

    teacher_by_name = {teacher.name: teacher for teacher in teachers}
    missing = [name for name in TEACHER_NAMES if name not in teacher_by_name]
    unexpected = sorted(name for name in teacher_by_name if name not in TEACHER_NAMES)
    if missing or unexpected:
        raise ValueError(f"lane-val teachers must be {TEACHER_NAMES}: missing={missing} unexpected={unexpected}")
    for teacher_name in TEACHER_NAMES:
        checkpoint_path = Path(teacher_by_name[teacher_name].checkpoint_path)
        if not checkpoint_path.is_file():
            raise TeacherFailureError(f"teacher checkpoint not found for {teacher_name}: {checkpoint_path}")

    predictions_by_teacher: dict[str, list[Mapping[str, Any]]] = {}
    for teacher_name in TEACHER_NAMES:
        predictions_by_teacher[teacher_name] = _run_lane_val_teacher_inference(
            teacher=teacher_by_name[teacher_name],
            base_records=base_records,
            run_config=run_config,
            log_fn=log_fn,
        )

    sample_results = build_lane_val_odpseudo_sample_results_from_predictions(
        base_records=base_records,
        base_lane_root=base_lane_root,
        predictions_by_teacher=predictions_by_teacher,
        class_policy=class_policy,
        allowed_dataset_keys=allowed_dataset_keys,
        expected_base_count=expected_base_count,
        run_id=resolved_run_id,
        created_at=resolved_created_at,
        max_rejected_candidates_per_sample=max_rejected_candidates_per_sample,
    )
    accepted_detection_count = sum(int(row["accepted_detection_count"]) for row in sample_results)
    rejected_candidate_count = sum(int(row.get("rejected_candidate_count", len(row["rejected_candidates"]))) for row in sample_results)
    emitted_rejected_candidate_count = sum(len(row["rejected_candidates"]) for row in sample_results)
    rejected_candidate_count_by_reason: Counter[str] = Counter()
    for row in sample_results:
        for reason, count in dict(row.get("rejected_candidate_count_by_reason", {})).items():
            rejected_candidate_count_by_reason[str(reason)] += int(count)
    prediction_count_by_teacher = {
        teacher_name: len(predictions_by_teacher[teacher_name])
        for teacher_name in TEACHER_NAMES
    }
    manifest = {
        "version": "pv26-eval-lane-val-odpseudo-sample-results-v1",
        "status": "ready",
        "run_id": resolved_run_id,
        "created_at": resolved_created_at,
        "source_kind": SOURCE_KIND,
        "base_lane_root": str(Path(base_lane_root).resolve()),
        "output_root": str(resolved_output_root),
        "sample_results_path": str(resolved_sample_results_path),
        "sample_count": len(sample_results),
        "accepted_detection_count": accepted_detection_count,
        "rejected_candidate_count": rejected_candidate_count,
        "emitted_rejected_candidate_count": emitted_rejected_candidate_count,
        "rejected_candidate_count_by_reason": dict(sorted(rejected_candidate_count_by_reason.items())),
        "max_rejected_candidates_per_sample": max_rejected_candidates_per_sample,
        "prediction_count_by_teacher": prediction_count_by_teacher,
        "teacher_checkpoints": {
            teacher_name: str(Path(teacher_by_name[teacher_name].checkpoint_path).resolve())
            for teacher_name in TEACHER_NAMES
        },
        "class_policy": {
            class_name: {
                "score_threshold": float(policy.score_threshold),
                "nms_iou_threshold": float(policy.nms_iou_threshold),
                "min_box_size": int(policy.min_box_size),
            }
            for class_name, policy in sorted(class_policy.items())
        },
    }
    resolved_sample_results_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(resolved_sample_results_path, sample_results, overwrite=overwrite)
    _atomic_write_json(resolved_manifest_path, manifest, overwrite=overwrite)
    return {
        "output_root": str(resolved_output_root),
        "sample_results_path": str(resolved_sample_results_path),
        "sample_results_manifest_path": str(resolved_manifest_path),
        "run_id": resolved_run_id,
        "sample_count": len(sample_results),
        "accepted_detection_count": accepted_detection_count,
        "rejected_candidate_count": rejected_candidate_count,
        "emitted_rejected_candidate_count": emitted_rejected_candidate_count,
        "prediction_count_by_teacher": prediction_count_by_teacher,
    }


def _normalized_eval_report_source_mask(
    report: Mapping[str, Any],
    *,
    source_mask: Mapping[str, Any] | None,
    variant: LaneValODPseudoVariant,
) -> dict[str, bool]:
    raw_mask = source_mask if source_mask is not None else report.get("source_mask", {"tl_attr": variant.tl_attr_enabled})
    if not isinstance(raw_mask, Mapping):
        raise TypeError("source_mask must be a mapping")
    if "tl_attr" not in raw_mask:
        raise ValueError("source_mask.tl_attr is required")
    if bool(raw_mask["tl_attr"]) != variant.tl_attr_enabled:
        expected = str(variant.tl_attr_enabled).lower()
        raise ValueError(f"{variant.dataset_key} eval report requires source_mask.tl_attr={expected}")
    return {str(key): bool(value) for key, value in raw_mask.items()}


def guard_lane_val_odpseudo_eval_report(
    report: Mapping[str, Any],
    *,
    source_mask: Mapping[str, Any] | None = None,
    variant: str | LaneValODPseudoVariant | None = None,
) -> dict[str, Any]:
    """Mark lane-val OD pseudo eval metrics with explicit v1/attr-v2 semantics."""

    if not isinstance(report, Mapping):
        raise TypeError("eval report must be a mapping")

    resolved_variant = resolve_lane_val_odpseudo_variant(variant)
    guarded = deepcopy(dict(report))
    dataset_key = guarded.get("dataset_key", resolved_variant.dataset_key)
    if dataset_key != resolved_variant.dataset_key:
        raise ValueError(f"eval report dataset_key must be {resolved_variant.dataset_key}")
    split = guarded.get("split", "val")
    if split != "val":
        raise ValueError(f"{resolved_variant.dataset_key} eval report must be val-only")

    metrics = guarded.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("eval report metrics must be a mapping")

    guarded["dataset_key"] = resolved_variant.dataset_key
    guarded["split"] = "val"
    guarded["source_mask"] = _normalized_eval_report_source_mask(
        guarded,
        source_mask=source_mask,
        variant=resolved_variant,
    )
    guarded["metric_semantics"] = deepcopy(resolved_variant.metric_semantics)
    guarded["metrics"] = deepcopy(dict(metrics))
    if not resolved_variant.tl_attr_enabled:
        guarded["metrics"]["traffic_light"] = {
            "disabled": True,
            "reason": TL_ATTR_METRIC_DISABLED_REASON,
        }
    return guarded


def _normalize_candidate_counts(value: Any, *, sample_id: str) -> dict[str, int]:
    payload = value or {teacher_name: 0 for teacher_name in TEACHER_NAMES}
    if not isinstance(payload, Mapping):
        raise TypeError(f"{sample_id}.candidate_count_by_teacher must be a mapping")
    counts: dict[str, int] = {}
    for teacher_name in TEACHER_NAMES:
        if teacher_name not in payload:
            raise ValueError(f"{sample_id}.candidate_count_by_teacher missing {teacher_name}")
        counts[teacher_name] = _coerce_nonnegative_int(
            payload[teacher_name],
            field_name=f"{sample_id}.candidate_count_by_teacher.{teacher_name}",
        )
    unexpected = sorted(str(key) for key in payload if str(key) not in TEACHER_NAMES)
    if unexpected:
        raise ValueError(f"{sample_id}.candidate_count_by_teacher has unexpected keys: {unexpected}")
    return counts


def _normalize_rejected_reason_counts(value: Any, *, sample_id: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{sample_id}.rejected_candidate_count_by_reason must be a mapping")
    counts: dict[str, int] = {}
    for raw_reason, raw_count in value.items():
        reason = _coerce_str(str(raw_reason), field_name=f"{sample_id}.rejected_candidate_count_by_reason.reason")
        if reason not in REJECTED_CANDIDATE_REASONS:
            raise ValueError(f"unsupported rejected candidate reason for {sample_id}: {reason}")
        counts[reason] = _coerce_nonnegative_int(
            raw_count,
            field_name=f"{sample_id}.rejected_candidate_count_by_reason.{reason}",
        )
    return dict(sorted(counts.items()))


def _normalize_yolo_rows(rows: Any, *, sample_id: str) -> tuple[str, ...]:
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"{sample_id}.accepted_yolo_rows must be a sequence")

    normalized_rows: list[str] = []
    for row_index, raw_row in enumerate(rows, start=1):
        row = _coerce_str(raw_row, field_name=f"{sample_id}.accepted_yolo_rows[{row_index}]")
        parts = row.split()
        if len(parts) != 5:
            raise ValueError(f"{sample_id}.accepted_yolo_rows[{row_index}] must be a YOLO 5-column row")
        try:
            int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{sample_id}.accepted_yolo_rows[{row_index}] must contain numeric YOLO values"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{sample_id}.accepted_yolo_rows[{row_index}] contains nonfinite values")
        normalized_rows.append(" ".join(parts))
    return tuple(normalized_rows)


def _normalize_accepted_detections(rows: Any, *, sample_id: str) -> tuple[dict[str, Any], ...]:
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"{sample_id}.accepted_detections must be a sequence")

    normalized: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"{sample_id}.accepted_detections[{row_index}] must be an object")
        detection = deepcopy(dict(raw_row))
        raw_id = detection.get("id")
        if raw_id is not None:
            if isinstance(raw_id, bool):
                raise ValueError(f"{sample_id}.accepted_detections[{row_index}].id must match row order")
            try:
                detection_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{sample_id}.accepted_detections[{row_index}].id must match row order") from exc
            if detection_id != row_index:
                raise ValueError(
                    f"{sample_id}.accepted_detections[{row_index}].id must match row order: "
                    f"{detection_id} != {row_index}"
                )
        detection["id"] = row_index
        class_name = _coerce_str(
            detection.get("class_name"),
            field_name=f"{sample_id}.accepted_detections[{row_index}].class_name",
        )
        if class_name not in OD_CLASS_TO_ID:
            raise ValueError(f"{sample_id}.accepted_detections[{row_index}].class_name is not a PV26 OD class")
        normalized.append(detection)
    return tuple(normalized)


def _normalize_sample_results(
    sample_results: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    sample_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    known_sample_ids = set(sample_ids)
    if isinstance(sample_results, Mapping):
        result_by_id: dict[str, Mapping[str, Any]] = {}
        for raw_sample_id, raw_payload in sample_results.items():
            sample_id = _coerce_str(str(raw_sample_id), field_name="sample_results key")
            if not isinstance(raw_payload, Mapping):
                raise TypeError(f"sample_results[{sample_id}] must be an object")
            result_by_id[sample_id] = {"sample_id": sample_id, **dict(raw_payload)}
    else:
        result_by_id = {}
        for raw_payload in sample_results:
            if not isinstance(raw_payload, Mapping):
                raise TypeError("sample result must be an object")
            sample_id = _coerce_str(raw_payload.get("sample_id"), field_name="sample_result.sample_id")
            if sample_id in result_by_id:
                raise ValueError(f"duplicate sample result for {sample_id}")
            result_by_id[sample_id] = raw_payload

    expected = set(sample_ids)
    actual = set(result_by_id)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"sample_results must exactly match base validation sample ids: missing={missing} extra={extra}")

    normalized_results: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        payload = result_by_id[sample_id]
        teacher_run_status = _coerce_str(
            payload.get("teacher_run_status"),
            field_name=f"{sample_id}.teacher_run_status",
        )
        accepted_yolo_rows = _normalize_yolo_rows(payload.get("accepted_yolo_rows", ()), sample_id=sample_id)
        accepted_detections = _normalize_accepted_detections(
            payload.get("accepted_detections", ()),
            sample_id=sample_id,
        )
        accepted_detection_count = _coerce_nonnegative_int(
            payload.get("accepted_detection_count", len(accepted_yolo_rows)),
            field_name=f"{sample_id}.accepted_detection_count",
        )
        if accepted_detection_count != len(accepted_yolo_rows):
            raise ValueError(
                f"{sample_id}.accepted_detection_count must match accepted_yolo_rows count: "
                f"{accepted_detection_count} != {len(accepted_yolo_rows)}"
            )
        if len(accepted_detections) != accepted_detection_count:
            raise ValueError(
                f"{sample_id}.accepted_detections must match accepted_detection_count: "
                f"{len(accepted_detections)} != {accepted_detection_count}"
            )

        det_file_status = _coerce_str(
            payload.get("det_file_status", "empty" if accepted_detection_count == 0 else "nonempty"),
            field_name=f"{sample_id}.det_file_status",
        )
        if det_file_status not in {"empty", "nonempty", "missing", "failed"}:
            raise ValueError(f"unsupported det_file_status for {sample_id}: {det_file_status}")
        if det_file_status in {"missing", "failed"}:
            raise TeacherFailureError(
                f"{sample_id} has det_file_status={det_file_status}; missing or failed det generation "
                f"cannot be materialized as a ready {DATASET_KEY} sample"
            )
        if accepted_detection_count == 0 and det_file_status != "empty":
            raise ValueError(f"{sample_id}.det_file_status must be empty when accepted_detection_count=0")
        if accepted_detection_count > 0 and det_file_status != "nonempty":
            raise ValueError(f"{sample_id}.det_file_status must be nonempty when detections are accepted")

        failure_count = _coerce_nonnegative_int(
            payload.get("failure_count", 0),
            field_name=f"{sample_id}.failure_count",
        )
        nonfinite_fatal_count = _coerce_nonnegative_int(
            payload.get("nonfinite_fatal_count", 0),
            field_name=f"{sample_id}.nonfinite_fatal_count",
        )
        oom_count = _coerce_nonnegative_int(payload.get("oom_count", 0), field_name=f"{sample_id}.oom_count")
        if (
            teacher_run_status != "completed"
            or failure_count
            or nonfinite_fatal_count
            or oom_count
        ):
            raise TeacherFailureError(
                f"{sample_id} has teacher-level failure and cannot be written as a ready empty det file: "
                f"teacher_run_status={teacher_run_status} det_file_status={det_file_status} "
                f"failure_count={failure_count} nonfinite_fatal_count={nonfinite_fatal_count} oom_count={oom_count}"
            )

        rejected_candidates = tuple(
            normalize_rejected_candidate_row(
                row,
                known_sample_ids=known_sample_ids,
                allow_teacher_failure=False,
            )
            for row in payload.get("rejected_candidates", ())
        )
        emitted_reason_counts = _rejected_candidate_reason_counts(rejected_candidates)
        rejected_reason_counts = _normalize_rejected_reason_counts(
            payload.get("rejected_candidate_count_by_reason"),
            sample_id=sample_id,
        )
        if not rejected_reason_counts:
            rejected_reason_counts = emitted_reason_counts
        for reason, emitted_count in emitted_reason_counts.items():
            if rejected_reason_counts.get(reason, 0) < emitted_count:
                raise ValueError(
                    f"{sample_id}.rejected_candidate_count_by_reason must cover emitted rejected rows for {reason}"
                )
        rejected_candidate_count = _coerce_nonnegative_int(
            payload.get("rejected_candidate_count", sum(rejected_reason_counts.values())),
            field_name=f"{sample_id}.rejected_candidate_count",
        )
        if rejected_candidate_count != sum(rejected_reason_counts.values()):
            raise ValueError(
                f"{sample_id}.rejected_candidate_count must match rejected_candidate_count_by_reason total"
            )
        emitted_rejected_candidate_count = _coerce_nonnegative_int(
            payload.get("emitted_rejected_candidate_count", len(rejected_candidates)),
            field_name=f"{sample_id}.emitted_rejected_candidate_count",
        )
        if emitted_rejected_candidate_count != len(rejected_candidates):
            raise ValueError(
                f"{sample_id}.emitted_rejected_candidate_count must match rejected_candidates length"
            )

        normalized_results.append(
            {
                "sample_id": sample_id,
                "teacher_run_status": teacher_run_status,
                "accepted_yolo_rows": accepted_yolo_rows,
                "accepted_detections": accepted_detections,
                "accepted_detection_count": accepted_detection_count,
                "det_file_status": det_file_status,
                "candidate_count_by_teacher": _normalize_candidate_counts(
                    payload.get("candidate_count_by_teacher"),
                    sample_id=sample_id,
                ),
                "failure_count": failure_count,
                "nonfinite_fatal_count": nonfinite_fatal_count,
                "oom_count": oom_count,
                "rejected_candidate_count": rejected_candidate_count,
                "emitted_rejected_candidate_count": emitted_rejected_candidate_count,
                "rejected_candidate_count_by_reason": rejected_reason_counts,
                "rejected_candidates": rejected_candidates,
            }
        )
    return tuple(normalized_results)


def _rejected_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    bbox = tuple("null" if value is None else f"{float(value):.12g}" for value in row["bbox"])
    score = "null" if row["score"] is None else f"{float(row['score']):.12g}"
    return (
        str(row["sample_id"]),
        str(row["teacher_name"]),
        str(row["class_name"]),
        str(row["reason"]),
        score,
        bbox,
    )


def _manifest_sample_rows(
    *,
    base_records: tuple[LaneValBaseRecord, ...],
    sample_results: tuple[dict[str, Any], ...],
    output_root: Path,
    variant: LaneValODPseudoVariant,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, result in zip(base_records, sample_results, strict=True):
        suffix = record.image_path.suffix.lower()
        rows.append(
            {
                "final_sample_id": record.sample_id,
                "source_dataset_key": variant.dataset_key,
                "split": "val",
                "source_kind": SOURCE_KIND,
                "variant": variant.name,
                "scene_path": str((output_root / "labels_scene" / "val" / f"{record.sample_id}.json").resolve()),
                "image_path": str((output_root / "images" / "val" / f"{record.sample_id}{suffix}").resolve()),
                "det_path": str((output_root / "labels_det" / "val" / f"{record.sample_id}.txt").resolve()),
                "source_scene_path": str(record.scene_path),
                "source_image_path": str(record.image_path),
                "source_det_path": None,
                "teacher_run_status": result["teacher_run_status"],
                "accepted_detection_count": result["accepted_detection_count"],
                "det_file_status": result["det_file_status"],
                "candidate_count_by_teacher": dict(result["candidate_count_by_teacher"]),
                "rejected_candidate_count": result.get("rejected_candidate_count", len(result["rejected_candidates"])),
                "emitted_rejected_candidate_count": result.get(
                    "emitted_rejected_candidate_count",
                    len(result["rejected_candidates"]),
                ),
                "rejected_candidate_count_by_reason": dict(result.get("rejected_candidate_count_by_reason", {})),
                "failure_count": result["failure_count"],
            }
        )
    rows.sort(key=lambda item: (str(item["split"]), str(item["final_sample_id"])))
    return rows


def _nonfinite_candidate_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("reason") == "nonfinite_prediction")


def preflight_lane_val_odpseudo_records(
    *,
    base_records: Iterable[LaneValBaseRecord],
    base_lane_root: Path,
    output_root: Path,
    sample_results: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    teacher_checkpoints: Mapping[str, Any],
    signal_attr_checkpoint: Any | None = None,
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
    variant: str | LaneValODPseudoVariant | None = None,
    expected_base_count: int | None = None,
) -> LaneValODPseudoPreflightResult:
    resolved_variant = resolve_lane_val_odpseudo_variant(variant)
    normalized_records = validate_base_records(
        base_records,
        base_lane_root=base_lane_root,
        allowed_dataset_keys=allowed_dataset_keys,
    )
    normalized_records = tuple(sorted(normalized_records, key=lambda item: (item.split, item.sample_id)))
    if expected_base_count is not None and len(normalized_records) != int(expected_base_count):
        raise ValueError(
            f"{resolved_variant.dataset_key} base lane val count must be exactly "
            f"{int(expected_base_count)}: {len(normalized_records)}"
        )
    sample_ids = tuple(record.sample_id for record in normalized_records)
    normalized_teacher_checkpoints = _normalize_teacher_checkpoints(teacher_checkpoints)
    if resolved_variant.tl_attr_enabled:
        if signal_attr_checkpoint is None:
            raise TeacherFailureError("signal_attr checkpoint is required for attr_v2 lane-val OD pseudo build")
        normalized_teacher_checkpoints[SIGNAL_ATTR_TEACHER_NAME] = _normalize_signal_attr_checkpoint(
            signal_attr_checkpoint
        )
    normalized_sample_results = _normalize_sample_results(sample_results, sample_ids=sample_ids)

    rejected_candidates = tuple(
        sorted(
            (
                candidate
                for result in normalized_sample_results
                for candidate in result["rejected_candidates"]
            ),
            key=_rejected_candidate_sort_key,
        )
    )
    nonfinite_candidate_count = _nonfinite_candidate_count(rejected_candidates)
    resolved_output_root = Path(output_root).resolve()
    sample_rows = _manifest_sample_rows(
        base_records=normalized_records,
        sample_results=normalized_sample_results,
        output_root=resolved_output_root,
        variant=resolved_variant,
    )
    manifest = {
        "version": resolved_variant.manifest_version,
        "dataset_key": resolved_variant.dataset_key,
        "variant": resolved_variant.name,
        "split": "val",
        "status": "ready",
        "metric_semantics": deepcopy(resolved_variant.metric_semantics),
        "sample_count": len(sample_rows),
        "failure_count": 0,
        "nonfinite_candidate_count": nonfinite_candidate_count,
        "nonfinite_fatal_count": 0,
        "oom_count": 0,
        "teacher_checkpoints": normalized_teacher_checkpoints,
        "audit_policy": audit_policy,
        "samples": sample_rows,
    }
    validate_ready_manifest_against_base_records(
        manifest,
        base_records=normalized_records,
        base_lane_root=base_lane_root,
        variant=resolved_variant,
    )
    return LaneValODPseudoPreflightResult(
        base_records=normalized_records,
        sample_results=normalized_sample_results,
        manifest=manifest,
        rejected_candidates=rejected_candidates,
    )


def preflight_lane_val_odpseudo_eval_root(
    *,
    base_lane_root: Path,
    output_root: Path,
    sample_results: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    teacher_checkpoints: Mapping[str, Any],
    signal_attr_checkpoint: Any | None = None,
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
    variant: str | LaneValODPseudoVariant | None = None,
    expected_base_count: int | None = None,
) -> LaneValODPseudoPreflightResult:
    base_records = discover_base_lane_val_records(
        base_lane_root,
        allowed_dataset_keys=allowed_dataset_keys,
    )
    return preflight_lane_val_odpseudo_records(
        base_records=base_records,
        base_lane_root=base_lane_root,
        output_root=output_root,
        sample_results=sample_results,
        teacher_checkpoints=teacher_checkpoints,
        signal_attr_checkpoint=signal_attr_checkpoint,
        audit_policy=audit_policy,
        allowed_dataset_keys=allowed_dataset_keys,
        variant=variant,
        expected_base_count=expected_base_count,
    )


def validate_ready_manifest_against_base_records(
    manifest: Mapping[str, Any],
    *,
    base_records: Iterable[LaneValBaseRecord],
    base_lane_root: Path,
    variant: str | LaneValODPseudoVariant | None = None,
) -> None:
    resolved_variant = resolve_lane_val_odpseudo_variant(variant)
    records = validate_base_records(base_records, base_lane_root=base_lane_root)
    ordered_sample_ids = [record.sample_id for record in sorted(records, key=lambda item: (item.split, item.sample_id))]
    if manifest.get("dataset_key") != resolved_variant.dataset_key:
        raise ValueError(f"manifest dataset_key must be {resolved_variant.dataset_key}")
    if manifest.get("variant", resolved_variant.name) != resolved_variant.name:
        raise ValueError(f"manifest variant must be {resolved_variant.name}")
    if manifest.get("split") != "val":
        raise ValueError("manifest split must be val")
    if manifest.get("status") != "ready":
        raise ValueError("manifest status must be ready")
    if manifest.get("metric_semantics") != resolved_variant.metric_semantics:
        raise ValueError(f"manifest metric_semantics must be {resolved_variant.metric_semantics}")
    sample_count = _coerce_nonnegative_int(manifest.get("sample_count"), field_name="manifest.sample_count")
    if sample_count <= 0:
        raise ValueError("ready manifest sample_count must be > 0")
    samples = manifest.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("manifest.samples must be a sequence")
    if sample_count != len(samples):
        raise ValueError(f"manifest sample_count does not match samples length: {sample_count} != {len(samples)}")
    manifest_sample_ids = [
        _coerce_str(row.get("final_sample_id"), field_name="manifest.samples[].final_sample_id")
        for row in samples
        if isinstance(row, Mapping)
    ]
    if manifest_sample_ids != ordered_sample_ids:
        raise ValueError(
            f"manifest sample order must preserve base val order: {manifest_sample_ids} != {ordered_sample_ids}"
        )

    root = Path(base_lane_root).resolve()
    record_by_sample_id = {record.sample_id: record for record in records}
    for row in samples:
        if not isinstance(row, Mapping):
            raise TypeError("manifest sample row must be an object")
        sample_id = _coerce_str(row.get("final_sample_id"), field_name="manifest.sample.final_sample_id")
        record = record_by_sample_id[sample_id]
        if row.get("source_dataset_key") != resolved_variant.dataset_key:
            raise ValueError(f"manifest row source_dataset_key must be {resolved_variant.dataset_key}: {sample_id}")
        if row.get("variant", resolved_variant.name) != resolved_variant.name:
            raise ValueError(f"manifest row variant must be {resolved_variant.name}: {sample_id}")
        if row.get("source_kind") != SOURCE_KIND:
            raise ValueError(f"manifest row source_kind must be {SOURCE_KIND}: {sample_id}")
        if row.get("split") != "val":
            raise ValueError(f"manifest row split must be val: {sample_id}")
        source_scene_path = Path(
            _coerce_str(row.get("source_scene_path"), field_name=f"{sample_id}.source_scene_path")
        ).resolve()
        source_image_path = Path(
            _coerce_str(row.get("source_image_path"), field_name=f"{sample_id}.source_image_path")
        ).resolve()
        if source_scene_path != record.scene_path:
            raise ValueError(f"manifest source_scene_path does not match base record for {sample_id}")
        if source_image_path != record.image_path:
            raise ValueError(f"manifest source_image_path does not match base record for {sample_id}")
        if not _is_relative_to(source_scene_path, root / "labels_scene" / "val"):
            raise ValueError(f"manifest source_scene_path is foreign for {sample_id}: {source_scene_path}")
        if not _is_relative_to(source_image_path, root / "images" / "val"):
            raise ValueError(f"manifest source_image_path is foreign for {sample_id}: {source_image_path}")
        if row.get("teacher_run_status") != "completed":
            raise TeacherFailureError(f"ready manifest sample must have completed teacher status: {sample_id}")
        accepted_detection_count = _coerce_nonnegative_int(
            row.get("accepted_detection_count"),
            field_name=f"{sample_id}.accepted_detection_count",
        )
        det_file_status = _coerce_str(row.get("det_file_status"), field_name=f"{sample_id}.det_file_status")
        if accepted_detection_count == 0 and det_file_status != "empty":
            raise TeacherFailureError(f"empty accepted detections must be recorded as det_file_status=empty: {sample_id}")
        if accepted_detection_count > 0 and det_file_status != "nonempty":
            raise TeacherFailureError(f"accepted detections must be recorded as det_file_status=nonempty: {sample_id}")
        if _coerce_nonnegative_int(row.get("failure_count"), field_name=f"{sample_id}.failure_count") != 0:
            raise TeacherFailureError(f"ready manifest sample failure_count must be 0: {sample_id}")


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"target path already exists: {target_path}")
    if copy_images:
        shutil.copy2(source_path, target_path)
        return
    try:
        target_path.hardlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _write_text_no_overwrite(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"target path already exists: {path}")
    path.write_text(contents, encoding="utf-8")


def _build_output_scene(
    *,
    source_scene: Mapping[str, Any],
    record: LaneValBaseRecord,
    result: Mapping[str, Any],
    final_image_name: str,
    variant: LaneValODPseudoVariant,
) -> dict[str, Any]:
    final_scene = deepcopy(dict(source_scene))
    final_scene.setdefault("source", {})
    final_scene.setdefault("image", {})
    final_scene.setdefault("tasks", {})
    if not isinstance(final_scene["source"], dict):
        raise ValueError(f"scene source must be an object: {record.scene_path}")
    if not isinstance(final_scene["image"], dict):
        raise ValueError(f"scene image must be an object: {record.scene_path}")
    if not isinstance(final_scene["tasks"], dict):
        raise ValueError(f"scene tasks must be an object: {record.scene_path}")

    original_image_name = str(final_scene["image"].get("original_file_name") or final_scene["image"].get("file_name"))
    final_scene["source"]["dataset"] = variant.dataset_key
    final_scene["source"]["split"] = "val"
    final_scene["source"]["final_sample_id"] = record.sample_id
    final_scene["source"]["source_kind"] = SOURCE_KIND
    final_scene["source"]["variant"] = variant.name
    final_scene["source"]["base_dataset"] = record.source_dataset_key
    final_scene["image"]["original_file_name"] = original_image_name
    final_scene["image"]["file_name"] = final_image_name
    final_scene["tasks"]["has_det"] = 1 if int(result["accepted_detection_count"]) > 0 else 0
    final_scene["tasks"]["has_tl_attr"] = 0
    final_scene["detections"] = list(result["accepted_detections"])
    if not variant.tl_attr_enabled:
        final_scene["traffic_lights"] = []
    return final_scene


def build_lane_val_odpseudo_eval_root(
    *,
    base_lane_root: Path,
    output_root: Path,
    sample_results: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    teacher_checkpoints: Mapping[str, Any],
    signal_attr_checkpoint: Any | None = None,
    signal_attr_sidecar: Any | None = None,
    copy_images: bool = False,
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
    variant: str | LaneValODPseudoVariant | None = None,
    expected_base_count: int | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> LaneValODPseudoBuildSummary:
    resolved_variant = resolve_lane_val_odpseudo_variant(variant)
    if resolved_variant.tl_attr_enabled:
        if signal_attr_sidecar is None:
            raise TeacherFailureError("signal_attr sidecar is required for attr_v2 lane-val OD pseudo build")
        if signal_attr_checkpoint is None:
            signal_attr_checkpoint = getattr(signal_attr_sidecar, "checkpoint_path", None)
    preflight = preflight_lane_val_odpseudo_eval_root(
        base_lane_root=base_lane_root,
        output_root=output_root,
        sample_results=sample_results,
        teacher_checkpoints=teacher_checkpoints,
        signal_attr_checkpoint=signal_attr_checkpoint,
        audit_policy=audit_policy,
        allowed_dataset_keys=allowed_dataset_keys,
        variant=resolved_variant,
        expected_base_count=expected_base_count,
    )
    resolved_output_root = Path(output_root).resolve()
    resolved_run_id = run_id or f"{SOURCE_KIND}_{resolved_variant.name}"
    resolved_created_at = created_at or _now_iso()
    sidecar_reason_counts: dict[str, int] = {}
    sidecar_traffic_light_count = 0
    sidecar_valid_count = 0
    sidecar_invalid_count = 0

    for record, result in zip(preflight.base_records, preflight.sample_results, strict=True):
        source_scene = _load_scene(record.scene_path)
        final_image_name = f"{record.sample_id}{record.image_path.suffix.lower()}"
        scene_path = resolved_output_root / "labels_scene" / "val" / f"{record.sample_id}.json"
        det_path = resolved_output_root / "labels_det" / "val" / f"{record.sample_id}.txt"
        image_path = resolved_output_root / "images" / "val" / final_image_name
        final_scene = _build_output_scene(
            source_scene=source_scene,
            record=record,
            result=result,
            final_image_name=final_image_name,
            variant=resolved_variant,
        )
        if resolved_variant.tl_attr_enabled:
            sidecar_stats = signal_attr_sidecar.apply_to_scene(
                final_scene,
                record.image_path,
                run_id=resolved_run_id,
                created_at=resolved_created_at,
            )
            sidecar_traffic_light_count += int(sidecar_stats.traffic_light_count)
            sidecar_valid_count += int(sidecar_stats.valid_count)
            sidecar_invalid_count += int(sidecar_stats.invalid_count)
            for reason, count in dict(sidecar_stats.reason_counts).items():
                sidecar_reason_counts[str(reason)] = sidecar_reason_counts.get(str(reason), 0) + int(count)
        _write_json(scene_path, final_scene, overwrite=False)
        det_text = "\n".join(result["accepted_yolo_rows"])
        _write_text_no_overwrite(det_path, f"{det_text}\n" if det_text else "")
        _link_or_copy(record.image_path, image_path, copy_images=copy_images)

    meta_root = resolved_output_root / "meta"
    rejected_candidates_path = meta_root / REJECTED_DETECTIONS_NAME
    manifest_path = meta_root / FINAL_DATASET_MANIFEST_NAME
    if resolved_variant.tl_attr_enabled:
        preflight.manifest["signal_attr_sidecar"] = {
            "enabled": True,
            "teacher_name": SIGNAL_ATTR_TEACHER_NAME,
            "traffic_light_count": sidecar_traffic_light_count,
            "valid_count": sidecar_valid_count,
            "invalid_count": sidecar_invalid_count,
            "reason_counts": dict(sorted(sidecar_reason_counts.items())),
        }
    _write_jsonl(rejected_candidates_path, preflight.rejected_candidates, sort_keys=True)
    _write_json(manifest_path, preflight.manifest, overwrite=False)
    return {
        "output_root": str(resolved_output_root),
        "manifest_path": str(manifest_path),
        "rejected_candidates_path": str(rejected_candidates_path),
        "sample_count": int(preflight.manifest["sample_count"]),
        "nonfinite_candidate_count": int(preflight.manifest["nonfinite_candidate_count"]),
    }


__all__ = [
    "ALLOWED_BASE_LANE_DATASET_KEYS",
    "ATTR_DATASET_KEY",
    "ATTR_METRIC_SEMANTICS",
    "ATTR_MANIFEST_VERSION",
    "AUDIT_POLICY",
    "DATASET_KEY",
    "DEFAULT_EXPECTED_BASE_VAL_COUNT",
    "FINAL_DATASET_MANIFEST_NAME",
    "MANIFEST_VERSION",
    "METRIC_SEMANTICS",
    "REJECTED_CANDIDATE_REASONS",
    "REJECTED_DETECTIONS_NAME",
    "SOURCE_KIND",
    "TL_ATTR_METRIC_DISABLED_REASON",
    "TEACHER_NAMES",
    "VARIANT_ATTR_V2",
    "VARIANT_V1",
    "VARIANTS_BY_NAME",
    "LaneValBaseRecord",
    "LaneValODPseudoVariant",
    "LaneValODPseudoBuildSummary",
    "LaneValODPseudoError",
    "LaneValODPseudoPreflightResult",
    "LaneValODPseudoSampleResultsSummary",
    "TeacherFailureError",
    "build_lane_val_odpseudo_sample_results_from_predictions",
    "build_lane_val_odpseudo_eval_root",
    "completed_empty_sample_result",
    "discover_base_lane_val_records",
    "guard_lane_val_odpseudo_eval_report",
    "normalize_rejected_candidate_row",
    "preflight_lane_val_odpseudo_eval_root",
    "preflight_lane_val_odpseudo_records",
    "rejected_candidate_semantics",
    "resolve_lane_val_odpseudo_variant",
    "run_lane_val_odpseudo_teacher_sample_results",
    "validate_base_records",
    "validate_ready_manifest_against_base_records",
]
