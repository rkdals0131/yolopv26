from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence, TypedDict

from common.io import read_json as _read_json
from common.io import write_json as _write_json
from common.io import write_jsonl as _write_jsonl


DATASET_KEY = "pv26_eval_lane_val_odpseudo_v1"
SOURCE_KIND = "lane_val_odpseudo"
FINAL_DATASET_MANIFEST_NAME = "final_dataset_manifest.json"
REJECTED_DETECTIONS_NAME = "rejected_detections.jsonl"
MANIFEST_VERSION = "pv26-eval-lane-val-odpseudo-v1"
METRIC_SEMANTICS = "teacher_pseudo_agreement"
AUDIT_POLICY = "audit_gated_v1"
TL_ATTR_METRIC_DISABLED_REASON = "source_mask.tl_attr=false"
TEACHER_NAMES = ("mobility", "signal", "obstacle")
ALLOWED_BASE_LANE_DATASET_KEYS = ("aihub_lane_seoul",)
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


class LaneValODPseudoError(ValueError):
    """Raised when the eval-root preflight cannot produce a ready artifact."""


class TeacherFailureError(LaneValODPseudoError):
    """Raised for teacher-level failures that must not be materialized as empty det files."""


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

    records: list[LaneValBaseRecord] = []
    for scene_path in sorted(val_scene_root.glob("*.json"), key=lambda item: ("val", item.stem)):
        scene = _load_scene(scene_path)
        dataset_key, split = _scene_source(scene, scene_path=scene_path)
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


def _normalized_eval_report_source_mask(
    report: Mapping[str, Any],
    *,
    source_mask: Mapping[str, Any] | None,
) -> dict[str, bool]:
    raw_mask = source_mask if source_mask is not None else report.get("source_mask", {"tl_attr": False})
    if not isinstance(raw_mask, Mapping):
        raise TypeError("source_mask must be a mapping")
    if "tl_attr" not in raw_mask:
        raise ValueError("source_mask.tl_attr is required")
    if bool(raw_mask["tl_attr"]):
        raise ValueError(f"{DATASET_KEY} eval report requires source_mask.tl_attr=false")
    return {str(key): bool(value) for key, value in raw_mask.items()}


def guard_lane_val_odpseudo_eval_report(
    report: Mapping[str, Any],
    *,
    source_mask: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark attr-disabled lane-val OD pseudo eval metrics with explicit semantics."""

    if not isinstance(report, Mapping):
        raise TypeError("eval report must be a mapping")

    guarded = deepcopy(dict(report))
    dataset_key = guarded.get("dataset_key", DATASET_KEY)
    if dataset_key != DATASET_KEY:
        raise ValueError(f"eval report dataset_key must be {DATASET_KEY}")
    split = guarded.get("split", "val")
    if split != "val":
        raise ValueError(f"{DATASET_KEY} eval report must be val-only")

    metrics = guarded.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("eval report metrics must be a mapping")

    guarded["dataset_key"] = DATASET_KEY
    guarded["split"] = "val"
    guarded["source_mask"] = _normalized_eval_report_source_mask(guarded, source_mask=source_mask)
    guarded["metric_semantics"] = METRIC_SEMANTICS
    guarded["metrics"] = deepcopy(dict(metrics))
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
        accepted_detections = payload.get("accepted_detections", ())
        if not isinstance(accepted_detections, Sequence) or isinstance(accepted_detections, (str, bytes)):
            raise TypeError(f"{sample_id}.accepted_detections must be a sequence")
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

        normalized_results.append(
            {
                "sample_id": sample_id,
                "teacher_run_status": teacher_run_status,
                "accepted_yolo_rows": accepted_yolo_rows,
                "accepted_detections": tuple(dict(item) for item in accepted_detections),
                "accepted_detection_count": accepted_detection_count,
                "det_file_status": det_file_status,
                "candidate_count_by_teacher": _normalize_candidate_counts(
                    payload.get("candidate_count_by_teacher"),
                    sample_id=sample_id,
                ),
                "failure_count": failure_count,
                "nonfinite_fatal_count": nonfinite_fatal_count,
                "oom_count": oom_count,
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, result in zip(base_records, sample_results, strict=True):
        suffix = record.image_path.suffix.lower()
        rows.append(
            {
                "final_sample_id": record.sample_id,
                "source_dataset_key": DATASET_KEY,
                "split": "val",
                "source_kind": SOURCE_KIND,
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
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
) -> LaneValODPseudoPreflightResult:
    normalized_records = validate_base_records(
        base_records,
        base_lane_root=base_lane_root,
        allowed_dataset_keys=allowed_dataset_keys,
    )
    normalized_records = tuple(sorted(normalized_records, key=lambda item: (item.split, item.sample_id)))
    sample_ids = tuple(record.sample_id for record in normalized_records)
    normalized_teacher_checkpoints = _normalize_teacher_checkpoints(teacher_checkpoints)
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
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset_key": DATASET_KEY,
        "split": "val",
        "status": "ready",
        "metric_semantics": METRIC_SEMANTICS,
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
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
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
        audit_policy=audit_policy,
        allowed_dataset_keys=allowed_dataset_keys,
    )


def validate_ready_manifest_against_base_records(
    manifest: Mapping[str, Any],
    *,
    base_records: Iterable[LaneValBaseRecord],
    base_lane_root: Path,
) -> None:
    records = validate_base_records(base_records, base_lane_root=base_lane_root)
    ordered_sample_ids = [record.sample_id for record in sorted(records, key=lambda item: (item.split, item.sample_id))]
    if manifest.get("dataset_key") != DATASET_KEY:
        raise ValueError(f"manifest dataset_key must be {DATASET_KEY}")
    if manifest.get("split") != "val":
        raise ValueError("manifest split must be val")
    if manifest.get("status") != "ready":
        raise ValueError("manifest status must be ready")
    if manifest.get("metric_semantics") != METRIC_SEMANTICS:
        raise ValueError(f"manifest metric_semantics must be {METRIC_SEMANTICS}")
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
        if row.get("source_dataset_key") != DATASET_KEY:
            raise ValueError(f"manifest row source_dataset_key must be {DATASET_KEY}: {sample_id}")
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
    final_scene["source"]["dataset"] = DATASET_KEY
    final_scene["source"]["split"] = "val"
    final_scene["source"]["final_sample_id"] = record.sample_id
    final_scene["source"]["source_kind"] = SOURCE_KIND
    final_scene["source"]["base_dataset"] = record.source_dataset_key
    final_scene["image"]["original_file_name"] = original_image_name
    final_scene["image"]["file_name"] = final_image_name
    final_scene["tasks"]["has_det"] = 1 if int(result["accepted_detection_count"]) > 0 else 0
    final_scene["tasks"]["has_tl_attr"] = 0
    final_scene["detections"] = list(result["accepted_detections"])
    return final_scene


def build_lane_val_odpseudo_eval_root(
    *,
    base_lane_root: Path,
    output_root: Path,
    sample_results: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    teacher_checkpoints: Mapping[str, Any],
    copy_images: bool = False,
    audit_policy: str = AUDIT_POLICY,
    allowed_dataset_keys: Iterable[str] = ALLOWED_BASE_LANE_DATASET_KEYS,
) -> LaneValODPseudoBuildSummary:
    preflight = preflight_lane_val_odpseudo_eval_root(
        base_lane_root=base_lane_root,
        output_root=output_root,
        sample_results=sample_results,
        teacher_checkpoints=teacher_checkpoints,
        audit_policy=audit_policy,
        allowed_dataset_keys=allowed_dataset_keys,
    )
    resolved_output_root = Path(output_root).resolve()

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
        )
        _write_json(scene_path, final_scene, overwrite=False)
        det_text = "\n".join(result["accepted_yolo_rows"])
        _write_text_no_overwrite(det_path, f"{det_text}\n" if det_text else "")
        _link_or_copy(record.image_path, image_path, copy_images=copy_images)

    meta_root = resolved_output_root / "meta"
    rejected_candidates_path = meta_root / REJECTED_DETECTIONS_NAME
    manifest_path = meta_root / FINAL_DATASET_MANIFEST_NAME
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
    "AUDIT_POLICY",
    "DATASET_KEY",
    "FINAL_DATASET_MANIFEST_NAME",
    "MANIFEST_VERSION",
    "METRIC_SEMANTICS",
    "REJECTED_CANDIDATE_REASONS",
    "REJECTED_DETECTIONS_NAME",
    "SOURCE_KIND",
    "TL_ATTR_METRIC_DISABLED_REASON",
    "TEACHER_NAMES",
    "LaneValBaseRecord",
    "LaneValODPseudoBuildSummary",
    "LaneValODPseudoError",
    "LaneValODPseudoPreflightResult",
    "TeacherFailureError",
    "build_lane_val_odpseudo_eval_root",
    "completed_empty_sample_result",
    "discover_base_lane_val_records",
    "guard_lane_val_odpseudo_eval_report",
    "normalize_rejected_candidate_row",
    "preflight_lane_val_odpseudo_eval_root",
    "preflight_lane_val_odpseudo_records",
    "rejected_candidate_semantics",
    "validate_base_records",
    "validate_ready_manifest_against_base_records",
]
