from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common.io import write_jsonl_sorted
from common.pv26_schema import (
    ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY,
    ETRI_KCITY_LEFTIMG_DATASET_KEY,
    ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY,
    ETRI_MULTICAMERA_LEFTIMG_DATASET_KEY,
    OD_CLASS_TO_ID,
)
from common.geometry import canonicalize_stop_line_points

from ..shared.io import link_or_copy, now_iso, write_json, write_text
from ..shared.raw import extract_annotations, extract_bbox, extract_points, normalize_text, probe_image_size, safe_slug
from ..shared.summary import counter_to_dict

try:
    from PIL import Image
except ImportError:  # pragma: no cover - covered by environments without PIL.
    Image = None


DATASET_KEY = ETRI_KCITY_LEFTIMG_DATASET_KEY
ATTRPSEUDO_DATASET_KEY = ETRI_KCITY_LEFTIMG_ATTRPSEUDO_DATASET_KEY
MULTICAMERA_DATASET_KEY = ETRI_MULTICAMERA_LEFTIMG_DATASET_KEY
MULTICAMERA_ATTRPSEUDO_DATASET_KEY = ETRI_MULTICAMERA_LEFTIMG_ATTRPSEUDO_DATASET_KEY
SOURCE_KIND = "etri_kcity_leftimg"
ATTRPSEUDO_SOURCE_KIND = "etri_kcity_leftimg_attrpseudo"
MULTICAMERA_SOURCE_KIND = "etri_multicamera_leftimg"
MULTICAMERA_ATTRPSEUDO_SOURCE_KIND = "etri_multicamera_leftimg_attrpseudo"
FINAL_DATASET_MANIFEST_NAME = "final_dataset_manifest.json"
HELD_LABELS_NAME = "held_labels.jsonl"
TL_ATTR_TEACHER_DEBUG_NAME = "tl_attr_teacher_debug.jsonl"
VALID_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
SEMANTIC_LABEL_EXTENSIONS = {".json", ".png", ".bmp", ".tif", ".tiff"}

RAW_SCAN_REASON_RIGHT_IMG = "rightImg"
RAW_SCAN_REASON_MONO_CAMERA = "MonoCamera"
RAW_SCAN_REASON_LIDAR = "lidar_annotation"
RAW_SCAN_REASON_PV26_OUTPUT = "pv26_output"

CANDIDATE_REASON_MISSING_SEMANTIC_LABEL = "missing_semantic_label"
CANDIDATE_REASON_SAMPLE_ID_MISMATCH = "image_label_sample_id_mismatch"
CANDIDATE_REASON_INVALID_SPLIT = "invalid_split"
CANDIDATE_REASON_MISSING_IMAGE_SIZE = "missing_image_size"
CANDIDATE_REASON_LABEL_PARSE_FAILURE = "label_parse_failure"

READY_STATUS = "ready"
BLOCKED_STATUS = "blocked"
RELEASE_BLOCKER_ZERO_SAMPLES = "zero_samples"

_SAMPLE_ID_SUFFIXES = (
    "_leftimg8bit",
    "_leftimg",
    "-leftimg8bit",
    "-leftimg",
    "_rightimg8bit",
    "_rightimg",
    "-rightimg8bit",
    "-rightimg",
    "_gtfine_labelids",
    "_gtfine_labeltrainids",
    "_gtfine_polygons",
    "_gtcoarse_labelids",
    "_labeltrainids",
    "_labelids",
    "_instanceids",
    "_semantic_label",
    "_semantic",
    "-semantic",
    "_semantics",
    "_label",
    "-label",
    "_mask",
    "-mask",
)
_CLASS_KEYS = (
    "label",
    "raw_class",
    "class_name",
    "class",
    "category",
    "label_name",
    "semantic_class",
    "semantic_label",
)
HELD_LABEL_REASON_UNMAPPED = "unmapped_label"
RELEASE_VERSION = "etri-kcity-leftimg-release-v1"
ATTRPSEUDO_RELEASE_VERSION = "etri-kcity-leftimg-attrpseudo-v1"
RELEASE_PATH_TOKEN = "20221124_kcity"
ETRI_LANE_CENTERLINE_POLICY = "etri_lane_polygon_row_slice_centerline_v1"
ETRI_STOP_LINE_CENTERLINE_POLICY = "etri_stop_line_polygon_centerline_v1"
ETRI_CROSSWALK_AREA_POLICY = "etri_crosswalk_area_polygon_v1"
ETRI_LANE_CENTERLINE_MAX_POINTS = 48
ETRI_LANE_CENTERLINE_SAMPLE_STEP_PX = 20.0
ETRI_SAFE_LABEL_MAPPING = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "caravan": "vehicle",
    "person": "pedestrian",
    "rubber_cone": "traffic_cone",
    "traffic_light": "traffic_light",
    "traffic_sign": "sign",
    "sign": "sign",
    "whsol": "lane",
    "whdot": "lane",
    "yesol": "lane",
    "blsol": "lane",
    "bldot": "lane",
    "stop_line": "stop_line",
    "crosswalk": "crosswalk",
}
ETRI_LANE_STYLE_BY_LABEL = {
    "whsol": ("white_lane", "solid"),
    "whdot": ("white_lane", "dotted"),
    "yesol": ("yellow_lane", "solid"),
    "blsol": ("blue_lane", "solid"),
    "bldot": ("blue_lane", "dotted"),
}
_IMAGE_SIZE_KEYS = ("image_size", "imsize", "size")
_IMAGE_SECTION_KEYS = ("image", "images", "metadata")
_LABEL_SAMPLE_ID_KEYS = ("sample_id", "raw_id", "frame_id", "image_id")


@dataclass(frozen=True)
class EtriDryRunSample:
    sample_id: str
    split: str
    image_path: Path
    semantic_label_path: Path
    width: int
    height: int
    raw_class_counts: Mapping[str, int]

    def to_manifest_item(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": str(self.image_path),
            "semantic_label_path": str(self.semantic_label_path),
            "width": self.width,
            "height": self.height,
            "raw_class_counts": dict(sorted(self.raw_class_counts.items())),
        }


@dataclass(frozen=True)
class EtriDryRunResult:
    dataset_key: str
    dataset_root: Path
    default_split: str | None
    samples: tuple[EtriDryRunSample, ...]
    raw_scan_ignored_count_by_reason: Mapping[str, int]
    candidate_excluded_count_by_reason: Mapping[str, int]
    raw_class_inventory: Mapping[str, int]
    failures: tuple[dict[str, Any], ...]
    generated_at: str

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def failure_count(self) -> int:
        return int(sum(self.candidate_excluded_count_by_reason.values()))

    @property
    def is_ready(self) -> bool:
        return is_dry_run_ready(self)

    @property
    def release_blockers(self) -> tuple[dict[str, Any], ...]:
        return dry_run_release_blockers(self)

    @property
    def status(self) -> str:
        if self.is_ready:
            return READY_STATUS
        return BLOCKED_STATUS

    def require_ready(self) -> EtriDryRunResult:
        return require_dry_run_ready(self)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": "etri-kcity-leftimg-dry-run-v1",
            "generated_at": self.generated_at,
            "dataset_key": self.dataset_key,
            "dataset_root": str(self.dataset_root),
            "default_split": self.default_split,
            "status": self.status,
            "sample_count": self.sample_count,
            "failure_count": self.failure_count,
            "release_blockers": list(self.release_blockers),
            "raw_scan_ignored_count_by_reason": dict(sorted(self.raw_scan_ignored_count_by_reason.items())),
            "candidate_excluded_count_by_reason": dict(sorted(self.candidate_excluded_count_by_reason.items())),
            "raw_class_inventory": dict(sorted(self.raw_class_inventory.items())),
            "samples": [sample.to_manifest_item() for sample in self.samples],
            "failures": list(self.failures),
        }


class EtriDryRunNotReadyError(RuntimeError):
    def __init__(self, result: EtriDryRunResult) -> None:
        self.result = result
        self.blockers = dry_run_release_blockers(result)
        reasons = ", ".join(str(blocker["reason"]) for blocker in self.blockers) or BLOCKED_STATUS
        super().__init__(f"ETRI KCity dry-run is not release-ready: {reasons}")


def is_dry_run_ready(result: EtriDryRunResult) -> bool:
    return result.sample_count > 0 and not result.candidate_excluded_count_by_reason


def dry_run_release_blockers(result: EtriDryRunResult) -> tuple[dict[str, Any], ...]:
    blockers: list[dict[str, Any]] = []
    if result.sample_count <= 0:
        blockers.append(
            {
                "reason": RELEASE_BLOCKER_ZERO_SAMPLES,
                "count": result.sample_count,
                "detail": "dry-run accepted no KCity leftImg samples",
            }
        )
    for reason, count in sorted(result.candidate_excluded_count_by_reason.items()):
        blockers.append({"reason": str(reason), "count": int(count)})
    return tuple(blockers)


def require_dry_run_ready(result: EtriDryRunResult) -> EtriDryRunResult:
    if not is_dry_run_ready(result):
        raise EtriDryRunNotReadyError(result)
    return result


def scan_dry_run(
    dataset_root: Path,
    *,
    default_split: str | None = None,
    required_path_token: str | None = "kcity",
    sample_id_dataset_key: str = DATASET_KEY,
) -> EtriDryRunResult:
    root = dataset_root.expanduser().resolve()
    resolved_default_split = _resolve_default_split(default_split)
    raw_scan_ignored = Counter()
    candidate_excluded = Counter()
    raw_class_inventory = Counter()
    failures: list[dict[str, Any]] = []

    image_candidates: list[Path] = []
    semantic_labels: dict[str, list[Path]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ignored_reason = raw_scan_ignored_reason(path)
        if ignored_reason is not None:
            raw_scan_ignored[ignored_reason] += 1
            continue
        if is_leftimg_candidate(path, required_path_token=required_path_token):
            image_candidates.append(path)
        elif is_semantic_label_candidate(path, required_path_token=required_path_token):
            sample_id = sample_id_from_path(path)
            semantic_labels.setdefault(sample_id, []).append(path)

    samples: list[EtriDryRunSample] = []
    for image_path in sorted(image_candidates):
        sample_id = sample_id_from_path(image_path)
        label_path = _select_semantic_label_for_image(image_path, semantic_labels.get(sample_id, []))
        if label_path is None:
            _append_candidate_failure(
                failures,
                reason=CANDIDATE_REASON_MISSING_SEMANTIC_LABEL,
                image_path=image_path,
                sample_id=sample_id,
            )
            candidate_excluded[CANDIDATE_REASON_MISSING_SEMANTIC_LABEL] += 1
            continue
        try:
            sample = build_dry_run_sample(
                image_path=image_path,
                semantic_label_path=label_path,
                dataset_root=root,
                default_split=resolved_default_split,
                sample_id_dataset_key=sample_id_dataset_key,
            )
        except EtriCandidateError as exc:
            _append_candidate_failure(
                failures,
                reason=exc.reason,
                image_path=image_path,
                label_path=label_path,
                sample_id=sample_id,
                detail=exc.detail,
            )
            candidate_excluded[exc.reason] += 1
            continue
        samples.append(sample)
        raw_class_inventory.update(sample.raw_class_counts)

    return EtriDryRunResult(
        dataset_key=DATASET_KEY,
        dataset_root=root,
        default_split=resolved_default_split,
        samples=tuple(samples),
        raw_scan_ignored_count_by_reason=counter_to_dict(raw_scan_ignored),
        candidate_excluded_count_by_reason=counter_to_dict(candidate_excluded),
        raw_class_inventory=counter_to_dict(raw_class_inventory),
        failures=tuple(failures),
        generated_at=now_iso(),
    )


def build_dry_run_manifest(
    dataset_root: Path,
    *,
    default_split: str | None = None,
    required_path_token: str | None = "kcity",
) -> dict[str, Any]:
    return scan_dry_run(
        dataset_root,
        default_split=default_split,
        required_path_token=required_path_token,
    ).to_manifest()


def build_ready_dry_run_manifest(
    dataset_root: Path,
    *,
    default_split: str | None = None,
    required_path_token: str | None = "kcity",
) -> dict[str, Any]:
    result = scan_dry_run(
        dataset_root,
        default_split=default_split,
        required_path_token=required_path_token,
    )
    require_dry_run_ready(result)
    return result.to_manifest()


def write_dry_run_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    default_split: str | None = None,
    required_path_token: str | None = "kcity",
) -> Path:
    manifest = build_dry_run_manifest(
        dataset_root,
        default_split=default_split,
        required_path_token=required_path_token,
    )
    return write_json(output_path, manifest)


def write_ready_dry_run_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    default_split: str | None = None,
    required_path_token: str | None = "kcity",
) -> Path:
    manifest = build_ready_dry_run_manifest(
        dataset_root,
        default_split=default_split,
        required_path_token=required_path_token,
    )
    return write_json(output_path, manifest)


def build_dry_run_sample(
    *,
    image_path: Path,
    semantic_label_path: Path | None,
    dataset_root: Path,
    default_split: str | None = None,
    sample_id_dataset_key: str = DATASET_KEY,
) -> EtriDryRunSample:
    if semantic_label_path is None:
        raise EtriCandidateError(
            CANDIDATE_REASON_MISSING_SEMANTIC_LABEL,
            "leftImg candidate has no paired semantic label",
        )
    resolved_default_split = _resolve_default_split(default_split)
    image = image_path.expanduser().resolve()
    label = semantic_label_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()

    image_sample_id = sample_id_from_path(image)
    label_sample_id = sample_id_from_path(label)
    if image_sample_id != label_sample_id:
        raise EtriCandidateError(
            CANDIDATE_REASON_SAMPLE_ID_MISMATCH,
            f"image sample id {image_sample_id!r} != label sample id {label_sample_id!r}",
        )

    label_raw = _load_json_label(label)
    metadata_sample_id = _label_metadata_sample_id(label_raw)
    if metadata_sample_id is not None and metadata_sample_id != image_sample_id:
        raise EtriCandidateError(
            CANDIDATE_REASON_SAMPLE_ID_MISMATCH,
            f"image sample id {image_sample_id!r} != label metadata sample id {metadata_sample_id!r}",
        )

    split = _resolve_split(image, label, default_split=resolved_default_split)
    if split is None:
        raise EtriCandidateError(
            CANDIDATE_REASON_INVALID_SPLIT,
            "split is absent from raw paths and no default_split was provided",
        )

    image_size = _resolve_image_size(image, label_raw)
    if image_size is None:
        raise EtriCandidateError(
            CANDIDATE_REASON_MISSING_IMAGE_SIZE,
            "image size is unavailable from file probe and semantic metadata",
        )
    width, height = image_size
    raw_class_counts = _extract_raw_class_counts(label, label_raw)
    sample_relative_id = _safe_relative_stem(image, root)
    sample_id = safe_slug(f"{sample_id_dataset_key}_{split}_{sample_relative_id}")
    return EtriDryRunSample(
        sample_id=sample_id,
        split=split,
        image_path=image,
        semantic_label_path=label,
        width=width,
        height=height,
        raw_class_counts=counter_to_dict(raw_class_counts),
    )


def raw_scan_ignored_reason(path: Path) -> str | None:
    normalized_parts = [normalize_text(part) for part in path.parts]
    joined = "/".join(normalized_parts)
    if any(part.startswith("pv26_") for part in normalized_parts):
        return RAW_SCAN_REASON_PV26_OUTPUT
    if "lidar" in joined or path.suffix.lower() in {".pcd", ".bin"}:
        return RAW_SCAN_REASON_LIDAR
    if "monocamera" in joined or "mono_camera" in joined or "mono-camera" in joined:
        return RAW_SCAN_REASON_MONO_CAMERA
    if "rightimg" in joined or "right_img" in joined or "right-img" in joined:
        return RAW_SCAN_REASON_RIGHT_IMG
    return None


def is_kcity_leftimg_candidate(path: Path) -> bool:
    return is_leftimg_candidate(path, required_path_token="kcity")


def is_leftimg_candidate(path: Path, *, required_path_token: str | None = "kcity") -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    normalized_parts = [normalize_text(part) for part in path.parts]
    joined = "/".join(normalized_parts)
    required = normalize_text(required_path_token) if required_path_token is not None else None
    if required is not None and required not in joined:
        return False
    if "leftimg" not in joined and "left_img" not in joined:
        return False
    if _has_semantic_label_marker(path):
        return False
    return raw_scan_ignored_reason(path) is None


def is_semantic_label_candidate(path: Path, *, required_path_token: str | None = "kcity") -> bool:
    if path.suffix.lower() not in SEMANTIC_LABEL_EXTENSIONS:
        return False
    if raw_scan_ignored_reason(path) is not None:
        return False
    normalized_parts = [normalize_text(part) for part in path.parts]
    joined = "/".join(normalized_parts)
    required = normalize_text(required_path_token) if required_path_token is not None else None
    if required is not None and required not in joined:
        return False
    return _has_semantic_label_marker(path)


def sample_id_from_path(path: Path) -> str:
    normalized = normalize_text(path.stem)
    previous = None
    while normalized != previous:
        previous = normalized
        for suffix in _SAMPLE_ID_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
    return safe_slug(normalized)


class EtriCandidateError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class EtriMaterializationError(ValueError):
    """Raised when a mapped ETRI label cannot be safely converted to PV26 geometry."""


def materialize_kcity_val_release(
    dataset_root: Path,
    output_root: Path,
    *,
    copy_images: bool = False,
    expected_sample_count: int | None = None,
    release_path_token: str | Sequence[str] | None = RELEASE_PATH_TOKEN,
    exclude_path_tokens: Sequence[str] = (),
    allowed_splits: Iterable[str] | None = ("val",),
    output_split: str = "val",
    scan_required_path_token: str | None = "kcity",
    sample_limit: int | None = None,
    signal_attr_sidecar: Any | None = None,
    signal_attr_checkpoint_path: Path | None = None,
    dataset_key_override: str | None = None,
    attrpseudo_dataset_key_override: str | None = None,
    source_kind_override: str | None = None,
    attrpseudo_source_kind_override: str | None = None,
    sample_id_dataset_key: str = DATASET_KEY,
) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    generated_at = now_iso()
    attrpseudo_enabled = signal_attr_sidecar is not None
    if attrpseudo_enabled:
        dataset_key = attrpseudo_dataset_key_override or ATTRPSEUDO_DATASET_KEY
        source_kind = attrpseudo_source_kind_override or ATTRPSEUDO_SOURCE_KIND
    else:
        dataset_key = dataset_key_override or DATASET_KEY
        source_kind = source_kind_override or SOURCE_KIND
    manifest_version = ATTRPSEUDO_RELEASE_VERSION if attrpseudo_enabled else RELEASE_VERSION
    sidecar_checkpoint = _sidecar_checkpoint_path(signal_attr_sidecar, signal_attr_checkpoint_path)
    split = _resolve_default_split(output_split)
    if split is None:
        raise ValueError("output_split must be provided")
    dry_run = scan_dry_run(
        root,
        default_split=split,
        required_path_token=scan_required_path_token,
        sample_id_dataset_key=sample_id_dataset_key,
    )
    samples = tuple(
        sample
        for sample in dry_run.samples
        if _is_release_sample(
            sample,
            release_path_token=release_path_token,
            exclude_path_tokens=exclude_path_tokens,
            allowed_splits=allowed_splits,
        )
    )
    samples = tuple(sorted(samples, key=lambda item: item.sample_id))
    if sample_limit is not None:
        limit = int(sample_limit)
        if limit <= 0:
            raise ValueError("sample_limit must be > 0")
        samples = samples[:limit]
    if expected_sample_count is not None and len(samples) != int(expected_sample_count):
        raise EtriMaterializationError(
            f"ETRI KCity release sample count must be exactly {int(expected_sample_count)}: {len(samples)}"
        )
    if not samples:
        raise EtriMaterializationError("ETRI KCity release has no val leftImg samples")

    held_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    sidecar_reason_counts: Counter[str] = Counter()
    sidecar_traffic_light_count = 0
    sidecar_valid_count = 0
    sidecar_invalid_count = 0
    tl_attr_debug_rows: list[dict[str, Any]] = []
    for sample in samples:
        row, sample_counts, sample_held, sidecar_stats, sample_debug_rows = _materialize_release_sample(
            sample=sample,
            output_root=output,
            copy_images=copy_images,
            dataset_key=dataset_key,
            source_kind=source_kind,
            output_split=split,
            signal_attr_sidecar=signal_attr_sidecar,
            run_id=f"{source_kind}:{generated_at}",
            created_at=generated_at,
        )
        manifest_rows.append(row)
        class_counts.update(sample_counts)
        held_rows.extend(sample_held)
        tl_attr_debug_rows.extend(sample_debug_rows)
        if sidecar_stats is not None:
            sidecar_traffic_light_count += int(sidecar_stats.traffic_light_count)
            sidecar_valid_count += int(sidecar_stats.valid_count)
            sidecar_invalid_count += int(sidecar_stats.invalid_count)
            sidecar_reason_counts.update(dict(sidecar_stats.reason_counts))
        geometry_counts.update(
            {
                "detections": int(row["accepted_detection_count"]),
                "lanes": int(row["lane_count"]),
                "stop_lines": int(row["stop_line_count"]),
                "crosswalks": int(row["crosswalk_count"]),
                "traffic_lights": int(row["traffic_light_count"]),
            }
        )

    manifest_rows.sort(key=lambda item: (str(item["split"]), str(item["final_sample_id"])))
    held_rows.sort(key=lambda item: (str(item["sample_id"]), int(item["annotation_index"]), str(item["raw_label"])))
    tl_attr_debug_rows.sort(key=lambda item: (str(item["sample_id"]), int(item["detection_id"])))
    manifest_path = output / "meta" / FINAL_DATASET_MANIFEST_NAME
    held_path = output / "meta" / HELD_LABELS_NAME
    tl_attr_debug_path = output / "meta" / TL_ATTR_TEACHER_DEBUG_NAME
    tl_attr_coverage_count = sidecar_traffic_light_count if attrpseudo_enabled else 0
    tl_attr_status = (
        "not_available"
        if not attrpseudo_enabled
        else "complete"
        if sidecar_invalid_count == 0
        else "partial"
    )
    tl_attr_teacher = {
        "enabled": attrpseudo_enabled,
        "teacher_name": "signal_attr" if attrpseudo_enabled else None,
        "checkpoint_path": str(sidecar_checkpoint) if sidecar_checkpoint is not None else None,
        "traffic_light_count": sidecar_traffic_light_count,
        "coverage_count": tl_attr_coverage_count,
        "coverage_gap_count": 0 if attrpseudo_enabled else sidecar_traffic_light_count,
        "valid_count": sidecar_valid_count,
        "invalid_count": sidecar_invalid_count,
        "status": tl_attr_status,
        "validity_policy": "conservative_thresholded",
        "debug_rows_path": str(tl_attr_debug_path) if attrpseudo_enabled else None,
        "reason_counts": counter_to_dict(sidecar_reason_counts),
    }
    manifest = {
        "version": manifest_version,
        "generated_at": generated_at,
        "dataset_key": dataset_key,
        "split": split,
        "status": READY_STATUS,
        "source_kind": source_kind,
        "dataset_root": str(root),
        "output_root": str(output),
        "release_path_token": release_path_token,
        "exclude_path_tokens": list(exclude_path_tokens),
        "allowed_source_splits": sorted(_normalized_split_set(allowed_splits) or []),
        "output_split": split,
        "sample_count": len(manifest_rows),
        "failure_count": 0,
        "dataset_counts": {dataset_key: len(manifest_rows)},
        "class_counts": counter_to_dict(class_counts),
        "geometry_counts": counter_to_dict(geometry_counts),
        "held_label_count": len(held_rows),
        "held_label_counts": counter_to_dict(Counter(str(row["raw_label"]) for row in held_rows)),
        "held_labels_path": str(held_path),
        "tl_attr_teacher": tl_attr_teacher,
        "metric_semantics": {
            "det": "human_polygon_bbox_gt",
            "tl_attr": "signal_attr_teacher_pseudo" if attrpseudo_enabled else "not_available",
            "lane": "human_polygon_centerline_gt",
            "stop_line": "human_polygon_centerline_gt",
            "crosswalk": "human_polygon_area_gt",
        },
        "signal_attr_sidecar": {
            "enabled": attrpseudo_enabled,
            "teacher_name": "signal_attr" if attrpseudo_enabled else None,
            "checkpoint_path": str(sidecar_checkpoint) if sidecar_checkpoint is not None else None,
            "traffic_light_count": sidecar_traffic_light_count,
            "coverage_count": tl_attr_coverage_count,
            "coverage_gap_count": tl_attr_teacher["coverage_gap_count"],
            "valid_count": sidecar_valid_count,
            "invalid_count": sidecar_invalid_count,
            "status": tl_attr_status,
            "validity_policy": "conservative_thresholded",
            "debug_rows_path": str(tl_attr_debug_path) if attrpseudo_enabled else None,
            "reason_counts": counter_to_dict(sidecar_reason_counts),
        },
        "samples": manifest_rows,
    }
    write_jsonl_sorted(held_path, held_rows)
    if attrpseudo_enabled:
        write_jsonl_sorted(tl_attr_debug_path, tl_attr_debug_rows)
    write_json(manifest_path, manifest)
    return {
        "output_root": str(output),
        "manifest_path": str(manifest_path),
        "held_labels_path": str(held_path),
        "tl_attr_debug_rows_path": str(tl_attr_debug_path) if attrpseudo_enabled else None,
        "sample_count": len(manifest_rows),
        "held_label_count": len(held_rows),
        "class_counts": counter_to_dict(class_counts),
        "tl_attr_teacher": manifest["tl_attr_teacher"],
        "signal_attr_sidecar": manifest["signal_attr_sidecar"],
    }


def _is_release_sample(
    sample: EtriDryRunSample,
    *,
    release_path_token: str | Sequence[str] | None,
    exclude_path_tokens: Sequence[str],
    allowed_splits: Iterable[str] | None,
) -> bool:
    allowed = _normalized_split_set(allowed_splits)
    if allowed is not None and sample.split not in allowed:
        return False
    for token in exclude_path_tokens:
        if _sample_path_has_token(sample, token):
            return False
    if release_path_token is None:
        return True
    tokens = [release_path_token] if isinstance(release_path_token, str) else list(release_path_token)
    return any(_sample_path_has_token(sample, token) for token in tokens)


def _normalized_split_set(splits: Iterable[str] | None) -> set[str] | None:
    if splits is None:
        return None
    normalized: set[str] = set()
    for split in splits:
        resolved = _resolve_default_split(str(split))
        if resolved is not None:
            normalized.add(resolved)
    return normalized


def _sample_path_has_token(sample: EtriDryRunSample, token: str) -> bool:
    normalized_token = normalize_text(token)
    if not normalized_token:
        return False
    joined = "/".join(normalize_text(part) for part in (*sample.image_path.parts, *sample.semantic_label_path.parts))
    return normalized_token in joined


def _materialize_release_sample(
    *,
    sample: EtriDryRunSample,
    output_root: Path,
    copy_images: bool,
    dataset_key: str,
    source_kind: str,
    output_split: str,
    signal_attr_sidecar: Any | None,
    run_id: str,
    created_at: str,
) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]], Any | None, list[dict[str, Any]]]:
    raw = _load_json_label(sample.semantic_label_path)
    if raw is None:
        raise EtriMaterializationError(f"ETRI KCity release requires JSON semantic labels: {sample.semantic_label_path}")
    annotations = extract_annotations(raw)
    detections: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    stop_lines: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    held_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for annotation_index, annotation in enumerate(annotations):
        if bool(annotation.get("deleted", 0)):
            continue
        raw_label = _annotation_raw_label(annotation)
        normalized_label = normalize_text(raw_label)
        mapped = ETRI_SAFE_LABEL_MAPPING.get(normalized_label)
        if mapped is None:
            held_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "annotation_index": annotation_index,
                    "raw_label": raw_label,
                    "reason": HELD_LABEL_REASON_UNMAPPED,
                }
            )
            continue
        if mapped in OD_CLASS_TO_ID:
            bbox = _annotation_bbox(annotation, width=sample.width, height=sample.height, sample=sample, raw_label=raw_label)
            detections.append(
                {
                    "id": len(detections),
                    "class_name": mapped,
                    "bbox": _bbox_to_mapping(bbox),
                    "meta": {"raw_label": raw_label, "label_origin": "etri_kcity_raw"},
                }
            )
            class_counts[mapped] += 1
            continue
        points = _annotation_points(
            annotation,
            width=sample.width,
            height=sample.height,
            sample=sample,
            raw_label=raw_label,
            min_points=3 if mapped == "crosswalk" else 2,
        )
        if mapped == "lane":
            lane_class, lane_type = ETRI_LANE_STYLE_BY_LABEL[normalized_label]
            centerline_points = _lane_centerline_points(
                points,
                width=sample.width,
                height=sample.height,
                sample=sample,
                raw_label=raw_label,
            )
            lanes.append(
                {
                    "id": len(lanes),
                    "class_name": lane_class,
                    "source_style": lane_type,
                    "points": centerline_points,
                    "meta": {
                        "raw_label": raw_label,
                        "label_origin": "etri_kcity_raw",
                        "geometry_policy": ETRI_LANE_CENTERLINE_POLICY,
                        "source_polygon_point_count": len(points),
                    },
                }
            )
            class_counts["lane"] += 1
        elif mapped == "stop_line":
            centerline_points = _stop_line_centerline_points(
                points,
                width=sample.width,
                height=sample.height,
                sample=sample,
                raw_label=raw_label,
            )
            stop_lines.append(
                {
                    "id": len(stop_lines),
                    "class_name": "stop_line",
                    "points": centerline_points,
                    "meta": {
                        "raw_label": raw_label,
                        "label_origin": "etri_kcity_raw",
                        "geometry_policy": ETRI_STOP_LINE_CENTERLINE_POLICY,
                        "source_polygon_point_count": len(points),
                    },
                }
            )
            class_counts["stop_line"] += 1
        elif mapped == "crosswalk":
            crosswalks.append(
                {
                    "id": len(crosswalks),
                    "class_name": "crosswalk",
                    "points": points,
                    "meta": {
                        "raw_label": raw_label,
                        "label_origin": "etri_kcity_raw",
                        "geometry_policy": ETRI_CROSSWALK_AREA_POLICY,
                        "source_polygon_point_count": len(points),
                    },
                }
            )
            class_counts["crosswalk"] += 1

    image_output_name = f"{sample.sample_id}{sample.image_path.suffix.lower()}"
    scene_path = output_root / "labels_scene" / output_split / f"{sample.sample_id}.json"
    det_path = output_root / "labels_det" / output_split / f"{sample.sample_id}.txt"
    image_path = output_root / "images" / output_split / image_output_name
    scene = {
        "image": {
            "file_name": image_output_name,
            "original_file_name": sample.image_path.name,
            "width": int(sample.width),
            "height": int(sample.height),
        },
        "source": {
            "dataset": dataset_key,
            "split": output_split,
            "raw_split": sample.split,
            "source_kind": source_kind,
            "source_image_path": str(sample.image_path),
            "source_label_path": str(sample.semantic_label_path),
        },
        "tasks": {
            "has_det": int(bool(detections)),
            "has_lane": int(bool(lanes)),
            "has_stop_line": int(bool(stop_lines)),
            "has_crosswalk": int(bool(crosswalks)),
            "has_tl_attr": 0,
        },
        "detections": detections,
        "lanes": lanes,
        "stop_lines": stop_lines,
        "crosswalks": crosswalks,
        "traffic_lights": [],
    }
    sidecar_stats = None
    tl_attr_debug_rows: list[dict[str, Any]] = []
    if signal_attr_sidecar is not None:
        sidecar_stats = signal_attr_sidecar.apply_to_scene(
            scene,
            sample.image_path,
            run_id=run_id,
            created_at=created_at,
        )
        tl_attr_debug_rows = _validate_and_summarize_tl_attr_rows(scene, sample=sample)
        _assert_tl_attr_stats_match_rows(sidecar_stats, tl_attr_debug_rows, sample=sample)
    write_json(scene_path, scene)
    write_text(det_path, _det_label_text(detections, width=sample.width, height=sample.height))
    if copy_images:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(sample.image_path.read_bytes())
    else:
        link_or_copy(sample.image_path, image_path)

    manifest_row = {
        "final_sample_id": sample.sample_id,
        "source_dataset_key": dataset_key,
        "split": output_split,
        "source_raw_split": sample.split,
        "source_kind": source_kind,
        "scene_path": str(scene_path.resolve()),
        "image_path": str(image_path.resolve()),
        "det_path": str(det_path.resolve()),
        "source_scene_path": str(sample.semantic_label_path),
        "source_image_path": str(sample.image_path),
        "source_det_path": None,
        "teacher_run_status": "signal_attr_sidecar_applied" if sidecar_stats is not None else "not_applicable_raw_source",
        "accepted_detection_count": len(detections),
        "det_file_status": "nonempty" if detections else "empty",
        "lane_count": len(lanes),
        "stop_line_count": len(stop_lines),
        "crosswalk_count": len(crosswalks),
        "traffic_light_count": (
            int(sidecar_stats.traffic_light_count)
            if sidecar_stats is not None
            else sum(1 for detection in detections if str(detection.get("class_name")) == "traffic_light")
        ),
        "tl_attr_coverage_count": int(sidecar_stats.traffic_light_count) if sidecar_stats is not None else 0,
        "tl_attr_valid_count": int(sidecar_stats.valid_count) if sidecar_stats is not None else 0,
        "tl_attr_invalid_count": int(sidecar_stats.invalid_count) if sidecar_stats is not None else 0,
        "tl_attr_reason_counts": dict(sidecar_stats.reason_counts) if sidecar_stats is not None else {},
        "tl_attr_status": (
            "not_available"
            if sidecar_stats is None
            else "complete"
            if int(sidecar_stats.invalid_count) == 0
            else "partial"
        ),
        "held_label_count": len(held_rows),
        "failure_count": 0,
    }
    return manifest_row, class_counts, held_rows, sidecar_stats, tl_attr_debug_rows


def _validate_and_summarize_tl_attr_rows(scene: Mapping[str, Any], *, sample: EtriDryRunSample) -> list[dict[str, Any]]:
    detections = scene.get("detections")
    traffic_lights = scene.get("traffic_lights")
    if not isinstance(detections, list):
        raise EtriMaterializationError(f"ETRI scene detections must be a list: {sample.sample_id}")
    if not isinstance(traffic_lights, list):
        raise EtriMaterializationError(f"ETRI scene traffic_lights must be a list after tl_attr teacher: {sample.sample_id}")
    traffic_detection_ids = {
        index for index, detection in enumerate(detections) if str(detection.get("class_name") or "") == "traffic_light"
    }
    seen_detection_ids: set[int] = set()
    rows: list[dict[str, Any]] = []
    for row_index, item in enumerate(traffic_lights):
        if not isinstance(item, Mapping):
            raise EtriMaterializationError(f"ETRI traffic_lights[{row_index}] must be an object: {sample.sample_id}")
        try:
            detection_id = int(item.get("detection_id"))
        except (TypeError, ValueError) as exc:
            raise EtriMaterializationError(
                f"ETRI traffic_lights[{row_index}].detection_id must be an integer: {sample.sample_id}"
            ) from exc
        if detection_id not in traffic_detection_ids:
            raise EtriMaterializationError(
                f"ETRI traffic_lights[{row_index}].detection_id must point to traffic_light detection: {sample.sample_id}"
            )
        if detection_id in seen_detection_ids:
            raise EtriMaterializationError(f"duplicate ETRI tl_attr detection_id: {sample.sample_id}:{detection_id}")
        seen_detection_ids.add(detection_id)
        meta = item.get("meta") if isinstance(item.get("meta"), Mapping) else {}
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_image_path": str(sample.image_path),
                "source_label_path": str(sample.semantic_label_path),
                "detection_id": detection_id,
                "bbox": item.get("bbox"),
                "tl_attr_valid": int(item.get("tl_attr_valid", 0)),
                "collapse_reason": str(item.get("collapse_reason") or ""),
                "tl_bits": item.get("tl_bits"),
                "base_color": item.get("base_color"),
                "arrow": item.get("arrow"),
                "base_color_confidence": item.get("base_color_confidence"),
                "arrow_probability": item.get("arrow_probability"),
                "crop_box": meta.get("crop_box"),
                "clipped_box": meta.get("clipped_box"),
            }
        )
    missing_detection_ids = sorted(traffic_detection_ids - seen_detection_ids)
    if missing_detection_ids:
        raise EtriMaterializationError(
            f"ETRI tl_attr teacher must emit one row per traffic_light detection: "
            f"{sample.sample_id} missing={missing_detection_ids}"
        )
    return rows


def _assert_tl_attr_stats_match_rows(stats: Any, rows: Sequence[Mapping[str, Any]], *, sample: EtriDryRunSample) -> None:
    traffic_light_count = int(getattr(stats, "traffic_light_count"))
    valid_count = int(getattr(stats, "valid_count"))
    invalid_count = int(getattr(stats, "invalid_count"))
    row_valid_count = sum(int(row.get("tl_attr_valid", 0)) for row in rows)
    if traffic_light_count != len(rows):
        raise EtriMaterializationError(
            f"ETRI tl_attr teacher stats traffic_light_count mismatch: "
            f"{sample.sample_id} stats={traffic_light_count} rows={len(rows)}"
        )
    if valid_count != row_valid_count or invalid_count != len(rows) - row_valid_count:
        raise EtriMaterializationError(
            f"ETRI tl_attr teacher stats valid/invalid mismatch: "
            f"{sample.sample_id} stats=({valid_count},{invalid_count}) rows=({row_valid_count},{len(rows) - row_valid_count})"
        )


def _sidecar_checkpoint_path(signal_attr_sidecar: Any | None, fallback: Path | None) -> Path | None:
    if fallback is not None:
        return Path(fallback).resolve()
    if signal_attr_sidecar is None:
        return None
    checkpoint_path = getattr(signal_attr_sidecar, "checkpoint_path", None)
    if checkpoint_path is None:
        return None
    return Path(checkpoint_path).resolve()


def _annotation_raw_label(annotation: Mapping[str, Any]) -> str:
    for key in _CLASS_KEYS:
        value = annotation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise EtriMaterializationError("ETRI annotation is missing objects[].label/class field")


def _annotation_bbox(
    annotation: dict[str, Any],
    *,
    width: int,
    height: int,
    sample: EtriDryRunSample,
    raw_label: str,
) -> list[float]:
    bbox = extract_bbox(annotation, width, height)
    if bbox is not None:
        return bbox
    points = _annotation_points(
        annotation,
        width=width,
        height=height,
        sample=sample,
        raw_label=raw_label,
        min_points=2,
    )
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise EtriMaterializationError(f"invalid bbox geometry for {sample.sample_id} label={raw_label!r}")
    return [round(value, 3) for value in bbox]


def _annotation_points(
    annotation: dict[str, Any],
    *,
    width: int,
    height: int,
    sample: EtriDryRunSample,
    raw_label: str,
    min_points: int,
) -> list[list[float]]:
    points = extract_points(annotation)
    if not points:
        for geometry_key in ("polyline", "polygon"):
            geometry = annotation.get(geometry_key)
            if isinstance(geometry, list):
                points = geometry
                break
    if not points:
        bbox = extract_bbox(annotation, width, height)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if len(points) < min_points:
        raise EtriMaterializationError(
            f"invalid vector geometry for {sample.sample_id} label={raw_label!r}: "
            f"expected at least {min_points} points"
        )
    cleaned: list[list[float]] = []
    for point_index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise EtriMaterializationError(
                f"invalid point geometry for {sample.sample_id} label={raw_label!r} point={point_index}"
            )
        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError) as exc:
            raise EtriMaterializationError(
                f"nonfinite point geometry for {sample.sample_id} label={raw_label!r} point={point_index}"
            ) from exc
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise EtriMaterializationError(
                f"nonfinite point geometry for {sample.sample_id} label={raw_label!r} point={point_index}"
            )
        cleaned.append([
            round(max(0.0, min(x_value, float(width))), 3),
            round(max(0.0, min(y_value, float(height))), 3),
        ])
    return cleaned


def _lane_centerline_points(
    points: list[list[float]],
    *,
    width: int,
    height: int,
    sample: EtriDryRunSample,
    raw_label: str,
) -> list[list[float]]:
    polygon = _dedupe_consecutive_points(points)
    if len(polygon) < 3:
        if len(polygon) < 2:
            raise EtriMaterializationError(
                f"invalid lane centerline geometry for {sample.sample_id} label={raw_label!r}: "
                "expected at least 2 points"
            )
        return polygon
    centerline = _polygon_row_slice_centerline(polygon, width=width, height=height)
    if len(centerline) < 2:
        centerline = _bbox_centerline_points(polygon, width=width, height=height)
    if len(centerline) < 2:
        raise EtriMaterializationError(
            f"invalid lane centerline geometry for {sample.sample_id} label={raw_label!r}: "
            "could not derive centerline from polygon"
        )
    return centerline


def _stop_line_centerline_points(
    points: list[list[float]],
    *,
    width: int,
    height: int,
    sample: EtriDryRunSample,
    raw_label: str,
) -> list[list[float]]:
    centerline = [
        _clamped_point(float(point[0]), float(point[1]), width=width, height=height)
        for point in canonicalize_stop_line_points(points).reshape(-1, 2).tolist()
    ]
    centerline = _dedupe_consecutive_points(centerline)
    if len(centerline) < 2:
        raise EtriMaterializationError(
            f"invalid stop_line centerline geometry for {sample.sample_id} label={raw_label!r}: "
            "expected at least 2 points"
        )
    return centerline[:2]


def _dedupe_consecutive_points(points: list[list[float]]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in points:
        if cleaned and abs(cleaned[-1][0] - point[0]) <= 1.0e-6 and abs(cleaned[-1][1] - point[1]) <= 1.0e-6:
            continue
        cleaned.append([float(point[0]), float(point[1])])
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= 1.0e-6 and abs(cleaned[0][1] - cleaned[-1][1]) <= 1.0e-6:
        cleaned.pop()
    return cleaned


def _polygon_row_slice_centerline(
    polygon: list[list[float]],
    *,
    width: int,
    height: int,
) -> list[list[float]]:
    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    axis = "y" if y_span >= 1.0 else "x"
    span = y_span if axis == "y" else x_span
    if span <= 1.0e-6:
        return []
    target_count = min(
        ETRI_LANE_CENTERLINE_MAX_POINTS,
        max(2, int(math.ceil(span / ETRI_LANE_CENTERLINE_SAMPLE_STEP_PX)) + 1),
    )
    coordinates = _scan_coordinates(min(y_values) if axis == "y" else min(x_values), max(y_values) if axis == "y" else max(x_values), target_count)
    centerline: list[list[float]] = []
    for coordinate in coordinates:
        intersections = _polygon_scanline_intersections(polygon, coordinate, axis=axis)
        if len(intersections) < 2:
            continue
        if len(intersections) % 2:
            intersections = intersections[:-1]
        if len(intersections) < 2:
            continue
        intervals = [
            (intersections[index], intersections[index + 1])
            for index in range(0, len(intersections) - 1, 2)
        ]
        start, end = max(intervals, key=lambda item: item[1] - item[0])
        midpoint = (start + end) * 0.5
        if axis == "y":
            centerline.append(_clamped_point(midpoint, coordinate, width=width, height=height))
        else:
            centerline.append(_clamped_point(coordinate, midpoint, width=width, height=height))
    centerline = _dedupe_consecutive_points(centerline)
    centerline.sort(key=lambda point: (-point[1], point[0]))
    return centerline


def _scan_coordinates(min_value: float, max_value: float, target_count: int) -> list[float]:
    if target_count <= 1:
        return [(min_value + max_value) * 0.5]
    span = max_value - min_value
    if span <= 1.0e-6:
        return [(min_value + max_value) * 0.5]
    margin = min(max(span * 0.02, 1.0e-3), span * 0.25)
    start = min_value + margin
    end = max_value - margin
    if end <= start:
        return [(min_value + max_value) * 0.5]
    step = (end - start) / float(target_count - 1)
    return [start + step * float(index) for index in range(target_count)]


def _polygon_scanline_intersections(
    polygon: list[list[float]],
    coordinate: float,
    *,
    axis: str,
) -> list[float]:
    intersections: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        start_axis = start[1] if axis == "y" else start[0]
        end_axis = end[1] if axis == "y" else end[0]
        if abs(start_axis - end_axis) <= 1.0e-9:
            continue
        if not ((start_axis <= coordinate < end_axis) or (end_axis <= coordinate < start_axis)):
            continue
        ratio = (coordinate - start_axis) / (end_axis - start_axis)
        start_cross = start[0] if axis == "y" else start[1]
        end_cross = end[0] if axis == "y" else end[1]
        intersections.append(start_cross + ratio * (end_cross - start_cross))
    intersections.sort()
    return intersections


def _bbox_centerline_points(
    points: list[list[float]],
    *,
    width: int,
    height: int,
) -> list[list[float]]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    if y2 - y1 >= 1.0:
        x = (x1 + x2) * 0.5
        return [_clamped_point(x, y2, width=width, height=height), _clamped_point(x, y1, width=width, height=height)]
    if x2 - x1 >= 1.0:
        y = (y1 + y2) * 0.5
        return [_clamped_point(x1, y, width=width, height=height), _clamped_point(x2, y, width=width, height=height)]
    return []


def _clamped_point(x_value: float, y_value: float, *, width: int, height: int) -> list[float]:
    return [
        round(max(0.0, min(float(x_value), float(width))), 3),
        round(max(0.0, min(float(y_value), float(height))), 3),
    ]


def _bbox_to_mapping(bbox: list[float]) -> dict[str, float]:
    return {
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3]),
    }


def _det_label_text(detections: list[dict[str, Any]], *, width: int, height: int) -> str:
    rows = []
    for detection in detections:
        bbox = detection["bbox"]
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        center_x = ((x1 + x2) * 0.5) / float(width)
        center_y = ((y1 + y2) * 0.5) / float(height)
        box_w = (x2 - x1) / float(width)
        box_h = (y2 - y1) / float(height)
        rows.append(
            f"{OD_CLASS_TO_ID[str(detection['class_name'])]} "
            f"{center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"
        )
    return ("\n".join(rows) + "\n") if rows else ""


def _resolve_default_split(default_split: str | None) -> str | None:
    if default_split is None:
        return None
    normalized = normalize_text(default_split)
    if normalized not in VALID_SPLITS:
        raise ValueError(f"default_split must be one of {VALID_SPLITS}, got {default_split!r}")
    return normalized


def _resolve_split(image_path: Path, label_path: Path, *, default_split: str | None) -> str | None:
    image_split = _infer_split(image_path)
    label_split = _infer_split(label_path)
    if image_split is not None and label_split is not None and image_split != label_split:
        return None
    if image_split is not None:
        return image_split
    if label_split is not None:
        return label_split
    return default_split


def _infer_split(path: Path) -> str | None:
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    for part in path.parts:
        split = aliases.get(normalize_text(part))
        if split is not None:
            return split
    return None


def _select_semantic_label_for_image(image_path: Path, label_paths: list[Path]) -> Path | None:
    if not label_paths:
        return None
    image_split = _infer_split(image_path)
    if image_split is None:
        return sorted(label_paths)[0]
    same_split = [label_path for label_path in label_paths if _infer_split(label_path) == image_split]
    if same_split:
        return sorted(same_split)[0]
    unspecified = [label_path for label_path in label_paths if _infer_split(label_path) is None]
    if unspecified:
        return sorted(unspecified)[0]
    return sorted(label_paths)[0]


def _resolve_image_size(image_path: Path, label_raw: dict[str, Any] | None) -> tuple[int, int] | None:
    metadata_size = _image_size_from_metadata(label_raw)
    try:
        probed_size = probe_image_size(image_path)
    except Exception:
        return metadata_size
    if probed_size is not None:
        return probed_size
    return metadata_size


def _image_size_from_metadata(raw: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(raw, dict):
        return None
    for key in _IMAGE_SIZE_KEYS:
        parsed = _parse_size(raw.get(key))
        if parsed is not None:
            return parsed
    for section_key in _IMAGE_SECTION_KEYS:
        section = raw.get(section_key)
        if not isinstance(section, dict):
            continue
        parsed = _parse_size(section)
        if parsed is not None:
            return parsed
        for key in _IMAGE_SIZE_KEYS:
            parsed = _parse_size(section.get(key))
            if parsed is not None:
                return parsed
    return None


def _parse_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if width is not None and height is not None:
            return int(width), int(height)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
    return None


def _load_json_label(label_path: Path) -> dict[str, Any] | None:
    if label_path.suffix.lower() != ".json":
        return None
    try:
        raw = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EtriCandidateError(CANDIDATE_REASON_LABEL_PARSE_FAILURE, str(exc)) from exc
    if not isinstance(raw, dict):
        raise EtriCandidateError(CANDIDATE_REASON_LABEL_PARSE_FAILURE, "semantic JSON root must be a mapping")
    return raw


def _label_metadata_sample_id(raw: dict[str, Any] | None) -> str | None:
    if not isinstance(raw, dict):
        return None
    for key in _LABEL_SAMPLE_ID_KEYS:
        value = raw.get(key)
        if value:
            return sample_id_from_path(Path(str(value)))
    for section_key in _IMAGE_SECTION_KEYS:
        section = raw.get(section_key)
        if not isinstance(section, dict):
            continue
        for key in ("file_name", "filename", "name", *_LABEL_SAMPLE_ID_KEYS):
            value = section.get(key)
            if value:
                return sample_id_from_path(Path(str(value)))
    return None


def _extract_raw_class_counts(label_path: Path, raw: dict[str, Any] | None) -> Counter[str]:
    if isinstance(raw, dict):
        counter = Counter()
        _collect_raw_classes(raw, counter)
        return counter
    if label_path.suffix.lower() == ".json":
        return Counter()
    return _extract_mask_value_counts(label_path)


def _collect_raw_classes(value: Any, counter: Counter[str]) -> None:
    if isinstance(value, dict):
        for key in _CLASS_KEYS:
            raw_class = value.get(key)
            if isinstance(raw_class, str) and raw_class.strip():
                counter[raw_class.strip()] += 1
                break
        for child in value.values():
            _collect_raw_classes(child, counter)
    elif isinstance(value, list):
        for child in value:
            _collect_raw_classes(child, counter)


def _extract_mask_value_counts(label_path: Path) -> Counter[str]:
    if Image is None:
        return Counter()
    try:
        with Image.open(label_path) as image:
            grayscale = image.convert("L")
            return Counter(f"pixel_value:{value}" for value in grayscale.getdata())
    except Exception:
        return Counter()


def _has_semantic_label_marker(path: Path) -> bool:
    stem = normalize_text(path.stem)
    stem_markers = ("semantic", "label", "gtfine", "gtcoarse", "mask")
    if any(marker in stem for marker in stem_markers):
        return True
    parent_markers = {"semantic", "semantics", "label", "labels", "gtfine", "gtcoarse", "mask", "masks"}
    return any(normalize_text(part) in parent_markers for part in path.parts[:-1])


def _safe_relative_stem(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).with_suffix("")
    except ValueError:
        relative = path.with_suffix("")
    return str(relative)


def _append_candidate_failure(
    failures: list[dict[str, Any]],
    *,
    reason: str,
    image_path: Path,
    sample_id: str,
    label_path: Path | None = None,
    detail: str | None = None,
) -> None:
    failure: dict[str, Any] = {
        "reason": reason,
        "sample_id": sample_id,
        "image_path": str(image_path),
    }
    if label_path is not None:
        failure["semantic_label_path"] = str(label_path)
    if detail:
        failure["detail"] = detail
    failures.append(failure)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.od_bootstrap.source.etri_kcity.dry_run",
        description="Run the ETRI KCity leftImg dry-run release gate.",
    )
    parser.add_argument("dataset_root", type=Path, help="Raw ETRI KCity dataset root to scan.")
    parser.add_argument("--output", type=Path, default=None, help="Write the ready dry-run manifest to this path.")
    parser.add_argument("--default-split", choices=VALID_SPLITS, default=None, help="Split to use when raw paths do not include one.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = scan_dry_run(Path(args.dataset_root), default_split=args.default_split)
    manifest = result.to_manifest()
    print(json.dumps(manifest, indent=2, ensure_ascii=True, default=str))
    if not result.is_ready:
        return 1
    if args.output is not None:
        write_json(Path(args.output), manifest)
    return 0


__all__ = [
    "BLOCKED_STATUS",
    "CANDIDATE_REASON_INVALID_SPLIT",
    "CANDIDATE_REASON_LABEL_PARSE_FAILURE",
    "CANDIDATE_REASON_MISSING_IMAGE_SIZE",
    "CANDIDATE_REASON_MISSING_SEMANTIC_LABEL",
    "CANDIDATE_REASON_SAMPLE_ID_MISMATCH",
    "DATASET_KEY",
    "ATTRPSEUDO_DATASET_KEY",
    "ATTRPSEUDO_SOURCE_KIND",
    "MULTICAMERA_ATTRPSEUDO_DATASET_KEY",
    "MULTICAMERA_DATASET_KEY",
    "ETRI_SAFE_LABEL_MAPPING",
    "EtriMaterializationError",
    "EtriCandidateError",
    "EtriDryRunNotReadyError",
    "EtriDryRunResult",
    "EtriDryRunSample",
    "HELD_LABEL_REASON_UNMAPPED",
    "HELD_LABELS_NAME",
    "TL_ATTR_TEACHER_DEBUG_NAME",
    "RAW_SCAN_REASON_LIDAR",
    "RAW_SCAN_REASON_MONO_CAMERA",
    "RAW_SCAN_REASON_PV26_OUTPUT",
    "RAW_SCAN_REASON_RIGHT_IMG",
    "MULTICAMERA_ATTRPSEUDO_SOURCE_KIND",
    "MULTICAMERA_SOURCE_KIND",
    "RELEASE_PATH_TOKEN",
    "ATTRPSEUDO_RELEASE_VERSION",
    "RELEASE_VERSION",
    "READY_STATUS",
    "RELEASE_BLOCKER_ZERO_SAMPLES",
    "VALID_SPLITS",
    "build_dry_run_manifest",
    "build_dry_run_sample",
    "build_ready_dry_run_manifest",
    "dry_run_release_blockers",
    "is_dry_run_ready",
    "is_kcity_leftimg_candidate",
    "is_leftimg_candidate",
    "is_semantic_label_candidate",
    "main",
    "materialize_kcity_val_release",
    "raw_scan_ignored_reason",
    "require_dry_run_ready",
    "sample_id_from_path",
    "scan_dry_run",
    "write_dry_run_manifest",
    "write_ready_dry_run_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
