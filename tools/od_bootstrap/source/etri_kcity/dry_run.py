from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..shared.io import now_iso, write_json
from ..shared.raw import normalize_text, probe_image_size, safe_slug
from ..shared.summary import counter_to_dict

try:
    from PIL import Image
except ImportError:  # pragma: no cover - covered by environments without PIL.
    Image = None


DATASET_KEY = "etri_kcity_multicamera_leftimg"
VALID_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
SEMANTIC_LABEL_EXTENSIONS = {".json", ".png", ".bmp", ".tif", ".tiff"}

RAW_SCAN_REASON_RIGHT_IMG = "rightImg"
RAW_SCAN_REASON_MONO_CAMERA = "MonoCamera"
RAW_SCAN_REASON_LIDAR = "lidar_annotation"

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
    "raw_class",
    "class_name",
    "class",
    "category",
    "label_name",
    "semantic_class",
    "semantic_label",
)
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


def scan_dry_run(dataset_root: Path, *, default_split: str | None = None) -> EtriDryRunResult:
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
        if is_kcity_leftimg_candidate(path):
            image_candidates.append(path)
        elif is_semantic_label_candidate(path):
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


def build_dry_run_manifest(dataset_root: Path, *, default_split: str | None = None) -> dict[str, Any]:
    return scan_dry_run(dataset_root, default_split=default_split).to_manifest()


def build_ready_dry_run_manifest(dataset_root: Path, *, default_split: str | None = None) -> dict[str, Any]:
    result = scan_dry_run(dataset_root, default_split=default_split)
    require_dry_run_ready(result)
    return result.to_manifest()


def write_dry_run_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    default_split: str | None = None,
) -> Path:
    manifest = build_dry_run_manifest(dataset_root, default_split=default_split)
    return write_json(output_path, manifest)


def write_ready_dry_run_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    default_split: str | None = None,
) -> Path:
    manifest = build_ready_dry_run_manifest(dataset_root, default_split=default_split)
    return write_json(output_path, manifest)


def build_dry_run_sample(
    *,
    image_path: Path,
    semantic_label_path: Path | None,
    dataset_root: Path,
    default_split: str | None = None,
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
    sample_id = safe_slug(f"{DATASET_KEY}_{split}_{sample_relative_id}")
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
    if "lidar" in joined or path.suffix.lower() in {".pcd", ".bin"}:
        return RAW_SCAN_REASON_LIDAR
    if "monocamera" in joined or "mono_camera" in joined or "mono-camera" in joined:
        return RAW_SCAN_REASON_MONO_CAMERA
    if "rightimg" in joined or "right_img" in joined or "right-img" in joined:
        return RAW_SCAN_REASON_RIGHT_IMG
    return None


def is_kcity_leftimg_candidate(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    normalized_parts = [normalize_text(part) for part in path.parts]
    joined = "/".join(normalized_parts)
    if "kcity" not in joined:
        return False
    if "leftimg" not in joined and "left_img" not in joined:
        return False
    if _has_semantic_label_marker(path):
        return False
    return raw_scan_ignored_reason(path) is None


def is_semantic_label_candidate(path: Path) -> bool:
    if path.suffix.lower() not in SEMANTIC_LABEL_EXTENSIONS:
        return False
    if raw_scan_ignored_reason(path) is not None:
        return False
    normalized_parts = [normalize_text(part) for part in path.parts]
    joined = "/".join(normalized_parts)
    if "kcity" not in joined:
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
    "EtriCandidateError",
    "EtriDryRunNotReadyError",
    "EtriDryRunResult",
    "EtriDryRunSample",
    "RAW_SCAN_REASON_LIDAR",
    "RAW_SCAN_REASON_MONO_CAMERA",
    "RAW_SCAN_REASON_RIGHT_IMG",
    "READY_STATUS",
    "RELEASE_BLOCKER_ZERO_SAMPLES",
    "VALID_SPLITS",
    "build_dry_run_manifest",
    "build_dry_run_sample",
    "build_ready_dry_run_manifest",
    "dry_run_release_blockers",
    "is_dry_run_ready",
    "is_kcity_leftimg_candidate",
    "is_semantic_label_candidate",
    "main",
    "raw_scan_ignored_reason",
    "require_dry_run_ready",
    "sample_id_from_path",
    "scan_dry_run",
    "write_dry_run_manifest",
    "write_ready_dry_run_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
