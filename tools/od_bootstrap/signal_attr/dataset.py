from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

from common.io import write_json_sorted, write_jsonl_sorted
from common.pv26_schema import AIHUB_TRAFFIC_DATASET_KEY, TL_BITS

from ..source.raw_common import TRAFFIC_DATASET_KEY, PairRecord
from ..source.shared.io import load_json
from ..source.shared.raw import discover_pairs, extract_annotations, extract_bbox, normalize_text, safe_slug
from ..source.shared.scene import sample_id as source_sample_id
from .aihub_policy import AIHUB_TL_INVALID_REASONS, collapse_aihub_traffic_light_attr, combo_name
from .crop import (
    DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    SIGNAL_ATTR_CROP_REASON_INVALID_ROI,
    SignalAttrCropConfig,
    SignalAttrCropResult,
    crop_signal_attr_roi,
    signal_attr_crop_config_to_dict,
)

SIGNAL_ATTR_DATASET_VERSION = "signal-attr-aihub-crops-v1"
SIGNAL_ATTR_DATASET_STATUS_READY = "ready"
SIGNAL_ATTR_DATASET_STATUS_EMPTY = "empty"
SIGNAL_ATTR_INVALID_BBOX_REASON = "traffic_light_invalid_bbox"
SIGNAL_ATTR_CANONICAL_INVALID_REASON = "canonical_tl_attr_invalid"
SIGNAL_ATTR_DATASET_REJECT_REASONS = (
    *AIHUB_TL_INVALID_REASONS,
    SIGNAL_ATTR_CANONICAL_INVALID_REASON,
    SIGNAL_ATTR_INVALID_BBOX_REASON,
    SIGNAL_ATTR_CROP_REASON_INVALID_ROI,
)
SIGNAL_ATTR_DATASET_SPLITS = ("train", "val")


@dataclass(frozen=True)
class SignalAttrDatasetPaths:
    output_root: Path
    manifest_path: Path
    crop_config_path: Path
    rejected_rows_path: Path
    label_paths: Mapping[str, Path]


def materialize_aihub_signal_attr_crop_dataset_from_root(
    dataset_root: Path,
    output_root: Path,
    *,
    source_dataset: str = AIHUB_TRAFFIC_DATASET_KEY,
    crop_config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    splits: Sequence[str] = SIGNAL_ATTR_DATASET_SPLITS,
    workers: int = 1,
    log_every: int = 500,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report = discover_pairs(TRAFFIC_DATASET_KEY, Path(dataset_root))
    return materialize_aihub_signal_attr_crop_dataset(
        report.pairs,
        output_root,
        source_dataset=source_dataset,
        crop_config=crop_config,
        splits=splits,
        workers=workers,
        log_every=log_every,
        log_fn=log_fn,
    )


def materialize_aihub_signal_attr_crop_dataset_from_canonical_root(
    canonical_root: Path,
    output_root: Path,
    *,
    source_dataset: str = AIHUB_TRAFFIC_DATASET_KEY,
    crop_config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    splits: Sequence[str] = SIGNAL_ATTR_DATASET_SPLITS,
    workers: int = 1,
    log_every: int = 500,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    split_names = _normalize_splits(splits)
    worker_count = _positive_int(workers, field_name="signal_attr.workers")
    progress_every = _positive_int(log_every, field_name="signal_attr.log_every")
    label_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    rejected_rows: list[dict[str, Any]] = []

    for split in split_names:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)

    scene_paths = _iter_canonical_scene_paths(Path(canonical_root), split_names)
    total_items = len(scene_paths)
    if log_fn is not None:
        log_fn(
            f"[teacher:signal_attr] dataset start input=canonical_scene samples={total_items} "
            f"workers={worker_count}"
        )
    start_time = time.monotonic()
    completed = 0
    accepted_total = 0
    rejected_total = 0

    def _consume(scene_path: Path, label_rows: list[dict[str, Any]], scene_rejections: list[dict[str, Any]]) -> None:
        nonlocal accepted_total, rejected_total
        split = scene_path.parent.name
        label_rows_by_split[split].extend(label_rows)
        rejected_rows.extend(scene_rejections)
        accepted_total += len(label_rows)
        rejected_total += len(scene_rejections)

    if total_items:
        if worker_count == 1:
            for scene_path in scene_paths:
                label_rows, scene_rejections = _materialize_canonical_scene_rows(
                    scene_path,
                    Path(canonical_root),
                    output,
                    source_dataset=source_dataset,
                    crop_config=crop_config,
                )
                _consume(scene_path, label_rows, scene_rejections)
                completed += 1
                _log_dataset_progress(
                    log_fn,
                    completed=completed,
                    total=total_items,
                    accepted=accepted_total,
                    rejected=rejected_total,
                    started_at=start_time,
                    log_every=progress_every,
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="signal_attr_dataset") as executor:
                future_to_scene = {
                    executor.submit(
                        _materialize_canonical_scene_rows,
                        scene_path,
                        Path(canonical_root),
                        output,
                        source_dataset=source_dataset,
                        crop_config=crop_config,
                    ): scene_path
                    for scene_path in scene_paths
                }
                for future in as_completed(future_to_scene):
                    scene_path = future_to_scene[future]
                    label_rows, scene_rejections = future.result()
                    _consume(scene_path, label_rows, scene_rejections)
                    completed += 1
                    _log_dataset_progress(
                        log_fn,
                        completed=completed,
                        total=total_items,
                        accepted=accepted_total,
                        rejected=rejected_total,
                        started_at=start_time,
                        log_every=progress_every,
                    )

    crop_config_payload = signal_attr_crop_config_to_dict(crop_config)
    paths = _dataset_paths(output, split_names)

    _sort_materialized_rows(label_rows_by_split, rejected_rows)
    for split, rows in label_rows_by_split.items():
        write_jsonl_sorted(paths.label_paths[split], rows)
    write_jsonl_sorted(paths.rejected_rows_path, rejected_rows)
    write_json_sorted(paths.crop_config_path, crop_config_payload)

    manifest = _build_manifest(
        output,
        source_dataset=source_dataset,
        splits=split_names,
        crop_config=crop_config_payload,
        label_rows_by_split=label_rows_by_split,
        rejected_rows=rejected_rows,
        paths=paths,
        input_root=Path(canonical_root),
        input_format="canonical_scene",
    )
    write_json_sorted(paths.manifest_path, manifest)
    if log_fn is not None:
        elapsed = max(time.monotonic() - start_time, 1.0e-6)
        log_fn(
            f"[teacher:signal_attr] dataset done samples={total_items} accepted={manifest['accepted_count']} "
            f"rejected={manifest['rejected_count']} elapsed={elapsed:.1f}s"
        )
    return manifest


def materialize_aihub_signal_attr_crop_dataset(
    pairs: Iterable[PairRecord],
    output_root: Path,
    *,
    source_dataset: str = AIHUB_TRAFFIC_DATASET_KEY,
    crop_config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    splits: Sequence[str] = SIGNAL_ATTR_DATASET_SPLITS,
    workers: int = 1,
    log_every: int = 500,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    split_names = _normalize_splits(splits)
    worker_count = _positive_int(workers, field_name="signal_attr.workers")
    progress_every = _positive_int(log_every, field_name="signal_attr.log_every")
    split_set = set(split_names)
    label_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    rejected_rows: list[dict[str, Any]] = []

    for split in split_names:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)

    sorted_pairs = _sorted_pairs(pair for pair in pairs if pair.split in split_set)
    total_items = len(sorted_pairs)
    if log_fn is not None:
        log_fn(
            f"[teacher:signal_attr] dataset start input=raw_pairs samples={total_items} "
            f"workers={worker_count}"
        )
    start_time = time.monotonic()
    completed = 0
    accepted_total = 0
    rejected_total = 0

    def _consume(pair: PairRecord, label_rows: list[dict[str, Any]], pair_rejections: list[dict[str, Any]]) -> None:
        nonlocal accepted_total, rejected_total
        label_rows_by_split[pair.split].extend(label_rows)
        rejected_rows.extend(pair_rejections)
        accepted_total += len(label_rows)
        rejected_total += len(pair_rejections)

    if total_items:
        if worker_count == 1:
            for pair in sorted_pairs:
                label_rows, pair_rejections = _materialize_pair_rows(
                    pair,
                    output,
                    source_dataset=source_dataset,
                    crop_config=crop_config,
                )
                _consume(pair, label_rows, pair_rejections)
                completed += 1
                _log_dataset_progress(
                    log_fn,
                    completed=completed,
                    total=total_items,
                    accepted=accepted_total,
                    rejected=rejected_total,
                    started_at=start_time,
                    log_every=progress_every,
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="signal_attr_dataset") as executor:
                future_to_pair = {
                    executor.submit(
                        _materialize_pair_rows,
                        pair,
                        output,
                        source_dataset=source_dataset,
                        crop_config=crop_config,
                    ): pair
                    for pair in sorted_pairs
                }
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    label_rows, pair_rejections = future.result()
                    _consume(pair, label_rows, pair_rejections)
                    completed += 1
                    _log_dataset_progress(
                        log_fn,
                        completed=completed,
                        total=total_items,
                        accepted=accepted_total,
                        rejected=rejected_total,
                        started_at=start_time,
                        log_every=progress_every,
                    )

    crop_config_payload = signal_attr_crop_config_to_dict(crop_config)
    paths = _dataset_paths(output, split_names)

    _sort_materialized_rows(label_rows_by_split, rejected_rows)
    for split, rows in label_rows_by_split.items():
        write_jsonl_sorted(paths.label_paths[split], rows)
    write_jsonl_sorted(paths.rejected_rows_path, rejected_rows)
    write_json_sorted(paths.crop_config_path, crop_config_payload)

    manifest = _build_manifest(
        output,
        source_dataset=source_dataset,
        splits=split_names,
        crop_config=crop_config_payload,
        label_rows_by_split=label_rows_by_split,
        rejected_rows=rejected_rows,
        paths=paths,
        input_format="raw_pairs",
    )
    write_json_sorted(paths.manifest_path, manifest)
    if log_fn is not None:
        elapsed = max(time.monotonic() - start_time, 1.0e-6)
        log_fn(
            f"[teacher:signal_attr] dataset done samples={total_items} accepted={manifest['accepted_count']} "
            f"rejected={manifest['rejected_count']} elapsed={elapsed:.1f}s"
        )
    return manifest


def _materialize_pair_rows(
    pair: PairRecord,
    output_root: Path,
    *,
    source_dataset: str,
    crop_config: SignalAttrCropConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pair.image_path is None:
        raise ValueError(f"AIHUB signal attr pair is missing image path: {pair.label_path}")

    raw = load_json(pair.label_path)
    source_id = source_sample_id(source_dataset, pair, safe_slug=safe_slug)
    label_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    traffic_light_index = 0

    with Image.open(pair.image_path) as image:
        image.load()
        width = int(image.width)
        height = int(image.height)
        for annotation_index, annotation in enumerate(extract_annotations(raw)):
            raw_class = normalize_text(annotation.get("class"))
            if raw_class not in {"traffic_light", "light"}:
                continue

            row_base = _row_base(
                pair,
                source_dataset=source_dataset,
                source_id=source_id,
                annotation=annotation,
                annotation_index=annotation_index,
                traffic_light_index=traffic_light_index,
            )
            traffic_light_index += 1

            label = collapse_aihub_traffic_light_attr(annotation)
            row_base.update(_label_payload(label.tl_bits, label.base_color, label.arrow, label.collapse_reason))
            bbox = extract_bbox(annotation, width, height)
            if bbox is None:
                rejected_rows.append(
                    {
                        **row_base,
                        "bbox": None,
                        "reject_reason": SIGNAL_ATTR_INVALID_BBOX_REASON,
                    }
                )
                continue
            row_base["bbox"] = bbox

            if not label.tl_attr_valid:
                rejected_rows.append({**row_base, "reject_reason": label.collapse_reason})
                continue

            crop = crop_signal_attr_roi(image, bbox, config=crop_config)
            if not crop.valid or crop.crop_image is None:
                rejected_rows.append({**row_base, **_crop_payload(crop), "reject_reason": crop.reason})
                continue

            crop_id = f"{source_id}_tl{row_base['traffic_light_index']:04d}"
            crop_relpath = Path("images") / pair.split / f"{crop_id}.jpg"
            crop_output_path = output_root / crop_relpath
            crop_output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.crop_image.save(crop_output_path, format="JPEG", quality=95)
            label_rows.append(
                {
                    **row_base,
                    **_crop_payload(crop),
                    "sample_id": crop_id,
                    "crop_path": crop_relpath.as_posix(),
                }
            )

    return label_rows, rejected_rows


def _iter_canonical_scene_paths(canonical_root: Path, splits: Sequence[str]) -> list[Path]:
    labels_scene_root = Path(canonical_root) / "labels_scene"
    if not labels_scene_root.is_dir():
        return []
    split_order = {split: index for index, split in enumerate(splits)}
    paths: list[Path] = []
    split_set = set(splits)
    for scene_path in labels_scene_root.rglob("*.json"):
        if scene_path.parent.name in split_set:
            paths.append(scene_path)
    return sorted(
        paths,
        key=lambda item: (
            split_order.get(item.parent.name, len(split_order)),
            item.parent.name,
            item.stem,
            str(item),
        ),
    )


def _materialize_canonical_scene_rows(
    scene_path: Path,
    canonical_root: Path,
    output_root: Path,
    *,
    source_dataset: str,
    crop_config: SignalAttrCropConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene = load_json(scene_path)
    if not isinstance(scene, Mapping):
        raise TypeError(f"canonical signal attr scene must be an object: {scene_path}")
    source = scene.get("source") if isinstance(scene.get("source"), Mapping) else {}
    scene_source_dataset = str(source.get("dataset") or "").strip()
    if scene_source_dataset != source_dataset:
        return [], []
    split = scene_path.parent.name
    source_split = str(source.get("split") or split).strip()
    if source_split != split:
        raise ValueError(f"canonical scene source.split must match labels_scene split: {scene_path}")

    image_path = _canonical_scene_image_path(scene, scene_path=scene_path, canonical_root=canonical_root, split=split)
    traffic_lights = scene.get("traffic_lights", [])
    if not isinstance(traffic_lights, Sequence) or isinstance(traffic_lights, (str, bytes)):
        raise TypeError(f"canonical scene traffic_lights must be a list: {scene_path}")

    label_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    source_id = scene_path.stem
    with Image.open(image_path) as image:
        image.load()
        for traffic_light_index, item in enumerate(traffic_lights):
            if not isinstance(item, Mapping):
                raise TypeError(f"canonical scene traffic_lights[{traffic_light_index}] must be an object: {scene_path}")
            row_base = _canonical_row_base(
                scene_path,
                image_path,
                split=split,
                source_dataset=source_dataset,
                source_id=source_id,
                traffic_light=item,
                traffic_light_index=traffic_light_index,
            )
            label = _canonical_label_payload(item)
            row_base.update(label)
            bbox = _canonical_traffic_light_bbox(scene, item)
            if bbox is None:
                rejected_rows.append(
                    {
                        **row_base,
                        "bbox": None,
                        "reject_reason": SIGNAL_ATTR_INVALID_BBOX_REASON,
                    }
                )
                continue
            row_base["bbox"] = bbox

            if not int(item.get("tl_attr_valid", 0)):
                rejected_rows.append(
                    {
                        **row_base,
                        "reject_reason": _canonical_reject_reason(str(item.get("collapse_reason") or "")),
                    }
                )
                continue
            if label["base_color"] == "multi":
                rejected_rows.append({**row_base, "reject_reason": SIGNAL_ATTR_CANONICAL_INVALID_REASON})
                continue

            crop = crop_signal_attr_roi(image, bbox, config=crop_config)
            if not crop.valid or crop.crop_image is None:
                rejected_rows.append({**row_base, **_crop_payload(crop), "reject_reason": crop.reason})
                continue

            crop_id = f"{source_id}_tl{traffic_light_index:04d}"
            crop_relpath = Path("images") / split / f"{crop_id}.jpg"
            crop_output_path = output_root / crop_relpath
            crop_output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.crop_image.save(crop_output_path, format="JPEG", quality=95)
            label_rows.append(
                {
                    **row_base,
                    **_crop_payload(crop),
                    "sample_id": crop_id,
                    "crop_path": crop_relpath.as_posix(),
                }
            )
    return label_rows, rejected_rows


def _row_base(
    pair: PairRecord,
    *,
    source_dataset: str,
    source_id: str,
    annotation: Mapping[str, Any],
    annotation_index: int,
    traffic_light_index: int,
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "split": pair.split,
        "source_sample_id": source_id,
        "source_relative_id": pair.relative_id,
        "source_image_path": str(pair.image_path),
        "source_label_path": str(pair.label_path),
        "source_image_file_name": pair.image_file_name,
        "annotation_index": int(annotation_index),
        "traffic_light_index": int(traffic_light_index),
        "raw_class": normalize_text(annotation.get("class")),
        "type": annotation.get("type"),
    }


def _canonical_scene_image_path(scene: Mapping[str, Any], *, scene_path: Path, canonical_root: Path, split: str) -> Path:
    image_payload = scene.get("image") if isinstance(scene.get("image"), Mapping) else {}
    file_name = str(image_payload.get("file_name") or "").strip()
    if not file_name:
        raise ValueError(f"canonical scene image.file_name missing: {scene_path}")
    if Path(file_name).is_absolute() or Path(file_name).name != file_name:
        raise ValueError(f"canonical scene image.file_name must be a file name: {scene_path}")
    image_path = Path(canonical_root) / "images" / split / file_name
    if not image_path.is_file():
        raise FileNotFoundError(f"canonical signal attr image missing: {image_path} ({scene_path})")
    return image_path


def _canonical_row_base(
    scene_path: Path,
    image_path: Path,
    *,
    split: str,
    source_dataset: str,
    source_id: str,
    traffic_light: Mapping[str, Any],
    traffic_light_index: int,
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "split": split,
        "source_sample_id": source_id,
        "source_relative_id": source_id,
        "source_image_path": str(image_path),
        "source_label_path": str(scene_path),
        "source_scene_path": str(scene_path),
        "source_image_file_name": image_path.name,
        "annotation_index": int(traffic_light_index),
        "traffic_light_index": int(traffic_light_index),
        "detection_id": traffic_light.get("detection_id"),
        "raw_class": "traffic_light",
        "type": traffic_light.get("type"),
    }


def _label_payload(
    tl_bits: Mapping[str, int],
    base_color: str,
    arrow: int,
    collapse_reason: str,
) -> dict[str, Any]:
    return {
        "tl_bits": {str(key): int(value) for key, value in tl_bits.items()},
        "base_color": str(base_color),
        "arrow": int(arrow),
        "collapse_reason": str(collapse_reason),
    }


def _canonical_label_payload(traffic_light: Mapping[str, Any]) -> dict[str, Any]:
    raw_bits = traffic_light.get("tl_bits")
    if not isinstance(raw_bits, Mapping):
        raw_bits = {}
    tl_bits = {bit: int(raw_bits.get(bit, 0) or 0) for bit in TL_BITS}
    active_colors = [bit for bit in ("red", "yellow", "green") if tl_bits.get(bit)]
    if len(active_colors) == 1:
        base_color = active_colors[0]
    elif active_colors:
        base_color = "multi"
    else:
        base_color = "off"
    return _label_payload(
        tl_bits,
        base_color,
        int(tl_bits.get("arrow", 0)),
        _canonical_reject_reason(str(traffic_light.get("collapse_reason") or "valid")),
    )


def _canonical_reject_reason(reason: str) -> str:
    value = str(reason).strip() or "valid"
    if value == "valid" or value in AIHUB_TL_INVALID_REASONS:
        return value
    return SIGNAL_ATTR_CANONICAL_INVALID_REASON


def _canonical_traffic_light_bbox(scene: Mapping[str, Any], traffic_light: Mapping[str, Any]) -> list[float] | None:
    bbox = _coerce_bbox(traffic_light.get("bbox"))
    if bbox is not None:
        return bbox
    detection_id = traffic_light.get("detection_id")
    if detection_id is None:
        return None
    try:
        detection_index = int(detection_id)
    except (TypeError, ValueError):
        return None
    detections = scene.get("detections", [])
    if not isinstance(detections, Sequence) or detection_index < 0 or detection_index >= len(detections):
        return None
    detection = detections[detection_index]
    if not isinstance(detection, Mapping):
        return None
    return _coerce_bbox(detection.get("bbox"))


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return bbox


def _crop_payload(crop: SignalAttrCropResult) -> dict[str, Any]:
    return {
        "crop_box": _list_or_none(crop.crop_box),
        "clipped_box": _list_or_none(crop.clipped_box),
    }


def _list_or_none(values: Sequence[Any] | None) -> list[Any] | None:
    if values is None:
        return None
    return list(values)


def _build_manifest(
    output_root: Path,
    *,
    source_dataset: str,
    splits: Sequence[str],
    crop_config: Mapping[str, Any],
    label_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    rejected_rows: Sequence[Mapping[str, Any]],
    paths: SignalAttrDatasetPaths,
    input_format: str,
    input_root: Path | None = None,
) -> dict[str, Any]:
    accepted_rows = [row for split in splits for row in label_rows_by_split[split]]
    accepted_by_split = {split: len(label_rows_by_split[split]) for split in splits}
    rejected_by_split = {split: 0 for split in splits}
    reject_reason_counts = {reason: 0 for reason in SIGNAL_ATTR_DATASET_REJECT_REASONS}
    reject_reason_counts_by_split = {
        split: {reason: 0 for reason in SIGNAL_ATTR_DATASET_REJECT_REASONS} for split in splits
    }
    for row in rejected_rows:
        split = str(row.get("split"))
        reason = str(row.get("reject_reason"))
        if reason not in reject_reason_counts:
            raise ValueError(f"unexpected signal attr reject reason: {reason}")
        if split in rejected_by_split:
            rejected_by_split[split] += 1
        reject_reason_counts[reason] += 1
        if split in reject_reason_counts_by_split and reason in reject_reason_counts_by_split[split]:
            reject_reason_counts_by_split[split][reason] += 1

    combo_counts_by_split = {split: _combo_counts(label_rows_by_split[split]) for split in splits}

    return {
        "version": SIGNAL_ATTR_DATASET_VERSION,
        "status": SIGNAL_ATTR_DATASET_STATUS_READY if accepted_rows else SIGNAL_ATTR_DATASET_STATUS_EMPTY,
        "source_dataset": source_dataset,
        "input_format": input_format,
        "input_root": str(input_root) if input_root is not None else None,
        "output_root": str(output_root),
        "splits": list(splits),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "accepted_count_by_split": accepted_by_split,
        "rejected_count_by_split": rejected_by_split,
        "reject_reason_counts": reject_reason_counts,
        "reject_reason_counts_by_split": reject_reason_counts_by_split,
        "tl_combo_counts_by_split": combo_counts_by_split,
        "closed_reject_reasons": list(SIGNAL_ATTR_DATASET_REJECT_REASONS),
        "crop_config": dict(crop_config),
        "crop_config_path": _relative_to_output(paths.crop_config_path, output_root),
        "rejected_rows_path": _relative_to_output(paths.rejected_rows_path, output_root),
        "label_paths": {
            split: _relative_to_output(path, output_root) for split, path in paths.label_paths.items()
        },
        "image_dirs": {split: f"images/{split}" for split in splits},
    }


def _combo_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        tl_bits = row.get("tl_bits")
        if isinstance(tl_bits, Mapping):
            counts[combo_name({str(key): int(value) for key, value in tl_bits.items()})] += 1
    return dict(sorted(counts.items()))


def _dataset_paths(output_root: Path, splits: Sequence[str]) -> SignalAttrDatasetPaths:
    return SignalAttrDatasetPaths(
        output_root=output_root,
        manifest_path=output_root / "meta" / "signal_attr_dataset_manifest.json",
        crop_config_path=output_root / "meta" / "crop_config.json",
        rejected_rows_path=output_root / "meta" / "rejected_rows.jsonl",
        label_paths={split: output_root / "labels" / f"{split}.jsonl" for split in splits},
    )


def _positive_int(value: int, *, field_name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return resolved


def _log_dataset_progress(
    log_fn: Callable[[str], None] | None,
    *,
    completed: int,
    total: int,
    accepted: int,
    rejected: int,
    started_at: float,
    log_every: int,
) -> None:
    if log_fn is None:
        return
    if completed != 1 and completed != total and completed % log_every != 0:
        return
    elapsed = max(time.monotonic() - started_at, 1.0e-6)
    rate = completed / elapsed
    log_fn(
        f"[teacher:signal_attr] dataset progress {completed}/{total} samples "
        f"({rate:.1f} samples/s, accepted={accepted}, rejected={rejected})"
    )


def _sort_materialized_rows(
    label_rows_by_split: Mapping[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
) -> None:
    for rows in label_rows_by_split.values():
        rows.sort(key=_signal_attr_row_sort_key)
    rejected_rows.sort(key=_signal_attr_row_sort_key)


def _signal_attr_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("source_sample_id") or ""),
        int(row.get("traffic_light_index") or 0),
        str(row.get("reject_reason") or ""),
        str(row.get("sample_id") or ""),
    )


def _relative_to_output(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for split in splits:
        value = str(split).strip()
        if not value:
            continue
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("signal attr dataset materialization requires at least one split")
    return tuple(normalized)


def _sorted_pairs(pairs: Iterable[PairRecord]) -> list[PairRecord]:
    split_order = {split: index for index, split in enumerate(SIGNAL_ATTR_DATASET_SPLITS)}
    return sorted(
        pairs,
        key=lambda pair: (
            split_order.get(pair.split, len(split_order)),
            pair.split,
            pair.relative_id,
            str(pair.label_path),
            str(pair.image_path or ""),
        ),
    )


__all__ = [
    "SIGNAL_ATTR_DATASET_REJECT_REASONS",
    "SIGNAL_ATTR_DATASET_SPLITS",
    "SIGNAL_ATTR_DATASET_STATUS_EMPTY",
    "SIGNAL_ATTR_DATASET_STATUS_READY",
    "SIGNAL_ATTR_DATASET_VERSION",
    "SIGNAL_ATTR_CANONICAL_INVALID_REASON",
    "SIGNAL_ATTR_INVALID_BBOX_REASON",
    "SignalAttrDatasetPaths",
    "materialize_aihub_signal_attr_crop_dataset",
    "materialize_aihub_signal_attr_crop_dataset_from_canonical_root",
    "materialize_aihub_signal_attr_crop_dataset_from_root",
]
