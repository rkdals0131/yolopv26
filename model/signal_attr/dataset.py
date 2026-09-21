"""Materialize product SignalAttr crops from raw AIHub traffic labels."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import math
import re
import time
from typing import Any, Callable, Mapping

from PIL import Image

from common.io import read_json, write_json_sorted, write_jsonl_sorted
from model.data.dataset import FocusedSource, _source_records
from .aihub_policy import extract_product_signal_attr_target
from .crop import DEFAULT_SIGNAL_ATTR_CROP_CONFIG, SignalAttrCropConfig, crop_signal_attr_roi, signal_attr_crop_config_to_dict


PRODUCT_SIGNAL_ATTR_DATASET_VERSION = "signal-attr-aihub-product-crops-v1"
SOURCE_DATASET = "aihub_traffic_seoul"


@dataclass(frozen=True)
class _Pair:
    split: str
    image_path: Path
    label_path: Path
    relative_id: str


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._") or "sample"


def _pairs(dataset_root: Path, max_samples_per_split: int | None) -> list[_Pair]:
    if max_samples_per_split is not None and max_samples_per_split < 1:
        raise ValueError("max_samples_per_split must be positive")
    source = FocusedSource("aihub_traffic", dataset_root, "traffic")
    pairs: list[_Pair] = []
    for split in ("train", "val"):
        for record in _source_records(source, split, max_samples_per_split):
            relative_id = _slug(str(record.label_path.relative_to(dataset_root).with_suffix("")))
            pairs.append(_Pair(split, record.image_path, record.label_path, relative_id))
    return pairs


def _bbox(annotation: Mapping[str, Any], width: int, height: int) -> list[float] | None:
    raw_box = annotation.get("box") or annotation.get("bbox")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw_box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    clipped = [
        max(0.0, min(x1, float(width))), max(0.0, min(y1, float(height))),
        max(0.0, min(x2, float(width))), max(0.0, min(y2, float(height))),
    ]
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _materialize_pair(
    pair: _Pair, output_root: Path, *, crop_config: SignalAttrCropConfig, all_off_is_valid: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_json(pair.label_path)
    annotations = raw.get("annotation")
    if not isinstance(annotations, list):
        raise ValueError(f"AIHub traffic annotation missing: {pair.label_path}")
    source_id = _slug(f"{SOURCE_DATASET}_{pair.split}_{pair.relative_id}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    traffic_index = 0
    with Image.open(pair.image_path) as image:
        image.load()
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping) or str(annotation.get("class") or "").lower() != "traffic_light":
                continue
            label = extract_product_signal_attr_target(annotation, all_off_is_valid=all_off_is_valid)
            row = {
                "source_dataset": SOURCE_DATASET,
                "split": pair.split,
                "source_sample_id": source_id,
                "source_relative_id": pair.relative_id,
                "source_image_path": str(pair.image_path),
                "source_label_path": str(pair.label_path),
                "source_image_file_name": pair.image_path.name,
                "annotation_index": annotation_index,
                "traffic_light_index": traffic_index,
                "raw_class": "traffic_light",
                "type": annotation.get("type"),
                "light_type": label.light_type,
                "left_arrow": label.left_arrow,
                "tl_bits": label.tl_bits,
                "base_color": label.base_color,
                "arrow": label.left_arrow,
                "collapse_reason": label.reason,
            }
            traffic_index += 1
            bbox = _bbox(annotation, image.width, image.height)
            row["bbox"] = bbox
            if bbox is None:
                rejected.append({**row, "reject_reason": "traffic_light_invalid_bbox"})
                continue
            if not label.state_valid:
                rejected.append({**row, "reject_reason": label.reason})
                continue
            crop = crop_signal_attr_roi(image, bbox, config=crop_config)
            if not crop.valid or crop.crop_image is None:
                rejected.append({**row, "reject_reason": crop.reason})
                continue
            crop_id = f"{source_id}_tl{row['traffic_light_index']:04d}"
            relative_crop = Path("images") / pair.split / f"{crop_id}.jpg"
            destination = output_root / relative_crop
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.crop_image.save(destination, format="JPEG", quality=95)
            accepted.append({
                **row, "sample_id": crop_id, "crop_path": relative_crop.as_posix(),
                "crop_box": list(crop.crop_box), "clipped_box": list(crop.clipped_box),
            })
    return accepted, rejected


def materialize_product_signal_attr_crop_dataset_from_root(
    dataset_root: Path,
    output_root: Path,
    *,
    all_off_is_valid: bool,
    crop_config: SignalAttrCropConfig = DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    workers: int = 1,
    log_every: int = 500,
    log_fn: Callable[[str], None] | None = None,
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    """Generate the existing crop JSONL format, limited by raw images per split if requested."""
    if workers < 1 or log_every < 1:
        raise ValueError("workers and log_every must be positive")
    root, output = Path(dataset_root), Path(output_root)
    pairs = _pairs(root, max_samples_per_split)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rejected: list[dict[str, Any]] = []
    completed = 0
    started = time.monotonic()

    def consume(result: tuple[list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        nonlocal completed
        accepted_rows, rejected_rows = result
        for row in accepted_rows:
            rows[row["split"]].append(row)
        rejected.extend(rejected_rows)
        completed += 1
        if log_fn is not None and (completed == 1 or completed == len(pairs) or completed % log_every == 0):
            log_fn(f"[signal_attr] crops {completed}/{len(pairs)} accepted={sum(map(len, rows.values()))} "
                   f"rejected={len(rejected)} elapsed={time.monotonic() - started:.1f}s")

    if workers == 1:
        for pair in pairs:
            consume(_materialize_pair(pair, output, crop_config=crop_config, all_off_is_valid=all_off_is_valid))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="signal_attr_crop") as executor:
            for start in range(0, len(pairs), workers * 2):
                jobs = [
                    executor.submit(
                        _materialize_pair, pair, output, crop_config=crop_config,
                        all_off_is_valid=all_off_is_valid,
                    )
                    for pair in pairs[start:start + workers * 2]
                ]
                for job in as_completed(jobs):
                    consume(job.result())

    for split in rows:
        rows[split].sort(key=lambda row: (row["source_sample_id"], row["traffic_light_index"]))
        write_jsonl_sorted(output / "labels" / f"{split}.jsonl", rows[split])
    rejected.sort(key=lambda row: (row["split"], row["source_sample_id"], row["traffic_light_index"]))
    write_jsonl_sorted(output / "meta" / "rejected_rows.jsonl", rejected)
    crop_payload = signal_attr_crop_config_to_dict(crop_config)
    write_json_sorted(output / "meta" / "crop_config.json", crop_payload)
    reason_counts = dict(sorted(Counter(str(row["reject_reason"]) for row in rejected).items()))
    rejected_by_split = {split: sum(row["split"] == split for row in rejected) for split in rows}
    reason_counts_by_split = {
        split: dict(sorted(Counter(str(row["reject_reason"]) for row in rejected if row["split"] == split).items()))
        for split in rows
    }
    combo_counts_by_split = {
        split: dict(sorted(Counter(
            "+".join(bit for bit, on in row["tl_bits"].items() if on) or "off" for row in items
        ).items()))
        for split, items in rows.items()
    }
    accepted_count = sum(map(len, rows.values()))
    manifest = {
        "version": PRODUCT_SIGNAL_ATTR_DATASET_VERSION,
        "status": "ready" if accepted_count else "empty",
        "source_dataset": SOURCE_DATASET,
        "input_format": "raw_pairs",
        "input_root": str(root),
        "output_root": str(output),
        "splits": ["train", "val"],
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "accepted_count_by_split": {split: len(items) for split, items in rows.items()},
        "rejected_count_by_split": rejected_by_split,
        "reject_reason_counts": reason_counts,
        "reject_reason_counts_by_split": reason_counts_by_split,
        "tl_combo_counts_by_split": combo_counts_by_split,
        "crop_config": crop_payload,
        "crop_config_path": "meta/crop_config.json",
        "rejected_rows_path": "meta/rejected_rows.jsonl",
        "label_paths": {split: f"labels/{split}.jsonl" for split in rows},
        "image_dirs": {split: f"images/{split}" for split in rows},
        "state_semantics": "left_arrow",
        "all_off_is_valid": bool(all_off_is_valid),
    }
    write_json_sorted(output / "meta" / "signal_attr_dataset_manifest.json", manifest)
    return manifest


__all__ = ["PRODUCT_SIGNAL_ATTR_DATASET_VERSION", "materialize_product_signal_attr_crop_dataset_from_root"]
