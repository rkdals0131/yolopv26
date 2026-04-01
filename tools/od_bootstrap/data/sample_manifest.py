from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

from .image_list import ImageListEntry, load_image_list, write_image_list


DEFAULT_SAMPLE_QUOTAS: dict[str, dict[str, int]] = {
    "bdd100k_det_100k": {"train": 500, "val": 200},
    "aihub_traffic_seoul": {"train": 500, "val": 200},
    "aihub_obstacle_seoul": {"train": 500, "val": 200},
}


def _normalize_sample_quotas(quotas: Mapping[str, Mapping[str, int]] | None) -> dict[str, dict[str, int]]:
    payload = quotas or DEFAULT_SAMPLE_QUOTAS
    normalized: dict[str, dict[str, int]] = {}
    for dataset_key, split_mapping in payload.items():
        dataset_name = str(dataset_key).strip()
        if not dataset_name:
            raise ValueError("dataset_key must not be empty")
        normalized[dataset_name] = {}
        for split, count in split_mapping.items():
            split_name = str(split).strip()
            quota = int(count)
            if not split_name:
                raise ValueError(f"split must not be empty for dataset {dataset_name}")
            if quota <= 0:
                raise ValueError(f"quota must be > 0 for {dataset_name}:{split_name}")
            normalized[dataset_name][split_name] = quota
    return normalized


def select_sample_entries(
    entries: list[ImageListEntry] | tuple[ImageListEntry, ...],
    *,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[ImageListEntry, ...]:
    normalized_quotas = _normalize_sample_quotas(quotas)
    selected: list[ImageListEntry] = []
    counts = Counter()
    ordered_entries = sorted(entries, key=lambda item: (item.dataset_key, item.split, item.sample_uid))
    for entry in ordered_entries:
        dataset_key = str(entry.dataset_key)
        split = str(entry.split)
        target = normalized_quotas.get(dataset_key, {}).get(split)
        if target is None:
            continue
        counter_key = (dataset_key, split)
        if counts[counter_key] >= target:
            continue
        selected.append(entry)
        counts[counter_key] += 1
    missing = []
    for dataset_key, split_mapping in normalized_quotas.items():
        for split, target in split_mapping.items():
            actual = counts[(dataset_key, split)]
            if actual < target:
                missing.append(f"{dataset_key}:{split}={actual}/{target}")
    if missing:
        raise RuntimeError(f"sample manifest is missing required entries: {', '.join(missing)}")
    return tuple(selected)


def summarize_entries(entries: list[ImageListEntry] | tuple[ImageListEntry, ...]) -> dict[str, object]:
    dataset_counts = Counter()
    split_counts = Counter()
    for entry in entries:
        dataset_counts[str(entry.dataset_key)] += 1
        split_counts[(str(entry.dataset_key), str(entry.split))] += 1
    return {
        "image_count": len(entries),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_counts": {
            f"{dataset_key}::{split}": count
            for (dataset_key, split), count in sorted(split_counts.items())
        },
    }


def build_sample_manifest(
    *,
    input_manifest_path: Path,
    output_manifest_path: Path,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    entries = load_image_list(input_manifest_path)
    selected_entries = select_sample_entries(entries, quotas=quotas)
    write_image_list(output_manifest_path, selected_entries)
    summary = summarize_entries(selected_entries)
    return {
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_manifest_path": str(Path(output_manifest_path).resolve()),
        "quotas": _normalize_sample_quotas(quotas),
        **summary,
    }


__all__ = [
    "DEFAULT_SAMPLE_QUOTAS",
    "ImageListEntry",
    "build_sample_manifest",
    "load_image_list",
    "select_sample_entries",
    "summarize_entries",
    "write_image_list",
]
