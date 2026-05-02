from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils.data import BatchSampler, DataLoader

from .dataset import (
    PV26CanonicalDataset,
    collate_pv26_encoded_batch,
    collate_pv26_encoded_eval_batch,
    collate_pv26_samples,
)
from .target_encoder import lane_supervised_valid_mask


DEFAULT_SAMPLER_RATIOS = {
    "bdd100k": 0.30,
    "aihub_traffic": 0.30,
    "aihub_lane": 0.25,
    "aihub_obstacle": 0.15,
}

DATASET_GROUP_BY_KEY = {
    "pv26_exhaustive_bdd100k_det_100k": "bdd100k",
    "pv26_exhaustive_aihub_traffic_seoul": "aihub_traffic",
    "pv26_exhaustive_aihub_obstacle_seoul": "aihub_obstacle",
    "bdd100k_det_100k": "bdd100k",
    "aihub_traffic_seoul": "aihub_traffic",
    "aihub_obstacle_seoul": "aihub_obstacle",
    "aihub_lane_seoul": "aihub_lane",
}


def dataset_group_for_key(dataset_key: str) -> str:
    try:
        return DATASET_GROUP_BY_KEY[dataset_key]
    except KeyError as exc:
        raise KeyError(f"unsupported dataset key for balanced sampler: {dataset_key}") from exc


@dataclass
class _GroupCursor:
    indices: list[int]
    rng: random.Random
    position: int = 0

    def draw(self) -> int:
        if not self.indices:
            raise ValueError("cannot draw from an empty group cursor")
        if self.position >= len(self.indices):
            self.rng.shuffle(self.indices)
            self.position = 0
        index = self.indices[self.position]
        self.position += 1
        return index


@dataclass
class _IndexCursor:
    indices: list[int]
    rng: random.Random
    position: int = 0

    def draw(self) -> int:
        if not self.indices:
            raise ValueError("cannot draw from an empty index cursor")
        if self.position >= len(self.indices):
            self.rng.shuffle(self.indices)
            self.position = 0
        index = self.indices[self.position]
        self.position += 1
        return index


def _batch_counts_from_ratios(
    ratios: dict[str, float],
    batch_size: int,
    available_groups: set[str],
) -> dict[str, int]:
    positive = {
        group: float(value)
        for group, value in ratios.items()
        if value > 0.0 and group in available_groups
    }
    if not positive:
        raise ValueError("balanced sampler requires at least one available group with positive ratio")

    total_weight = sum(positive.values())
    raw = {
        group: (weight / total_weight) * batch_size
        for group, weight in positive.items()
    }
    counts = {group: math.floor(value) for group, value in raw.items()}
    assigned = sum(counts.values())
    remainders = sorted(
        ((raw[group] - counts[group], group) for group in positive),
        reverse=True,
    )
    for _, group in remainders[: batch_size - assigned]:
        counts[group] += 1
    return counts


class PV26BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        num_batches: int | None = None,
        ratios: dict[str, float] | None = None,
        split: str | None = "train",
        seed: int = 26,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        self.ratios = dict(ratios or DEFAULT_SAMPLER_RATIOS)

        grouped_indices: dict[str, list[int]] = {}
        eligible_count = 0
        for index, record in enumerate(dataset.records):
            if split is not None and record.split != split:
                continue
            group = dataset_group_for_key(record.dataset_key)
            grouped_indices.setdefault(group, []).append(index)
            eligible_count += 1

        if eligible_count == 0:
            raise ValueError("balanced sampler found no eligible samples")

        self.available_groups = set(grouped_indices)
        self.batch_counts = _batch_counts_from_ratios(self.ratios, self.batch_size, self.available_groups)
        if sum(self.batch_counts.values()) != self.batch_size:
            raise AssertionError("balanced sampler batch counts must sum to batch size")

        sampled_count = sum(len(grouped_indices[group]) for group in self.batch_counts)
        if sampled_count <= 0:
            raise ValueError("balanced sampler found no samples for the configured positive-ratio groups")
        self.num_batches = num_batches or max(1, math.ceil(sampled_count / batch_size))

        base_rng = random.Random(seed)
        self._cursors: dict[str, _GroupCursor] = {}
        for group, indices in grouped_indices.items():
            shuffled = list(indices)
            base_rng.shuffle(shuffled)
            self._cursors[group] = _GroupCursor(indices=shuffled, rng=random.Random(base_rng.randint(0, 1_000_000)))
        self._shuffle_rng = random.Random(seed + 1)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            for group, count in self.batch_counts.items():
                cursor = self._cursors[group]
                for _ in range(count):
                    batch.append(cursor.draw())
            self._shuffle_rng.shuffle(batch)
            yield batch


class PV26SequentialBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        num_batches: int | None = None,
        split: str | None = "val",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = batch_size
        self.indices = [
            index
            for index, record in enumerate(dataset.records)
            if split is None or record.split == split
        ]
        if not self.indices:
            raise ValueError("eval sampler found no eligible samples")
        self.num_batches = num_batches or math.ceil(len(self.indices) / batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        max_items = min(len(self.indices), self.num_batches * self.batch_size)
        selected = self.indices[:max_items]
        for start in range(0, len(selected), self.batch_size):
            yield selected[start : start + self.batch_size]


class PV26RandomSubsetBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        num_batches: int | None = None,
        split: str | None = "val",
        seed: int = 26,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = batch_size
        indices = [
            index
            for index, record in enumerate(dataset.records)
            if split is None or record.split == split
        ]
        if not indices:
            raise ValueError("eval sampler found no eligible samples")
        rng = random.Random(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        self._cursor = _IndexCursor(indices=shuffled, rng=random.Random(rng.randint(0, 1_000_000)))
        self.num_batches = num_batches or math.ceil(len(indices) / batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        requested = min(len(self._cursor.indices), self.num_batches * self.batch_size)
        selected: list[int] = []
        seen: set[int] = set()
        while len(selected) < requested:
            candidate = self._cursor.draw()
            if candidate in seen:
                continue
            selected.append(candidate)
            seen.add(candidate)
        for start in range(0, len(selected), self.batch_size):
            yield selected[start : start + self.batch_size]


@lru_cache(maxsize=131072)
def _scene_task_flags(scene_path: str) -> dict[str, bool]:
    path = str(scene_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {"lane": False, "stop_line": False, "crosswalk": False}

    task_map = payload.get("tasks")
    if isinstance(task_map, dict):
        return {
            "lane": bool(task_map.get("has_lane", False)),
            "stop_line": bool(task_map.get("has_stop_line", False)),
            "crosswalk": bool(task_map.get("has_crosswalk", False)),
        }

    lanes = payload.get("lanes") or payload.get("lane_targets", {}).get("lanes") or []
    stop_lines = payload.get("stop_lines") or payload.get("lane_targets", {}).get("stop_lines") or []
    crosswalks = payload.get("crosswalks") or payload.get("lane_targets", {}).get("crosswalks") or []
    return {
        "lane": bool(lanes),
        "stop_line": bool(stop_lines),
        "crosswalk": bool(crosswalks),
    }


@lru_cache(maxsize=131072)
def _scene_lane_supervised(scene_path: str) -> bool:
    path = str(scene_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False
    lanes = payload.get("lanes") or payload.get("lane_targets", {}).get("lanes") or []
    if not lanes:
        return False
    normalized_rows = []
    for lane in lanes:
        normalized_rows.append(
            {
                "points_xy": lane.get("points_xy") or lane.get("points") or [],
                "visibility": lane.get("visibility"),
                "color": lane.get("color", lane.get("meta", {}).get("color", -1)),
                "lane_type": lane.get("lane_type", lane.get("meta", {}).get("lane_type", -1)),
            }
        )
    valid_mask = torch.ones((len(normalized_rows),), dtype=torch.bool)
    # Lane oversampling should prefer any scene with usable geometry, even when
    # semantic color/type labels are incomplete.
    return bool(lane_supervised_valid_mask(normalized_rows, valid_mask, require_semantics=False).any())


def _task_positive_available(record: SampleRecord, task_name: str) -> bool:
    resolved = str(task_name)
    if resolved == "lane":
        return _scene_lane_supervised(str(record.scene_path))
    return bool(_scene_task_flags(str(record.scene_path)).get(resolved, False))


class PV26TaskPositiveBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        task_name: str,
        positive_fraction: float = 0.5,
        num_batches: int | None = None,
        split: str | None = "train",
        seed: int = 26,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)
        self.task_name = str(task_name)
        fraction = max(0.0, min(1.0, float(positive_fraction)))
        self.positive_count = min(self.batch_size, max(1, int(round(self.batch_size * fraction))))
        self.negative_count = self.batch_size - self.positive_count

        positive_indices: list[int] = []
        negative_indices: list[int] = []
        for index, record in enumerate(dataset.records):
            if split is not None and record.split != split:
                continue
            if self.task_name == "lane":
                positive = _scene_lane_supervised(str(record.scene_path))
            else:
                flags = _scene_task_flags(str(record.scene_path))
                positive = bool(flags.get(self.task_name, False))
            if positive:
                positive_indices.append(index)
            else:
                negative_indices.append(index)
        if not positive_indices:
            raise ValueError(f"task-positive sampler found no positive samples for task={self.task_name!r}")
        if self.negative_count > 0 and not negative_indices:
            self.negative_count = 0
            self.positive_count = self.batch_size
        self.num_batches = num_batches or max(1, math.ceil(len(positive_indices) / max(self.positive_count, 1)))

        rng = random.Random(seed)
        pos_shuffled = list(positive_indices)
        neg_shuffled = list(negative_indices)
        rng.shuffle(pos_shuffled)
        rng.shuffle(neg_shuffled)
        self._positive_cursor = _IndexCursor(indices=pos_shuffled, rng=random.Random(rng.randint(0, 1_000_000)))
        self._negative_cursor = _IndexCursor(indices=neg_shuffled, rng=random.Random(rng.randint(0, 1_000_000))) if neg_shuffled else None
        self._shuffle_rng = random.Random(seed + 1)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.positive_count):
                batch.append(self._positive_cursor.draw())
            if self._negative_cursor is not None:
                for _ in range(self.negative_count):
                    batch.append(self._negative_cursor.draw())
            self._shuffle_rng.shuffle(batch)
            yield batch


def _canonical_positive_task_name(task_name: str) -> str:
    key = str(task_name).strip().lower()
    aliases = {
        "lane": "lane",
        "lane_only": "lane",
        "stop_line": "stop_line",
        "stopline": "stop_line",
        "stopline_only": "stop_line",
        "stop_line_only": "stop_line",
        "crosswalk": "crosswalk",
        "crosswalk_only": "crosswalk",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported task-positive task name: {task_name!r}") from exc


def _parse_task_positive_spec(task_positive_task: str | None) -> list[str]:
    if task_positive_task in (None, "", "none"):
        return []
    raw = str(task_positive_task).strip()
    lowered = raw.lower()
    if lowered.startswith("rotate:"):
        payload = raw.split(":", 1)[1]
        task_names = [_canonical_positive_task_name(item) for item in payload.split(",") if str(item).strip()]
        if not task_names:
            raise ValueError("rotate task-positive spec requires at least one task name")
        return task_names
    if lowered.startswith("multi:"):
        payload = raw.split(":", 1)[1]
        task_names = [_canonical_positive_task_name(item) for item in payload.split(",") if str(item).strip()]
        if not task_names:
            raise ValueError("multi task-positive spec requires at least one task name")
        return task_names
    return [_canonical_positive_task_name(raw)]


def _task_positive_mode(task_positive_task: str | None) -> str | None:
    if task_positive_task in (None, "", "none"):
        return None
    lowered = str(task_positive_task).strip().lower()
    if lowered.startswith("rotate:"):
        return "rotate"
    if lowered.startswith("multi:"):
        return "multi"
    return "single"


class PV26TaskPositiveRotationBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        task_names: list[str],
        positive_fraction: float = 0.5,
        num_batches: int | None = None,
        split: str | None = "train",
        seed: int = 26,
        allow_missing_tasks: bool = False,
    ) -> None:
        resolved_task_names = [_canonical_positive_task_name(item) for item in task_names]
        if not resolved_task_names:
            raise ValueError("task-positive rotation sampler requires at least one task")
        self.requested_task_names = list(resolved_task_names)
        self.unavailable_task_names: list[str] = []
        self._rotation_index = 0
        self._samplers = {}
        for index, task_name in enumerate(self.requested_task_names):
            try:
                self._samplers[task_name] = PV26TaskPositiveBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    task_name=task_name,
                    positive_fraction=positive_fraction,
                    num_batches=num_batches,
                    split=split,
                    seed=seed + (index * 101),
                )
            except ValueError:
                if not allow_missing_tasks:
                    raise
                self.unavailable_task_names.append(task_name)
        self.task_names = list(self._samplers.keys())
        if not self.task_names:
            raise ValueError("task-positive rotation sampler found no available positive tasks")
        self.current_task_name = self.task_names[0]
        self.num_batches = max(len(sampler) for sampler in self._samplers.values())

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        task_name = self.task_names[self._rotation_index % len(self.task_names)]
        self._rotation_index += 1
        self.current_task_name = task_name
        yield from iter(self._samplers[task_name])


class PV26TaskPositiveMultiBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PV26CanonicalDataset,
        *,
        batch_size: int,
        task_names: list[str],
        positive_fraction: float = 0.5,
        num_batches: int | None = None,
        split: str | None = "train",
        seed: int = 26,
        allow_missing_tasks: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        resolved_task_names = [_canonical_positive_task_name(item) for item in task_names]
        if not resolved_task_names:
            raise ValueError("task-positive multi sampler requires at least one task")
        self.requested_task_names = list(resolved_task_names)
        self.unavailable_task_names: list[str] = []
        self._task_cursors: dict[str, _IndexCursor] = {}
        self.task_names: list[str] = []
        fraction = max(0.0, min(1.0, float(positive_fraction)))
        self.batch_size = int(batch_size)

        for index, task_name in enumerate(self.requested_task_names):
            positive_indices: list[int] = []
            for record_index, record in enumerate(dataset.records):
                if split is not None and record.split != split:
                    continue
                if _task_positive_available(record, task_name):
                    positive_indices.append(record_index)
            if not positive_indices:
                if not allow_missing_tasks:
                    raise ValueError(f"task-positive multi sampler found no positive samples for task={task_name!r}")
                self.unavailable_task_names.append(task_name)
                continue
            rng = random.Random(seed + (index * 101))
            shuffled = list(positive_indices)
            rng.shuffle(shuffled)
            self._task_cursors[task_name] = _IndexCursor(indices=shuffled, rng=random.Random(rng.randint(0, 1_000_000)))
            self.task_names.append(task_name)
        if not self.task_names:
            raise ValueError("task-positive multi sampler found no available positive tasks")

        union_positive: set[int] = set()
        eligible_indices: list[int] = []
        for index, record in enumerate(dataset.records):
            if split is not None and record.split != split:
                continue
            eligible_indices.append(index)
            if any(_task_positive_available(record, task_name) for task_name in self.task_names):
                union_positive.add(index)
        negative_indices = [index for index in eligible_indices if index not in union_positive]
        self.positive_count = min(self.batch_size, max(len(self.task_names), int(round(self.batch_size * fraction))))
        self.negative_count = max(0, self.batch_size - self.positive_count)
        if self.negative_count > 0 and not negative_indices:
            self.negative_count = 0
            self.positive_count = self.batch_size
        self.num_batches = num_batches or max(1, math.ceil(len(union_positive) / max(self.positive_count, 1)))

        neg_rng = random.Random(seed + 909)
        neg_shuffled = list(negative_indices)
        neg_rng.shuffle(neg_shuffled)
        self._negative_cursor = _IndexCursor(indices=neg_shuffled, rng=random.Random(neg_rng.randint(0, 1_000_000))) if neg_shuffled else None
        self._shuffle_rng = random.Random(seed + 1)

    def __len__(self) -> int:
        return self.num_batches

    def _draw_unique(self, cursor: _IndexCursor, used: set[int]) -> int:
        candidate = cursor.draw()
        if candidate not in used:
            return candidate
        for _ in range(max(4, len(cursor.indices))):
            candidate = cursor.draw()
            if candidate not in used:
                return candidate
        return candidate

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            used: set[int] = set()
            base_quota = self.positive_count // len(self.task_names)
            remainder = self.positive_count % len(self.task_names)
            quotas = {
                task_name: base_quota + (1 if index < remainder else 0)
                for index, task_name in enumerate(self.task_names)
            }
            for task_name in self.task_names:
                cursor = self._task_cursors[task_name]
                for _ in range(quotas[task_name]):
                    candidate = self._draw_unique(cursor, used)
                    batch.append(candidate)
                    used.add(candidate)
            if self._negative_cursor is not None:
                for _ in range(self.negative_count):
                    candidate = self._draw_unique(self._negative_cursor, used)
                    batch.append(candidate)
                    used.add(candidate)
            self._shuffle_rng.shuffle(batch)
            yield batch


def build_pv26_train_dataloader(
    dataset: PV26CanonicalDataset,
    *,
    batch_size: int,
    num_batches: int | None = None,
    ratios: dict[str, float] | None = None,
    task_positive_task: str | None = None,
    task_positive_fraction: float | None = None,
    split: str | None = "train",
    seed: int = 26,
    num_workers: int = 0,
    pin_memory: bool = False,
    encode_batches: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    positive_task_names = _parse_task_positive_spec(task_positive_task)
    positive_task_mode = _task_positive_mode(task_positive_task)
    fallback_reason: str | None = None
    unavailable_positive_tasks: list[str] = []
    if positive_task_names:
        try:
            if positive_task_mode == "multi":
                sampler = PV26TaskPositiveMultiBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    task_names=positive_task_names,
                    positive_fraction=0.5 if task_positive_fraction is None else float(task_positive_fraction),
                    num_batches=num_batches,
                    split=split,
                    seed=seed,
                    allow_missing_tasks=True,
                )
                unavailable_positive_tasks = list(getattr(sampler, "unavailable_task_names", []))
            elif len(positive_task_names) == 1:
                sampler = PV26TaskPositiveBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    task_name=positive_task_names[0],
                    positive_fraction=0.5 if task_positive_fraction is None else float(task_positive_fraction),
                    num_batches=num_batches,
                    split=split,
                    seed=seed,
                )
            else:
                sampler = PV26TaskPositiveRotationBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    task_names=positive_task_names,
                    positive_fraction=0.5 if task_positive_fraction is None else float(task_positive_fraction),
                    num_batches=num_batches,
                    split=split,
                    seed=seed,
                    allow_missing_tasks=True,
                )
                unavailable_positive_tasks = list(getattr(sampler, "unavailable_task_names", []))
        except ValueError:
            fallback_reason = f"task_positive_unavailable:{task_positive_task}"
            sampler = PV26BalancedBatchSampler(
                dataset,
                batch_size=batch_size,
                num_batches=num_batches,
                ratios=ratios,
                split=split,
                seed=seed,
            )
    else:
        sampler = PV26BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            ratios=ratios,
            split=split,
            seed=seed,
        )
    if fallback_reason is None and unavailable_positive_tasks:
        fallback_reason = "task_positive_partially_unavailable:" + ",".join(str(item) for item in unavailable_positive_tasks)
    loader_kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_pv26_encoded_batch if encode_batches else collate_pv26_samples,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(dataset, **loader_kwargs)
    loader._sampling_metadata = {  # type: ignore[attr-defined]
        "requested_task_positive_task": None if task_positive_task in (None, "", "none") else str(task_positive_task),
        "requested_task_positive_mode": positive_task_mode,
        "resolved_task_positive_tasks": list(positive_task_names),
        "effective_task_positive_tasks": list(getattr(sampler, "task_names", [])) if positive_task_names else [],
        "unavailable_task_positive_tasks": list(unavailable_positive_tasks),
        "sampler_type": type(sampler).__name__,
        "fallback_reason": fallback_reason,
    }
    return loader


def build_pv26_eval_dataloader(
    dataset: PV26CanonicalDataset,
    *,
    batch_size: int,
    num_batches: int | None = None,
    split: str | None = "val",
    seed: int = 26,
    num_workers: int = 0,
    pin_memory: bool = False,
    encode_batches: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    sampler = PV26RandomSubsetBatchSampler(
        dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        split=split,
        seed=seed,
    )
    loader_kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_pv26_encoded_eval_batch if encode_batches else collate_pv26_samples,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


__all__ = [
    "DATASET_GROUP_BY_KEY",
    "DEFAULT_SAMPLER_RATIOS",
    "PV26BalancedBatchSampler",
    "PV26TaskPositiveBatchSampler",
    "PV26RandomSubsetBatchSampler",
    "PV26SequentialBatchSampler",
    "build_pv26_eval_dataloader",
    "build_pv26_train_dataloader",
    "dataset_group_for_key",
]
