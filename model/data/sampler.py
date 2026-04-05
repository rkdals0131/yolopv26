from __future__ import annotations

import math
import random
from dataclasses import dataclass

from torch.utils.data import BatchSampler, DataLoader

from .dataset import (
    PV26CanonicalDataset,
    collate_pv26_encoded_batch,
    collate_pv26_encoded_eval_batch,
    collate_pv26_samples,
)


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


def build_pv26_train_dataloader(
    dataset: PV26CanonicalDataset,
    *,
    batch_size: int,
    num_batches: int | None = None,
    ratios: dict[str, float] | None = None,
    split: str | None = "train",
    seed: int = 26,
    num_workers: int = 0,
    pin_memory: bool = False,
    encode_batches: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    sampler = PV26BalancedBatchSampler(
        dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        ratios=ratios,
        split=split,
        seed=seed,
    )
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
    return DataLoader(dataset, **loader_kwargs)


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
    "PV26RandomSubsetBatchSampler",
    "PV26SequentialBatchSampler",
    "build_pv26_eval_dataloader",
    "build_pv26_train_dataloader",
    "dataset_group_for_key",
]
