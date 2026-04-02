from __future__ import annotations

from dataclasses import dataclass

from .raw_common import PairRecord


@dataclass(frozen=True)
class StandardizeTask:
    dataset_kind: str
    output_dataset_key: str
    pair: PairRecord
    output_root: str
