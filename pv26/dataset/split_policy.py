from __future__ import annotations

import hashlib


def stable_split_for_group_key(
    group_key: str,
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> str:
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("invalid split ratios")

    h = hashlib.sha1(f"{seed}::{group_key}".encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / float(2**64 - 1)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"

