from __future__ import annotations

from collections import Counter


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


__all__ = [
    "counter_to_dict",
]
