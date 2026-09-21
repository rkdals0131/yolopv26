from __future__ import annotations

import math
from typing import Any


def flatten_scalar_tree(prefix: str, payload: Any) -> list[tuple[str, float]]:
    scalars: list[tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            scalars.extend(flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}/{index}" if prefix else str(index)
            scalars.extend(flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, bool):
        return [(prefix, 1.0 if payload else 0.0)]
    if isinstance(payload, (int, float)):
        numeric = float(payload)
        if math.isfinite(numeric):
            return [(prefix, numeric)]
    return scalars
