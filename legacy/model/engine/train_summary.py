from __future__ import annotations

from typing import Any


def resolve_summary_path(summary: dict[str, Any], path: str) -> float:
    current: Any = summary
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"summary path not found: {path}")
        current = current[part]
    return float(current)


__all__ = ["resolve_summary_path"]
