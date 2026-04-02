from __future__ import annotations

from typing import Any

from .raw_common import (
    _extract_annotations,
    _normalize_text,
    _safe_slug,
)


def extract_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_annotations(raw)


def normalize_text(value: Any) -> str:
    return _normalize_text(value)


def safe_slug(value: str) -> str:
    return _safe_slug(value)


__all__ = [
    "extract_annotations",
    "normalize_text",
    "safe_slug",
]
