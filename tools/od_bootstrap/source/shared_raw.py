from __future__ import annotations

from pathlib import Path
from typing import Any

from .raw_common import (
    DiscoveryReport,
    _discover_pairs,
    _env_path,
    _extract_annotations,
    _extract_attribute_map,
    _extract_bbox,
    _extract_filename,
    _extract_image_size,
    _extract_points,
    _extract_tl_state,
    _normalize_text,
    _now_iso,
    _probe_image_size,
    _repo_root,
    _safe_slug,
    _seg_dataset_root,
)


def discover_pairs(dataset_key: str, dataset_root: Path) -> DiscoveryReport:
    return _discover_pairs(dataset_key, dataset_root)


def env_path(name: str, default: Path) -> Path:
    return _env_path(name, default)


def extract_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_annotations(raw)


def extract_attribute_map(annotation: dict[str, Any]) -> dict[str, Any]:
    return _extract_attribute_map(annotation)


def extract_bbox(annotation: dict[str, Any], width: int, height: int) -> list[float] | None:
    return _extract_bbox(annotation, width, height)


def extract_filename(raw: dict[str, Any], fallback_name: str) -> str:
    return _extract_filename(raw, fallback_name)


def extract_image_size(raw: dict[str, Any], image_path) -> tuple[int, int]:
    return _extract_image_size(raw, image_path)


def extract_points(annotation: dict[str, Any]) -> list[list[float]]:
    return _extract_points(annotation)


def extract_tl_state(annotation: dict[str, Any]) -> str:
    return _extract_tl_state(annotation)


def normalize_text(value: Any) -> str:
    return _normalize_text(value)


# Preserve raw-source UTC timestamps; this intentionally differs from common.io.now_iso.
now_iso = _now_iso


def probe_image_size(path: Path) -> tuple[int, int]:
    return _probe_image_size(path)


def repo_root() -> Path:
    return _repo_root()


def safe_slug(value: str) -> str:
    return _safe_slug(value)


def seg_dataset_root(repo_root_path: Path | None = None) -> Path:
    return _seg_dataset_root(repo_root_path)


__all__ = [
    "discover_pairs",
    "env_path",
    "extract_annotations",
    "extract_attribute_map",
    "extract_bbox",
    "extract_filename",
    "extract_image_size",
    "extract_points",
    "extract_tl_state",
    "normalize_text",
    "now_iso",
    "probe_image_size",
    "repo_root",
    "safe_slug",
    "seg_dataset_root",
]
