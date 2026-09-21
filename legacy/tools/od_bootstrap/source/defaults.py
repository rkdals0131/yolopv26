from __future__ import annotations

from pathlib import Path

from common.paths import resolve_path

from .constants import (
    DEFAULT_AIHUB_OUTPUT_ROOT,
    DEFAULT_BDD_IMAGES_ROOT,
    DEFAULT_BDD_LABELS_ROOT,
    DEFAULT_BDD_ROOT,
)
from .types import SourcePrepConfig, SourceRoots


def build_default_source_prep_config(*, output_root: Path | None = None) -> SourcePrepConfig:
    roots = SourceRoots(
        bdd_root=DEFAULT_BDD_ROOT,
        bdd_images_root=DEFAULT_BDD_IMAGES_ROOT,
        bdd_labels_root=DEFAULT_BDD_LABELS_ROOT,
    )
    resolved_output_root = Path(output_root or DEFAULT_AIHUB_OUTPUT_ROOT.parent / "pv26_od_bootstrap").resolve()
    return SourcePrepConfig(roots=roots, output_root=resolved_output_root)


def resolve_source_path(value: str | Path, *, base_dir: Path) -> Path:
    return resolve_path(value, base_dir=base_dir)


__all__ = [
    "build_default_source_prep_config",
    "resolve_source_path",
]
