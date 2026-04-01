from __future__ import annotations

from pathlib import Path

from tools.od_bootstrap.common import resolve_path
from tools.od_bootstrap.data.source_prep import (
    BOOTSTRAP_SOURCE_KEYS,
    EXCLUDED_SOURCE_KEYS,
    CanonicalSourceBundle,
    SourcePrepConfig,
    SourcePrepResult,
    SourceRoots,
    build_default_source_prep_config,
    prepare_od_bootstrap_sources,
)


def resolve_source_path(value: str | Path, *, base_dir: Path) -> Path:
    return resolve_path(value, base_dir=base_dir)


__all__ = [
    "BOOTSTRAP_SOURCE_KEYS",
    "EXCLUDED_SOURCE_KEYS",
    "CanonicalSourceBundle",
    "SourcePrepConfig",
    "SourcePrepResult",
    "SourceRoots",
    "build_default_source_prep_config",
    "prepare_od_bootstrap_sources",
    "resolve_source_path",
]
