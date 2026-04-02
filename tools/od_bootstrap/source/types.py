from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    BOOTSTRAP_SOURCE_KEYS,
    DEFAULT_AIHUB_DOCS_ROOT,
    DEFAULT_AIHUB_OUTPUT_ROOT,
    DEFAULT_BDD_IMAGES_ROOT,
    DEFAULT_BDD_LABELS_ROOT,
    DEFAULT_BDD_ROOT,
    DEFAULT_DEBUG_VIS_SEED,
    EXCLUDED_SOURCE_KEYS,
)


@dataclass(frozen=True)
class SourceRoots:
    bdd_root: Path = DEFAULT_BDD_ROOT
    bdd_images_root: Path = DEFAULT_BDD_IMAGES_ROOT
    bdd_labels_root: Path = DEFAULT_BDD_LABELS_ROOT
    aihub_root: Path = DEFAULT_AIHUB_OUTPUT_ROOT.parent / "AIHUB"
    aihub_lane_root: Path | None = None
    aihub_obstacle_root: Path | None = None
    aihub_traffic_root: Path | None = None
    aihub_docs_root: Path | None = None


@dataclass(frozen=True)
class SourcePrepConfig:
    roots: SourceRoots
    output_root: Path
    workers: int = 1
    force_reprocess: bool = False
    write_source_readmes: bool = False
    debug_vis_count: int = 0
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED


@dataclass(frozen=True)
class CanonicalSourceBundle:
    bdd_root: Path
    aihub_root: Path
    output_root: Path
    bootstrap_source_keys: tuple[str, ...] = BOOTSTRAP_SOURCE_KEYS
    excluded_source_keys: tuple[str, ...] = EXCLUDED_SOURCE_KEYS


@dataclass(frozen=True)
class SourcePrepResult:
    bundle: CanonicalSourceBundle
    manifest_path: Path
    image_list_manifest_path: Path
    canonical_debug_vis_manifest_paths: dict[str, Path]
    bdd_outputs: dict[str, Path]
    aihub_outputs: dict[str, Path]


__all__ = [
    "BOOTSTRAP_SOURCE_KEYS",
    "EXCLUDED_SOURCE_KEYS",
    "CanonicalSourceBundle",
    "DEFAULT_AIHUB_DOCS_ROOT",
    "DEFAULT_DEBUG_VIS_SEED",
    "SourcePrepConfig",
    "SourcePrepResult",
    "SourceRoots",
]
