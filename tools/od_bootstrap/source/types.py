from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.paths import resolve_path
DEFAULT_DEBUG_VIS_SEED = 26

from .aihub import (
    DEFAULT_DOCS_ROOT as DEFAULT_AIHUB_DOCS_ROOT,
    DEFAULT_LANE_ROOT as DEFAULT_AIHUB_LANE_ROOT,
    DEFAULT_OBSTACLE_ROOT as DEFAULT_AIHUB_OBSTACLE_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_AIHUB_OUTPUT_ROOT,
    DEFAULT_TRAFFIC_ROOT as DEFAULT_AIHUB_TRAFFIC_ROOT,
)
from .bdd100k import (
    DEFAULT_BDD_ROOT,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_LABELS_ROOT,
)


AIHUB_LANE_DIRNAME = DEFAULT_AIHUB_LANE_ROOT.name
AIHUB_OBSTACLE_DIRNAME = DEFAULT_AIHUB_OBSTACLE_ROOT.name
AIHUB_TRAFFIC_DIRNAME = DEFAULT_AIHUB_TRAFFIC_ROOT.name
BOOTSTRAP_SOURCE_KEYS = ("bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul")
EXCLUDED_SOURCE_KEYS = ("aihub_lane_seoul",)


@dataclass(frozen=True)
class SourceRoots:
    bdd_root: Path = DEFAULT_BDD_ROOT
    bdd_images_root: Path = DEFAULT_IMAGES_ROOT
    bdd_labels_root: Path = DEFAULT_LABELS_ROOT
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


def build_default_source_prep_config(*, output_root: Path | None = None) -> SourcePrepConfig:
    roots = SourceRoots()
    resolved_output_root = Path(output_root or DEFAULT_AIHUB_OUTPUT_ROOT.parent / "pv26_od_bootstrap").resolve()
    return SourcePrepConfig(roots=roots, output_root=resolved_output_root)


def resolve_source_path(value: str | Path, *, base_dir: Path) -> Path:
    return resolve_path(value, base_dir=base_dir)


__all__ = [
    "AIHUB_LANE_DIRNAME",
    "AIHUB_OBSTACLE_DIRNAME",
    "AIHUB_TRAFFIC_DIRNAME",
    "BOOTSTRAP_SOURCE_KEYS",
    "EXCLUDED_SOURCE_KEYS",
    "CanonicalSourceBundle",
    "DEFAULT_AIHUB_DOCS_ROOT",
    "DEFAULT_DEBUG_VIS_SEED",
    "SourcePrepConfig",
    "SourcePrepResult",
    "SourceRoots",
    "build_default_source_prep_config",
    "resolve_source_path",
]
