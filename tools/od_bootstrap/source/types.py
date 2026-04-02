from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

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


class DebugVisSummaryRow(TypedDict):
    dataset_key: str
    split: str
    sample_id: str
    scene_path: str
    image_path: str
    det_count: int
    traffic_light_count: int
    traffic_sign_count: int
    lane_count: int
    stop_line_count: int
    crosswalk_count: int


class DebugVisManifestItem(TypedDict):
    dataset_key: str
    split: str
    sample_id: str
    scene_path: str
    image_path: str
    output_path: str


class DebugVisManifest(TypedDict):
    generated_at: str
    selection_count: int
    seed: int
    items: list[DebugVisManifestItem]


class DebugVisOutputs(TypedDict):
    debug_vis_dir: Path
    debug_vis_index: Path


__all__ = [
    "BOOTSTRAP_SOURCE_KEYS",
    "EXCLUDED_SOURCE_KEYS",
    "CanonicalSourceBundle",
    "DEFAULT_AIHUB_DOCS_ROOT",
    "DEFAULT_DEBUG_VIS_SEED",
    "DebugVisManifest",
    "DebugVisManifestItem",
    "DebugVisOutputs",
    "DebugVisSummaryRow",
    "SourcePrepConfig",
    "SourcePrepResult",
    "SourceRoots",
]
