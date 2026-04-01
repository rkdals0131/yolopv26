from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .debug_vis import DEFAULT_DEBUG_VIS_SEED, generate_canonical_debug_vis
from model.preprocess.aihub_standardize import (
    DEFAULT_DOCS_ROOT as DEFAULT_AIHUB_DOCS_ROOT,
    DEFAULT_LANE_ROOT as DEFAULT_AIHUB_LANE_ROOT,
    DEFAULT_OBSTACLE_ROOT as DEFAULT_AIHUB_OBSTACLE_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_AIHUB_OUTPUT_ROOT,
    DEFAULT_TRAFFIC_ROOT as DEFAULT_AIHUB_TRAFFIC_ROOT,
    run_standardization as run_aihub_standardization,
)
from model.preprocess.bdd100k_standardize import (
    DEFAULT_BDD_ROOT,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_LABELS_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_BDD_OUTPUT_ROOT,
    run_standardization as run_bdd_standardization,
)
from .image_list import discover_image_list_entries, write_image_list


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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_default_source_prep_config(*, output_root: Path | None = None) -> SourcePrepConfig:
    roots = SourceRoots()
    resolved_output_root = Path(output_root or DEFAULT_AIHUB_OUTPUT_ROOT.parent / "pv26_od_bootstrap").resolve()
    return SourcePrepConfig(roots=roots, output_root=resolved_output_root)


def _resolve_aihub_roots(roots: SourceRoots) -> tuple[Path, Path, Path, Path]:
    aihub_root = roots.aihub_root.resolve()
    lane_root = (roots.aihub_lane_root or (aihub_root / AIHUB_LANE_DIRNAME)).resolve()
    obstacle_root = (roots.aihub_obstacle_root or (aihub_root / AIHUB_OBSTACLE_DIRNAME)).resolve()
    traffic_root = (roots.aihub_traffic_root or (aihub_root / AIHUB_TRAFFIC_DIRNAME)).resolve()
    docs_root = (roots.aihub_docs_root or (aihub_root / "docs")).resolve()
    return lane_root, obstacle_root, traffic_root, docs_root


def prepare_od_bootstrap_sources(config: SourcePrepConfig) -> SourcePrepResult:
    bdd_root = config.roots.bdd_root.resolve()
    bdd_images_root = config.roots.bdd_images_root.resolve()
    bdd_labels_root = config.roots.bdd_labels_root.resolve()
    aihub_root = config.roots.aihub_root.resolve()
    lane_root, obstacle_root, traffic_root, docs_root = _resolve_aihub_roots(config.roots)
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    canonical_root = output_root / "canonical"
    bdd_output_root = canonical_root / "bdd100k_det_100k"
    aihub_output_root = canonical_root / "aihub_standardized"

    bdd_outputs = run_bdd_standardization(
        bdd_root=bdd_root,
        images_root=bdd_images_root,
        labels_root=bdd_labels_root,
        output_root=bdd_output_root,
        workers=config.workers,
        debug_vis_count=config.debug_vis_count,
        force_reprocess=config.force_reprocess,
        write_dataset_readme=config.write_source_readmes,
    )
    aihub_outputs = run_aihub_standardization(
        lane_root=lane_root,
        obstacle_root=obstacle_root,
        traffic_root=traffic_root,
        docs_root=docs_root if docs_root.is_dir() else DEFAULT_AIHUB_DOCS_ROOT,
        output_root=aihub_output_root,
        workers=config.workers,
        debug_vis_count=config.debug_vis_count,
        force_reprocess=config.force_reprocess,
        write_dataset_readmes=config.write_source_readmes,
    )

    manifest = {
        "version": "od-bootstrap-source-prep-v1",
        "generated_at": _now_iso(),
        "raw_roots": {
            "bdd_root": str(bdd_root),
            "aihub_root": str(aihub_root),
            "aihub_lane_root": str(lane_root),
            "aihub_obstacle_root": str(obstacle_root),
            "aihub_traffic_root": str(traffic_root),
        },
        "canonical_roots": {
            "bdd_root": str(bdd_output_root),
            "aihub_root": str(aihub_output_root),
        },
        "bootstrap_source_keys": list(BOOTSTRAP_SOURCE_KEYS),
        "excluded_source_keys": list(EXCLUDED_SOURCE_KEYS),
        "workers": int(config.workers),
        "force_reprocess": bool(config.force_reprocess),
        "write_source_readmes": bool(config.write_source_readmes),
        "debug_vis_count": int(config.debug_vis_count),
        "debug_vis_seed": int(config.debug_vis_seed),
    }
    manifest_path = _write_json(output_root / "meta" / "source_prep_manifest.json", manifest)

    bundle = CanonicalSourceBundle(
        bdd_root=bdd_output_root,
        aihub_root=aihub_output_root,
        output_root=output_root,
    )
    image_list_entries = discover_image_list_entries(
        (bdd_output_root, aihub_output_root),
        allowed_dataset_keys=BOOTSTRAP_SOURCE_KEYS,
    )
    image_list_manifest_path = write_image_list(output_root / "meta" / "bootstrap_image_list.jsonl", image_list_entries)
    canonical_debug_vis_outputs = generate_canonical_debug_vis(
        image_list_manifest_path=image_list_manifest_path,
        canonical_root=canonical_root,
        debug_vis_count=int(config.debug_vis_count),
        debug_vis_seed=int(config.debug_vis_seed),
    )
    return SourcePrepResult(
        bundle=bundle,
        manifest_path=manifest_path,
        image_list_manifest_path=image_list_manifest_path,
        canonical_debug_vis_manifest_paths={
            dataset_name: Path(str(payload["debug_vis_manifest"]))
            for dataset_name, payload in canonical_debug_vis_outputs.items()
        },
        bdd_outputs=bdd_outputs,
        aihub_outputs=aihub_outputs,
    )
