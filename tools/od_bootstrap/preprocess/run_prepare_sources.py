from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.common import resolve_path
from tools.od_bootstrap.preprocess.sources import SourcePrepConfig, SourceRoots, prepare_od_bootstrap_sources


DEFAULT_CONFIG_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "preprocess" / "sources.default.yaml"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare canonical OD bootstrap sources.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to a source prep YAML file.")
    parser.add_argument("--bdd-root", type=Path, default=None, help="Override BDD100K raw root.")
    parser.add_argument("--aihub-root", type=Path, default=None, help="Override AIHUB raw root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override bootstrap output root.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count.")
    parser.add_argument("--force-reprocess", action="store_true", help="Override with force_reprocess=true.")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("source prep config root must be a mapping")
    return payload


def _resolve_config(args: argparse.Namespace) -> SourcePrepConfig:
    config_path = Path(args.config).resolve()
    payload = _load_config(config_path)
    base_dir = config_path.parent
    raw = dict(payload.get("raw") or {})
    runtime = dict(payload.get("runtime") or {})

    bdd_root = resolve_path(args.bdd_root or raw.get("bdd_root"), base_dir=base_dir)
    aihub_root = resolve_path(args.aihub_root or raw.get("aihub_root"), base_dir=base_dir)
    output_root = resolve_path(args.output_root or payload.get("output_root"), base_dir=base_dir)
    workers = int(args.workers if args.workers is not None else runtime.get("workers", 1))
    force_reprocess = bool(args.force_reprocess or runtime.get("force_reprocess", False))
    return SourcePrepConfig(
        roots=SourceRoots(
            bdd_root=bdd_root,
            bdd_images_root=bdd_root / "bdd100k_images_100k" / "100k",
            bdd_labels_root=bdd_root / "bdd100k_labels" / "100k",
            aihub_root=aihub_root,
        ),
        output_root=output_root,
        workers=workers,
        force_reprocess=force_reprocess,
        write_source_readmes=bool(runtime.get("write_source_readmes", False)),
        debug_vis_count=int(runtime.get("debug_vis_count", 0)),
        debug_vis_seed=int(runtime.get("debug_vis_seed", 26)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _resolve_config(args)
    result = prepare_od_bootstrap_sources(config)
    print(
        json.dumps(
            {
                "bundle": {
                    "bdd_root": str(result.bundle.bdd_root),
                    "aihub_root": str(result.bundle.aihub_root),
                    "output_root": str(result.bundle.output_root),
                    "bootstrap_source_keys": list(result.bundle.bootstrap_source_keys),
                    "excluded_source_keys": list(result.bundle.excluded_source_keys),
                },
                "manifest_path": str(result.manifest_path),
                "image_list_manifest_path": str(result.image_list_manifest_path),
                "canonical_debug_vis_manifest_paths": {
                    dataset_name: str(path)
                    for dataset_name, path in result.canonical_debug_vis_manifest_paths.items()
                },
                "bdd_output_root": str(result.bdd_outputs["output_root"]),
                "aihub_output_root": str(result.aihub_outputs["output_root"]),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
