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
from tools.od_bootstrap.preprocess.sources import CanonicalSourceBundle
from tools.od_bootstrap.preprocess.teacher_dataset import build_teacher_datasets


DEFAULT_CONFIG_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "preprocess" / "teacher_datasets.default.yaml"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build teacher training datasets from canonical OD sources.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to a teacher dataset YAML file.")
    parser.add_argument("--canonical-root", type=Path, default=None, help="Override bootstrap canonical root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override teacher dataset output root.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count.")
    parser.add_argument("--log-every", type=int, default=None, help="Emit progress every N completed samples.")
    parser.add_argument("--copy-images", action="store_true", help="Override with copy_images=true.")
    return parser


def _load_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("teacher dataset config root must be a mapping")
    return payload


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return resolved


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    payload = _load_payload(config_path)
    base_dir = config_path.parent
    input_payload = dict(payload.get("input") or {})
    runtime_payload = dict(payload.get("runtime") or {})

    canonical_root = resolve_path(args.canonical_root or input_payload.get("canonical_root"), base_dir=base_dir)
    output_root = resolve_path(args.output_root or payload.get("output_root"), base_dir=base_dir)
    copy_images = bool(args.copy_images or runtime_payload.get("copy_images", False))
    workers = _coerce_positive_int(
        args.workers if args.workers is not None else runtime_payload.get("workers", 8),
        field_name="runtime.workers",
    )
    log_every = _coerce_positive_int(
        args.log_every if args.log_every is not None else runtime_payload.get("log_every", 250),
        field_name="runtime.log_every",
    )
    debug_vis_count = int(runtime_payload.get("debug_vis_count", 0))
    debug_vis_seed = int(runtime_payload.get("debug_vis_seed", 26))

    bundle = CanonicalSourceBundle(
        bdd_root=canonical_root / "canonical" / "bdd100k_det_100k",
        aihub_root=canonical_root / "canonical" / "aihub_standardized",
        output_root=canonical_root,
    )
    _log(
        f"[teacher-datasets] canonical_root={canonical_root} output_root={output_root} "
        f"workers={workers} copy_images={copy_images} log_every={log_every} "
        f"debug_vis_count={debug_vis_count} debug_vis_seed={debug_vis_seed}"
    )
    results = build_teacher_datasets(
        bundle,
        output_root,
        copy_images=copy_images,
        workers=workers,
        log_every=log_every,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        log_fn=_log,
    )
    print(
        json.dumps(
            {
                teacher_name: {
                    "dataset_root": str(result.dataset_root),
                    "manifest_path": str(result.manifest_path),
                    "debug_vis_manifest_path": str(result.debug_vis_manifest_path),
                    "sample_count": result.sample_count,
                    "detection_count": result.detection_count,
                    "class_counts": result.class_counts,
                }
                for teacher_name, result in results.items()
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
