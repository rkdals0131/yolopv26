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

from tools.od_bootstrap.preprocess.sources import CanonicalSourceBundle
from tools.od_bootstrap.preprocess.teacher_dataset import build_teacher_datasets


DEFAULT_CONFIG_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "preprocess" / "teacher_datasets.default.yaml"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build teacher training datasets from canonical OD sources.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to a teacher dataset YAML file.")
    parser.add_argument("--canonical-root", type=Path, default=None, help="Override bootstrap canonical root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override teacher dataset output root.")
    parser.add_argument("--copy-images", action="store_true", help="Override with copy_images=true.")
    return parser


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("teacher dataset config root must be a mapping")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    payload = _load_payload(config_path)
    base_dir = config_path.parent
    input_payload = dict(payload.get("input") or {})
    runtime_payload = dict(payload.get("runtime") or {})

    canonical_root = _resolve_path(args.canonical_root or input_payload.get("canonical_root"), base_dir=base_dir)
    output_root = _resolve_path(args.output_root or payload.get("output_root"), base_dir=base_dir)
    copy_images = bool(args.copy_images or runtime_payload.get("copy_images", False))

    bundle = CanonicalSourceBundle(
        bdd_root=canonical_root / "canonical" / "bdd100k_det_100k",
        aihub_root=canonical_root / "canonical" / "aihub_standardized",
        output_root=canonical_root,
    )
    results = build_teacher_datasets(bundle, output_root, copy_images=copy_images)
    print(
        json.dumps(
            {
                teacher_name: {
                    "dataset_root": str(result.dataset_root),
                    "manifest_path": str(result.manifest_path),
                    "data_yaml_path": str(result.data_yaml_path),
                    "sample_count": result.sample_count,
                    "detection_count": result.detection_count,
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
