from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.finalize.final_dataset import build_pv26_exhaustive_od_lane_dataset


DEFAULT_CONFIG_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "finalize" / "pv26_exhaustive_od_lane.default.yaml"


def _log_finalize(message: str) -> None:
    print(f"[od_bootstrap.finalize] {message}", flush=True)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("finalize config root must be a mapping")
    return payload


def run_finalize_scenario(config_path: Path) -> dict[str, Any]:
    payload = _load_payload(config_path)
    base_dir = config_path.parent
    input_payload = dict(payload.get("input") or {})
    output_payload = dict(payload.get("output") or {})
    _log_finalize(f"scenario={config_path}")
    return build_pv26_exhaustive_od_lane_dataset(
        exhaustive_od_root=_resolve_path(input_payload.get("exhaustive_od_root"), base_dir=base_dir),
        aihub_canonical_root=_resolve_path(input_payload.get("aihub_canonical_root"), base_dir=base_dir),
        output_root=_resolve_path(output_payload.get("root"), base_dir=base_dir),
        copy_images=bool(output_payload.get("copy_images", False)),
        log_fn=_log_finalize,
    )


def load_and_run_default_finalize() -> dict[str, Any]:
    return run_finalize_scenario(DEFAULT_CONFIG_PATH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the final PV26 exhaustive OD + lane dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a finalize YAML file.")
    args = parser.parse_args(argv)
    run_finalize_scenario(Path(args.config).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
