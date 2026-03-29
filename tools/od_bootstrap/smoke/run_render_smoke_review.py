from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.smoke.review import render_review_bundle


DEFAULT_MANIFEST_PATH = (
    REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset_smoke" / "meta" / "final_dataset_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "od_bootstrap_smoke" / "review" / "final_dataset"


def _parse_quota_args(items: list[str] | None) -> dict[str, int] | None:
    if not items:
        return None
    quotas: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"quota must use dataset_key=count syntax, got: {item!r}")
        dataset_key, count_text = item.split("=", 1)
        dataset_name = dataset_key.strip()
        if not dataset_name:
            raise ValueError(f"dataset_key must not be empty in quota: {item!r}")
        quotas[dataset_name] = int(count_text)
    return quotas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 300-image smoke review bundle from the final dataset manifest.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quota", action="append", default=None, help="Repeatable dataset_key=count override.")
    args = parser.parse_args(argv)
    quotas = _parse_quota_args(args.quota)
    summary = render_review_bundle(
        manifest_path=Path(args.manifest_path).resolve(),
        output_root=Path(args.output_root).resolve(),
        split=str(args.split),
        quotas=quotas,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
