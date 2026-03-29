from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.smoke.subset import build_smoke_image_list


DEFAULT_INPUT_MANIFEST_PATH = (
    REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "meta" / "bootstrap_image_list.jsonl"
)
DEFAULT_OUTPUT_MANIFEST_PATH = (
    REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "meta" / "bootstrap_image_list.smoke.jsonl"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the OD bootstrap smoke image-list subset.")
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST_PATH)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST_PATH)
    args = parser.parse_args(argv)
    summary = build_smoke_image_list(
        input_manifest_path=Path(args.input_manifest).resolve(),
        output_manifest_path=Path(args.output_manifest).resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
