from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.smoke.checkpoint_audit import write_checkpoint_audit


DEFAULT_OUTPUT_PATH = REPO_ROOT / "runs" / "od_bootstrap_smoke" / "checkpoint_audit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit smoke teacher checkpoints and alias state.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    summary = write_checkpoint_audit(Path(args.output).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
