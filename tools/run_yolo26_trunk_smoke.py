#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.trunk import build_yolo26n_trunk
from model.trunk import summarize_trunk_adapter


def main() -> int:
    adapter = build_yolo26n_trunk()
    print(json.dumps(summarize_trunk_adapter(adapter), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
