#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PV26_MODAL_PROFILE", "a100")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.modal.launcher import _main, app, modal_entrypoint, train_remote  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(_main())
