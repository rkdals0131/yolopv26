from __future__ import annotations

from pathlib import Path
import site
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from tools.check_env import main

if __name__ == "__main__":
    raise SystemExit(main())
