from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path
import site
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

_module = _import_module("tools.pv26_train.cli")
if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
