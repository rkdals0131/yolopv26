from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path
import site
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    site.addsitedir(str(REPO_ROOT))

_module = _import_module("tools.pv26_eval_harness")

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
