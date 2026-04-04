from __future__ import annotations

# Compatibility shim: keep legacy imports working while the canonical
# implementation lives in `tools.pv26_train.scenarios`.
from .scenarios import *  # noqa: F401,F403
from .scenarios import __all__  # noqa: F401
