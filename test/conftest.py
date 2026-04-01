from __future__ import annotations

import site
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(REPO_ROOT))
