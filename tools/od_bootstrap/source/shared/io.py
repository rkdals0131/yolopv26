from __future__ import annotations

import os
import shutil
from pathlib import Path

from common.io import ensure_parent_dir
from common.io import now_iso
from common.io import read_json as load_json
from common.io import write_json_sorted as write_json
from common.io import write_text


def link_or_copy(source_path: Path, target_path: Path) -> str:
    target = ensure_parent_dir(target_path)
    if target.exists():
        return "existing"
    try:
        os.link(source_path, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, target)
        return "copy"


__all__ = [
    "link_or_copy",
    "load_json",
    "now_iso",
    "write_json",
    "write_text",
]
