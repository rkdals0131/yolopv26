from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from common.io import now_iso as _common_now_iso
from common.io import write_json as _common_write_json
from common.io import write_text as _common_write_text


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return _common_now_iso()


def write_text(path: Path, contents: str) -> None:
    _common_write_text(path, contents)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _common_write_json(path, payload, sort_keys=True)


def link_or_copy(source_path: Path, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return "existing"
    try:
        os.link(source_path, target_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"


__all__ = [
    "link_or_copy",
    "load_json",
    "now_iso",
    "write_json",
    "write_text",
]
