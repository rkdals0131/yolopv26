from __future__ import annotations

import importlib
from pathlib import Path
import sys


def import_modal_sdk():
    """Import the external Modal SDK from inside this repo's ./modal script dir."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    removed: list[tuple[int, str]] = []
    for index, entry in reversed(list(enumerate(sys.path))):
        raw = entry or "."
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        if resolved in {script_dir, repo_root}:
            removed.append((index, entry))
            sys.path.pop(index)
    local_modal = sys.modules.get("modal")
    if local_modal is not None:
        module_file = getattr(local_modal, "__file__", None)
        module_paths = [Path(path).resolve() for path in getattr(local_modal, "__path__", [])]
        if module_file is None and any(path == script_dir or path == repo_root / "modal" for path in module_paths):
            sys.modules.pop("modal", None)
    try:
        return importlib.import_module("modal")
    finally:
        for index, entry in sorted(removed):
            sys.path.insert(index, entry)


modal = import_modal_sdk()
