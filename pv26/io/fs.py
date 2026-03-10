from __future__ import annotations

from pathlib import Path
from typing import List


def list_files_recursive(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p)
    return sorted(out)

