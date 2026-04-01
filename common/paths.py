from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_latest_root(root: str | Path, *, must_exist: bool = True) -> Path:
    base_root = Path(root).resolve()
    if base_root.is_file():
        return base_root
    if base_root.is_dir():
        return base_root
    if not must_exist:
        return base_root
    raise FileNotFoundError(base_root)

