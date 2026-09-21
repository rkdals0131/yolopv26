from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_optional_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value in {None, ""}:
        return None
    return resolve_path(value, base_dir=base_dir)


def resolve_latest_root(root: str | Path, *, must_exist: bool = True) -> Path:
    base_root = Path(root).resolve()
    if base_root.name == "latest":
        parent = base_root.parent
        if not parent.is_dir():
            if not must_exist:
                return base_root
            raise FileNotFoundError(parent)
        candidates = sorted((path for path in parent.iterdir() if path.is_dir()), key=lambda item: item.name)
        if candidates:
            return candidates[-1]
        if not must_exist:
            return base_root
        raise FileNotFoundError(f"no directories found under {parent}")
    if base_root.is_file():
        return base_root
    if base_root.is_dir():
        return base_root
    if not must_exist:
        return base_root
    raise FileNotFoundError(base_root)
