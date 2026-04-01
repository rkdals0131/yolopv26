from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
USER_PATHS_CONFIG_PATH = REPO_ROOT / "config" / "user_paths.yaml"
USER_HYPERPARAMETERS_CONFIG_PATH = REPO_ROOT / "config" / "user_hyperparameters.yaml"


def _read_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_yaml(path)


def load_user_paths_config() -> dict[str, Any]:
    return _read_optional_yaml(USER_PATHS_CONFIG_PATH)


def load_user_hyperparameters_config() -> dict[str, Any]:
    return _read_optional_yaml(USER_HYPERPARAMETERS_CONFIG_PATH)


def nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_repo_path(value: str | Path | None, *, repo_root: Path = REPO_ROOT) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path.resolve()


def resolve_repo_paths(values: Iterable[str | Path] | None, *, repo_root: Path = REPO_ROOT) -> tuple[Path, ...]:
    if values is None:
        return ()
    return tuple(
        resolved
        for resolved in (resolve_repo_path(value, repo_root=repo_root) for value in values)
        if resolved is not None
    )

