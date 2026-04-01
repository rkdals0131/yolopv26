from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_token(*, datetime_cls: Any = datetime) -> str:
    return datetime_cls.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    default: Any | None = None,
    sort_keys: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=default,
            sort_keys=sort_keys,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def append_jsonl(path: str | Path, payload: Any, *, ensure_ascii: bool = True) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=ensure_ascii) + "\n")
    return output_path


def write_text(path: str | Path, contents: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def link_or_copy(source_path: str | Path, target_path: str | Path) -> None:
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)
