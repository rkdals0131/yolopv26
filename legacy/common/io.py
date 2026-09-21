from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_token(*, datetime_cls: Any = datetime) -> str:
    return datetime_cls.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [payload for _, payload in iter_jsonl(path)]


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return
    for line_index, raw_line in enumerate(read_text(input_path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        yield line_index, json.loads(line)


def read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def remove_path(path: str | Path) -> None:
    target_path = Path(path)
    if target_path.is_symlink() or target_path.is_file():
        target_path.unlink()
        return
    if target_path.is_dir():
        shutil.rmtree(target_path)


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    default: Any | None = None,
    sort_keys: bool = False,
    overwrite: bool = True,
) -> Path:
    output_path = ensure_parent_dir(path)
    if not overwrite and output_path.exists():
        raise FileExistsError(f"target path already exists: {output_path}")
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


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    default: Any | None = None,
    sort_keys: bool = False,
) -> Path:
    """Durably replace a run configuration without exposing partial JSON."""
    output_path = ensure_parent_dir(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii,
                      default=default, sort_keys=sort_keys)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def write_json_sorted(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    default: Any | None = None,
    overwrite: bool = True,
) -> Path:
    return write_json(
        path,
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
        sort_keys=True,
        overwrite=overwrite,
    )


def append_jsonl(path: str | Path, payload: Any, *, ensure_ascii: bool = True, sort_keys: bool = False) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=ensure_ascii, sort_keys=sort_keys) + "\n")
    return output_path


def append_jsonl_sorted(path: str | Path, payload: Any, *, ensure_ascii: bool = True) -> Path:
    return append_jsonl(path, payload, ensure_ascii=ensure_ascii, sort_keys=True)


def write_jsonl(path: str | Path, rows: Iterable[Any], *, ensure_ascii: bool = True, sort_keys: bool = False) -> Path:
    output_path = ensure_parent_dir(path)
    serialized = "\n".join(json.dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys) for row in rows)
    output_path.write_text((serialized + "\n") if serialized else "", encoding="utf-8")
    return output_path


def write_jsonl_sorted(path: str | Path, rows: Iterable[Any], *, ensure_ascii: bool = True) -> Path:
    return write_jsonl(path, rows, ensure_ascii=ensure_ascii, sort_keys=True)


def write_text(path: str | Path, contents: str) -> Path:
    output_path = ensure_parent_dir(path)
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def link_or_copy(source_path: str | Path, target_path: str | Path) -> None:
    """Replace the target with a symlink when possible, else copy the file."""

    source = Path(source_path)
    target = ensure_parent_dir(target_path)
    if target.exists() or target.is_symlink():
        remove_path(target)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)
