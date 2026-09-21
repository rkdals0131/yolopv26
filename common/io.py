from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return []
    with input_path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


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


def write_jsonl(path: str | Path, rows: Iterable[Any], *, ensure_ascii: bool = True, sort_keys: bool = False) -> Path:
    output_path = ensure_parent_dir(path)
    serialized = "\n".join(json.dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys) for row in rows)
    output_path.write_text((serialized + "\n") if serialized else "", encoding="utf-8")
    return output_path


def write_jsonl_sorted(path: str | Path, rows: Iterable[Any], *, ensure_ascii: bool = True) -> Path:
    return write_jsonl(path, rows, ensure_ascii=ensure_ascii, sort_keys=True)
