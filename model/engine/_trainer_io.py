from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from common.io import append_jsonl as _common_append_jsonl
from common.io import now_iso as _common_now_iso
from common.io import timestamp_token as _common_timestamp_token
from common.io import write_json as _common_write_json
from common.io import write_jsonl as _common_write_jsonl
from common.train_runtime import maybe_build_summary_writer as _common_maybe_build_summary_writer


def _default_run_dir() -> Path:
    timestamp = _common_timestamp_token()
    return Path("runs") / "pv26_train" / f"pv26_fit_{timestamp}"


def _now_iso() -> str:
    return _common_now_iso()


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    return _common_write_json(path, payload)


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    return _common_append_jsonl(path, payload, sort_keys=True)


def _write_jsonl_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    return _common_write_jsonl(path, rows, sort_keys=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _maybe_build_summary_writer(log_dir: Path, *, purge_step: int | None = None):
    return _common_maybe_build_summary_writer(log_dir, purge_step=purge_step)
