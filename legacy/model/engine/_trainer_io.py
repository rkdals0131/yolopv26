from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import append_jsonl_sorted as _common_append_jsonl_sorted
from common.io import now_iso as _common_now_iso
from common.io import timestamp_token as _common_timestamp_token
from common.io import write_json as _common_write_json
from common.io import write_jsonl_sorted as _common_write_jsonl_sorted
from common.train_runtime import maybe_build_summary_writer as _common_maybe_build_summary_writer


def _default_run_dir() -> Path:
    timestamp = _common_timestamp_token()
    return Path("runs") / "pv26_train" / f"pv26_fit_{timestamp}"


_now_iso = _common_now_iso


_write_json = _common_write_json


_append_jsonl = _common_append_jsonl_sorted


_write_jsonl_rows = _common_write_jsonl_sorted


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


_maybe_build_summary_writer = _common_maybe_build_summary_writer
