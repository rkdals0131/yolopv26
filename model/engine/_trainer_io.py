from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable


def _default_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "pv26_train" / f"pv26_fit_{timestamp}"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    return output_path


def _write_jsonl_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in rows)
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _maybe_build_summary_writer(log_dir: Path, *, purge_step: int | None = None):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional dependency.
        return None, {
            "enabled": False,
            "status": "unavailable",
            "error": str(exc),
            "log_dir": str(log_dir),
            "purge_step": purge_step,
        }
    try:
        writer_kwargs: dict[str, Any] = {"log_dir": str(log_dir)}
        if purge_step is not None:
            writer_kwargs["purge_step"] = int(purge_step)
        writer = SummaryWriter(**writer_kwargs)
    except Exception as exc:  # pragma: no cover - filesystem or environment issue.
        return None, {
            "enabled": False,
            "status": "init_failed",
            "error": str(exc),
            "log_dir": str(log_dir),
            "purge_step": purge_step,
        }
    return writer, {
        "enabled": True,
        "status": "active",
        "error": None,
        "log_dir": str(log_dir),
        "purge_step": purge_step,
    }
