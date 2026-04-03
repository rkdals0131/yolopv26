from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from common.scalars import flatten_scalar_tree


TimingValueResolver = Callable[[Mapping[str, Any], str], Any]


def sync_timing_device(torch_module: Any, device: Any, enabled: bool) -> None:
    if not enabled or torch_module is None or not torch_module.cuda.is_available():
        return
    device_type = getattr(device, "type", None)
    if device_type != "cuda":
        return
    try:
        torch_module.cuda.synchronize(device)
    except Exception:
        torch_module.cuda.synchronize()


def quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def profile_stats(values: Iterable[float]) -> dict[str, float]:
    ordered = [float(value) for value in values]
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(ordered, 0.50),
        "p99": quantile(ordered, 0.99),
    }


def timing_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    keys: Iterable[str],
    value_resolver: TimingValueResolver | None = None,
) -> dict[str, Any]:
    items = list(records)
    if not items:
        return {"window_size": 0}
    resolve_value = value_resolver or (lambda record, key: record.get(key, 0.0))
    profile: dict[str, Any] = {"window_size": len(items)}
    for key in keys:
        profile[str(key)] = profile_stats(float(resolve_value(item, str(key)) or 0.0) for item in items)
    return profile


def format_duration(seconds: float | None, *, unavailable: str = "n/a") -> str:
    if seconds is None:
        return unavailable
    numeric = float(seconds)
    if not math.isfinite(numeric):
        return unavailable
    total_seconds = max(0, int(round(numeric)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def maybe_build_summary_writer(log_dir: Path, *, purge_step: int | None = None):
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


def write_tensorboard_scalars(writer: Any, prefix: str, payload: dict[str, Any], step: int) -> int:
    if writer is None:
        return 0
    count = 0
    for name, value in flatten_scalar_tree(prefix, payload):
        writer.add_scalar(name, value, global_step=int(step))
        count += 1
    return count

