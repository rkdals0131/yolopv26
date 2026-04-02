from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable, TextIO

PARALLEL_SUBMIT_LOG_INTERVAL = 5_000
PARALLEL_WAIT_HEARTBEAT_SECONDS = 15.0
PARALLEL_MAX_TASKS_PER_CHUNK = 16
PARALLEL_INFLIGHT_CHUNKS_PER_WORKER = 2


class LiveLogger:
    def __init__(self, stream: TextIO | None = None, throttle_seconds: float = 1.0) -> None:
        self.stream = stream or sys.stdout
        self.throttle_seconds = throttle_seconds
        self.stage_name = "idle"
        self.stage_started_at = time.monotonic()
        self.stage_total: int | None = None
        self.last_progress_at = 0.0

    def _emit(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.stream.write(f"[{timestamp}] {message}\n")
        self.stream.flush()

    def info(self, message: str) -> None:
        self._emit(message)

    def stage(self, name: str, why: str, total: int | None = None) -> None:
        self.stage_name = name
        self.stage_started_at = time.monotonic()
        self.stage_total = total
        self.last_progress_at = 0.0
        total_text = f", total={total}" if total is not None else ""
        self._emit(f"stage={name} 시작{total_text} | why={why}")

    def progress(self, completed: int, counters: dict[str, int], *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress_at < self.throttle_seconds:
            return
        elapsed = max(now - self.stage_started_at, 1e-6)
        rate = completed / elapsed
        eta_text = "eta=unknown"
        if self.stage_total is not None and completed > 0:
            remaining = max(self.stage_total - completed, 0)
            eta_seconds = remaining / rate if rate > 0 else 0.0
            eta_text = f"eta={eta_seconds:.1f}s"
        total_text = self.stage_total if self.stage_total is not None else "?"
        counters_text = " ".join(f"{key}={value}" for key, value in sorted(counters.items()))
        self._emit(
            f"stage={self.stage_name} progress={completed}/{total_text} rate={rate:.2f}/s {eta_text} {counters_text}".strip()
        )
        self.last_progress_at = now


def default_workers() -> int:
    available = os.cpu_count() or 4
    return max(1, min(32, available - 1))


def parallel_chunk_size(total_tasks: int, workers: int) -> int:
    if total_tasks <= 0:
        return 1
    target_inflight_tasks = max(1, workers * 32)
    return max(1, min(PARALLEL_MAX_TASKS_PER_CHUNK, (total_tasks + target_inflight_tasks - 1) // target_inflight_tasks))


def iter_task_chunks(tasks: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(tasks), chunk_size):
        yield tasks[start : start + chunk_size]


__all__ = [
    "LiveLogger",
    "PARALLEL_INFLIGHT_CHUNKS_PER_WORKER",
    "PARALLEL_MAX_TASKS_PER_CHUNK",
    "PARALLEL_SUBMIT_LOG_INTERVAL",
    "PARALLEL_WAIT_HEARTBEAT_SECONDS",
    "default_workers",
    "iter_task_chunks",
    "parallel_chunk_size",
]
