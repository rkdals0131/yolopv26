from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable, TextIO


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
        self._emit(f"[{name}] {why}{total_text}")

    def progress(self, completed: int, total: int | None = None, detail: str | None = None) -> None:
        now = time.monotonic()
        effective_total = total if total is not None else self.stage_total
        if effective_total is not None:
            message = f"[{self.stage_name}] {completed}/{effective_total}"
        else:
            message = f"[{self.stage_name}] {completed}"
        if detail:
            message = f"{message} {detail}"
        if completed == 0 or completed == effective_total or now - self.last_progress_at >= self.throttle_seconds:
            self._emit(message)
            self.last_progress_at = now

    def heartbeat(self, detail: str | None = None) -> None:
        elapsed = time.monotonic() - self.stage_started_at
        message = f"[{self.stage_name}] still running ({elapsed:.1f}s)"
        if detail:
            message = f"{message} {detail}"
        self._emit(message)


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def parallel_chunk_size(total_tasks: int, workers: int) -> int:
    if total_tasks <= 0:
        return 1
    normalized_workers = max(1, workers)
    return max(1, min(16, (total_tasks + normalized_workers - 1) // normalized_workers))


def iter_task_chunks(tasks: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    effective_chunk_size = max(1, chunk_size)
    for start in range(0, len(tasks), effective_chunk_size):
        yield tasks[start : start + effective_chunk_size]


__all__ = [
    "LiveLogger",
    "default_workers",
    "iter_task_chunks",
    "parallel_chunk_size",
]
