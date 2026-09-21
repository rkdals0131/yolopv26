from __future__ import annotations

import time
from typing import Any, Callable

import torch
from common.train_runtime import join_status_segments, progress_meter

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
except Exception:  # pragma: no cover - optional dependency fallback.
    Console = None
    Group = None
    Live = None
    Text = None

_ProfileBuilder = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _progress_console() -> Any:
    if Console is None:
        return None
    return Console(stderr=True)


class _RichProgressBar:
    def __init__(self, *, total: int | None, desc: str) -> None:
        if Group is None or Live is None or Text is None:
            raise RuntimeError("rich progress backend is unavailable")
        console = _progress_console()
        if console is None:
            raise RuntimeError("rich console backend is unavailable")
        self.console = console
        self.total = total
        self.desc = str(desc)
        self.current = 0
        self.status = ""
        self.detail = ""
        self._live = Live(
            self._renderable(),
            console=self.console,
            transient=False,
            auto_refresh=False,
        )
        self._live.start()
        self._closed = False

    def _renderable(self) -> Any:
        lines = [
            self._styled_line(
                join_status_segments(
                    self.desc,
                    progress_meter(int(self.current), int(self.total) if self.total is not None else None, width=10),
                    self.status,
                )
            )
        ]
        if self.detail:
            for line in str(self.detail).splitlines():
                if not line:
                    continue
                detail_line = Text("  ")
                detail_line.append_text(self._styled_line(line))
                lines.append(detail_line)
        return Group(*lines)

    def _styled_line(self, line: str) -> Text:
        output = Text(no_wrap=False)
        for index, segment in enumerate(part for part in str(line).split("  |  ") if part):
            if index > 0:
                output.append("  |  ")
            output.append_text(self._styled_segment(segment))
        return output

    def _styled_segment(self, segment: str) -> Text:
        output = Text()
        value = str(segment)
        if not value:
            return output
        output.append(value)
        return output

    def _refresh(self) -> None:
        if self._closed:
            return
        self._live.update(self._renderable(), refresh=True)

    def update(self, advance: int = 1) -> None:
        self.current += int(advance)
        self._refresh()

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        self.status = str(value)
        if refresh:
            self._refresh()

    def set_detail(self, value: str) -> None:
        self.detail = str(value)
        self._refresh()

    def write(self, message: str) -> None:
        self.console.print(message, soft_wrap=True, markup=False)
        self._refresh()

    def close(self) -> None:
        if self._closed:
            return
        self._live.stop()
        self._closed = True


def _should_use_rich_progress() -> bool:
    console = _progress_console()
    return bool(
        console is not None
        and getattr(console, "is_terminal", False)
        and Group is not None
        and Live is not None
        and Text is not None
    )


def build_progress_bar(*, total: int | None, desc: str) -> _RichProgressBar | None:
    if not _should_use_rich_progress():
        return None
    return _RichProgressBar(total=total, desc=desc)


def safe_len(loader: Any) -> int | None:
    try:
        return int(len(loader))
    except Exception:
        return None


def next_loader_batch(loader_iter: Any) -> tuple[bool, Any, float]:
    fetch_started_at = time.perf_counter()
    try:
        batch = next(loader_iter)
    except StopIteration:
        return False, None, 0.0
    fetch_ended_at = time.perf_counter()
    return True, batch, max(0.0, fetch_ended_at - fetch_started_at)


def update_timing_window(window: list[dict[str, Any]], item: dict[str, Any], *, profile_window: int) -> list[dict[str, Any]]:
    window.append(item)
    return window[-max(1, int(profile_window)) :]


def summarize_progress(
    *,
    started_at: float,
    batch_index: int,
    total_batches: int | None,
    timing_window: list[dict[str, Any]],
    profile_builder: _ProfileBuilder,
) -> tuple[dict[str, Any], float, float | None]:
    elapsed_sec = max(0.0, time.perf_counter() - started_at)
    remaining_batches = None
    if total_batches is not None:
        remaining_batches = max(0, int(total_batches) - int(batch_index))
    profile_summary = profile_builder(list(timing_window))
    eta_sec = None
    if remaining_batches is not None:
        eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_batches)
    return profile_summary, elapsed_sec, eta_sec


def should_log_progress(*, batch_index: int, total_batches: int | None, log_every_n_steps: int) -> bool:
    should_log = int(batch_index) % max(1, int(log_every_n_steps)) == 0
    if total_batches is not None and int(batch_index) == int(total_batches):
        should_log = True
    return should_log


def emit_progress_message(message: str, *, progress_bar: Any = None) -> None:
    if progress_bar is not None and hasattr(progress_bar, "set_detail"):
        progress_bar.set_detail(message)
        return
    if progress_bar is not None and hasattr(progress_bar, "write"):
        progress_bar.write(message)
        return
    print(message, flush=True)


def sync_profile_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "build_progress_bar",
    "emit_progress_message",
    "next_loader_batch",
    "safe_len",
    "should_log_progress",
    "summarize_progress",
    "sync_profile_device",
    "update_timing_window",
]
