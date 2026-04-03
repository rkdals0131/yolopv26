from __future__ import annotations

from typing import Any

import torch

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
except Exception:  # pragma: no cover - optional dependency fallback.
    Console = None
    Group = None
    Live = None
    Text = None

from .trainer_reporting import _join_segments, _progress_meter


def _progress_console() -> Any:
    if Console is None:
        return None
    return Console(stderr=True)


def _should_use_rich_progress() -> bool:
    console = _progress_console()
    return bool(
        console is not None
        and getattr(console, "is_terminal", False)
        and Group is not None
        and Live is not None
        and Text is not None
    )


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
                _join_segments(
                    self.desc,
                    _progress_meter(int(self.current), int(self.total) if self.total is not None else None, width=10),
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


def _emit_progress_message(message: str, *, progress_bar: Any = None) -> None:
    if progress_bar is not None and hasattr(progress_bar, "set_detail"):
        progress_bar.set_detail(message)
        return
    if progress_bar is not None and hasattr(progress_bar, "write"):
        progress_bar.write(message)
        return
    print(message, flush=True)


def _sync_profile_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
