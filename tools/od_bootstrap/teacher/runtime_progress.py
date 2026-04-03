from __future__ import annotations

from pathlib import Path
import time
from types import MethodType
from typing import Any

from common import io as common_io
from common.train_runtime import format_duration as _common_format_duration
from common.train_runtime import sync_timing_device as _common_sync_timing_device
from common.train_runtime import timing_profile as _common_timing_profile
try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
except Exception:  # pragma: no cover - optional dependency fallback.
    Console = None
    Progress = None
    BarColumn = None
    TaskProgressColumn = None
    TextColumn = None


def sync_timing_device(torch_module: Any, device: Any, enabled: bool) -> None:
    _common_sync_timing_device(torch_module, device, enabled)


def timing_profile(window: list[dict[str, float]]) -> dict[str, Any]:
    return _common_timing_profile(window, keys=("iteration_sec", "wait_sec", "compute_sec"))


append_jsonl = common_io.append_jsonl


def format_duration(seconds: float | None) -> str:
    return _common_format_duration(seconds, unavailable="unknown")


def _progress_console(*, console: Any = None) -> Any:
    if console is not None:
        return console
    if Console is None:
        return None
    return Console(stderr=True)


def rich_progress_available(*, console: Any = None, enabled: bool | None = None) -> bool:
    resolved_console = _progress_console(console=console)
    if resolved_console is None or Progress is None:
        return False
    if enabled is not None:
        return bool(enabled)
    return bool(getattr(resolved_console, "is_terminal", False))


class RichBootstrapProgressBar:
    def __init__(
        self,
        iterable: Any,
        *,
        total: int | None,
        description: str,
        console: Any = None,
    ) -> None:
        if Progress is None:
            raise RuntimeError("rich progress backend is unavailable")
        resolved_console = _progress_console(console=console)
        if resolved_console is None:
            raise RuntimeError("rich console backend is unavailable")
        self._iterable = iter(iterable)
        self.console = resolved_console
        self.desc = str(description)
        self.status = ""
        self.closed = False
        self._progress = Progress(
            TextColumn("{task.fields[description]}", markup=False),
            BarColumn(bar_width=10),
            TaskProgressColumn(),
            TextColumn("  |  "),
            TextColumn("{task.fields[status]}", markup=False),
            console=self.console,
            transient=False,
            auto_refresh=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("", total=total, description=self.desc, status=self.status)

    def __iter__(self) -> "RichBootstrapProgressBar":
        return self

    def __next__(self) -> Any:
        if self.closed:
            raise StopIteration
        try:
            item = next(self._iterable)
        except StopIteration:
            self.close()
            raise
        self._progress.update(self._task_id, advance=1)
        return item

    def set_description(self, value: str) -> None:
        self.desc = str(value)
        self._progress.update(self._task_id, description=self.desc)

    def set_bootstrap_postfix(self, postfix: str) -> None:
        self.status = str(postfix)
        self._progress.update(self._task_id, status=self.status)

    def set_postfix_str(self, postfix: str, refresh: bool = True) -> None:
        del refresh
        self.set_bootstrap_postfix(postfix)

    def write(self, message: str) -> None:
        self.console.print(message, soft_wrap=True, markup=False)

    def close(self) -> None:
        if self.closed:
            return
        self._progress.stop()
        self.closed = True


def build_rich_progress_bar(
    iterable: Any,
    *,
    total: int | None,
    description: str,
    console: Any = None,
    enabled: bool | None = None,
) -> RichBootstrapProgressBar | None:
    if not rich_progress_available(console=console, enabled=enabled):
        return None
    return RichBootstrapProgressBar(
        iterable,
        total=total,
        description=description,
        console=console,
    )


def emit_log(message: str, *, progress_bar: Any = None) -> None:
    if progress_bar is not None and hasattr(progress_bar, "write"):
        progress_bar.write(message)
        return
    print(message, flush=True)


timestamp_token = common_io.timestamp_token


def build_live_postfix(
    *,
    elapsed_sec: float,
    eta_sec: float | None,
    profile_summary: dict[str, Any],
) -> str:
    return "  |  ".join(
        [
            f"elapsed={format_duration(elapsed_sec)}",
            f"eta={format_duration(eta_sec)}",
            f"iter={profile_summary['iteration_sec']['mean'] * 1000.0:.1f}ms",
            f"wait={profile_summary['wait_sec']['mean'] * 1000.0:.1f}ms",
            f"compute={profile_summary['compute_sec']['mean'] * 1000.0:.1f}ms",
        ]
    )


def set_progress_postfix(pbar: Any, postfix: str) -> bool:
    if pbar is None:
        return False
    if hasattr(pbar, "set_bootstrap_postfix"):
        pbar.set_bootstrap_postfix(postfix)
        return True
    if hasattr(pbar, "set_postfix_str"):
        pbar.set_postfix_str(postfix, refresh=True)
        return True
    description = str(getattr(pbar, "desc", "") or "")
    base_description = description.split("  |  ")[0] if "  |  " in description else description
    merged_description = f"{base_description}  |  {postfix}" if base_description else postfix
    if hasattr(pbar, "set_description"):
        pbar.set_description(merged_description)
        return True
    if hasattr(pbar, "set_postfix"):
        pbar.set_postfix(profile=postfix)
        return True
    return False


def _render_progress_line(pbar: Any, *, final: bool, time_module: Any) -> str | None:
    if pbar.disable or (pbar.closed and not final):
        return None

    current_time = time_module.time()
    dt = current_time - pbar.last_print_t
    dn = pbar.n - pbar.last_print_n

    if not final and not pbar._should_update(dt, dn):
        return None

    if dt > pbar.MIN_RATE_CALC_INTERVAL:
        rate = dn / dt if dt else 0.0
        if rate < pbar.MAX_SMOOTHED_RATE:
            pbar.last_rate = pbar.RATE_SMOOTHING_FACTOR * rate + (1 - pbar.RATE_SMOOTHING_FACTOR) * pbar.last_rate
            rate = pbar.last_rate
    else:
        rate = pbar.last_rate

    if pbar.total and pbar.n >= pbar.total:
        overall_elapsed = current_time - pbar.start_t
        if overall_elapsed > 0:
            rate = pbar.n / overall_elapsed

    pbar.last_print_n = pbar.n
    pbar.last_print_t = current_time
    elapsed = current_time - pbar.start_t

    remaining_str = ""
    if pbar.total and 0 < pbar.n < pbar.total and elapsed > 0:
        est_rate = rate or (pbar.n / elapsed)
        remaining_str = f"<{pbar._format_time((pbar.total - pbar.n) / est_rate)}"

    if pbar.total:
        percent = (pbar.n / pbar.total) * 100
        n_str = pbar._format_num(pbar.n)
        t_str = pbar._format_num(pbar.total)
        if pbar.is_bytes and len(n_str) >= 2 and len(t_str) >= 2 and n_str[-2] == t_str[-2]:
            n_str = n_str.rstrip("KMGTPB")
    else:
        percent = 0.0
        n_str, t_str = pbar._format_num(pbar.n), "?"

    elapsed_str = pbar._format_time(elapsed)
    rate_str = pbar._format_rate(rate) or (pbar._format_rate(pbar.n / elapsed) if elapsed > 0 else "")
    bar = pbar._generate_bar()

    if pbar.total:
        if pbar.is_bytes and pbar.n >= pbar.total:
            progress_str = f"{pbar.desc}: {percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
        else:
            progress_str = f"{pbar.desc}: {percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
    else:
        progress_str = f"{pbar.desc}: {bar} {n_str} {rate_str} {elapsed_str}"

    return progress_str


def install_ultralytics_postfix_renderer(
    pbar: Any,
    *,
    time_module: Any = time,
    method_type: Any = MethodType,
) -> Any:
    if pbar is None or getattr(pbar, "_od_bootstrap_renderer_installed", False):
        return pbar
    if not hasattr(pbar, "_display"):
        return pbar

    def _display_with_bootstrap_postfix(self: Any, final: bool = False) -> None:
        progress_str = _render_progress_line(self, final=final, time_module=time_module)
        if progress_str is None:
            return
        postfix = str(getattr(self, "_od_bootstrap_postfix", "") or "").strip()
        try:
            if self.noninteractive:
                if postfix:
                    self.file.write(f"{progress_str} | {postfix}")
                else:
                    self.file.write(progress_str)
            else:
                prior_line_count = int(getattr(self, "_od_bootstrap_rendered_lines", 1))
                if prior_line_count > 1:
                    self.file.write("\r\033[1A\r\033[K")
                else:
                    self.file.write("\r\033[K")
                self.file.write(progress_str)
                if postfix:
                    self.file.write(f"\n\033[K{postfix}")
                    self._od_bootstrap_rendered_lines = 2
                else:
                    self._od_bootstrap_rendered_lines = 1
            self.file.flush()
        except Exception:
            pass

    def _set_bootstrap_postfix(self: Any, postfix: str) -> None:
        self._od_bootstrap_postfix = str(postfix)
        if not self.disable:
            self._display()

    pbar._od_bootstrap_postfix = ""
    pbar._od_bootstrap_renderer_installed = True
    pbar._od_bootstrap_rendered_lines = 1
    pbar._display = method_type(_display_with_bootstrap_postfix, pbar)
    pbar.set_bootstrap_postfix = method_type(_set_bootstrap_postfix, pbar)
    return pbar


def loader_profile_payload(loader: Any) -> dict[str, Any]:
    return {
        "batch_size": int(getattr(loader, "batch_size", 0) or 0),
        "num_workers": int(getattr(loader, "num_workers", 0) or 0),
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
        "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
        "prefetch_factor": getattr(loader, "prefetch_factor", None),
        "dataset_size": int(len(getattr(loader, "dataset", []))),
        "num_batches": int(len(loader)),
    }
